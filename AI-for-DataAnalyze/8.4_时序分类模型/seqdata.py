# -*- coding: utf-8 -*-
"""
seqdata.py —— 原始 96Hz 惯性序列的数据集构建（供 02_LSTM / 03_Transformer 共用）
输出：X (N, WIN, C) 序列、y 标签、groups 受试者编号（做留一受试者交叉验证用）
"""
from common import *

CACHE = os.path.join(OUT, "seq_dataset.npz")

# 样本集签名：01b / 02 / 03 三个模型都从这份序列建样本，用同一个签名，
# 8.5 才敢把它们放进同一张 ROC 图。01_random_forest 读的是另一张表（每人一行），
# 签名不同，8.5 会把它们分开比 —— 否则就是拿两批不相干的样本算指标。
SEQ_SIG = f"原始序列表(仅行走={ONLY_WALK}, 部位={BODY_PART}, 窗口={WIN})"

# ---------- torch 安全导入（02 / 03 / 04 共用这一份）----------
# ★ 不能在脚本顶层裸写 import torch：平台上一旦没装 torch，那会在**模块加载阶段**
#   就抛 ModuleNotFoundError，用户看到的是个跟数据和用法都无关的报错；而脚本里那些
#   「没数据就 [SKIP] 友好退出」全在下游，根本轮不到执行。
#   放在这里一份，三个脚本共用；torch=None 时各脚本自己打印提示再退出。
try:
    import torch, torch.nn as nn
except ImportError:
    torch = nn = None


def torch_skip(what):
    """没装 torch 时的统一提示。返回 True 表示「该跳过」。"""
    if nn is not None:
        return False
    print(f"[SKIP] 没装 torch，{what} 需要 PyTorch。")
    print("       装法：pip install torch    （离线环境怎么装见邮件里说的办法）")
    print("       其余脚本（01 / 01b）不需要 torch，不受影响。")
    return True


def list_sheets_safe(folder, prefix):
    """list_sheets 的容错版：读不动工作簿时返回 []，而不是把整条流程带崩。

    ★ 真实数据上「表能打开」和「sheet 能列出来」不是一回事（文件被占用、
      权限不足、xlsx 损坏、缺 openpyxl），而 seqdata 里读 sheet 的 try
      包不住列 sheet 这一步 —— 一崩就整个 build 失败。
    """
    try:
        return list_sheets(folder, prefix)
    except Exception as e:
        print(f"  [跳过] 无法列出 sheet（{type(e).__name__}: {e}）")
        return []


def cache_signature(raw_path):
    """缓存里记的「这批序列是怎么来的」。

    ★ 只记表路径是不够的：WIN / BODY_PART / ONLY_WALK / RAW_FS 任何一项改了，
      序列的内容就完全不同了，但表还是那张表 —— 于是缓存不会失效，
      01b/02/03 会继续吃按旧设置建的序列。样本数、AUC 全都"正常"，看不出来。
      对 03_transformer 更狠：位置编码按 WIN 开的形状，X 却还是旧长度，
      跑起来是 forward 里的一句广播 RuntimeError，跟真正原因隔了十万八千里。
      所以签名要把这些参数一起写进去，改了就自动重建。
    """
    return "|".join([raw_path, f"WIN={WIN}", f"PART={BODY_PART}",
                     f"WALK={ONLY_WALK}", f"FS={RAW_FS}"])


def build_seq_dataset(force=False):
    """扫描原始数据的所有 sheet，拼成深度学习可用的序列数据集（带本地缓存）。"""
    # ★ 先只解析一次表路径：缓存比对和写缓存都要用它。
    #   原来两处各自调 find_file，写缓存那处没有 try —— 一旦发生在写缓存那一刻
    #   找不到表（比如数据盘中途被卸载），前面训练全白跑，还抛一个跟原因无关的错。
    try:
        raw_path = find_file(DIR_RAW, RAW_MAIN)
    except FileNotFoundError:
        raw_path = ""
    sig = cache_signature(raw_path)

    if os.path.exists(CACHE) and not force:
        d = np.load(CACHE, allow_pickle=True)
        old = str(d["src"]) if "src" in d else ""
        if old and old == sig:
            print(f"[缓存] 载入 {CACHE}，形状 {d['X'].shape}")
            return d["X"], d["y"], d["g"]
        print("[缓存] 作废并重建 —— 建这份缓存的条件变了：")
        print(f"       旧：{old or '（这份缓存没记来源，按旧版处理）'}")
        print(f"       新：{sig}")

    if skip_if_missing(
            DIR_RAW, RAW_MAIN, "帕金森病患者与健康老年人行走运动参数表（原始 96Hz 序列）"):
        return None, None, None

    sheets = list_sheets_safe(DIR_RAW, RAW_MAIN)
    if ONLY_WALK:                       # 本章做步态分析，剔除闭眼站立组次
        sheets = [s for s in sheets if "行走" in str(s) or "WALK" in str(s).upper()]
    print(f"[序列] 候选 sheet {len(sheets)} 个")

    X, y, g = [], [], []
    for s in sheets:
        lab = sheet_label(s)
        if lab is None:
            print(f"  [跳过] 无法从「{s}」判定分组")
            continue
        try:
            X.append(load_sequence(DIR_RAW, RAW_MAIN, s))
            y.append(lab); g.append(sheet_subject(s))
        except Exception as e:
            print(f"  [跳过] {s}: {type(e).__name__}: {e}")

    if not X:
        # ★ 一个序列都没读成。（2026-09-25 实测：原始表里可能只有「闭眼站立」
        #   这类其它任务，ONLY_WALK=True 一过滤就空了。）
        #   这里必须拦住，否则 np.stack([]) 会抛 ValueError，
        #   而那个报错看不出真正原因。注意此时**不能**写缓存。
        print(f"\n[SKIP] 这张表里没有可用于序列建模的组次（读成序列的：0 个）")
        print(f"       过滤后候选 sheet {len(sheets)} 个，ONLY_WALK={ONLY_WALK}")
        if ONLY_WALK:
            print(f"       如果这张表里只有「闭眼站立」等其它任务，把 common.py 里的")
            print(f"       ONLY_WALK 改成 False，就会改用全部组次。")
        print(f"       先跑 python 00_inspect.py，看「sheet 名解析自检」那一段。")
        return None, None, None

    X = np.stack(X); y = np.array(y, dtype="int64"); g = np.array(g)

    # ★ 全零序列自检：'跳过' 与否的边界在这里。读表读成空、部位筛没筛中、
    #   数值列全被判成非数值 —— 任何一条都会让某段序列变成全零，
    #   而零填充不会报错，模型照样训练、图照样画，只是数据是假的。
    flat = X.reshape(X.shape[0], -1)
    n_zero = int((np.abs(flat).max(axis=1) == 0).sum())
    if n_zero:
        print(f"[序列] ⚠️ 有 {n_zero}/{X.shape[0]} 段序列全为 0（不是真实信号）。")
        print(f"       多半是 sheet 里筛不到 BODY_PART={BODY_PART!r} 的行，"
              f"或该段数值列没读到数。")
    if n_zero == X.shape[0]:
        print("\n[SKIP] 所有序列都是全零 —— 这等于在假数据上训练，不往下走。")
        print(f"       先跑 python 00_inspect.py 看「BodyPart 取值」那一行，")
        print(f"       再把 common.py 里的 BODY_PART 改成实际出现的部位名。")
        return None, None, None

    # ★ 只有一类就停下：单类标签下 AUC 会静默变 nan，而 Accuracy 还是 1.0，
    #   图照画、退出码 0 —— 后面所有结论都是空的。（详见 common.check_two_classes）
    check_two_classes(y, "原始序列表（sheet 名解析出来的分组）")

    np.savez_compressed(CACHE, X=X, y=y, g=g, src=sig)
    print(f"[序列] 建成数据集 X={X.shape}  PD={int(y.sum())}/HC={int((1-y).sum())}  "
          f"受试者 {len(set(g))} 名")
    print(f"[序列] 通道数 C={X.shape[2]}，窗口 {WIN} 点 ≈ {WIN / RAW_FS:.2f} s")
    return X, y, g


def make_loader(X, y, idx, batch=32, shuffle=True):
    """把 numpy 切片包成 PyTorch DataLoader。"""
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx], dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def get_device():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[设备] {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    return dev


# ---------- 手工特征提取（8.4 路径一：与 LSTM/Transformer 用同一批样本，保证可比）----------
from scipy.stats import skew, kurtosis
from scipy.signal import welch

TD_NAMES = ["均值", "标准差", "能量", "熵值", "最大值", "最小值",
            "范围", "偏度", "峭度", "四分位距"]
FD_NAMES = ["平均频率", "峰值频率", "重心频率", "均方根频率", "频率标准差",
            "冻结带功率3-8Hz", "运动带功率0.5-3Hz", "FI指数"]


def _td_feats(v):
    """时域特征（对应教材表 4）。"""
    p = np.abs(v) / (np.abs(v).sum() + 1e-12)
    # ★ 偏度/峭度要防常数序列：某轴全程不动的通道（std=0）会让 scipy 算出 nan，
    #   而 nan 传进随机森林会直接报 "Input contains NaN"，报错位置离真正原因很远。
    sk, ku = skew(v), kurtosis(v)
    sk = 0.0 if not np.isfinite(sk) else float(sk)
    ku = 0.0 if not np.isfinite(ku) else float(ku)
    return [v.mean(), v.std(), np.mean(v ** 2), -np.sum(p * np.log(p + 1e-12)),
            v.max(), v.min(), np.ptp(v), sk, ku,
            np.percentile(v, 75) - np.percentile(v, 25)]


def _fd_feats(v, fs=None):
    """频域特征（对应教材表 5），含帕金森关键的 FI 冻结指数。"""
    fs = fs or RAW_FS
    f, psd = welch(v, fs=fs, nperseg=min(256, len(v)))
    tot = psd.sum() + 1e-12
    mf = psd.mean()
    pf = f[np.argmax(psd)]
    cf = float((f * psd).sum() / tot)
    rmsf = float(np.sqrt((f ** 2 * psd).sum() / tot))
    sf = float(np.sqrt(((f - cf) ** 2 * psd).sum() / tot))
    frozen = float(psd[(f >= 3) & (f <= 8)].sum())      # 冻结带 3-8Hz（震颤）
    motion = float(psd[(f >= 0.5) & (f <= 3)].sum())    # 运动带 0.5-3Hz（步态主频）
    fi = frozen / motion if motion > 1e-12 else 0.0
    return [mf, pf, cf, rmsf, sf, frozen, motion, fi]


def extract_handcrafted(X, fs=None):
    """把 (N, T, C) 序列变成 (N, C×(时域10+频域8)) 的手工特征矩阵。"""
    feats = []
    for i in range(X.shape[0]):
        row = []
        for c in range(X.shape[2]):
            v = np.asarray(X[i, :, c], dtype=float)
            row += _td_feats(v) + _fd_feats(v, fs)
        feats.append(row)
    names = [f"{ch}_{n}" for ch in [f"ch{c}" for c in range(X.shape[2])]
             for n in TD_NAMES + FD_NAMES]
    return np.asarray(feats, dtype="float32"), names


# ---------- 统一的训练/评估循环（LSTM 与 Transformer 共用）----------
def train_torch_model(model, X, y, g, epochs=15, lr=1e-3, batch=32, tag="model"):
    """留一受试者交叉验证（LOSO）：每组受试者轮流当测试集，避免同一人泄露。"""
    import torch, torch.nn as nn
    torch.manual_seed(42)               # 固定随机种子，保证结果可复现
    dev = get_device()
    cv = make_cv(y, g)                  # 受试者多时取 5 折，少时全留一
    print(f"[划分] {type(cv).__name__}  {len(set(g))} 名受试者 -> {cv.get_n_splits()} 折")

    oof_prob = np.zeros(len(y))
    for k, (tr, te) in enumerate(cv.split(X, y, g), 1):
        trl = make_loader(X, y, tr, batch, True)
        net = model().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            net.train()
            for xb, yb in trl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward(); opt.step()
            print(f"  [{tag}] fold{k} epoch{ep + 1}/{epochs} loss={loss.item():.4f}")
        net.eval()
        with torch.no_grad():       # 整折一次性推理，顺序与标签严格对齐
            oof_prob[te] = torch.sigmoid(
                net(torch.tensor(X[te]).to(dev))).cpu().numpy()

    return oof_prob
