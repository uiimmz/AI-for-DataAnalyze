# -*- coding: utf-8 -*-
"""
04_trajectory_demo.py —— 8.4 后半段：纵向时序模型（轨迹预测与趋势分析）轻量 demo
思路：取一位受试者行走过程中的关节三维位置序列，用前 70% 步预测后 30% 步，
      与「线性外推」基线对比，用 RMSE 和 DTW 距离评价预测质量。
数据：year2帕金森病患者与健康老年人行走运动参数数据（原始 96Hz）
跑法：python 04_trajectory_demo.py
"""
from common import *
from seqdata import get_device, torch, nn, torch_skip

# ★ BODY 是「优先用哪个部位」，不是「必须」。原始表实拍到的列名是 `Hips-Joint-Posi-x/y/z`
#   （诺亦腾导出，部位写在列名里），所以默认值改成 Hips、与 common.py 的 BODY_PART 一致。
#   万一表里没有这个部位，脚本会用表里实际有的部位并把图题改成实际部位名 ——
#   而不是照样顶着 "RightUpLeg" 的标题去画 Hips 的曲线（那种图进了教材就是错的）。
RATIO, BODY = 0.7, "Hips"
# 位置列名各版本不一，按正则认。
# ★ 2026-09-25 按原始表实拍的表头改过：真实列名是 `Hips-Joint-Posi-x`
#   （部位-Joint-物理量-轴，**全程用连字符**）。原来只认 `Posi_[xyz]`，
#   即要求 Posi 和轴之间是下划线 —— 于是真实表里一列都匹配不上，
#   加载轨迹直接报「找不到关节位置列（Posi_x/y/z）」并中止。
#   现在下划线和连字符都认，也可以没有分隔符（`POSx` 这种写法）——
#   与 common.py 的 _channel_cols 保持一致，免得同一份表在两个脚本里认出的东西不一样。
#   注意**不能**把 Velo（关节速度）也算进来：
#   那份表里 Posi 和 Velo 紧挨着列，认宽了就会拿速度当坐标用。
POSI_RE = r"(?:Posi|Position|Pos|关节位置)[_\-]?([xyz])$"


def load_track(folder=DIR_RAW, prefix=RAW_MAIN, body=BODY):
    """取一个「行走」sheet 里该身体部位的三维位置轨迹，返回 ((T, 3), 部位名)。

    部位名是**实际取到的**那个（列名里没有 body 时就是表里有的部位），
    给图题用 —— 免得顶着 "RightUpLeg" 的标题画 Hips 的曲线。


    ★ 三处以前是错的：
      1) 每试一个 sheet 就 pd.read_excel **整表**读一遍 —— 115 个 sheet 的表要读上百次，
         真实数据上能跑十几分钟，而且界面上看不出它在干嘛。
         现在第一遍只读 3 行去找「哪张 sheet 有位置列」，只把那一张整表读进来。
         找不到时的代价也从「全量读几十张表」降到「每张读 3 行」。
      2) 不限定任务。本章是步态分析，闭眼站立的轨迹不该拿来当「行走轨迹」，
         而 sheet 顺序上它往往排在前面，于是静默用了站立数据。现在优先挑含「行走」的。
      3) 一个 sheet 里没有这个部位就直接 continue，连「表里到底有哪些列」都不打印，
         最后只剩一句「找不到位置列」。现在把整份列名打出来 —— 一次运行就能定位。
    """
    xl = pd.ExcelFile(find_file(folder, prefix))
    names = [s for s in xl.sheet_names if s != xl.sheet_names[0]]
    walk = [s for s in names if "行走" in str(s) or "WALK" in str(s).upper()]
    if not walk:
        print("[轨迹] ⚠️ 表里没有含「行走」的 sheet —— 下面用的不是步态数据，"
              "图只能当版式示意。")

    # 第一遍：只读表头，找出哪张 sheet 有位置列（避免为了找列名把几十张表全读一遍）
    seen_cols, hit = [], None
    for s in (walk or names):
        try:
            head = pd.read_excel(xl, sheet_name=s, nrows=3)
        except Exception:
            continue
        if not head.shape[1]:
            continue
        cols = [c for c in head.columns if re.search(POSI_RE, str(c), re.I)]
        if cols:
            # 优先取 body 那个部位；表里没有就用实际有的（把部位名带回去当图题）
            pref = [c for c in cols if str(body).strip().lower() in str(c).lower()]
            use = sorted(map(str, pref or cols))
            got = use[0].split("-")[0] if "-" in use[0] else ""
            hit = (s, use, (str(body) if pref else got))
            break
        seen_cols.append((s, [str(c) for c in head.columns]))

    if hit is None:
        print(f"\n[轨迹] 没有任何 sheet 含位置列（正则 {POSI_RE}）。")
        print(f"       body 参数={body!r}")
        for s, cs in seen_cols[:2]:
            print(f"\n       例 sheet「{s}」的全部 {len(cs)} 列：")
            for i, c in enumerate(cs):
                print(f"         [{i:>3}] {c!r}")
        print("\n       把上面这些列名贴回来，我据此改 POSI_RE / BODY。")
        raise ValueError("找不到关节位置列（形如 Hips-Joint-Posi-x 或 Posi_x）")

    s, cols, label = hit
    df = pd.read_excel(xl, sheet_name=s)
    if "BodyPart" in df.columns:                 # 长表结构：按身体部位筛行
        seen_parts = sorted(map(str, df["BodyPart"].dropna().unique()))
        df = df[df["BodyPart"].astype(str) == body]
        if df.empty:
            print(f"\n[轨迹] sheet「{s}」里没有 BodyPart == {body!r} 的行。"
                  f"实际部位：{seen_parts[:20]}")
            print("       按需改 common.py 的 BODY_PART。")
            raise ValueError(f"sheet「{s}」里没有部位 {body!r}")
        label = str(body)                        # 长表：部位由行筛出来的，图题用它
    num = mask_lost(df, df[cols[:3]].apply(pd.to_numeric, errors="coerce"))
    t = num.interpolate().fillna(0).values
    report_quality(t, f"轨迹 sheet「{s}」")
    print(f"[轨迹] sheet「{s}」部位 {label}，{t.shape[0]} 帧，列 {cols[:3]}")
    return t.astype("float32"), label


DTW_MAX = 300          # DTW 的最长序列长度，超了就抽稀


def _decimate(a, cap=DTW_MAX):
    """序列太长就等间隔抽稀（位置量本身不受抽稀影响，不需要换算回去）。"""
    a = np.asarray(a, dtype=float)
    if len(a) <= cap:
        return a
    idx = np.round(np.arange(cap) * (len(a) / float(cap))).astype(int)
    return a[idx.clip(0, len(a) - 1)]


def dtw(a, b):
    """动态时间规整距离（已按路径长度归一，即「每步平均代价」）。

    ★ 必须限长。一次行走试次按 96Hz 录几分钟就是上万帧，n·m 上亿次循环 ——
      这个 demo 会跑几个小时，而且界面上看不出它在干嘛。抽稀到 300 点后
      代价固定在 9 万次，对「两条曲线形状有多像」这个用途足够。
      归一化用 /max(n,m) 已经是「每步平均距离」，与序列长度无关，
      所以抽稀前后是同一个尺度，两个方法之间仍然可比 —— 不需要再乘抽稀倍数。
    """
    a, b = _decimate(a), _decimate(b)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = np.linalg.norm(a[i - 1] - b[j - 1]) + min(D[i - 1, j],
                                                               D[i, j - 1],
                                                               D[i - 1, j - 1])
    return D[n, m] / max(n, m)


class TrajLSTM(nn.Module):
    """序列到序列：编码历史轨迹，回归未来轨迹。"""

    def __init__(self, c=3, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(c, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden, c)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.out(h)          # (B, T, 3) 逐步预测下一时刻


if nn is None:                      # 没装 torch：类定义不了，占个位，main 里会先退出
    TrajLSTM = None


if __name__ == "__main__":
    if skip_if_missing(DIR_RAW, RAW_MAIN, "行走运动参数表（轨迹预测要用的关节位置序列）"):
        sys.exit(0)
    if torch_skip("04_trajectory_demo.py"):
        sys.exit(0)
    try:
        t, body_label = load_track()
    except ValueError as e:
        # load_track 已经把「实际列名/部位」都打印出来了，再甩一屏 traceback 只是噪音。
        print(f"\n[终止] {e}")
        print("       这个脚本要的是关节三维位置（Posi/POS 之类）序列；")
        print("       如果原始表里确实没有位置字段，本章的轨迹预测就做不了 ——")
        print("       把 00_inspect.py 报告里原始表的列名贴回来，我确认后再定。")
        sys.exit(2)
    k = int(len(t) * RATIO)
    hist, future = t[:k], t[k:]
    if len(hist) < 10 or len(future) < 2:
        print(f"[SKIP] 轨迹太短（共 {len(t)} 帧，历史 {len(hist)} / 待预测 {len(future)}），"
              f"不够做预测对比。")
        sys.exit(0)
    print(f"[轨迹] 历史 {len(hist)} 帧 -> 滚动预测后 {len(future)} 帧")
    torch.manual_seed(42)       # ★ 固定种子：不固定的话这张图和 RMSE 每次都不一样
    dev = get_device()

    # --- 模型预测：用历史轨迹训练逐步预测器 ---
    net = TrajLSTM().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xb = torch.tensor(hist[:-1][None]).to(dev)      # (1, T-1, 3)
    yb = torch.tensor(hist[1:][None]).to(dev)       # (1, T-1, 3)
    for ep in range(200):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(xb), yb)
        loss.backward(); opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  [轨迹] epoch{ep + 1}/200 loss={loss.item():.5f}")

    # ★ 这里是本脚本最容易错的地方，原来就错了：
    #   net(hist) 的每个输出是「预测下一个时刻」，第 i 个输出预测的是 hist[i+1]。
    #   原来的写法 pred = net(hist)[-len(future):] 取的是**最后 len(future) 个输出**，
    #   也就是「对 hist 尾部那些已知点的拟合」，而 future 是 k 之后的未知点 ——
    #   两者几乎没有重叠，RMSE/DTW 曲线和表格比的根本不是预测准不准，
    #   但图上看起来照样像回事。正确做法是**自回归滚动**：
    #   用已知历史起步，每步把模型自己的输出接回输入，一步步预测到 future 的末尾。
    net.eval()
    with torch.no_grad():
        seq = torch.tensor(hist[None]).to(dev)
        outs = []
        for _ in range(len(future)):
            nxt = net(seq)[:, -1:]                  # 只取最后一步的下一步预测
            outs.append(nxt)
            seq = torch.cat([seq, nxt], dim=1)      # 预测值接回输入（自回归）
        pred = torch.cat(outs, dim=1).cpu().numpy()[0]
    # 预测与真实必须一一对应。不写这句的话，将来谁改动了滚动逻辑，
    # 又会静默地拿两段对不上的序列去算 RMSE —— 数字照样出得来。
    assert pred.shape == future.shape, \
        f"预测形状 {pred.shape} 与真实形状 {future.shape} 不一致，不能直接比"

    # --- 基线：线性外推 ---
    v = np.diff(hist, axis=0)[-5:].mean(0)          # 用末端平均速度外推
    base = hist[-1] + v * np.arange(1, len(future) + 1)[:, None]

    rows = [{"方法": "LSTM 序列预测", "RMSE": np.sqrt(np.mean((pred - future) ** 2)),
             "DTW": dtw(pred, future)},
            {"方法": "线性外推基线", "RMSE": np.sqrt(np.mean((base - future) ** 2)),
             "DTW": dtw(base, future)}]
    save_metrics(rows, "8.4_轨迹预测对比.csv")

    # --- 可视化：三维轨迹 + 误差曲线 ---
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(*hist.T, color="#2E6F9E", lw=2, label="历史（前70%）")
    ax.plot(*future.T, color="#4CAF50", lw=2, label="真实（后30%）")
    ax.plot(*pred.T, color="#C0504D", lw=2, ls="--", label="LSTM 预测")
    ax.set_title(f"{body_label} 三维轨迹预测"); ax.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(np.linalg.norm(pred - future, axis=1), label="LSTM 预测误差")
    ax2.plot(np.linalg.norm(base - future, axis=1), label="线性外推误差")
    ax2.set_xlabel("预测步（帧）"); ax2.set_ylabel("三点欧氏误差")
    ax2.set_title("逐帧预测误差对比"); ax2.legend()
    savefig(fig, "8.4_轨迹预测.png")
    print("\n完成。结果已存到 results/")
