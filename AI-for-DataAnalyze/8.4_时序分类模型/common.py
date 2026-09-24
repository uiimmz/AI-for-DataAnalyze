# -*- coding: utf-8 -*-
"""
common.py —— 8.4 / 8.5 全部脚本的公用配置与工具函数
放在本目录下，其它脚本用  from common import *  引入。
★ 只改下面 CONFIG 区即可，其余代码不用动。
"""
import os, re, sys, glob
# Windows 控制台默认 GBK，遇到编码不了的字符会抛异常中断脚本；
# 改成「替换」而不是「报错」，保证长跑脚本不会因为一个符号崩掉。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====================== CONFIG 配置区 ======================
# 数据根目录：自动探测，Windows 虚拟桌面(Z:) 和 Linux 远程分析平台(/eaas/...) 通用。
# 想手动指定就设环境变量：export PD_DATA_ROOT="/eaas/default/groups/.../share/二期-多模态帕金森..."
# 注：不再按「项目文件夹叫什么名字」找，改成按「里面有没有 year*/表1_」找，
#     所以「退行性 / 退化性」这类写法差异、以及 zip 解压后多一层少一层，都能自动适应。

# 远程分析平台上你账号的数据目录（已知的确定位置，放前面优先命中）。
# ★ 2026-09-25 按平台实际报错里的绝对路径更正：
#   /eaas/default/groups/casestudy_cnu/home/share/chapter8/chapter8_4/...
#   —— 这条里**没有** phdauser004 那层。下面那些带 phdauser004 的是别的入口，留着兜底。
#   注意：/eaas/default/groups/*/home/*/share 这种 glob **匹配不到** home/share
#   （少一层），所以必须在这里写全，不能指望自动 glob。
KNOWN_SHARES = ["/eaas/default/groups/casestudy_cnu/home/share",
                "/eaas/default/groups/casestudy_cnu/home/phdauser004/share",
                "/eaas/default/groups/casestudy_cnu/phdauser004/share"]
# 00_unpack.py 解压 zip 后的工作目录（家目录，通常可写）
WORK_DIR = os.environ.get("PD_WORK", os.path.expanduser("~/pd_data"))

# 递归探测时要跳过的重目录，避免在 ~ 下扫到 conda / 缓存 / 应用数据里浪费时间
_SKIP_DIRS = {".git", ".cache", ".conda", ".local", ".config", ".ipynb_checkpoints",
              "anaconda3", "miniconda3", "miniforge3", "node_modules",
              "site-packages", "__pycache__", ".venv", "venv", "envs", "pkgs",
              # Windows 家目录下的一堆应用数据（本机自测时踩过：能扫出 400+ 个无关 zip）
              "AppData", "Application Data", "Local Settings", "Windows",
              "snap", ".vscode-server", ".npm", ".cursor-server"}


def _looks_like_data(d):
    """这个目录是不是「直接装着数据集」——有 year* 文件夹，或直接有 表1_ 开头的表。"""
    try:
        names = os.listdir(d)
    except OSError:
        return False
    return (any(n.startswith("year") for n in names)
            or any(n.startswith("表1_") for n in names))


def _find_data_dirs(base, maxdepth=4, limit=12):
    """从 base 往下逐层找出所有「直接装着 year*/表1_」的目录（由浅到深）。

    ★ 必须返回**全部**，不能只返回第一个。
      share 里同时解着好几个数据集是常态（肝脏 CT、中医文本、认知筛查……），
      只取第一个会漏掉真正要用的那个，而且漏得很隐蔽：
      兜底搜索是广度优先、先命中浅层的，于是「独立那个 zip 解出来的同名表」
      会比「二期 zip 里同一张表」先被找到 —— 静默用错数据源。
    """
    if not base or not os.path.isdir(base):
        return []
    from collections import deque
    q, seen = deque([(os.path.abspath(base), 0)]), set()
    hits = []
    while q and len(hits) < limit:
        d, depth = q.popleft()
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        if _looks_like_data(d):
            hits.append(d)
            continue                 # 命中就不再往下，免得把数据集内部也算成一层
        if depth < maxdepth:
            try:
                for n in os.listdir(d):
                    p = os.path.join(d, n)
                    if n in _SKIP_DIRS or n.startswith("."):
                        continue
                    if os.path.isdir(p):
                        q.append((p, depth + 1))
            except OSError:
                pass
    return hits


def _root_spec():
    """候选数据位置 + 各自的搜索深度。
    深度按目录性质给：数据目录可以往深里找几层；家目录只准看直接子项，
    否则会扫进 conda / 应用缓存（本机自测踩过：能翻出 400+ 个无关 zip）。"""
    spec = [(os.environ.get("PD_DATA_ROOT", ""), 3),
            (WORK_DIR, 3)]                              # ← 解包后的工作目录，最优先
    for k in KNOWN_SHARES:                              # ← 你账号在平台上的确定位置
        spec.append((k, 3))
    # ★ 2026-09-25：这里原来只写了「神经退**化**性」，而 zip/目录实际叫「神经退**行**性」——
    #   字面路径对不上就是个死候选，虚拟桌面上会一路退到 `~` 去找而找不到数据。
    #   两种写法都列上，谁在先都行（探测是按「里面有没有 year*/表1_」认的，不靠目录名）。
    for _z in ("神经退行性", "神经退化性"):
        spec.append((r"Z:\二期-多模态帕金森" + _z + "疾病的步态动力学与可穿戴健康监测", 1))
    for g in glob.glob("/eaas/default/groups/*/home/*/share"):   # 远程分析平台标准位置
        spec.append((g, 2))
    for g in glob.glob("/eaas/*/groups/*/home/*/share"):
        spec.append((g, 2))
    spec += [(os.path.expanduser("~/share"), 2),
             (os.path.expanduser("~"), 1)]              # ← 只看家目录直接子项
    for g in glob.glob("/mnt/*") + glob.glob("/data/*"):
        spec.append((g, 2))
    return spec


def _candidate_roots():
    """把所有候选位置展开成一串「待搜索的根目录」，按优先级去重。
    ★ 两个 zip 各装一个数据集，解压后是两个平级目录——
      所以这里不能只挑一个根，要把它们都收进来，让 find_file 逐个去认。
    ★ 找出数据集目录后按「标准布局的完整度」排序：同时装着「诊断特征」和
      「行走运动参数」的那个（= 二期 zip 解出来的）排前面，只有一个的排后面。
      否则同名表两份都规范摆放时，先搜到谁全看 listdir 顺序，等于碰运气。"""
    # ★ 用 realpath 归一：平台上 home/share 与 home/phdauser004/share 是同一份
    #   存储的两个入口，只用 abspath 认不出，会把同一批文件搜两遍。
    roots, hits = [], []
    for c, depth in _root_spec():
        if not c or not os.path.isdir(c):
            continue
        c = os.path.realpath(c)
        if c not in roots:
            roots.append(c)
        for hit in _find_data_dirs(c, maxdepth=depth):    # ★ 全部收进来
            if hit not in roots and hit not in hits:
                hits.append(hit)

    def _completeness(d):
        return -sum(os.path.isdir(os.path.join(d, n)) for n in _DATASET_DIRS)

    hits.sort(key=_completeness)          # 越全的排越前（稳定排序，同级保持原序）
    roots.extend(hits)
    return roots


_DATASET_DIRS = ("year2帕金森病诊断步态特征数据",                    # 已提取好的步态特征表
                 "year2帕金森病患者与健康老年人行走运动参数数据")     # 原始 96Hz 惯性数据

ALL_ROOTS = _candidate_roots()
DATA_ROOT = ALL_ROOTS[0] if ALL_ROOTS else ""

DIR_FEAT, DIR_RAW = _DATASET_DIRS

# ★ 2026-09-25 拍照逐张核对，「year2帕金森病诊断步态特征数据」下有四张表：
#     表1 (592×9) / 表2 (592×269) / 表3 (592×1067)
#         —— 每人多行（102 人的 试次×左右脚），列是 Participants / Foot /
#            Group_index（取值 1/2/3/4，是**组次序号不是诊断**），**没有诊断列**。
#     表4 (19×393)
#         —— ParticipantID（HC_01…HC_08、PD_01…PD_11，共 19 人）+ 392 个步态特征列，
#            **一人一行、诊断就在 ID 前缀里**。
#   01_random_forest.py 是「每人一行 + 普通 StratifiedKFold」，只有表4 符合；
#   用表1/2/3 会让同一人同时落进训练集和测试集，指标虚高（脚本里的 assert 会拦下）。
#   另外表4 的这 19 人正好就是原始表那 19 人（115 sheet = 1 + 19 人 × 6 组次），
#   所以 01 和 01b/02/03 是同一批样本，8.5 里能真正放在一起比。
FEAT_MAIN = "表4_2022-2023帕金森病诊断步态特征数据"
RAW_MAIN  = "表1_2022-2023帕金森病患者与健康老年人行走运动参数数据"

RAW_FS    = 96      # 原始采样率 Hz（说明.txt：诺亦腾 Perception Neuron Studio）
WIN       = 200     # 深度学习用的序列窗口长度（点）≈ 2.1 s
BODY_PART = "Hips"  # 做序列建模取哪个身体部位，勘察后可改
ONLY_WALK = True    # 只用「行走」组次（本章是步态分析，排除闭眼站立）

# ★ 原始表每个部位都带一列 `Hips-Sensor-Lost`（0/1）：1 表示该帧追踪丢失，
#   这一帧的惯性量是**无效**的（实拍的例子里 Lost 帧附近 Acce/Velo 会整段变成 0）。
#   0 不是「测出来是 0」而是「没测到」——直接拿去算时域/频域特征会得到假的低能量、
#   假的冻结指数，而曲线上看不出来。
#   ⚠️ 默认保持 False（= 照单全收，与之前行为一致），原因见下：
#   照片只能确认这一列存在，还**不能**确认 Lost=1 时哪些通道一定不可用
#   （我看到的零值片段和 Lost=1 的行号并不重合，可能 Lost 只表示骨骼解算丢失、
#    而 IMU 三轴仍然有效）。这属于会改动教材数字的口径问题，不该由脚本悄悄替你定。
#   想要「Lost 帧按缺失值插值」就把这里改成 True，并在正文里写明该口径。
DROP_LOST = False


def mask_lost(df, num):
    """把 `*-Lost` 标记为 1 的行在 num（同长度的数值 DataFrame）里置成 NaN。

    返回置好 NaN 的 num；DROP_LOST=False、没有 Lost 列、或对不上长度时原样返回。
    只认**当前部位**的 Lost 列 —— 各部位各有一列，全部 OR 起来会把
    「脚丢了」误当成「髋丢了」，把好数据也一起抹掉。
    """
    if not DROP_LOST or not isinstance(num, pd.DataFrame) or len(num) != len(df):
        return num
    all_lost = [c for c in df.columns if re.search(r"[-_]lost$", str(c), re.I)]
    if not all_lost:
        return num
    p = str(BODY_PART).strip().lower()
    lost = [c for c in all_lost if p in str(c).lower()]
    if not lost and len(all_lost) == 1:        # 只有一列，无所谓部位，直接用
        lost = all_lost
    if not lost:
        print(f"  [丢失] 表里有 Lost 列 {all_lost}，但没有 {BODY_PART!r} 那一列，"
              f"本次不做丢失帧处理")
        return num
    m = np.zeros(len(num), dtype=bool)
    for c in lost:
        m |= (pd.to_numeric(df[c], errors="coerce").fillna(0).values != 0)
    if m.any():
        print(f"  [丢失] {int(m.sum())}/{len(num)} 帧被 {lost} 标为 Lost=1，"
              f"已按缺失值插值")
    # ★ 必须按**行**置 NaN。写成 num.mask(m) 会因为 m 是长度 n 的一维数组、
    #   而 num 是 (n, C) 的二维表而抛 "Array conditional must be same shape as self"
    #   —— 这条路径默认关着，只有把 DROP_LOST 打开才会走到，正好是最容易没测到的地方。
    out = num.copy()
    out.loc[m] = np.nan
    return out


_QUALITY_ONCE = set()


def report_quality(x, where):
    """打印序列的数据质量线索，供人工判断 —— 不改动任何数值。

    ★ 存在的意义：整段被补成 0 的序列、或大段 0 的序列，照样能训出模型、
      画出图，只是结果是假的。这里把「有多少帧是精确的 0」摆到台面上。
    ★ 整段全 0 每次都报（那是真异常）；只是部分帧为 0 的话每个部位只说一次，
      否则 57 个 sheet 会把运行日志刷得看不见别的东西。
    """
    x = np.asarray(x)
    if not x.size:
        return
    zero_row = np.all(x == 0, axis=1)
    n = int(zero_row.sum())
    if n >= len(x):
        print(f"  [质量] ⚠️ {where}：整段 {len(x)} 帧全是 0 —— 这条序列没有任何信息，"
              f"请核对 BODY_PART / 通道列名。")
    elif n and where not in _QUALITY_ONCE:
        _QUALITY_ONCE.add(where)
        print(f"  [质量] ⚠️ {where}：{n}/{len(x)} 帧全通道为 0（数据不足补零，"
              f"或原始表里就是 0）。若原始表有 `*-Lost` 列且想把这些帧按缺失值处理，"
              f"把 common.py 顶部的 DROP_LOST 改成 True。")

# 结果输出目录（代码目录不可写时自动退回家目录）
def _ensure_out(d):
    """能建且能写就返回该目录，否则返回 None。"""
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".w_probe")
        open(probe, "w").close()
        os.remove(probe)
        return d
    except OSError:
        return None


_OUT_TRIES = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
              os.path.join(os.path.expanduser("~"), "pd_results"),
              os.path.join(os.environ.get("PD_WORK") or os.path.expanduser("~"),
                           "pd_results"),
              os.path.join(os.getcwd(), "results"),
              "/tmp/pd_results"]
OUT = next((d for d in (_ensure_out(t) for t in _OUT_TRIES) if d), None)

# ★ OUT 一个都建不出来时必须当场停下。原来自动退回家目录，家目录也写不了时
#   OUT=None，然后 os.path.join(None, "seq_dataset.npz") 会在**导入阶段**抛
#   TypeError —— 报错发生在 import 那一行，跟真正的原因（没地方写文件）毫无关系，
#   而且 01/01b/02/03/04 全都是这个错，看上去像代码坏了。
if OUT is None:
    raise SystemExit(
        "[致命] 这几个目录都建不出/写不了：\n  " + "\n  ".join(_OUT_TRIES) +
        "\n       结果没地方存，先解决写入权限，或指定一个可写目录：\n"
        "         export PD_WORK=/你有写权限的目录\n"
        "       然后重跑。")
print(f"[环境] 结果目录 OUT = {OUT}")

# 中文图表风格（Windows 字体在前，Linux 的 CJK 字体兜底）
_CJK_WANT = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
             "Source Han Sans SC", "Source Han Sans CN", "WenQuanYi Zen Hei",
             "WenQuanYi Micro Hei", "Droid Sans Fallback", "AR PL UMing CN",
             "Noto Serif CJK SC", "SimSun", "Arial Unicode MS"]


def setup_cjk_font():
    """挑一个**真实存在**的中文字体；一个都没有就大声报出来。

    ★ 原来只是把候选名单丢给 matplotlib，没人验证过哪个真的装了。
      平台上一个都没有时，matplotlib 会**静默**回退到 DejaVu Sans ——
      图照样存盘、退出码照样 0，只是图里所有中文都变成方框 □□□，
      提示只有几十条 UserWarning，夹在正常输出里根本看不见。
      教材插图里全是方框，等发现就晚了。
    ★ 没有 root 装字体的退路：把任意中文 ttf/otf 放到能读的地方，然后
        export PD_FONT=/path/to/NotoSansCJKsc-Regular.otf
      脚本会先把它注册进来再挑。
    """
    from matplotlib import font_manager
    extra = os.environ.get("PD_FONT", "").strip()
    if extra:
        if os.path.exists(extra):
            try:
                font_manager.fontManager.addfont(extra)
                nm = font_manager.FontProperties(fname=extra).get_name()
                print(f"[字体] 已按 PD_FONT 加载：{nm}（{extra}）")
            except Exception as e:
                print(f"[字体] ⚠️ PD_FONT={extra} 加载失败（{e}），继续找已装的")
        else:
            print(f"[字体] ⚠️ PD_FONT 指向的文件不存在：{extra}")
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["axes.unicode_minus"] = False
    for f in _CJK_WANT:
        if f in have:
            plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            print(f"[字体] 中文字体 = {f}")
            return f
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    print("\n" + "!" * 66)
    print("[字体] ⚠️ 这台机器上一个中文字体都没有 —— 下面出的每张图里，")
    print("       中文标题/图例/坐标轴都会变成方框 □□□（图仍会存盘、退出码仍是 0）。")
    print("       装一个再重跑，任选其一：")
    print("         sudo apt install fonts-noto-cjk        # 有 root")
    print("         conda install -c conda-forge fonts-anaconda   # 有 conda")
    print("       没有 root 也行：随便下个中文 ttf/otf 存到自己目录，然后")
    print("         export PD_FONT=/你的路径/NotoSansCJKsc-Regular.otf")
    print("!" * 66 + "\n")
    return None


setup_cjk_font()
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
# =========================================================


# ---------- 文件定位 ----------
_OK_EXT = (".xlsx", ".xls", ".csv")


def _match_in(d, prefix):
    """在目录 d 下按前缀找数据文件（先精确前缀，再模糊匹配）。"""
    cand = [p for p in glob.glob(os.path.join(d, prefix + ".*"))
            if os.path.splitext(p)[1].lower() in _OK_EXT]
    if not cand:  # 前缀不完全一致时退化为模糊匹配
        cand = [p for p in glob.glob(os.path.join(d, "*" + prefix[1:] + "*.*"))
                if os.path.splitext(p)[1].lower() in _OK_EXT]
    return sorted(cand)


def _search_under(base, prefix, maxdepth=4):
    """兜底：在 base 下面任意深度找一个匹配 prefix 的数据文件（zip 解压后目录层级不定时用）。"""
    from collections import deque
    q, seen = deque([(base, 0)]), set()
    while q:
        d, depth = q.popleft()
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        hit = _match_in(d, prefix)
        if hit:
            return hit[0]
        if depth < maxdepth:
            try:
                for n in os.listdir(d):
                    p = os.path.join(d, n)
                    if n in _SKIP_DIRS or n.startswith("."):
                        continue
                    if os.path.isdir(p):
                        q.append((p, depth + 1))
            except OSError:
                pass
    return None


_FILE_CACHE = {}


def find_file(folder, prefix):
    """按文件名前缀找数据文件，自动适配 xlsx/xls/csv。

    ★ 两个 zip 各装一个数据集、解压成两个平级目录，所以这里挨个根目录去认，
      哪个根里有就到哪个根里取——不要求两个数据集同处一个 DATA_ROOT。

    ★ 分两轮，顺序不能反：
      第一轮只认「root/数据集目录/表」这种标准布局，谁先命中就用谁；
      第二轮才退化为「在根目录下任意深度找一个同名表」。
      反过来（每个根都先深搜再换下一个根）会出问题：兜底搜索是广度优先、
      先命中浅层的，于是「独立那个 zip 解出来的同名表」会比
      「二期 zip 里同一张表」先被找到 —— 静默用错数据源。
    """
    key = (folder, prefix)
    p = _FILE_CACHE.get(key)
    if p and os.path.exists(p):        # 命中缓存（一张表常被读很多次）
        return p

    for root in ALL_ROOTS:             # 第一轮：标准布局
        d = os.path.join(root, folder)
        if os.path.isdir(d):
            cand = _match_in(d, prefix)
            if cand:
                _FILE_CACHE[key] = cand[0]
                return cand[0]

    for root in ALL_ROOTS:             # 第二轮：布局对不上，才任意深度找
        hit = _search_under(root, prefix)
        if hit:
            _FILE_CACHE[key] = hit
            return hit

    # 都没找到 —— 把搜过的地方和实际内容列出来，方便一眼看出问题
    if not ALL_ROOTS:
        raise FileNotFoundError(
            "一个候选数据目录都不存在。请按顺序排查：\n"
            "  1) 数据还是 zip 的话，先解包：python 00_unpack.py\n"
            "  2) 手动指定：export PD_DATA_ROOT='<你的数据目录>'  再重跑\n"
            "  3) 或直接修改 common.py 顶部的 KNOWN_SHARES。")
    lines = [f"找不到以「{prefix}」开头的数据文件。搜过这些地方："]
    for root in ALL_ROOTS:
        d = os.path.join(root, folder)
        if os.path.isdir(d):
            lines.append(f"  ● {d}\n      实际内容: {sorted(os.listdir(d))[:12]}")
        else:
            try:
                names = sorted(os.listdir(root))[:12]
            except OSError:
                names = ["(列不出)"]
            lines.append(f"  ○ {root}   (无「{folder}」子目录)\n      实际内容: {names}")
    if any(glob.glob(os.path.join(r, "*.zip")) for r in ALL_ROOTS):
        lines.append("  —— 目录里还有 zip，先跑 python 00_unpack.py 解包")
    raise FileNotFoundError("\n".join(lines))


def _inventory():
    """把当前实际认得的数据集目录列成一句话，方便一眼看出缺的是哪一份。"""
    found = []
    for root in ALL_ROOTS:
        try:
            for n in sorted(os.listdir(root)):
                if n.startswith("year") and n not in found:
                    found.append(n)
        except OSError:
            pass
    return "、".join(found) if found else "（没认出任何 year*/表1_ 数据集目录）"


def has_table(folder, prefix):
    """这份数据里到底有没有这张表。找不到只返回 False，不抛异常。"""
    try:
        find_file(folder, prefix)
        return True
    except FileNotFoundError:
        return False


def skip_if_missing(folder, prefix, what):
    """缺表就打印一句人话然后让调用方退出，而不是甩一屏 traceback。

    ★ 只有「二期」一个 zip 时很常见：那份数据里未必同时含
      「诊断特征表」和「行走运动参数表」，缺哪张，依赖它的脚本就该安静跳过。
    返回 True 表示「本脚本该跳过」。
    """
    if has_table(folder, prefix):
        return False
    print(f"\n[SKIP] 当前数据里没有「{what}」（本脚本跳过，其余脚本不受影响）")
    print(f"       找的是：{folder}/{prefix}.*")
    print(f"       这份数据里只有：{_inventory()}")
    return True


def read_table(folder, prefix, **kw):
    """读取数据表（自动区分 csv / excel）。

    ★ csv 要依次试几种编码：Windows 上导出的 csv 多半是 GBK/GB18030，
      pandas 默认按 UTF-8 读，遇到中文列名会直接 UnicodeDecodeError，
      报错还只说「codec can't decode byte ...」，看不出是编码问题。
      （本套脚本自己写出的 csv 一律 utf-8-sig，读回来也在这个序列里。）
    """
    path = find_file(folder, prefix)
    print(f"[读入] {os.path.basename(path)}")
    if path.lower().endswith(".csv"):
        if "encoding" in kw:
            return pd.read_csv(path, **kw)
        last = None
        for enc in ("utf-8-sig", "gb18030", "utf-8", "big5"):
            try:
                return pd.read_csv(path, encoding=enc, **kw)
            except (UnicodeDecodeError, UnicodeError) as e:
                last = e
        raise UnicodeDecodeError(
            "csv", b"", 0, 1,
            f"{os.path.basename(path)} 用 utf-8/gb18030/big5 都读不出来（{last}）；"
            f"请在 read_table 里显式传 encoding=...")
    return pd.read_excel(path, **kw)


def list_sheets(folder, prefix):
    """列出 Excel 的所有 sheet 名。"""
    return pd.ExcelFile(find_file(folder, prefix)).sheet_names


# ---------- 特征表的列自动探测 ----------
ID_KEYS = ["受试者ID", "患者ID", "编号", "序号", "ID", "subject", "姓名",
           # ★ 2026-09-25 实测：二期这两张表用的是英文列名
           #   Participants（592 行 / 102 人，值形如 xj-co-001）
           #   ParticipantID（19 行 / 19 人，值形如 HC_01）
           #   加在「ID」前面，否则子串匹配会先撞上别的列
           "Participants", "ParticipantID", "participant"]
LABEL_KEYS = ["分组", "组别", "标签", "诊断", "类别", "类型", "label", "group", "class"]

# ★ 标签列的显式映射，默认 None（不启用）。
#   正常情况用不上 —— 只有当标签列不是二分类、需要人工指定对应关系时才设它。
#   例：LABEL_MAP = {"1": 1, "2": 0, "3": None}   # None = 丢弃该行
#   为什么要这么防：靠关键词猜标签时，名字里带 "group" 的列会被当成分组列。
#   二期特征表里有个 Group_index（取值 1/2/3/4，是组次不是诊断），
#   一旦被猜中，再按「含 PD/1 即患者」硬转，就会得到一组完全没意义的标签，
#   而样本数、AUC 都「看着正常」，极难发现。所以这里宁可停下报错。
LABEL_MAP = None

# ★ 有些表**根本没有标签列**，诊断写在受试者 ID 的前缀里 —— 表4 就是这样：
#     ParticipantID = HC_01…HC_08 / PD_01…PD_11（19 人）。
#   这时 guess_label 找不到任何标签列，LABEL_MAP 也救不了（没有列可以映射），
#   原来会抛「探测不到标签列，请手动指定 label_col」—— 可这张表根本没有能指定的列，
#   照着提示去查只会一无所获。
#   所以在这里声明「ID 前缀 -> 类别」；get_xy 只在找不到标签列时才启用它。
#   前缀后面必须紧跟非字母（HC_01 / HC-01 / HC01 都认，"HCP" 不算 HC），
#   免得将来多一个前缀被静默并进 HC。
ID_PREFIX_LABEL = {"HC": 0, "PD": 1}


def guess_col(df, keys):
    """按关键词猜列名（先精确、后包含）。"""
    for k in keys:
        for c in df.columns:
            if str(c).strip().lower() == k.lower():
                return c
    for k in keys:
        for c in df.columns:
            if k.lower() in str(c).lower():
                return c
    return None


def guess_label(df, exclude=()):
    """猜标签列：先按关键词，再退化到「取值像分组」的列。

    ★ 穷举那一步（关键词没命中时）只挑**非数值列**。
      原来看「只有 2~3 个唯一值」就认，这对连续型的特征表是灾难：
      表4 有 393 列步态特征，随便一个数值列（比如某个 MEAN）都能只有 2~3 个
      不同取值（小数被截断过、或本来就是分档的），于是标签列会被猜成某个步态指标，
      再经 to_binary 变成一组纯粹随机的 0/1 —— 指标、ROC、混淆矩阵全都"正常"，
      而结论完全是噪音。分组信息几乎总在字符串列（PD/HC、患者/对照），
      数值列里的 2 个取值更可能是测量档位而不是诊断结果。
    """
    c = guess_col(df, LABEL_KEYS)
    if c and c not in exclude:
        return c
    for c in df.columns:
        if c in exclude or pd.api.types.is_numeric_dtype(df[c]):
            continue
        u = df[c].dropna().unique()
        if 1 < len(u) <= 3:
            return c
    return None


def to_binary(s):
    """把标签列统一成 1=帕金森患者(PD)、0=健康老年人(HC)。"""
    t = s.astype(str).str.upper().str.strip()
    pd_pat = r"PD|帕金森|患者|阳性|^1$|^1\.0$|YES|TRUE"
    return t.str.contains(pd_pat, regex=True).astype(int)


def label_from_id(ids, mapping):
    """按受试者 ID 的**前缀**定分组。返回 (y 的 Series, 认不出的 ID 列表)。

    ★ 前缀后面必须紧跟「非字母」，或者就是结尾：
        HC_01 / HC-01 / HC01 / HC  -> 认
        HCP_01                     -> 不认（不能被静默并进 HC）
      不写这条边界的话，将来数据集里多一个 HCP_xx，"HCP_01" 会因为以 "HC" 开头
      被当成健康对照 —— 而且没有任何提示。
    """
    t = ids.astype(str).str.upper().str.strip()
    y = pd.Series(np.nan, index=t.index, dtype="float")
    for pref, lab in mapping.items():
        p = str(pref).upper()
        hit = t.str.match(rf"^{re.escape(p)}(?![A-Z])")
        y[hit & y.isna()] = lab          # 先声明的先赢（防止前缀互相包含）
    unknown = sorted(set(t[y.isna()]))
    return y, unknown


def get_xy(df, label_col=None, id_col=None):
    """从特征表切出 X(数值特征) / y(0-1标签) / groups(受试者ID)。"""
    id_col = id_col or guess_col(df, ID_KEYS)
    label_col = label_col or guess_label(df, exclude=(id_col,))

    if label_col is None and ID_PREFIX_LABEL and id_col:
        # 表4 这类没有标签列、诊断写在 ID 前缀里的表（见 ID_PREFIX_LABEL 处的说明）。
        y_s, unknown = label_from_id(df[id_col], ID_PREFIX_LABEL)
        if unknown:
            raise ValueError(
                f"ID 列 '{id_col}' 里有认不出分组的前缀：{unknown[:10]}"
                f"{' 等' if len(unknown) > 10 else ''}。"
                f"common.py 顶部的 ID_PREFIX_LABEL={ID_PREFIX_LABEL} 只认 "
                f"{sorted(ID_PREFIX_LABEL)} 开头的 ID。"
                f"缺哪个前缀就往里补哪个，**不要**改用别的列去硬凑标签 —— "
                f"标签错了指标照样算得出来，只是全部没有意义。")
        y = y_s.astype(int).values
        print(f"[标签] 表里没有标签列，改按 ID 列 '{id_col}' 的前缀判定分组"
              f"（{ID_PREFIX_LABEL}）")
    elif label_col is None:
        raise ValueError("探测不到标签列，请手动指定 label_col")
    elif LABEL_MAP is not None:                        # 人工指定的映射，优先
        raw = df[label_col].astype(str)
        table = {str(k): v for k, v in LABEL_MAP.items()}
        missing = sorted(set(raw) - set(table))
        if missing:
            raise ValueError(f"LABEL_MAP 没覆盖标签列的全部取值，还缺：{missing}")
        mapped = raw.map(table)
        keep = mapped.notna()
        if not keep.any():
            raise ValueError("LABEL_MAP 把所有行都映射成 None 了，检查一下取值")
        n_drop = int((~keep).sum())
        df = df.loc[keep].reset_index(drop=True)
        y = mapped.loc[keep].astype(int).values
        if n_drop:
            print(f"[标签] 按 LABEL_MAP 丢弃 {n_drop} 行（映射为 None 的取值）")
    else:
        u = df[label_col].dropna().unique()
        if len(u) > 2:
            # ★ 宁可停下，也不要把「非二分类」硬转成一对没意义的标签。
            vc = df[label_col].value_counts(dropna=False)
            print(f"\n[停下] 标签列 '{label_col}' 有 {len(u)} 个取值，不是二分类：")
            for v, k in vc.items():
                print(f"        {v!r}: {k} 行")
            print("       按「含 PD/帕金森/1 就是患者」硬转会得到没有意义的标签，")
            print("       而样本数、AUC 都看着正常，很难发现，所以这里不往下走。")
            print("       弄清真正的分组依据后，在 common.py 顶部显式设置 LABEL_MAP，")
            print('       例：LABEL_MAP = {"1": 1, "2": 0}    # 不要的取值给 None')
            print("       或者给脚本传 label_col=<真正的标签列名>。")
            raise ValueError(f"标签列 '{label_col}' 不是二分类（{len(u)} 个取值）")
        y = to_binary(df[label_col]).values

    num = df.select_dtypes(include=[np.number])
    drop = [c for c in (label_col, id_col) if c in num.columns]
    X = num.drop(columns=drop)
    X = X.loc[:, X.nunique() > 1]                      # 剔除常数列
    g = df[id_col].astype(str).values if id_col else np.arange(len(df)).astype(str)
    print(f"[数据] 特征列 {X.shape[1]} 个 | 样本 {X.shape[0]} 例 | "
          f"PD {int(y.sum())} / HC {int((1 - y).sum())} | 受试者 {len(set(g))} 名")
    print(f"[列名] ID列='{id_col}'  标签列='{label_col}'")
    return X.values.astype("float32"), y, g, list(X.columns)


# ---------- 原始序列读取（96Hz 惯性数据）----------
def sheet_subject(sheet):
    """从 sheet 名解析受试者编号，如 'PD_01_行走_1' -> 'PD_01'。"""
    m = re.match(r"^\s*([A-Za-z]+\s*[_-]?\s*\d+)", str(sheet))
    return m.group(1) if m else str(sheet).split("_")[0]


def sheet_label(sheet):
    """sheet 名 -> 1=PD / 0=HC。识别不出来时返回 None。"""
    s = str(sheet).upper()
    if re.match(r"^\s*PD", s) or "帕金森" in s:
        return 1
    if re.match(r"^\s*HC", s) or "健康" in s or "对照" in s:
        return 0
    return None


def _channel_cols(df, kind):
    """挑出某个物理量的 x/y/z 三轴列名。kind='imu' 找加速度/角速度，'posi' 找位置。

    ★ 列名不能写死。这份数据的来源有 MATLAB 导出的特征表（Hip_MEAN_of_ACCx 之类）
      和诺亦腾原始表两套，命名习惯不同：Acce_x / AcceX / ACCx / GYROy / Gyro_y /
      POSx / Posi_x 都出现过。写死一种，换一份数据就一个通道都认不出来 ——
      而报错信息会指向「没数据」，把人往完全错的方向带。
      所以：先按「整列名就是 加速度/角速度+轴」精确认（原始长表的常见形态），
      认不够 3 轴再退一步按「结尾是 _ACCx / ACCx 这种」认。

    ★ 2026-09-25 按原始表实拍的表头（`Hips-Sens-…` / `Hips-Joint-…` 这种
      「部位-类型-物理量」的连字符命名）补了两条：
        1) 词干要认全称：Accel / Acceleration / Gyro / Gyroscope，不能只认 Acc/Gyr；
        2) 轴字母后面**不再要求就是行尾** —— 带单位的写法（如 `AccX(m/s^2)`）
           原来一个都认不出来。改成「后面不能再跟字母」，即 `AccX` / `AccX(m/s^2)`
           都认，而 `AccXxx` 这种仍然不认，免得把不相干的列算成通道。
    """
    # ★ 2026-09-25 按原始表实拍的完整表头定的词干。
    #   诺亦腾导出的列名是 `部位-Sensor-Acce-x` / `部位-Sensor-Gyro-x`
    #   （连字符分隔，物理量用缩写 `Acce` / `Gyro` / `Velo` / `Posi`）。
    #   ⚠️ 原来的 `acc(?:el(?:erometer|eration)?)?` **认不出 `Acce`**：
    #      "Acce" = acc + e，而那个可选组只有 el 开头一条路，于是整列被漏掉。
    #      实测后果：真实表里 6 个 IMU 通道只能认出 3 个 Gyro，
    #      加速度三轴全部丢失 → _pick_axes 直接报「认不出加速度」而中止。
    #      补上 `|e` 分支后 acc / acce / accel / acceleration / accelerometer 全认。
    #   `posi` 只认位置（Pos / Posi / Position），**不含 Velo** ——
    #      04 的轨迹预测要的是关节位置，如果把 Velo 也算进来就会拿速度当坐标。
    stems = {"imu": r"acc(?:el(?:erometer|eration)?|e)?|gyr(?:o(?:scope)?)?",
             "posi": r"pos(?:i(?:tion)?)?|关节位置|transl(?:ation)?"}[kind]
    exact = re.compile(rf"^(?:{stems})[_\-]?([xyz])(?![a-z])", re.I)
    tail = re.compile(rf"[_\-](?:{stems})[_\-]?([xyz])(?![a-z])", re.I)
    # ★ 两种形态是**相加**的，不是二选一。
    #   原来写成「精确形态不够 3 个就整个丢掉，换成结尾形态」，于是
    #   `Posi_x`（精确形态本身命中）在只给这一列时会先命中、再被丢掉，
    #   结尾形态又要求前面有分隔符 —— 最终返回空，报「认不出位置列」。
    #   整表列多的时候结尾形态能兜住，所以这个错一直看不出来；
    #   一旦表变小（或只查一个物理量）就显形。合并后是原来的超集，只会多认不会少认。
    ex = [c for c in df.columns if exact.match(str(c).strip())]
    tl = [c for c in df.columns if tail.search(str(c).strip())]
    hits, seen = [], set()
    for c in ex + tl:                      # 精确形态优先，再补结尾形态，去重保序
        if c not in seen:
            seen.add(c)
            hits.append(c)
    return hits


def _prefer_part(cols, part):
    """宽表里身体部位写在**列名**里（Hips-Sens-AccX / Foot-Sens-AccX）。

    只认前 3 个的话，取到的是列顺序最靠前那个部位的通道，是哪个部位纯看导出顺序 ——
    静默取错部位，序列照样建得出来、模型照样训得动。所以先用 BODY_PART 过滤；
    一个都没匹配上（说明是「部位在行里」的长表）就原样返回，交给调用方去判断。
    """
    if not part:
        return cols
    p = str(part).strip().lower()
    hit = [c for c in cols if p in str(c).lower()]
    return hit or cols


_AXES_SEEN = {}          # 只在取用列发生变化时打印一次，避免 57 个 sheet 刷屏


def _pick_axes(df, sheet):
    """取加速度 3 轴 + 角速度 3 轴，凑成 6 通道。认不出就报出实际列名。"""
    got = _prefer_part(_channel_cols(df, "imu"), BODY_PART)
    gyro = [c for c in got if re.search(r"gyr", str(c), re.I)]
    acc = [c for c in got if c not in gyro]
    if len(acc) < 3 or len(gyro) < 3:
        raise ValueError(
            f"sheet「{sheet}」里认不出加速度/角速度的 x/y/z 列。"
            f"（BODY_PART={BODY_PART!r} 过滤后 加速度 {len(acc)} 轴 / 角速度 {len(gyro)} 轴）"
            f"实际列名（前 25 个）：{[str(c) for c in df.columns][:25]}。"
            f"把 00_inspect.py 报告里原始表的列名贴回来，我据此改 common.py 的 _channel_cols")
    axes = acc[:3] + gyro[:3]
    # ★ 把实际取用的列名打出来。这一行是「静默取错通道」唯一的现场证据：
    #   部位取错、把 Joint 列当成 Sensor 列，图上看不出，只有这里能看见。
    key = tuple(map(str, axes))
    if _AXES_SEEN.get("k") != key:
        _AXES_SEEN["k"] = key
        joined = " ".join(map(str, axes)).lower()
        note = ""
        if str(BODY_PART).strip().lower() not in joined:
            note = (f"   ⚠️ 这些列名里没有 {BODY_PART!r}，是按列顺序取的前 6 个 —— "
                    f"取到的可能不是你要的部位，请核对")
        print(f"[通道] 取用列：{list(map(str, axes))}{note}")
    return axes


def load_sequence(folder, prefix, sheet, part=None, win=None):
    """读一个 sheet，返回定长 (win, C) 的传感器序列矩阵；数据不足则零填充。"""
    df = pd.read_excel(find_file(folder, prefix), sheet_name=sheet)
    p = part or BODY_PART
    if "BodyPart" in df.columns:                       # 长表结构：按身体部位筛选
        kept = df[df["BodyPart"].astype(str) == p]
        if kept.empty:
            # ★ 这里必须拦住。不拦的话下面会把空表补成一段**全零序列**，
            #   模型照样"训练成功"、图照样画出来，而数据是假的 —— 极难发现。
            raise ValueError(
                f"sheet「{sheet}」里没有 BodyPart == {p!r} 的行；"
                f"实际部位：{sorted(map(str, df['BodyPart'].dropna().unique()))[:20]}。"
                f"请按 00_inspect.py 打印的部位取值改 common.py 的 BODY_PART")
        df = kept
    cols = _pick_axes(df, sheet)                        # 加速度/角速度各 3 轴
    # 先把可能存在的 Lost 帧标成缺失值（默认关，见 DROP_LOST 处的说明），
    # 再插值 —— 顺序不能反：interpolate 只补 NaN，对已经是 0 的帧无能为力。
    num = mask_lost(df, df[cols].apply(pd.to_numeric, errors="coerce"))
    x = num.interpolate().fillna(0).values
    if x.size == 0:
        raise ValueError(f"sheet「{sheet}」部位 {p!r} 下没有任何数据行，无法构成序列")
    w = win or WIN
    x = x[:w] if len(x) >= w else np.pad(x, ((0, w - len(x)), (0, 0)))
    report_quality(x, f"sheet「{sheet}」")
    return x.astype("float32")


# ---------- 交叉验证划分 ----------
def make_cv(y, g, max_splits=5):
    """留一受试者交叉验证的划分器：优先用分层分组版，退回普通分组版。

    ★ 为什么要分层：GroupKFold 不洗牌，且只保证「每折样本数」均衡、不管类别。
      而这类数据集必然是「PD 受试者在前、HC 受试者在后」地排着，
      于是它可能把一整段同类受试者切进同一折 —— 一旦某折测试集全是 PD，
      训练集里 HC 就占多数，模型只要输出训练集先验就能"猜对"，
      留出的那折却全是 PD，结果预测与标签系统性反向（AUC 掉到 0 附近）。
      受试者少时尤其致命。StratifiedGroupKFold 在「同一个人不跨折」的前提下
      额外把两类摊平，能挡住这个问题；老版本 sklearn 没有它才退回 GroupKFold。
    """
    from sklearn.model_selection import GroupKFold
    import numpy as _np
    n_sub = len(set(g))
    # ★ 折数不能超过「受试者数」和「少数类样本数」，否则 sklearn 直接抛
    #   "n_splits=N cannot be greater than the number of members in each class"。
    #   受试者只剩 1 个人时更没得划分，这里先拦住并说清原因。
    if n_sub < 2:
        raise ValueError(
            f"只有 {n_sub} 名受试者，没法做留一受试者交叉验证。"
            f"请确认序列数据（seqdata 的 g）里确实是多人，"
            f"而不是被 ONLY_WALK / BodyPart 之类的过滤条件筛得只剩一个人的片段。")
    y_bin = _np.asarray(y)
    n_minor = int(min((y_bin == 1).sum(), (y_bin == 0).sum())) or 1
    n = max(2, min(n_sub, n_minor, max_splits))
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=42)
    except ImportError:                 # scikit-learn < 1.0
        return GroupKFold(n_splits=n)


# ---------- 评估指标（8.5 会用）----------
def check_two_classes(y, where="数据"):
    """确认标签里 PD 和 HC 两类都在。只有一类就当场停下。

    ★ 为什么必须停：sklearn 从 1.6 起，单类标签下 roc_auc_score 不再报错，
      只发一个 UndefinedMetricWarning 然后**返回 nan**。于是：
      Accuracy 仍然是 1.0、混淆矩阵照样画得出来、02_cm_calibration 还会在
      图上写「准确率 1.000」，脚本退出码 0 —— 一整组看着完全正常的图，
      而里面没有一个数字是有意义的。AUC=nan 只出现在 CSV 的一列里，
      图上看不见。这种情况必须先停下，让人去查 sheet 名/标签列。
    """
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 and n0:
        return
    print(f"\n[停下] {where}的标签只有一类：PD {n1} 例 / HC {n0} 例（共 {len(y)} 例）。")
    print("       单类标签下 AUC 会静默变成 nan，而 Accuracy 仍是 1.0，")
    print("       混淆矩阵、校准曲线、雷达图照样画得很正常 —— 必须先弄清分组再跑。")
    print("       常见原因：sheet 名解析不出 PD/HC 前缀（如全是 HC_01 这种）、")
    print("       或 ONLY_WALK 过滤后只剩一类的组次。先看 00_inspect.py 的")
    print("       「sheet 名解析自检」那一段。")
    raise ValueError(f"{where}的标签不是二分类：PD {n1} / HC {n0}")


from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve,
                             brier_score_loss)


def evaluate(y_true, y_pred, y_prob=None, name="model"):
    """算一组分类指标，返回一行字典。"""
    check_two_classes(y_true, f"模型「{name}」的评估")
    r = {"模型": name,
         "Accuracy": accuracy_score(y_true, y_pred),
         "Precision": precision_score(y_true, y_pred, zero_division=0),
         "Recall": recall_score(y_true, y_pred, zero_division=0),
         "F1": f1_score(y_true, y_pred, zero_division=0)}
    if y_prob is not None:
        r["AUC"] = roc_auc_score(y_true, y_prob)
        r["Brier"] = brier_score_loss(y_true, y_prob)
    return r


def save_metrics(rows, fname):
    """把指标表存到 results/ 并打印。"""
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, fname), index=False, encoding="utf-8-sig")
    show = [c for c in df.columns if df[c].dtype.kind == "f"]
    print(df[["模型"] + show].round(3).to_string(index=False) if "模型" in df else df.round(3))
    return df


def savefig(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p); plt.close(fig)
    print(f"[图] {p}")


# ---------- 预测结果存取（8.5 靠这两个文件把各模型对齐比较）----------
def save_pred(key, y, prob, sig=""):
    """把某模型的标签与预测概率存盘。key 用英文，供 8.5 识别（rf/lstm/transformer…）

    ★ sig 是「样本集签名」——一句话说清这批样本是哪来的。
      8.5 要把多个模型放进同一张 ROC 图，前提是它们逐行对应同一批样本。
      光比标签数组长度（或内容）是不够的：两份样本数相同但顺序不同的结果，
      标签可能碰巧一模一样，8.5 就会把两批不相干的预测当成同一批去比，
      指标、ROC、混淆矩阵全都算错，而且没有任何报错。
      所以这里显式记下来源，8.5 按签名分组。
    """
    y = np.asarray(y)
    np.save(os.path.join(OUT, f"8.4_{key}_labels.npy"), y)
    np.save(os.path.join(OUT, f"8.4_{key}_prob.npy"), np.asarray(prob))
    with open(os.path.join(OUT, f"8.4_{key}_sig.txt"), "w", encoding="utf-8") as f:
        f.write(f"{sig or '未知'}\nn={len(y)}\nPD={int((y == 1).sum())}\nHC={int((y == 0).sum())}\n")
    print(f"[存] 8.4_{key}_labels.npy / 8.4_{key}_prob.npy  （样本集：{sig or '未知'}，{len(y)} 例）")
