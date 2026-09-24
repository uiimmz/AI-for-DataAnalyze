# -*- coding: utf-8 -*-
"""
eval_common.py —— 8.5 三个脚本共用的：路径、指标计算、模型概率收集、绘图风格
"""
import os, sys, glob
# Windows 控制台默认 GBK，遇到编码不了的字符会直接抛异常中断脚本；
# 这里改成「替换」而不是「报错」，保证长跑脚本不会因为一个符号崩掉。
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             cohen_kappa_score)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")                                     # 8.5 自己的输出
os.makedirs(OUT, exist_ok=True)

# ★ 8.4 的输出目录不能写死成「../8.4_时序分类模型/results」。
#   两处都会让它对不上：
#     1) 8.4 的 common.py 在代码目录不可写时会退到 ~/pd_results（平台上很常见），
#        于是预测文件根本不在代码目录下；
#     2) 平台上 8.4 的目录实际叫 chapter8/chapter8_4，前缀不叫「8.4」。
#   对不上时的表现是 8.5 打印「没有从 8.4 读到任何结果，请先跑 8.4」——
#   而用户明明刚跑完，只会以为是 8.4 坏了。
_84_CANDS = []
if "--results" in sys.argv:
    _84_CANDS.append(sys.argv[sys.argv.index("--results") + 1])
if os.environ.get("PD_RESULTS"):
    _84_CANDS.append(os.environ["PD_RESULTS"])
if os.environ.get("PD_WORK"):
    _84_CANDS.append(os.path.join(os.environ["PD_WORK"], "pd_results"))
_PARENT = os.path.dirname(HERE)
for _pat in ("8.4*", "*8.4*", "chapter8_4", "*时序分类模型*"):
    for _d in sorted(glob.glob(os.path.join(_PARENT, _pat))):
        if os.path.isdir(_d):
            _84_CANDS.append(os.path.join(_d, "results"))
_84_CANDS += [os.path.join(_PARENT, "8.4_时序分类模型", "results"),
              os.path.join(os.path.expanduser("~"), "pd_results")]
_84_SEEN, _tmp = set(), []
for _c in _84_CANDS:
    _c = os.path.abspath(_c) if _c else ""
    if _c and _c not in _84_SEEN:
        _84_SEEN.add(_c)
        _tmp.append(_c)
_84_CANDS = _tmp

OUT84 = next((c for c in _84_CANDS if glob.glob(os.path.join(c, "8.4_*_prob.npy"))), "")
if not OUT84:
    print("[路径] 这些地方都没有 8.4 的预测结果（8.4_*_prob.npy）：")
    for c in _84_CANDS[:8]:
        print(f"         {c}")
    OUT84 = _84_CANDS[0] if _84_CANDS else ""
else:
    print(f"[路径] 8.4 的结果目录 = {OUT84}")

_CJK_WANT = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP",
             "Source Han Sans SC", "Source Han Sans CN", "WenQuanYi Zen Hei",
             "WenQuanYi Micro Hei", "Droid Sans Fallback", "AR PL UMing CN",
             "Noto Serif CJK SC", "SimSun", "Arial Unicode MS"]


def setup_cjk_font():
    """挑一个**真实存在**的中文字体；一个都没有就大声报出来（同 8.4 common.py）。

    ★ 不验证的话，平台上一个中文字体都没有时会静默回退 DejaVu Sans，
      图照样存盘、退出码照样 0，只是图里中文全变方框 □□□ —— 教材插图就废了。
    ★ 没有 root 的退路：export PD_FONT=/path/to/中文字体.otf
    """
    from matplotlib import font_manager
    extra = os.environ.get("PD_FONT", "").strip()
    if extra:
        if os.path.exists(extra):
            try:
                font_manager.fontManager.addfont(extra)
                print("[字体] 已按 PD_FONT 加载："
                      f"{font_manager.FontProperties(fname=extra).get_name()}")
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
    print("       装一个再重跑：sudo apt install fonts-noto-cjk")
    print("       没有 root 就 export PD_FONT=/你的路径/中文字体.otf")
    print("!" * 66 + "\n")
    return None


setup_cjk_font()
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120


def check_two_classes(y, where):
    """只有一类就当场停下。

    ★ sklearn 从 1.6 起，单类标签下 roc_auc_score 不再报错，只发一个
      UndefinedMetricWarning 然后**返回 nan**；roc_curve 也一样返回全 nan。
      于是 02_cm_calibration 会照样画出一张图，标题写着「LSTM 准确率 1.000」，
      校准曲线是一条贴着 0 的直线，脚本退出码 0 —— 一整组看着完全正常的图，
      图上没有任何地方显示 AUC 是 nan。这种情况必须先停下。
    """
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 and n0:
        return
    print(f"\n[终止] {where} 的标签只有一类：PD {n1} / HC {n0}（共 {len(y)} 例）。")
    print("       单类标签下 AUC 是 nan，而 Accuracy 仍是 1.0、混淆矩阵照样能画，")
    print("       图上看不出任何异常 —— 所以这里不往下走。")
    print("       先回 8.4 跑 00_inspect.py，看「sheet 名解析自检」那一段的")
    print("       分组计数是不是 PD/HC 都在。")
    sys.exit(3)


def full_metrics(y, p, name, thr=0.5):
    """一次算出 8.5 需要的全部指标（含混淆矩阵四格与特异度）。"""
    check_two_classes(y, f"模型「{name}」的评估")
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"模型": name, "Accuracy": accuracy_score(y, yhat),
            "Precision": precision_score(y, yhat, zero_division=0),
            "Recall": recall_score(y, yhat, zero_division=0),
            "F1": f1_score(y, yhat, zero_division=0),
            "AUC": roc_auc_score(y, p), "Brier": brier_score_loss(y, p),
            "Kappa": cohen_kappa_score(y, yhat),
            "特异度": tn / (tn + fp) if (tn + fp) else 0,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def bootstrap_ci(y, p, n=1000, seed=42):
    """自助法估计 AUC 的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    if not aucs:
        # ★ 单类标签时每一次重采样都被上面那句 continue 掉，aucs 是空的，
        #   np.percentile([], ...) 会抛 IndexError（报错内容是「index -1 is out of
        #   bounds for axis 0 with size 0」，跟 AUC 毫无关系），
        #   而且是在指标 CSV 已经写出去之后才炸，留下一半结果。
        check_two_classes(y, "AUC 置信区间")
        return np.array([np.nan, np.nan])
    return np.percentile(aucs, [2.5, 97.5])


# 文件名 key -> 展示名（用中文，且把两个随机森林区分开，避免图例混淆）
NAME_MAP = {"rf": "随机森林(特征表)", "rf_seq": "随机森林(时序特征)",
            "lstm": "LSTM", "transformer": "Transformer"}
# 模型归属的技术路线，图例/正文里用来区分
ROUTE = {"rf": "手工特征表", "rf_seq": "手工时频特征",
         "lstm": "深度学习", "transformer": "深度学习"}
# 展示名 -> 技术路线（03_compare_all 用它给雷达图起一个「画了什么就说什么」的标题）
ROUTE_BY_NAME = {NAME_MAP[k]: v for k, v in ROUTE.items()}


def routes_of(names):
    """这些模型实际覆盖了哪几条技术路线（按 ROUTE 的定义顺序去重）。

    ★ 雷达图的标题原来写死成「三条技术路线综合对比」，而 01（特征表）与
      01b/02/03（原始序列）样本集不同时，8.5 只在最大的那一组里比 ——
      于是图上只有深度学习那一条，标题却照样写着「三条」。图进了教材就是错的。
    """
    seen, out = set(), []
    for n in names:
        r = ROUTE_BY_NAME.get(n)
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _read_sig(key):
    """读某模型的样本集签名（8.4 的 save_pred 写的）。"""
    f = os.path.join(OUT84, f"8.4_{key}_sig.txt")
    if not os.path.exists(f):
        return ""
    try:
        with open(f, encoding="utf-8") as fh:
            return fh.read().strip().splitlines()[0].strip()
    except OSError:
        return ""


def collect_models():
    """从 8.4 的 results/ 载入各模型的 (标签, 概率)。

    ★ 关键：不同脚本的样本集可能不一样（特征表是"每人一行"、原始序列是
      "每 sheet 一行"），样本集不同的模型**不能**放进同一张 ROC 图。
      所以按「样本集签名」分组，签名是 8.4 的 save_pred 写下来的来源说明。

      为什么不能只比标签数组：两份样本数相同、但顺序不同的结果，标签完全可能
      逐字节一样，于是会被并进同一组，指标、ROC、混淆矩阵全部按错位的行去算，
      而且一个警告都不会有。原来的做法正是只看 y.tobytes()。
      老版本没写签名文件，退回按「长度 + 标签内容」分组，并提示一下。

    返回：{组签名: {模型名: (y, prob)}}
    """
    files = sorted(glob.glob(os.path.join(OUT84, "8.4_*_prob.npy")))
    groups, no_sig, skipped = {}, [], []
    for f in files:
        key = os.path.basename(f).replace("8.4_", "").replace("_prob.npy", "")
        lab_f = os.path.join(OUT84, f"8.4_{key}_labels.npy")
        if not os.path.exists(lab_f):
            print(f"  [跳过] {key}：缺少配套的 labels 文件，请重跑 8.4 对应脚本")
            skipped.append(key)
            continue
        y, p = np.load(lab_f), np.load(f)
        # ★ 标签与概率必须一一对应。save_pred 是先写 labels 再写 prob，
        #   中间被杀掉（平台上跑 torch 很久，很容易）就会留下新的 labels 配旧的 prob。
        #   这时算出来的指标全是错的，而如果不查长度，连一句提示都没有。
        if len(y) != len(p):
            print(f"  [跳过] {key}：labels {len(y)} 个 vs prob {len(p)} 个，对不上。"
                  f"多半是上一次跑到一半被打断了，重跑 8.4 对应脚本即可。")
            skipped.append(key)
            continue
        sig = _read_sig(key)
        if not sig:
            no_sig.append(key)
            sig = f"（无签名）n={len(y)}"
        groups.setdefault(sig, {})[NAME_MAP.get(key, key)] = (y, p)

    if no_sig:
        print(f"  [注意] 这些模型没有样本集签名文件（老版本 8.4 写的）：{no_sig}")
        print(f"         只能按样本数分组。若它们其实来自不同的样本集，结果会不正确；")
        print(f"         重跑一遍 8.4 的对应脚本就会补上签名。")

    if groups:
        print(f"[数据] 从 {OUT84} 载入：")
        for sig, d in groups.items():
            y0 = next(iter(d.values()))[0]
            print(f"        {len(d)} 个模型 / {len(y0)} 例样本 —— {sig}")
        if len(groups) > 1:
            print("        （样本集不同，各脚本只在最大的那一组内做对比）")
        return groups

    if skipped:
        # ★ 找到了文件、但每一个都被跳过（缺 labels / 长度对不上）。这时**不能**
        #   往下说「请先跑 8.4」—— 用户刚跑完，真正的原因上面已经打印过了，
        #   再说一句「请先跑 8.4」只会把人支到错的方向去查。
        print(f"\n[终止] 8.4 的 results/ 里找到了 {len(files)} 个概率文件，"
              f"但都没法用（原因见上）：{skipped}")
        print(f"       这不是「还没跑 8.4」，而是那些文件本身不完整或对不上。")
        print(f"       重跑 8.4 对应的脚本（01b / 02 / 03）即可。")
        sys.exit(4)

    # 没跑过 8.4（或 8.4 的脚本因数据里缺表而跳过了）时的退路。
    # ★ 默认直接停下：随机数据照样能画出很漂亮的 ROC、混淆矩阵和热力图，
    #   万一被当成真实结果写进教材就麻烦了。只想看图表版式请显式加 --demo。
    if "--demo" not in sys.argv:
        print("\n[终止] 没有从 8.4 读到任何模型预测结果。")
        print("       请先到 ../8.4_时序分类模型/ 下依次跑 01b_random_forest_seq.py、")
        print("       02_lstm.py、03_transformer.py（需 torch），它们会把预测存进 results/。")
        print("       如果那边提示「跳过」，说明这份数据里没有对应的表。")
        print("       只想看图表版式（用随机数据，不可作为结果）：加参数 --demo")
        sys.exit(1)
    print("\n[演示模式] 未找到 8.4 的输出，改用随机数据。")
    print("           ★ 本模式产物仅供查看版式，不可作为任何结果引用。")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    demo = {"随机森林": (y, np.clip(y * .60 + rng.normal(.30, .22, 80), 0, 1)),
            "LSTM": (y, np.clip(y * .55 + rng.normal(.32, .25, 80), 0, 1)),
            "Transformer": (y, np.clip(y * .58 + rng.normal(.31, .23, 80), 0, 1))}
    return {y.tobytes(): demo}


def main_group(groups):
    """挑出模型最多的一组作为主对比组（即样本集相同的那些模型）。

    ★ 标签只有一类时在这里就停下，不能等到算指标那一步再停 ——
      02_cm_calibration 是先画混淆矩阵/校准曲线、最后才算漏诊率的，
      等它算到那一步，前面几张图**已经写进 results/ 了**，
      残留的「准确率 1.000」的混淆矩阵会被当成结果拿走。
    """
    sig = max(groups, key=lambda k: len(groups[k]))
    d = groups[sig]
    y = next(iter(d.values()))[0]
    check_two_classes(y, f"主对比组的标签（{sig}）")
    return {n: p for n, (_, p) in d.items()}, y


def fmt_groups(groups):
    """把分组情况打印成人话，方便在正文里交代样本集差异。"""
    for i, (sig, d) in enumerate(groups.items(), 1):
        y0 = next(iter(d.values()))[0]
        print(f"  组{i}：{len(y0)} 例样本（PD {int(y0.sum())} / HC {int(len(y0) - y0.sum())}）"
              f" —— {list(d)}")


# ★ 演示模式必须往另一个文件名写。原来 demo 和真实运行用的是同一批路径，
#   一次 --demo（比如在 Windows 上预览版式）就会把已经算好的真实图**覆盖掉**，
#   而 demo 的数据是随机数造出来的、AUC 还有 0.9 左右，肉眼看不出是假的，
#   文件名里也没有任何标记。加个 DEMO_ 前缀，就不会再误伤真结果。
DEMO = "--demo" in sys.argv


def _tag(name):
    return ("DEMO_" + name) if DEMO else name


def savefig(fig, name):
    p = os.path.join(OUT, _tag(name))
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[图] {p}" + ("   ← 演示数据，不是结果" if DEMO else ""))


def save_csv(df, name):
    p = os.path.join(OUT, _tag(name))
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[表] {p}" + ("   ← 演示数据，不是结果" if DEMO else ""))
