# -*- coding: utf-8 -*-
"""
02_cm_calibration.py —— 8.5 混淆矩阵、校准曲线、阈值分析、学习曲线
对应教材：模型评价体系（ROC/AUC、校准曲线）与模型优化策略（阈值调整）
跑法：python 02_cm_calibration.py
"""
from eval_common import *
from sklearn.calibration import calibration_curve
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_curve)


def plot_confusion(y, models, ax_list):
    """画各模型的混淆矩阵（带 TP/FP/FN/TN 中文注释）。"""
    cmaps = ["Blues", "Greens", "Oranges", "Purples"]
    for i, (ax, (n, p)) in enumerate(zip(ax_list, models.items())):
        cmap = cmaps[i % len(cmaps)]
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        im = ax.imshow(cm, cmap=cmap, vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{[['TN', 'FP'], ['FN', 'TP']][i][j]}",
                        ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() * .55 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["预测 HC", "预测 PD"])
        ax.set_yticklabels(["实际 HC", "实际 PD"])
        ax.set_title(f"{n}\n准确率 {np.trace(cm) / cm.sum():.3f}")
        plt.colorbar(im, ax=ax, fraction=.046)


def threshold_sweep(y, p):
    """遍历判决阈值，找 F1 最大的阈值。"""
    fpr, tpr, thr = roc_curve(y, p)
    # ★ roc_curve 的 thresholds 数组第一项恒为 np.inf（sklearn 的设计）。
    #   不剔掉的话：inf 会写进 CSV；更糟的是当所有 F1 都是 0 时，
    #   idxmax() 返回第 0 行 —— 于是图上、控制台、图例里都会出现
    #   「F1 最优阈值 = inf」这种没法解释的结论。
    thr = [t for t in thr if np.isfinite(t)]
    rows = []
    for t in thr:
        pred = (p >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                        zero_division=0)
        rows.append({"阈值": t, "Precision": pr, "Recall": rc, "F1": f1})
    df = pd.DataFrame(rows)
    if df.empty:                     # 极端情况：没有任何有限阈值
        df = pd.DataFrame([{"阈值": 0.5, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}])
    return df


if __name__ == "__main__":
    models, y = main_group(collect_models())

    # ---------- 1. 混淆矩阵 ----------
    k = len(models)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 4.4))
    plot_confusion(y, models, np.atleast_1d(axes))
    fig.tight_layout(); savefig(fig, "8.5_混淆矩阵.png")

    # ---------- 2. 校准曲线 ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for n, p in models.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", lw=2,
                label=f"{n} (Brier={brier_score_loss(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="理想校准线")
    ax.set_xlabel("预测概率"); ax.set_ylabel("实际阳性比例")
    ax.set_title("校准曲线（可靠性图）"); ax.legend(fontsize=8); ax.grid(alpha=.3)

    # ---------- 3. 概率分布 ----------
    ax = axes[1]
    for n, p in models.items():
        ax.hist(p[y == 0], bins=15, alpha=.45, label=f"{n} · HC")
        ax.hist(p[y == 1], bins=15, alpha=.45, label=f"{n} · PD")
    ax.axvline(0.5, color="r", ls="--", lw=1.5, label="默认阈值 0.5")
    ax.set_xlabel("预测为 PD 的概率"); ax.set_ylabel("样本数")
    ax.set_title("预测概率分布"); ax.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, "8.5_校准与概率分布.png")

    # ---------- 4. 阈值分析（以第一个模型为例）----------
    name0 = list(models)[0]
    sw = threshold_sweep(y, models[name0])
    best = sw.loc[sw["F1"].idxmax()]
    print(f"\n【阈值分析 · {name0}】默认阈值 0.5，"
          f"F1 最优阈值 = {best['阈值']:.3f}（F1={best['F1']:.3f}，"
          f"Precision={best['Precision']:.3f}，Recall={best['Recall']:.3f}）")
    save_csv(sw.round(4), "8.5_阈值扫描.csv")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(sw["阈值"], sw["Precision"], label="Precision")
    ax.plot(sw["阈值"], sw["Recall"], label="Recall")
    ax.plot(sw["阈值"], sw["F1"], label="F1", lw=2.4)
    ax.axvline(best["阈值"], color="r", ls="--", label=f"F1 最优阈值 {best['阈值']:.2f}")
    ax.axvline(0.5, color="gray", ls=":", label="默认阈值 0.5")
    ax.set_xlabel("判决阈值"); ax.set_ylabel("得分")
    ax.set_title(f"{name0} 阈值—指标曲线"); ax.legend(); ax.grid(alpha=.3)
    savefig(fig, "8.5_阈值分析.png")

    # ---------- 5. 各模型的漏诊率 / 误诊率 ----------
    # ★ 分母为 0 时不能拿 max(...,1) 糊过去。原来分母无论是 0 还是 1，
    #   漏诊率都算成 0.0，于是「一个 PD 都没有」的数据会显示成
    #   「所有模型漏诊率 0.0」，读起来就是「没有漏掉任何帕金森患者」——
    #   而实际上根本没有帕金森患者可漏。改为写 NaN，图上/表里一眼能看出无定义。
    n_pd, n_hc = int(np.sum(y)), int(len(y) - np.sum(y))
    err = pd.DataFrame([{"模型": n,
                         "漏诊率FN": (m["FN"] / n_pd) if n_pd else np.nan,
                         "误诊率FP": (m["FP"] / n_hc) if n_hc else np.nan}
                        for n, p in models.items()
                        for m in [full_metrics(y, p, n)]])
    save_csv(err.round(4), "8.5_漏诊误诊率.csv")
    print("\n【漏诊/误诊率】漏诊 = 帕金森被当成健康；误诊 = 健康被当成帕金森")
    print(err.round(3).to_string(index=False))
    if not n_pd or not n_hc:
        print(f"   （注意：PD {n_pd} 例 / HC {n_hc} 例，分母为 0 的那一列写的是 NaN，"
              f"表示无定义，不是 0）")
