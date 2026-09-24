# -*- coding: utf-8 -*-
"""
01_metrics_roc.py —— 8.5 分类性能指标计算 + ROC 曲线
对应教材：Accuracy / Precision / Recall / F1 四维指标 + ROC-AUC + Bootstrap 置信区间
跑法：python 01_metrics_roc.py   （需先跑过 8.4 的建模脚本）
配套公式（教材 8.5）：
    Accuracy  = (TP+TN) / (TP+TN+FP+FN)
    Precision = TP / (TP+FP)
    Recall    = TP / (TP+FN)
    F1        = 2·Precision·Recall / (Precision+Recall)
"""
from eval_common import *
from sklearn.metrics import roc_curve

if __name__ == "__main__":
    groups = collect_models()
    print("\n【样本集分组】只在同一组内做对比（样本集不同不能直接比）")
    fmt_groups(groups)

    models, y = main_group(groups)
    print(f"\n→ 主对比组：{len(y)} 例样本，模型 {list(models)}\n")

    # ---------- 1. 指标表 ----------
    rows = [full_metrics(y, p, n) for n, p in models.items()]
    df = pd.DataFrame(rows)
    save_csv(df, "8.5_全模型指标.csv")
    print("\n【全模型分类性能】")
    print(df[["模型", "Accuracy", "Precision", "Recall", "F1",
              "AUC", "Brier", "特异度", "Kappa"]].round(3).to_string(index=False))

    # ---------- 2. AUC 自助法置信区间 ----------
    ci = pd.DataFrame([{"模型": n, "AUC": roc_auc_score(y, p),
                        "CI下限": bootstrap_ci(y, p)[0],
                        "CI上限": bootstrap_ci(y, p)[1]} for n, p in models.items()])
    save_csv(ci, "8.5_AUC置信区间.csv")
    print("\n【AUC 95% 置信区间（Bootstrap 1000 次）】")
    print(ci.round(3).to_string(index=False))

    # ---------- 3. 可视化 ----------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]                                   # ROC 曲线
    for n, p in models.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=2, label=f"{n} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测 (AUC=0.5)")
    ax.set_xlabel("假阳性率 FPR (1-特异度)"); ax.set_ylabel("真阳性率 TPR (灵敏度)")
    ax.set_title("各模型 ROC 曲线对比"); ax.legend(loc="lower right"); ax.grid(alpha=.3)

    ax = axes[1]                                   # 四维指标分组柱状图
    df.set_index("模型")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(
        ax=ax, rot=0, width=.8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("得分"); ax.set_title("四维分类指标对比")
    ax.legend(loc="lower right", fontsize=8); ax.grid(axis="y", alpha=.3)

    ax = axes[2]                                   # AUC ± 置信区间
    ypos = np.arange(len(ci))
    # ★ 一律转成 numpy 再画。只有一个模型时 ci["AUC"] 这样的 1 元素 Series 传给
    #   matplotlib，它会去 float() 一个 Series —— pandas 2.x 只是警告，
    #   但明说「将来会抛 TypeError」，pandas 3 上这句就直接崩。
    auc_v = np.asarray(ci["AUC"], dtype=float)
    lo_v = np.asarray(ci["CI下限"], dtype=float)
    hi_v = np.asarray(ci["CI上限"], dtype=float)
    ax.barh(ypos, auc_v, color="#2E6F9E", alpha=.85)
    ax.errorbar(auc_v, ypos, xerr=[auc_v - lo_v, hi_v - auc_v],
                fmt="none", ecolor="k", capsize=4)
    ax.set_yticks(ypos); ax.set_yticklabels(ci["模型"])
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUC（误差棒为 95% CI）")
    ax.set_title("AUC 及置信区间"); ax.grid(axis="x", alpha=.3)

    fig.tight_layout()
    savefig(fig, "8.5_ROC与指标对比.png")
