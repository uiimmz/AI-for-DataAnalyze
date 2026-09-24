# -*- coding: utf-8 -*-
"""
03_compare_all.py —— 8.5 多模型横向对比 + 统计显著性检验
对应教材：手工特征机器学习 vs 深度学习自动提特征 两条路线的性能综合对比
统计检验：McNemar 检验（两个模型的分类对错是否显著不同）
          Bootstrap 近似检验（两个模型的 AUC 差是否有显著差异）
          ★ 这里是 Bootstrap，不是 DeLong。教材正文若要写 DeLong，
            得用解析法（协方差矩阵 + z 统计量），不能拿本脚本的输出当 DeLong 引用。
跑法：python 03_compare_all.py
"""
from eval_common import *
from itertools import combinations
from scipy import stats
from sklearn.metrics import roc_auc_score


def mcnemar(y, p1, p2, thr=0.5):
    """McNemar 检验：只看两个模型预测不一致的样本，检验其是否有系统性差异。"""
    c1 = (p1 >= thr).astype(int) == y
    c2 = (p2 >= thr).astype(int) == y
    b = int(np.sum(c1 & ~c2))       # 模型1对、模型2错
    c = int(np.sum(~c1 & c2))       # 模型1错、模型2对
    if b + c == 0:
        return b, c, np.nan
    # 连续性校正的卡方统计量（b+c < 25 时用精确二项检验）
    if b + c < 25:
        p = stats.binomtest(b, b + c, 0.5).pvalue
    else:
        p = stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), df=1)
    return b, c, p


def bootstrap_auc_diff(y, p1, p2):
    """两模型 AUC 差值的 Bootstrap 检验（**不是** DeLong 检验）。

    ★ 名字必须叫对。DeLong 是解析地算 AUC 的方差-协方差再构造 z 统计量；
      这里做的是对样本重采样、看 AUC 差值的分布，属于 Bootstrap 法。
      两者结论通常接近，但写进教材就是另一回事了 —— 方法名写错等于结论不可复现。
      教材里若要写 DeLong，请另用 pROC / 自己实现协方差那一套；
      本脚本的输出请按「Bootstrap 近似」引用。
    """
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    diffs = np.array(diffs)
    if diffs.size == 0:
        # 单类标签时每次重采样都被 continue 掉。np.percentile([], ...) 会抛
        # IndexError（内容是「size 0」，跟 AUC 毫无关系），而且发生在汇总表已经存盘之后。
        check_two_classes(y, "AUC 差值的 Bootstrap 检验")
        return np.nan, np.array([np.nan, np.nan]), np.nan
    # 双尾 p 值：差值分布中越过 0 的比例
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5]), min(p, 1.0)


if __name__ == "__main__":
    groups = collect_models()
    print("\n【样本集分组】")
    fmt_groups(groups)
    models, y = main_group(groups)
    names = list(models)
    print(f"\n→ 主对比组：{len(y)} 例样本，模型 {names}")
    if len(names) < 2:
        print(f"\n[注意] 主对比组里只有 {len(names)} 个模型，没法做两两检验。")
        print(f"       想看对比请把 8.4 的 01b / 02 / 03 都跑一遍（02、03 需要 torch）。")
        print(f"       下面仍然会出汇总表和雷达图，但没有显著性检验那一段。")

    # ---------- 1. 汇总指标表 ----------
    df = pd.DataFrame([full_metrics(y, p, n) for n, p in models.items()])
    df = df.sort_values("AUC", ascending=False).reset_index(drop=True)
    save_csv(df, "8.5_模型对比汇总.csv")

    print("【多模型综合对比（按 AUC 降序）】")
    print(df[["模型", "Accuracy", "Precision", "Recall", "F1", "AUC", "特异度", "Kappa"]]
          .round(3).to_string(index=False))

    best = df.iloc[0]
    print(f"\n→ 综合表现最优：{best['模型']}（AUC={best['AUC']:.3f}，"
          f"F1={best['F1']:.3f}，Accuracy={best['Accuracy']:.3f}）")

    # ---------- 2. 两两显著性检验 ----------
    rows = []
    for a, b in combinations(names, 2):
        mb, mc, pm = mcnemar(y, models[a], models[b])
        dm, ci, pd_ = bootstrap_auc_diff(y, models[a], models[b])
        pm_ok, pd_ok = not np.isnan(pm), not np.isnan(pd_)
        rows.append({"模型A": a, "模型B": b,
                     "AUC差(A-B)": roc_auc_score(y, models[a]) - roc_auc_score(y, models[b]),
                     "AUC差95%CI下限": ci[0], "AUC差95%CI上限": ci[1],
                     "Bootstrap近似p": pd_,
                     "McNemar_b": mb, "McNemar_c": mc, "McNemar_p": pm,
                     # ★ 两个检验各判各的，不能只看一个。原来只看 AUC 的 p，
                     #   于是「McNemar 显著、Bootstrap 不显著」也会写成「否」，
                     #   把「分类对错有系统差异」这件事整个丢掉。
                     "AUC差显著(α=.05)": ("是" if pd_ok and pd_ < .05 else
                                          "否" if pd_ok else "无法判定"),
                     "McNemar显著(α=.05)": ("是" if pm_ok and pm < .05 else
                                            "否" if pm_ok else "无法判定")})
    sig = pd.DataFrame(rows)
    if sig.empty:
        # ★ 只有一个模型时 combinations 不产生任何组合，pd.DataFrame([]) 没有任何列，
        #   to_csv 会写一个 5 字节的空文件（只有 BOM），控制台打印 "Empty DataFrame"，
        #   既不说为什么空，也没有列名 —— 后面读这张表的人完全无从判断。
        #   这里明确写出「未做检验」的说明表。
        sig = pd.DataFrame([{"说明": f"主对比组只有 {len(names)} 个模型，无法做两两检验；"
                                     f"请把 8.4 的 01b/02/03 都跑一遍"}])
        save_csv(sig, "8.5_模型两两显著性检验.csv")
        print("\n【模型两两差异显著性检验】")
        print(sig.to_string(index=False))
    else:
        save_csv(sig.round(4), "8.5_模型两两显著性检验.csv")
        print("\n【模型两两差异显著性检验】")
        print(sig.round(3).to_string(index=False))

    # ---------- 3. 性能雷达图 ----------
    met = ["Accuracy", "Precision", "Recall", "F1", "AUC", "特异度"]
    ang = np.linspace(0, 2 * np.pi, len(met), endpoint=False).tolist()
    ang += ang[:1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={"projection": "polar"})
    # ★ 左边这块原来会把标题写死成「手工特征 + 树模型（随机森林）」，
    #   而在「主对比组里没有随机森林」时（rf 属于另一个样本集，或 01/01b 没跑成），
    #   它就把 df 全画上 —— 于是图上出现一块标题写着随机森林、曲线其实是 LSTM 的
    #   雷达图，还印进教材。标题必须跟着实际画了什么走。
    rf_sub = df[df["模型"].str.contains("随机森林")]
    panels = []
    if not rf_sub.empty:
        panels.append((rf_sub, "手工特征 + 树模型（随机森林）"))
    # 标题跟着**实际画进去的模型**走。写死「三条技术路线」的话，一旦 01（特征表）
    # 因为样本集不同被分到另一组，这上面只有深度学习一条，标题却是三条。
    _rt = routes_of(list(df["模型"]))
    panels.append((df, (" + ".join(_rt) + " 综合对比") if _rt else "各模型综合对比"))
    for ax, (sub, title) in zip(axes, panels):
        for _, r in sub.iterrows():
            v = [r[m] for m in met]; v += v[:1]
            ax.plot(ang, v, "o-", lw=2, label=r["模型"])
            ax.fill(ang, v, alpha=.08)
        ax.set_xticks(ang[:-1]); ax.set_xticklabels(met)
        ax.set_ylim(0, 1); ax.set_title(title, pad=18); ax.legend(loc="lower right",
                                                                 fontsize=8)
    for ax in axes[len(panels):]:        # 没有随机森林时多余的轴：关掉，别留个空框
        ax.axis("off")
    fig.tight_layout(); savefig(fig, "8.5_性能雷达图.png")

    # ---------- 4. 综合排名热力图 ----------
    fig, ax = plt.subplots(figsize=(8, max(2.4, .7 * len(df))))
    mat = df.set_index("模型")[met].values
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(met))); ax.set_xticklabels(met)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["模型"])
    for i in range(len(df)):
        for j in range(len(met)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=10)
    ax.set_title("各模型多指标得分热力图")
    plt.colorbar(im, ax=ax, fraction=.03)
    savefig(fig, "8.5_综合排名热力图.png")
