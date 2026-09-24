# -*- coding: utf-8 -*-
"""
01b_random_forest_seq.py —— 8.4 路径一（可比版）：手工时频特征 + 随机森林
为什么要单独有这个脚本：
    01_random_forest.py 读的是「诊断步态特征表」，样本是"每位受试者一行"；
    02/03 读的是「原始96Hz序列」，样本是"每个 sheet 一行"。
    两者样本集不同，没法放在同一张 ROC 图上对比。
    本脚本从**同一批原始序列**里手工提取时域+频域特征（对应教材表4、表5），
    再喂随机森林 —— 这样三个模型吃的是完全相同的样本，对比才成立。
跑法：python 01b_random_forest_seq.py
"""
from common import *
from seqdata import (build_seq_dataset, extract_handcrafted, TD_NAMES, FD_NAMES,
                     SEQ_SIG)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RANDOM_STATE = 42


if __name__ == "__main__":
    # ---------- 1. 取与 LSTM/Transformer 完全相同的样本 ----------
    X_seq, y, g = build_seq_dataset()
    if X_seq is None:
        sys.exit(0)

    # ---------- 2. 手工提特征 ----------
    Xf, names = extract_handcrafted(X_seq)
    print(f"[特征] 手工特征矩阵 {Xf.shape} = {X_seq.shape[2]} 通道 × "
          f"({len(TD_NAMES)} 时域 + {len(FD_NAMES)} 频域)")

    # ---------- 3. 留一受试者交叉验证（与深度模型同一套划分逻辑）----------
    cv = make_cv(y, g)
    print(f"[划分] {type(cv).__name__}  {len(set(g))} 名受试者 -> {cv.get_n_splits()} 折")
    pipe = make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))

    y_prob = cross_val_predict(pipe, Xf, y, cv=cv, groups=g,
                               method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    save_metrics([evaluate(y, y_pred, y_prob, "随机森林")], "8.4_随机森林(序列)指标.csv")

    # ---------- 4. 特征重要性 ----------
    pipe.fit(Xf, y)
    imp = pd.Series(pipe[-1].feature_importances_, index=names).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT, "8.4_随机森林(序列)_特征重要性.csv"), encoding="utf-8-sig")
    print("\n[特征重要性 Top15]")
    print(imp.head(15).round(4).to_string())

    # ---------- 5. 时域 vs 频域 贡献占比 ----------
    # ★ regex=False：FD_NAMES 里有 "冻带功率0.5-3Hz" 这种名字，里面带 . 和 -，
    #   按正则解释是能匹配任意字符的元字符，虽然现在不会误伤，但纯属埋雷。
    is_fd = imp.index.str.contains("|".join(FD_NAMES), regex=False)
    share = {"时域特征": float(imp[~is_fd].sum()), "频域特征": float(imp[is_fd].sum())}
    print(f"\n[时/频域贡献] 时域 {share['时域特征']:.3f} | 频域 {share['频域特征']:.3f}")
    if sum(share.values()) <= 0:
        # ★ 饼图的数值全为 0 时 matplotlib 会抛
        #   ValueError: cannot convert float NaN to integer（算百分比那步炸的），
        #   报错点离原因很远，而且此时指标 CSV 已经写出去了。
        #   全 0 意味着这棵树一次都没分裂（标签只有一类，或特征全零/常量）。
        print("[跳过] 特征重要性全为 0（树没有分裂）—— 通常说明标签只有一类、"
              "或特征全是常量。饼图不画了。")
        fig, ax = plt.subplots(figsize=(7, 5))
        imp.head(15)[::-1].plot.barh(ax=ax, color="#2E6F9E")
        ax.set_title("随机森林特征重要性 Top15（手工时频特征）")
        savefig(fig, "8.4_随机森林(序列)_特征重要性.png")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        imp.head(15)[::-1].plot.barh(ax=axes[0], color="#2E6F9E")
        axes[0].set_title("随机森林特征重要性 Top15（手工时频特征）")
        axes[1].pie(share.values(), labels=share.keys(), autopct="%.1f%%",
                    colors=["#2E6F9E", "#C0504D"], startangle=90)
        axes[1].set_title("时域 / 频域特征贡献占比")
        savefig(fig, "8.4_随机森林(序列)_特征重要性.png")

    save_pred("rf_seq", y, y_prob, SEQ_SIG)   # 供 8.5 与 LSTM/Transformer 同图对比
    print("\n完成。结果已存到 results/")
