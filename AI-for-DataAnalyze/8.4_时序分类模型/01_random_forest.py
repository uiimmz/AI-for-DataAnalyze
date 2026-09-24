# -*- coding: utf-8 -*-
"""
01_random_forest.py —— 8.4 路径一：手工提取特征 + 随机森林
对应教材：随机森林分类器（Bagging + 特征随机性 + 多数投票）
数据：year2帕金森病诊断步态特征数据（Matlab 已提取好的步态特征表）
跑法：python 01_random_forest.py
"""
from common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RANDOM_STATE, N_SPLITS = 42, 5

if skip_if_missing(DIR_FEAT, FEAT_MAIN, "帕金森病诊断步态特征表"):
    sys.exit(0)

# ---------------- 1. 读数据 ----------------
df = read_table(DIR_FEAT, FEAT_MAIN)
try:
    X, y, groups, feat_names = get_xy(df)
except ValueError as e:
    # get_xy 已经把「为什么不能往下跑」逐条打印了，再甩一屏 traceback 只是噪音。
    print(f"\n[终止] {e}")
    print("       这条路径（特征表 + 每人一行）跑不了就先跑 01b_random_forest_seq.py，")
    print("       它和 02/03 用同一批样本，8.5 的对比照样成立。")
    sys.exit(2)

assert len(set(groups)) == len(groups), (
    f"表「{FEAT_MAIN}」里同一受试者出现了多行（{len(df)} 行 / {len(set(groups))} 人）。"
    f"这个脚本用的是普通 StratifiedKFold，没有按受试者分组，同一人多行会让人"
    f"同时出现在训练集和测试集里，指标虚高。要么换成「每人一行」的聚合表，"
    f"要么改用 make_cv(y, groups)（受试者分组交叉验证）。")

# ---------------- 2. 交叉验证评估 ----------------
# ★ 折数要压到「少数类样本数」以内，否则 sklearn 直接抛
#   ValueError: n_splits=5 cannot be greater than the number of members in each class。
#   小样本表一跑就撞上，报错还看不出是折数问题。
#   表4 实测是 19 人（HC 8 / PD 11），少数类 8 人，还算宽裕 —— 但换数据就不一定了。
n_minor = int(min((y == 1).sum(), (y == 0).sum())) or 1
n_splits = max(2, min(N_SPLITS, n_minor))
if n_splits != N_SPLITS:
    print(f"[划分] 少数类只有 {n_minor} 例，折数从 {N_SPLITS} 降到 {n_splits}")
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
pipe = make_pipeline(StandardScaler(), RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1))

y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
save_metrics([evaluate(y, y_pred, y_prob, "随机森林")], "8.4_随机森林指标.csv")

# ---------------- 3. 超参调优（可选，8.5 讨论优化策略用）----------------
grid = {"randomforestclassifier__max_depth": [3, 5, 8, None],
        "randomforestclassifier__min_samples_leaf": [1, 2, 4]}
gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X, y)
print(f"[调优] 最优参数：{gs.best_params_}  CV-AUC={gs.best_score_:.3f}")

# ---------------- 4. 全量训练 + 特征重要性 ----------------
pipe.fit(X, y)
imp = pd.Series(pipe[-1].feature_importances_, index=feat_names).sort_values(ascending=False)
imp.to_csv(os.path.join(OUT, "8.4_随机森林_特征重要性.csv"), encoding="utf-8-sig")
print("\n[特征重要性 Top15]")
print(imp.head(15).round(4).to_string())

# ---------------- 5. 可视化 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
imp.head(15)[::-1].plot.barh(ax=axes[0], color="#2E6F9E")
axes[0].set_title("随机森林特征重要性 Top15"); axes[0].set_xlabel("重要性")

ax = axes[1]
imp.head(20).plot.bar(ax=ax, color="#C0504D")
ax.set_title("Top20 特征重要性分布"); ax.set_ylabel("重要性")
ax.tick_params(axis="x", rotation=75)
for lb in ax.get_xticklabels():
    lb.set_ha("right")
savefig(fig, "8.4_随机森林_特征重要性.png")

# 供 8.5 使用。签名与 01b/02/03 不同 —— 本模型是「特征表、每人一行」，
# 样本集和序列那三个不是一回事，8.5 会按签名把它们分开比。
save_pred("rf", y, y_prob, f"特征表:{FEAT_MAIN}")
print("\n完成。结果已存到 results/")
