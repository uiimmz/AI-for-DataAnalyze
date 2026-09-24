# 帕金森病步态时序分类与模型性能综合评价

**Parkinsonian Gait Time-Series Classification & Model Evaluation**

> 《生成式 AI 驱动的数据分析》第 8 章配套代码 —— 从可穿戴惯性传感器（IMU）的步态序列出发，
> 走完「手工特征 → 时序深度学习」两条技术路线，并对多个模型做一套完整的性能与统计评价。
>
> Companion code for Chapter 8 of *Data Analysis Driven by Generative AI* — from wearable IMU
> gait sequences, through both hand-crafted-feature and deep time-series pipelines, to a full
> performance and statistical evaluation of the resulting models.

[中文](#中文文档) · [English](#english-documentation)

---

# 中文文档

## 一、这个项目是什么

同一批步态数据（帕金森病患者 PD / 健康老年人 HC），用**两条技术路线**各建一套分类模型，再横向比较：

| 路线 | 思路 | 模型 | 样本单位 |
|---|---|---|---|
| 手工特征表 | 用现成的步态时空参数表，每人一行 | 随机森林 | 每位受试者 |
| 手工时频特征 | 从原始 96Hz 序列里自己提时频特征 | 随机森林 | 每个试次 |
| 深度学习 | 原始序列端到端，自动提特征 | LSTM / Transformer 编码器 | 每个试次 |

除了分类，还包含一个**纵向时序 demo**：用历史轨迹自回归滚动预测未来轨迹，作为「生成式」思路在时序上的最小示例。

代码由两部分组成，**必须放在同级目录下**（`8.5` 要去 `8.4` 的输出目录里读模型预测）：

```
8.4_时序分类模型/          建模与训练
8.5_模型性能综合评价/       指标、检验与可视化（不读原始数据，只读 8.4 的输出）
```

## 二、目录结构

```
8.4_时序分类模型/
  common.py                    公用配置 + 数据读取 + 指标函数（其它脚本都 import 它）
  seqdata.py                   原始 96Hz 序列数据集构建 + 手工时频特征 + PyTorch 训练循环
  00_unpack.py                 解包数据集 zip 并识别目录结构
  00_inspect.py                数据勘察：表结构、列名自检、sheet 名解析自检
  01_random_forest.py          路径一：手工特征表 + 随机森林（样本 = 每位受试者一行）
  01b_random_forest_seq.py     路径一（可比版）：手工时频特征 + 随机森林（样本 = 每个试次一行）
  02_lstm.py                   路径二：原始序列 + LSTM
  03_transformer.py            路径三：原始序列 + Transformer 编码器
  04_trajectory_demo.py        纵向时序：轨迹预测与趋势分析 demo

8.5_模型性能综合评价/
  eval_common.py               公用：路径探测、指标计算、按样本集签名分组、绘图风格
  01_metrics_roc.py            四维指标 + ROC 曲线 + AUC 的 Bootstrap 置信区间
  02_cm_calibration.py         混淆矩阵 + 校准曲线 + 阈值分析 + 漏诊误诊率
  03_compare_all.py            多模型横向对比 + McNemar 检验 + Bootstrap 近似检验 + 雷达图 + 排序热力图
```

> 运行顺序即上表从上到下。`00_unpack.py` 只在数据还是 zip 时需要跑；`00_inspect.py` 建议每次都先跑，它会把列名和 sheet 名解析情况打出来。

## 三、环境依赖

```bash
pip install pandas numpy scikit-learn scipy matplotlib openpyxl torch
```

**还需要一个中文字体**（Linux 服务器上经常一个都没有）：

```bash
sudo apt install fonts-noto-cjk              # 有 root
conda install -c conda-forge fonts-anaconda  # 有 conda
```

> ⚠️ **不装字体，图里所有中文会变成方框 □□□，但脚本照样正常结束、退出码是 0。**
> 表现是一堆 `UserWarning: Glyph ... missing from font(s) DejaVu Sans`，混在长输出里极易漏看，
> 然后拿到一整批中文全是方框的插图。代码开跑时会主动打印选中的字体名；
> 一个都找不到时打出一整屏 `!!!!` 警告 —— **跑之前先确认 `[字体] 中文字体 = …` 这行出现了。**
>
> 没有 root 的退路：下载任意中文 ttf/otf，然后 `export PD_FONT=/你的路径/NotoSansCJKsc-Regular.otf`。

## 四、数据

**本仓库不包含任何数据。** 研究数据涉及患者信息，请自行获取并遵守所属机构的数据使用协议。

代码期望的数据形态如下，**换成结构相同的自有数据即可运行**（无需改代码）：

- 一个**特征表**：每位受试者一行，含 ID 列（如 `ParticipantID`，取值形如 `HC_01` / `PD_01`）与若干步态时空参数列。
- 一个**原始序列表**：宽表形态，每个试次一个 sheet，sheet 名形如 `HC_01-行走-1`，列为

  ```
  <部位>-Sensor-Acce-{x,y,z}    三轴加速度
  <部位>-Sensor-Gyro-{x,y,z}    三轴角速度
  <部位>-Joint-Posi-{x,y,z}     三维关节位置
  <部位>-Sensor-Lost            丢失帧标记（可选）
  ```

**标签从 ID 前缀推导**（`HC` → 0，`PD` → 1），不依赖表里存在诊断列。若你的数据命名不同，改 `common.py` 顶部 CONFIG 区的 `ID_PREFIX_LABEL` 即可。

`common.py` 顶部 CONFIG 区是唯一需要改的地方：

| 配置项 | 默认值 | 含义 |
|---|---|---|
| `FEAT_MAIN` | 特征表名 | 手工特征表 |
| `RAW_MAIN` | 原始表名 | 原始惯性序列表 |
| `RAW_FS` | `96` | 采样率 (Hz) |
| `WIN` | `200` | 滑窗长度（帧） |
| `BODY_PART` | `"Hips"` | 取哪个佩戴部位 |
| `ONLY_WALK` | `True` | 只用「行走」试次 |
| `ID_PREFIX_LABEL` | `{"HC": 0, "PD": 1}` | ID 前缀 → 标签 |
| `DROP_LOST` | `False` | 是否把 `Lost=1` 的帧按缺失值插值 |

数据根目录**自动探测**，按「这个目录里有没有 `year*/` 或 `表1_` 开头的表」识别，不依赖文件夹叫什么名字。
想手动指定：`export PD_DATA_ROOT="/你的/数据/根目录"`。找不到时会**明确报错并列出搜过的位置**，不会静默改用随机数据。

## 五、输出

跑完 8.4 后，每个模型会在 `results/` 里留下三个文件：

```
8.4_<key>_prob.npy     预测概率
8.4_<key>_labels.npy   对应标签
8.4_<key>_sig.txt      样本集签名（说明这批样本是怎么来的）
```

`key` ∈ `{rf, rf_seq, lstm, transformer}`。**8.5 只读这些文件，不碰原始数据** —— 所以 8.5 可以独立重跑、反复调图，不必重新训练。

8.5 产出的图与表：ROC 曲线、混淆矩阵、校准曲线、阈值分析、漏诊误诊率、性能雷达图、综合排名热力图，以及全模型指标、AUC 置信区间、两两显著性检验等 CSV。

## 六、方法学说明（重要）

这一节是刻意写详细的 —— 下面每条都是**「不查就会得到看似正常、实则错误的结果」**的地方。

**1. 两两比较用的是 Bootstrap 近似检验，不是 DeLong 检验。**
`03_compare_all.py` 对样本重采样看 AUC 差值的分布；DeLong 是解析地算 AUC 的方差-协方差再构造 z 统计量。两者结论通常接近，但**方法名写错等于结论不可复现**。若要引用 DeLong，请另用 pROC 或自行实现。脚本内也写了同样的提醒。

**2. 不同脚本的样本集不一样，不能混进同一张 ROC。**
`01` 是「每位受试者一行」，`01b`/`02`/`03` 是「每个试次一行」—— 同一批人，样本数不同（示例数据中为 19 vs 57）。
若强行放在一起比较，指标、ROC、混淆矩阵会按错位的行去算，**且一个警告都不会有**。
因此 `save_pred` 会写一个**样本集签名**，8.5 按签名分组、只在最大的那一组内做对比，并明确打印每组包含哪些模型。

> 为什么不能只比对标签数组：两份样本数相同、但顺序不同的结果，标签完全可能逐字节一样，于是被并进同一组 —— 这是**静默**的。

**3. 单类标签会当场停下，不会继续出图。**
sklearn 1.6 起，单类标签下 `roc_auc_score` 不再报错，只发一个 `UndefinedMetricWarning` 然后**返回 nan**；而 Accuracy 仍是 1.0、混淆矩阵照样能画。于是一整组「准确率 1.000」的图看着毫无异常。代码在算指标前就检查类别数并 `exit 3`。

**4. 标签与概率必须一一对应。**
`save_pred` 先写 labels 再写 prob，中途被打断会留下「新 labels 配旧 prob」。加载时若长度不一致，该模型会被跳过并说明原因，而不是照算。

**5. `Lost` 帧的默认口径。**
原始表可能有 `<部位>-Sensor-Lost` 列。`DROP_LOST` 默认为 `False`，即**按原值（通常为 0）保留**，不做插值。
这个口径需要数据使用者按 `Lost` 列的实际含义自行决定 —— 代码里留成了显式开关，不会替你猜。

**6. 图表字体。** 见上文「环境依赖」，不装中文字体会静默产出方框。

## 七、引用与许可

若本代码对你的研究有帮助，请引用本教材：*《生成式 AI 驱动的数据分析》第 8 章*。

> **许可协议：待定。** 在仓库所有者补充 LICENSE 之前，请勿假定任何授权条款。
> 患者数据不在本仓库内，也不受本仓库许可协议约束。

---

# English Documentation

## 1. Overview

Given one gait dataset (Parkinson's disease patients, **PD**, vs. healthy older controls, **HC**), this project builds classification models along **two technical routes** and compares them:

| Route | Approach | Model | Sample unit |
|---|---|---|---|
| Hand-crafted feature table | Ready-made spatiotemporal gait parameters, one row per person | Random Forest | per subject |
| Hand-crafted time-frequency features | Features extracted from the raw 96 Hz sequences | Random Forest | per trial |
| Deep learning | End-to-end on raw sequences | LSTM / Transformer encoder | per trial |

Beyond classification there is a **forecasting demo**: autoregressive rollout of future trajectory from history — a minimal "generative" example on time series.

The code has two parts and they **must sit in sibling directories** (`8.5` reads model predictions from `8.4`'s output directory):

```
8.4_时序分类模型/          modelling & training
8.5_模型性能综合评价/       metrics, tests & visualisation (reads only 8.4's outputs, never raw data)
```

## 2. Layout

```
8.4_时序分类模型/
  common.py                    shared config + data loading + metric helpers (imported by every script)
  seqdata.py                   raw 96 Hz dataset builder + hand-crafted time-frequency features + PyTorch loop
  00_unpack.py                 unzip the dataset archive and identify its layout
  00_inspect.py                data inspection: column check, sheet-name parsing self-test
  01_random_forest.py          route 1: feature table + Random Forest (one row per subject)
  01b_random_forest_seq.py     route 1 (comparable version): time-frequency features + RF (one row per trial)
  02_lstm.py                   route 2: raw sequences + LSTM
  03_transformer.py            route 3: raw sequences + Transformer encoder
  04_trajectory_demo.py        longitudinal demo: trajectory forecasting and trend analysis

8.5_模型性能综合评价/
  eval_common.py               shared: path discovery, metrics, grouping by sample-set signature, plot style
  01_metrics_roc.py            metrics + ROC + Bootstrap confidence interval for AUC
  02_cm_calibration.py         confusion matrix + calibration curve + threshold analysis + miss/false-alarm rates
  03_compare_all.py            cross-model comparison + McNemar + Bootstrap approximation + radar + ranking heatmap
```

Run them top to bottom. `00_unpack.py` is only needed while the data is still zipped; run `00_inspect.py` first every time — it prints the column and sheet-name parsing results.

## 3. Requirements

```bash
pip install pandas numpy scikit-learn scipy matplotlib openpyxl torch
```

**A CJK font is also required** (Linux servers frequently have none):

```bash
sudo apt install fonts-noto-cjk
```

> ⚠️ **Without it, every Chinese label renders as a tofu box □□□ — while the scripts still exit 0.**
> You get a flood of `UserWarning: Glyph ... missing from font(s) DejaVu Sans`, easy to miss in long output,
> and a complete set of figures with no readable Chinese. The code prints the font it selected at start-up,
> and emits a full screen of `!!!!` warnings when it finds none — **check that `[字体] 中文字体 = …` appears.**
>
> No root? Download any CJK ttf/otf and `export PD_FONT=/path/to/NotoSansCJKsc-Regular.otf`.

## 4. Data

**No data is included in this repository.** The research data contains patient information; obtain it
yourself and comply with your institution's data-use agreement.

The expected shape is below — **substitute your own data with the same structure and it runs unchanged**:

- a **feature table**: one row per subject, with an ID column (`ParticipantID`, values like `HC_01` / `PD_01`) and spatiotemporal gait parameters;
- a **raw sequence table**: wide format, one sheet per trial, sheet names like `HC_01-行走-1`, columns such as

  ```
  <part>-Sensor-Acce-{x,y,z}    tri-axial acceleration
  <part>-Sensor-Gyro-{x,y,z}    tri-axial angular velocity
  <part>-Joint-Posi-{x,y,z}     3-D joint position
  <part>-Sensor-Lost            lost-frame flag (optional)
  ```

**Labels are derived from the ID prefix** (`HC` → 0, `PD` → 1); no diagnostic column is required.
If your naming differs, edit `ID_PREFIX_LABEL` in the CONFIG block at the top of `common.py`.

The data root is **auto-detected** (it looks for a `year*/` folder or a `表1_`-prefixed table, so the directory
name does not matter). To set it explicitly: `export PD_DATA_ROOT="/your/data/root"`. When nothing is found it
**fails loudly and lists every location probed** rather than silently switching to random data.

## 5. Outputs

After running 8.4, each model leaves three files in `results/`:

```
8.4_<key>_prob.npy     predicted probabilities
8.4_<key>_labels.npy   matching labels
8.4_<key>_sig.txt      sample-set signature (how this batch of samples was produced)
```

with `key` ∈ `{rf, rf_seq, lstm, transformer}`. **8.5 reads only these files and never touches raw data** —
so you can re-run 8.5 freely to re-tune plots without retraining.

8.5 produces ROC curves, confusion matrices, calibration curves, threshold analysis, miss/false-alarm rates,
a performance radar chart and a ranking heatmap, plus CSVs of full metrics, AUC confidence intervals and
pairwise significance tests.

## 6. Methodological Notes (important)

This section is deliberately long — every item below is a place where **an unchecked run yields results that
look fine and are wrong**.

**1. Pairwise comparison is a Bootstrap approximation, NOT DeLong.**
`03_compare_all.py` resamples the data to obtain the distribution of AUC differences; DeLong computes the
analytic variance-covariance of AUC and forms a z-statistic. Conclusions usually agree, but **using the wrong
method name makes the result irreproducible**. To cite DeLong, use pROC or implement it yourself.

**2. Different scripts use different sample sets — they must not share one ROC plot.**
`01` is one row per subject; `01b`/`02`/`03` are one row per trial — the same people, different sample counts
(19 vs 57 in the example data). Forcing them together computes metrics, ROC and confusion matrices on
misaligned rows, **with no warning whatsoever**. Hence `save_pred` writes a **sample-set signature**, and 8.5
groups by signature, compares only within the largest group, and prints which models each group contains.

> Why not compare label arrays alone: two results with the same sample count but different order can have
> byte-identical labels, so they get merged into one group — **silently**.

**3. Single-class labels halt the run.** Since scikit-learn 1.6, `roc_auc_score` no longer raises on a
single-class target — it emits an `UndefinedMetricWarning` and **returns nan**, while Accuracy stays 1.0 and
the confusion matrix still plots. Code checks the class count before computing metrics and exits with code 3.

**4. Labels and probabilities must correspond one-to-one.** `save_pred` writes labels before probabilities; an
interrupted run leaves new labels paired with stale probabilities. On load, a length mismatch causes that
model to be skipped with an explanation rather than scored anyway.

**5. Default handling of `Lost` frames.** The raw table may carry a `<part>-Sensor-Lost` column. `DROP_LOST`
defaults to `False`, i.e. frames are kept at their original value (usually 0) rather than interpolated. The
correct convention depends on what the column actually means in your data, so it is left as an explicit switch.

**6. Figure fonts.** See Requirements above — a missing CJK font silently produces tofu boxes.

## 7. Citation & License

If this code helps your research, please cite *Chapter 8 of Data Analysis Driven by Generative AI*.

> **License: TBD.** Until the repository owner adds a LICENSE file, assume no grant of rights.
> Patient data is not part of this repository and is not covered by its license.
