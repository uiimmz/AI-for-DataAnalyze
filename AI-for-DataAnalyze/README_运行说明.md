# 8.4 / 8.5 代码运行说明（虚拟桌面）

## 一、这些代码是干什么的

对应教材《生成式 AI 驱动的数据分析》第 8 章「时序数据分析：帕金森患者步态运动参数」：

| 小节 | 内容 | 对应目录 |
|---|---|---|
| **8.4** | 时序分类模型：随机森林、LSTM、Transformer 对比；纵向时序模型（轨迹预测与趋势分析） | `8.4_时序分类模型/` |
| **8.5** | 模型分类性能综合评价：四维指标、ROC-AUC、混淆矩阵、校准曲线、显著性检验 | `8.5_模型性能综合评价/` |

**两条技术路线**（与《数据组3案例分析》的写法保持一致，便于这套教材前后呼应）：
- **手工提特征 + 机器学习** → 读 `year2帕金森病诊断步态特征数据`（Matlab 已提好的步态特征表），喂**随机森林**
- **深度学习自动提特征** → 读 `year2帕金森病患者与健康老年人行走运动参数数据`（原始 96Hz 惯性序列），喂 **LSTM / Transformer**

---

## 二、文件清单

```
8.4_时序分类模型/
  common.py                   公用配置 + 数据读取 + 指标函数（其它脚本都 import 它）
  seqdata.py                  原始 96Hz 序列数据集构建 + 手工时频特征 + PyTorch 训练循环
  00_unpack.py                ⓪ 解包 + 认数据集（数据是 zip 没解的话先跑这个）
  00_inspect.py               ① 数据勘察（先跑这个）
  01_random_forest.py         ② 手工特征表 + 随机森林（样本=每位受试者一行）
  01b_random_forest_seq.py    ③ 手工时频特征 + 随机森林（样本=每个 sheet 一行）★可比版
  02_lstm.py                  ④ 原始序列 + LSTM
  03_transformer.py           ⑤ 原始序列 + Transformer
  04_trajectory_demo.py       ⑥ 纵向时序：轨迹预测与趋势分析 demo
  results/                    所有输出（图 PNG / 表 CSV）都落在这里（运行后自动生成）

8.5_模型性能综合评价/
  eval_common.py         公用：路径、指标、模型概率收集与**样本集分组**、绘图风格
  01_metrics_roc.py      ① 四维指标 + ROC + AUC 置信区间
  02_cm_calibration.py   ② 混淆矩阵 + 校准曲线 + 阈值分析
  03_compare_all.py      ③ 多模型汇总 + McNemar 检验 + Bootstrap 近似检验 + 雷达图
  results/               输出目录（运行后自动生成）
```

> ⚠️ **`03_compare_all.py` 做的是 Bootstrap 近似检验，不是 DeLong。**
> DeLong 是解析地算 AUC 的方差-协方差再构造 z 统计量；这个脚本是对样本重采样看
> AUC 差值的分布。两者结论通常接近，但**方法名写错等于结论不可复现** ——
> 正文若要写 DeLong，得另用 pROC 或自己实现协方差那一套，
> 不能拿本脚本的输出当 DeLong 引用。脚本里也写了同样的提醒。

### ⚠️ 为什么有 01 和 01b 两个随机森林

这是**样本集可比性**的问题，写正文时值得交代一句：

| 脚本 | 数据源 | 一个样本 = | 样本数（实测） |
|---|---|---|---|
| `01_random_forest.py` | **表4**（Matlab 已提好的诊断步态特征表） | 一位受试者 | **19**（HC 8 / PD 11） |
| `01b_random_forest_seq.py` | 原始 96Hz 惯性序列 | 一个 sheet（一次采集） | **57**（19 人 × 3 个行走组次） |
| `02_lstm.py` / `03_transformer.py` | 原始 96Hz 惯性序列 | 一个 sheet（一次采集） | **57** |

> **为什么 `01` 必须用表4**：`year2帕金森病诊断步态特征数据/` 下有四张表。表1/表2/表3
> 是**每人多行**（102 人的 试次×左右脚），列是 `Participants`/`Foot`/`Group_index`，
> 而 `Group_index` 取值 1~4 是**组次序号不是诊断** —— 这几张表**根本没有诊断列**，
> 而且用它们会让同一个人同时落进训练集和测试集，指标虚高。
> 表4 是**一人一行**，诊断写在 `ParticipantID` 的前缀里（`HC_01…HC_08` / `PD_01…PD_11`）。
> `01_random_forest.py` 是「每人一行 + 普通 StratifiedKFold」，只有表4 符合。

**`01` 和 `02/03` 的样本集不同，不能直接放在一张 ROC 图上比。**
`01b` 从与 LSTM/Transformer **完全相同的那批序列**里手工提时域+频域特征（对应教材表 4、表 5，
含帕金森关键的 FI 冻结指数），再喂随机森林——**这个才是"随机森林 vs LSTM vs Transformer"三模型
公平对比的入口**。

`8.5` 的脚本已自动按样本集分组，只在**同一组内**做对比，不会把不同样本集的模型混在一起算检验。

---

## 三、在虚拟桌面上怎么跑

### 1. 放文件

把 `8.4_时序分类模型` 和 `8.5_模型性能综合评价` **两个文件夹整个**拷到虚拟桌面任意位置，例如：

```
C:\Users\<你的用户名>\Desktop\帕金森分析\
    ├── 8.4_时序分类模型\
    └── 8.5_模型性能综合评价\
```

> ⚠️ 两个文件夹必须保持**同级**关系，因为 8.5 要去 8.4 的 `results/` 里读模型输出。
>
> ⚠️ **只传这两个文件夹。** 旁边那个 `_t/` 是本地自测用的脚手架（会造假数据），
> **不要**打包上传；`results/`、`__pycache__/` 也不用传，跑的时候会自动生成。
>
> ⚠️ **更新代码时要 13 个 `.py` 一起传，不要只传改动过的那几个。**
> 脚本和 `common.py` / `eval_common.py` 是一整套，混用新旧版本会以各种形式炸：
> 比如新的 `01_random_forest.py` 调 `save_pred(key, y, prob, sig)`（4 个参数）配上旧的
> `common.py`（3 个参数），报 `TypeError: save_pred() takes 3 positional arguments but 4 were given`；
> 更麻烦的是**不报错**的那种 —— 旧 `common.py` 找不到标签列时会拿 `Group_index` 当诊断标签，
> 于是照样跑完、照样出图，而 `Group_index` 只是组次序号（1~4），
> 结果是 AUC 0.517 这种"看着正常其实全错"的数字。传完对一下开头的
> `[字体] 中文字体 = …` / `[读入] 表4_…` / `[标签] …按 ID 列 'ParticipantID' 的前缀判定` 三行在不在。

### 2. 确认数据路径

**不用改代码** —— 数据根目录是**自动探测**的，按「这个目录里有没有 `year*/` 或 `表1_`」
认，不依赖文件夹叫什么名字。它会依次找：环境变量 `PD_DATA_ROOT` → `~/pd_data`（解包工作目录）
→ 平台上你账号的确定位置 `/eaas/default/groups/casestudy_cnu/home/share` → `Z:\` → `/mnt`、`/data` → 家目录。

想手动指定就设环境变量，再重跑：

```bash
export PD_DATA_ROOT="/eaas/default/groups/casestudy_cnu/home/share/二期-多模态帕金森神经退行性疾病的步态动力学与可穿戴健康监测"
```

> ★ 注意路径里**没有** `phdauser004` 那一层 —— 就是 `…/casestudy_cnu/home/share/`。
> 而且 `…/home/*/share` 这种通配符**匹配不到** `home/share`（少一层），
> 所以这个位置是写死在代码里的，不能指望自动 glob 兜住。

> 找不到数据时**不会**静默用随机数据，会明确报错并列出它找过哪些位置 —— 照着改就行。

### 3. 装依赖（第一次跑之前）

```bat
pip install pandas numpy scikit-learn scipy matplotlib openpyxl torch
```

PyTorch 要装 GPU 版的话用官网给的命令（你那边已有 GPU + PyTorch，可跳过）。

**还要装一个中文字体**（Linux 平台上经常一个都没有）：

```bash
sudo apt install fonts-noto-cjk          # 有 root
conda install -c conda-forge fonts-anaconda   # 有 conda
```

> ⚠️ **不装字体的话，图里所有中文都会变成方框 □□□，但脚本照样正常结束、退出码是 0。**
> 表现是一堆 `UserWarning: Glyph 38543 ... missing from font(s) DejaVu Sans` ——
> 这些警告混在几百行输出里很容易漏掉，然后你拿走的就是一整套中文全是方框的教材插图。
> 代码里的 `setup_cjk_font()` 会在开跑时**主动打印**它选中的字体名；一个都找不到时会打出
> 一整屏 `!!!!` 警告。**跑之前先确认那行 `[字体] 中文字体 = …` 出现了。**
>
> 没有 root 的退路：随便下个中文 ttf/otf 放到自己目录，然后
> `export PD_FONT=/你的路径/NotoSansCJKsc-Regular.otf` 再跑。

### 4. 按顺序运行

```bat
cd /d "C:\Users\<你的用户名>\Desktop\帕金森分析\8.4_时序分类模型"

python 00_unpack.py             :: ⓪ 数据还是 zip 的话先跑这个（已解包可跳过）
python 00_inspect.py            :: ① 先跑这个！看清数据结构
python 01_random_forest.py      :: ② 随机森林（吃诊断步态特征表）
python 01b_random_forest_seq.py :: ③ 随机森林（吃原始序列的手工时频特征）★
python 02_lstm.py               :: ④ LSTM
python 03_transformer.py        :: ⑤ Transformer
python 04_trajectory_demo.py    :: ⑥ 轨迹预测 demo

cd /d "..\8.5_模型性能综合评价"
python 01_metrics_roc.py
python 02_cm_calibration.py
python 03_compare_all.py
```

> **中文乱码怎么办**：cmd 里先执行 `chcp 65001` 再跑；或者直接用 Spyder / PyCharm 运行。
> **建议**：00～04 每个脚本跑完后，把 `results/` 里生成的 CSV 和 PNG 收好，就是教材插图和数据来源。

### 5. 注意：8.5 依赖 8.4 的输出

`8.5` 的脚本会自动去 `../8.4_时序分类模型/results/` 找 `8.4_*_prob.npy`。
所以**先跑完 8.4 的 01 / 01b / 02 / 03**，再跑 8.5。

找不到时 8.5 **会直接停下退出（exit 1）**，不会自己编数据往下画。
（随机数据照样能画出很漂亮的 ROC 和混淆矩阵，万一被当成真实结果写进教材就麻烦了。）
只想预览图表版式的话显式加参数：`python 01_metrics_roc.py --demo`，
产物文件名会带 `DEMO_` 前缀，不会覆盖真结果。

---

## 四、数据结构（已逐项确认，无需再勘察）

2026-09-25 已按实拍的表/表头逐项核对，`common.py` 与四个脚本都已按真实结构改好：

| 项 | 确认结果 |
|---|---|
| 主特征表 | **表4**（19 人 = HC 8 + PD 11），诊断在 `ParticipantID` 前缀里，无独立标签列 |
| 表1/2/3 | 每人多行（102 人 × 试次 × 左右脚），`Group_index` 是组次序号不是诊断，**无诊断列** → 不能用作分类标签 |
| 原始表 | **宽表**，无 `BodyPart` 列；部位写在列名里，形如 `<部位>-Sensor-Acce-x`、`<部位>-Joint-Posi-x` |
| sheet 命名 | `HC_01-行走-1`；第一个 sheet 是空的占位 `Sheet1`；共 115 个（19 人 × 6 组次 + 1） |
| 部位 | `BODY_PART = "Hips"` |

> `00_inspect.py` 仍会生成 `results/00_勘察报告.txt`，跑完扫一眼能再次确认。

### ⚠️ 一个待你拍板的口径：`Hips-Sensor-Lost`

原始表每个部位有一列 `*-Sensor-Lost`（0/1，1 = 该帧追踪丢失），Lost 帧附近
**加速度/关节速度会整段变成 0**。那些 0 是「没测到」而不是「测得 0」，直接拿去算
时域/频域特征会得到假的低能量、假的冻结指数，而曲线上看不出来。

但**还不能确认** Lost=1 时是不是所有通道都不可用（可能它只表示骨骼解算丢失、
IMU 三轴仍然有效），所以没有替你定，处理方式保留在 `common.py` 顶部：

```python
DROP_LOST = False     # 保持现状：照单全收，结果与之前一致
                      # 改成 True：把 Lost 帧当缺失值，插值补上
```

无论开关，脚本都会提示「有 N/M 帧全通道为 0」，跑完看一眼日志就知道影响多大。
**确认语义后告诉我，我再定成最终值。**

---

## 五、每个脚本在教材正文里讲什么

| 脚本 | 正文可引用的知识点 |
|---|---|
| `01_random_forest.py` | Bagging 抽样、特征随机性、多数投票；临床步态特征的重要性排序（哪些参数最能区分 PD/HC） |
| `01b_random_forest_seq.py` | 时域（均值/标准差/能量/熵/偏度/峭度/IQR）+ 频域（重心频率/FI 冻结指数）手工特征体系；时域 vs 频域贡献占比 |
| `02_lstm.py` | 输入门/遗忘门/输出门、梯度消失问题、双向 LSTM 捕捉前后文步态模式 |
| `03_transformer.py` | 自注意力对关键时间步加权、位置编码、与 LSTM 循环结构的对比 |
| `04_trajectory_demo.py` | 序列到序列的轨迹预测、RMSE 与 DTW 距离、线性外推基线对照 |
| `8.5/01` | Accuracy/Precision/Recall/F1 四维指标公式、ROC 曲线、Bootstrap 置信区间 |
| `8.5/02` | 混淆矩阵、TP/FP/FN/TN 临床含义（漏诊 vs 误诊）、校准曲线、阈值调整 |
| `8.5/03` | 多模型横向对比、McNemar 检验、**Bootstrap 近似检验（不是 DeLong）**、雷达图与热力图 |

**临床解释要点**（写正文时可直接用）：本案例中 **Recall（召回率）= 不漏诊**，**Precision（精确率）= 不误诊**。
帕金森筛查场景下漏诊代价更高，因此可适当下调判决阈值提高 Recall——`8.5/02_cm_calibration.py`
的阈值分析正好给出了这条权衡曲线的证据。
