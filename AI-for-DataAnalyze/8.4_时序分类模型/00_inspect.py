# -*- coding: utf-8 -*-
"""
00_inspect.py —— 数据勘察（第一步，先跑这个）
作用：把数据目录里两个数据集的真实结构打印出来，确认列名后再跑后面的建模脚本。
      数据根目录自动探测（虚拟桌面 Z 盘 / Linux 远程分析平台 /eaas/ 都能用）。
跑法：python 00_inspect.py
      屏幕会打印，同时把完整输出存到 results/00_勘察报告.txt
      把那个 txt 发回给我，我据此把 common.py 的 CONFIG 定死。
"""
import sys
from common import *

SEP = "=" * 70

# 把输出同时写入文件（避免中文控制台乱码，也方便直接发回）
_TEE = open(os.path.join(OUT, "00_勘察报告.txt"), "w", encoding="utf-8")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = _Tee(sys.__stdout__, _TEE)


def _first_data_sheet(xl):
    """找第一个「有内容」的 sheet。

    ★ 不能用 sheet_names[0]：这类 workbook 常常第一个 sheet 是个空的占位
      「Sheet1」（0 行 0 列），在它上面 describe() 会抛
      ValueError: Cannot describe a DataFrame without columns，
      把整段勘察带崩 —— 2026-09-25 实测就是这样，原始序列表的信息全丢了。
    """
    for s in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=s, nrows=50)
        except Exception:
            continue
        if df.shape[0] > 0 and df.shape[1] > 0:
            return s
    return xl.sheet_names[0]


def _sheet_tasks(names):
    """按「任务」汇总 sheet 名，如 'HC_01-闭眼站立-1' -> '闭眼站立'。
    115 个 sheet 逐个列不方便看，汇总一下就知道有哪些任务、各多少组次。"""
    from collections import Counter
    cnt = Counter()
    for n in names:
        parts = [p for p in re.split(r"[-_]", str(n)) if p]
        mid = [p for p in parts if not re.fullmatch(r"[A-Za-z]{1,4}|\d+", p)]
        cnt["-".join(mid) if mid else str(n)] += 1
    return cnt


def dump_id_values(df):
    """把「像 ID」的列的全部取值列出来。

    ★ 诊断标签经常就编码在编号前缀里（HC_01/PD_01、xj-co-001/xj-pd-001），
      只印 3 个样例根本看不出来（排序后 co 永远排在 pd 前面）。
    """
    for c in df.columns:
        low = str(c).lower()
        if not any(k in low for k in ("id", "participant", "subject", "编号", "受试者")):
            continue
        u = sorted(map(str, df[c].dropna().unique()))
        print(f"\n  [ID列] {c!r}  共 {len(u)} 个不同值")
        if len(u) <= 60:
            print(f"    全部取值：{u}")
        else:
            from collections import Counter
            pref = Counter(re.sub(r"\d+$", "", x) for x in u)
            print(f"    前缀分布：{dict(pref)}")
            print(f"    前 20 个：{u[:20]}")
            print(f"    后 10 个：{u[-10:]}")


def dump_table(folder, prefix, n=5, max_cols=150):
    print(f"\n{SEP}\n【表】{prefix}\n{SEP}")
    xl = pd.ExcelFile(find_file(folder, prefix))
    print(f"sheet 数量：{len(xl.sheet_names)}")
    if len(xl.sheet_names) <= 40:
        print(f"sheet 名（全部）：{xl.sheet_names}")
    else:
        print(f"sheet 名（前 6 个）：{xl.sheet_names[:6]}")
        print(f"sheet 名（后 3 个）：{xl.sheet_names[-3:]}")
        print(f"\nsheet 任务汇总（共 {len(xl.sheet_names)} 个）：")
        for t, k in _sheet_tasks(xl.sheet_names).most_common():
            print(f"    {t!r}: {k} 个")

    s = _first_data_sheet(xl)
    if s != xl.sheet_names[0]:
        print(f"\n★ 第一个 sheet「{xl.sheet_names[0]}」是空的，已改用「{s}」")
    df = pd.read_excel(xl, sheet_name=s)
    print(f"\nsheet「{s}」形状：{df.shape}")
    if df.shape[1] == 0:
        print("（这个 sheet 是空的，后面的统计跳过）")
        return xl
    cols = list(df.columns)
    show = cols if len(cols) <= max_cols else cols[:40]
    print(f"列名（共 {len(cols)} 个{'' if len(cols) <= max_cols else '，中间省略'}）：")
    for i, c in enumerate(show):
        u = df[c].dropna().unique()
        sample = ", ".join(map(str, u[:3]))
        print(f"  [{i:>3}] {c!r:<42} dtype={str(df[c].dtype):<8} "
              f"nunique={len(u):<6} 例: {sample[:60]}")
    if len(cols) > max_cols:
        print(f"  ...（省略 {len(cols) - 45} 列）...")
        for i in range(len(cols) - 5, len(cols)):
            c = cols[i]
            u = df[c].dropna().unique()
            print(f"  [{i:>3}] {c!r:<42} dtype={str(df[c].dtype):<8} "
                  f"nunique={len(u):<6} 例: {', '.join(map(str, u[:3]))[:60]}")
    print(f"\n前 {n} 行：")
    print(df.head(n).to_string())
    print(f"\n数值列统计：")
    print(df.describe().T.to_string())
    dump_id_values(df)
    return xl


if __name__ == "__main__":
    print(f"[环境] platform={sys.platform}  python={sys.version.split()[0]}  "
          f"user={os.environ.get('USER') or os.environ.get('USERNAME') or '?'}")
    print(f"[环境] 工作目录 = {os.getcwd()}")
    print(f"[环境] 结果目录 = {OUT}")
    print(f"DATA_ROOT = {DATA_ROOT!r}")
    print(f"候选数据根 {len(ALL_ROOTS)} 个（find_file 会挨个去认）：")
    for r in ALL_ROOTS:
        print(f"   ● {r}")
        try:
            for d in sorted(os.listdir(r))[:20]:
                print(f"        - {d}")
        except OSError as e:
            print(f"        (列不出：{e})")
    if not ALL_ROOTS:
        print("\n一个候选目录都不存在 —— 下面这些地方的实际情况请贴回来：")
        for p in ["/eaas/default/groups", os.path.expanduser("~"),
                  os.path.expanduser("~/share")]:
            ok = os.path.isdir(p)
            print(f"   {p}  ->  {'存在' if ok else '不存在'}")
            if ok:
                try:
                    for d in sorted(os.listdir(p))[:20]:
                        print(f"        - {d}")
                except OSError as e:
                    print(f"        (列不出：{e})")

    # 0) 两个数据集目录里到底有哪些文件
    #    ★ 必须列出来：脚本只按固定的表名去认，目录里有没有别的表
    #      （比如「行走」是不是另有表2、表3）光看现有输出看不出来。
    for tag, folder in (("特征数据集", DIR_FEAT), ("原始数据集", DIR_RAW)):
        print(f"\n{SEP}\n[{tag}] {folder}\n{SEP}")
        shown = False
        for d in ALL_ROOTS:
            p = os.path.join(d, folder)
            if not os.path.isdir(p):
                continue
            shown = True
            print(f"  目录：{p}")
            try:
                for f in sorted(os.listdir(p)):
                    fp = os.path.join(p, f)
                    sz = f"{os.path.getsize(fp)/1e6:.2f} MB" if os.path.isfile(fp) else "<目录>"
                    print(f"     - {f:<70} {sz}")
            except OSError as e:
                print(f"     (列不出：{e})")
        if not shown:
            print("  （哪个候选根下都没有这个目录）")

    # 1) 主特征表：表1~表4
    for p in ["表1_2022-2023帕金森病诊断步态特征数据",
              "表2_2022-2023帕金森病诊断步态特征数据",
              "表3_2022-2023帕金森病诊断步态特征数据",
              "表4_2022-2023帕金森病诊断步态特征数据"]:
        try:
            dump_table(DIR_FEAT, p)
        except Exception as e:
            print(f"\n[跳过] {p} -> {type(e).__name__}: {e}")

    # 2) 自动探测标签列/ID列是否命中
    try:
        df = read_table(DIR_FEAT, FEAT_MAIN)
        print(f"\n{SEP}\n【自动探测结果】\n{SEP}")
        idc = guess_col(df, ID_KEYS)
        print("ID  列 ->", idc)
        print("标签列 ->", guess_label(df))
        lc = guess_label(df, exclude=(idc,))
        if lc is not None:
            vc = df[lc].value_counts(dropna=False)
            print(f"\n★ 标签列 {lc!r} 的取值分布（这个必须自己核对）：")
            for v, k in vc.items():
                print(f"      {v!r}: {k} 行")
            if len(vc) > 2:
                print(f"\n  ⚠️ 这一列有 {len(vc)} 个取值，不是二分类。"
                      f"\n     直接按「含 PD/1 就是患者」去转，会得到一组没有意义的标签。"
                      f"\n     请确认：真正的 PD/HC 是靠哪一列、还是靠 ID 前缀区分的？")
            print(f"\n  换算成 0/1 后：PD {int(to_binary(df[lc]).sum())} / "
                  f"HC {int((1 - to_binary(df[lc])).sum())}  ← 若与预期不符，别往下跑")
        elif ID_PREFIX_LABEL and idc:
            # 表4 这类：没有标签列，诊断编码在 ID 前缀里。
            y_s, unknown = label_from_id(df[idc], ID_PREFIX_LABEL)
            print(f"\n★ 表里**没有标签列** —— 改按 ID 列 {idc!r} 的前缀判定分组"
                  f"（ID_PREFIX_LABEL={ID_PREFIX_LABEL}）")
            print(f"      PD {int((y_s == 1).sum())} / HC {int((y_s == 0).sum())}"
                  f" / 认不出 {len(unknown)} 行")
            print(f"      ID 列全部取值：{sorted(map(str, df[idc].dropna().unique()))}")
            if unknown:
                print(f"      ⚠️ 认不出的：{unknown[:10]} —— get_xy 会直接报错停下")
        else:
            print("\n★ 既没有标签列、也没配 ID_PREFIX_LABEL —— "
                  "01_random_forest.py 会在 get_xy 处停下。")
    except Exception as e:
        print(f"[探测失败] {type(e).__name__}: {e}")

    # 3) 原始 96Hz 惯性数据
    #    ★ 这里刻意拆成**三段各自独立的 try**。原来是一整个大 try，里面任何一句出问题
    #      都会把后面的「sheet 名解析自检」整段带走 —— 而那一段恰恰是判断
    #      「有没有行走组次 / 分组能不能从 sheet 名解析出来」的唯一依据。
    #      2026-09-25 实测就是这么丢的：报告里只剩一句「原始数据读取失败」，
    #      列名、部位、任务分布全都没打出来，只能靠拍照。
    xl = None
    try:
        xl = dump_table(DIR_RAW, RAW_MAIN)
    except Exception as e:
        print(f"\n[原始表打开失败] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    if xl is not None:
        # 3a) 第一个有内容的 sheet 的列名 —— 全部打印，不截断
        try:
            s0 = _first_data_sheet(xl)
            print(f"\n{SEP}\n【原始序列】sheet「{s0}」的列\n{SEP}")
            raw = pd.read_excel(xl, sheet_name=s0, nrows=2000)
            print(f"共 {raw.shape[1]} 列：")
            for i, c in enumerate(raw.columns):
                print(f"  [{i:>3}] {str(c)!r}")
            if "BodyPart" in raw.columns:
                print(f"\nBodyPart 取值（全部）："
                      f"{sorted(map(str, raw['BodyPart'].dropna().unique()))[:60]}")
            else:
                print("\nBodyPart 取值：无该列"
                      "（部位大概是写在**列名**里的宽表，如 Hips-Sens-AccX）")
            # ★ 时间列名各家导出不一样，必须自己找。原来写死 raw.get('Timestamp')：
            #   表里没有这一列时它返回 None，pd.to_numeric(None) 得到的是一个 float，
            #   再 .dropna() 就是 AttributeError —— 一句采样率统计把整段带崩。
            tcol = next((c for c in raw.columns
                         if str(c).strip().lower() in
                         ("timestamp", "time", "时间", "时间戳", "帧时间", "frametime")), None)
            if tcol is None:
                print("\n采样间隔中位数：表里没有时间列"
                      "（找过 Timestamp / Time / 时间 / 时间戳 / 帧时间）")
            else:
                dt = np.diff(pd.to_numeric(raw[tcol], errors="coerce").dropna().values)
                if len(dt):
                    print(f"\n采样间隔中位数：{np.median(dt)}"
                          f"  （时间列={tcol!r}，约 {1.0 / np.median(dt):.2f} Hz）")
                else:
                    print(f"\n采样间隔中位数：{tcol!r} 这一列没读到数值")
        except Exception as e:
            print(f"\n[原始序列读列失败] {type(e).__name__}: {e}")

        # 3b) 「行走」sheet 的列要单独打一遍 —— 第一个有内容的 sheet 往往是闭眼站立，
        #     而本章只用行走组次。两类 sheet 的列名万一不一样，只看第一个是看不出来的。
        try:
            walk = [s for s in xl.sheet_names
                    if "行走" in str(s) or "WALK" in str(s).upper()]
            if walk:
                w = pd.read_excel(xl, sheet_name=walk[0], nrows=2000)
                print(f"\n{SEP}\n【原始序列·行走】sheet「{walk[0]}」共 {w.shape[1]} 列\n{SEP}")
                for i, c in enumerate(w.columns):
                    print(f"  [{i:>3}] {str(c)!r}")
            else:
                print("\n★ 一个含「行走」的 sheet 都没有 —— ONLY_WALK=True 会把数据全过滤掉。")
        except Exception as e:
            print(f"\n[行走 sheet 读列失败] {type(e).__name__}: {e}")

        # 3c) sheet 名解析自检：直接回答「有没有行走组次」「分组能不能解析出来」
        try:
            names = [s for s in xl.sheet_names if s != xl.sheet_names[0]]
            subs, labs, bad = {}, {}, []
            for s in names:
                g = sheet_label(s)
                if g is None:
                    bad.append(s)
                    continue
                labs[g] = labs.get(g, 0) + 1
                subs[sheet_subject(s)] = subs.get(sheet_subject(s), 0) + 1
            print(f"\n【sheet 名解析自检】共 {len(names)} 个数据 sheet")
            print(f"  能判出分组的：{len(names) - len(bad)} 个   判不出的：{len(bad)} 个")
            if bad:
                print(f"  判不出的例子：{bad[:5]}")
            print(f"  分组计数：{labs}   （1=PD 0=HC）")
            print(f"  受试者数：{len(subs)}")
            print(f"  每人组次分布：{sorted(set(subs.values()))}")
            walk = [s for s in names if "行走" in str(s)]
            print(f"  ★ 含「行走」的 sheet：{len(walk)} 个"
                  f"{'  ← 一个都没有！ONLY_WALK=True 会把数据全过滤掉' if not walk else ''}")
            if walk:
                print(f"    例：{walk[:5]}")
        except Exception as e:
            print(f"\n[sheet 名解析自检失败] {type(e).__name__}: {e}")

    print(f"\n{SEP}\n勘察完毕。完整输出已保存到："
          f"{os.path.join(OUT, '00_勘察报告.txt')}\n把这个 txt 发回即可。\n{SEP}")
    sys.stdout = sys.__stdout__
    _TEE.close()
