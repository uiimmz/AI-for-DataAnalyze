# -*- coding: utf-8 -*-
"""
00_unpack.py —— 数据解包（Linux 平台的第一步，先跑这个）

为什么需要这一步：
  平台上的数据是两个 zip，而 01/02/03/04 等脚本都按「文件夹 + xlsx」查找。
  另外数据目录可能只读，所以统一解到可写的工作目录（默认 ~/pd_data）。

跑法：python 00_unpack.py
      - 在数据目录里找 *.zip（默认就是你账号的 share 目录）
      - 解压到 ~/pd_data/<zip名>/
      - 中文文件名乱码会自动纠正（Windows 压的 zip 常见问题）
      - 如果 zip 里还套着 zip，会自动再解一层
      - 最后打印「数据到底在哪」，并给出要 export 的路径

参数：python 00_unpack.py --force            已有结果也重新解一遍
      PD_DATA_ROOT=/别的/目录 python 00_unpack.py   # 数据不在默认位置时
"""
import os, sys, glob, shutil, zipfile

try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("PD_WORK", os.path.expanduser("~/pd_data"))
FORCE = "--force" in sys.argv
ALL_ZIPS = "--all" in sys.argv        # 不加就只解含目标数据集的那个 zip

# ---- 查找 zip 的候选目录：(路径, 最大搜索深度) ----
# 深度按目录性质给：数据目录可以往深里找几层；家目录只准看直接子项，
# 否则会扫进 conda / AppData / 应用缓存，一次翻出几百个无关 zip（本机自测踩过）。
_HOME = os.path.expanduser("~")
ROOTS = [(os.environ.get("PD_DATA_ROOT", ""), 3),
         # 2026-09-24 终端实测：真实路径中间还有一层 home/
         ("/eaas/default/groups/casestudy_cnu/home/phdauser004/share", 3),
         ("/eaas/default/groups/casestudy_cnu/phdauser004/share", 3),   # 旧写法，留着兜底
         (os.path.join(_HOME, "share"), 2),
         (os.path.join(_HOME, "pd_data"), 2),
         (_HOME, 1)]
for g in glob.glob("/eaas/default/groups/*/home/*/share"):
    ROOTS.append((g, 2))
for g in glob.glob("/eaas/*/groups/*/home/*/share"):
    ROOTS.append((g, 2))
for g in glob.glob("/mnt/*") + glob.glob("/data/*"):
    ROOTS.append((g, 2))

# 去重：上面两个 /eaas 通配会跟写死的路径命中同一个目录，不去重会重复遍历。
# ★ 用 realpath 而不是 abspath：2026-09-24 实测发现平台上
#     /eaas/default/groups/casestudy_cnu/home/share
#     /eaas/default/groups/casestudy_cnu/home/phdauser004/share
#   是同一份存储的两个入口（脚本打印的 os.getcwd() 是前者，提示符是后者）。
#   abspath 只是拼字符串，认不出这种别名，会把同一批 zip 列两遍。
_seen_root, _uniq = set(), []
for _r, _d in ROOTS:
    if not _r:
        continue
    _k = os.path.realpath(_r)
    if _k in _seen_root:
        continue
    _seen_root.add(_k)
    _uniq.append((_r, _d))
ROOTS = _uniq

# 熔断阈值：找到这么多 zip 说明目录选错了，宁可停下让人确认，也不要乱解压
MAX_ZIPS = 50
MAX_SHOW_DEPTH = 2          # 目录树最多展示几层（0 起算）

_SKIP = {".git", ".cache", ".conda", ".local", ".config", ".ipynb_checkpoints",
         "anaconda3", "miniconda3", "miniforge3", "node_modules",
         "site-packages", "__pycache__", "venv", ".venv", "envs", "pkgs",
         "AppData", "Application Data", "Local Settings", "Windows",
         "snap", ".vscode-server", ".npm", ".cursor-server"}


# ---------- 中文文件名修复 ----------
def _decoded_name(info):
    """Windows 压出来的 zip，中文名常按 cp437 存；这里还原成 UTF-8 字符串。"""
    if info.flag_bits & 0x800:          # 已声明 UTF-8，直接用
        return info.filename
    raw = info.filename
    for enc in ("gbk", "utf-8", "big5"):
        try:
            return raw.encode("cp437").decode(enc)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return raw


def _safe_target(dest, name):
    """把 zip 里的路径映射成本地路径，顺手挡掉 ../ 穿越。"""
    parts = [p for p in name.replace("\\", "/").split("/")
             if p not in ("", ".", "..")]
    return os.path.join(dest, *parts) if parts else None


def unzip_to(zf_path, dest):
    """解压一个 zip，返回解出的文件数。"""
    n = 0
    with zipfile.ZipFile(zf_path) as zf:
        for info in zf.infolist():
            target = _safe_target(dest, _decoded_name(info))
            if target is None:
                continue
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
    return n


# ---------- 找 zip / 找数据 ----------
def _rel_depth(dp, root):
    """dp 相对 root 的层数（root 自身为 0）。用 relpath，Windows/Linux 都对。"""
    rel = os.path.relpath(dp, root)
    return 0 if rel == os.curdir else rel.count(os.sep) + 1


def find_zips(roots, maxdepth=3):
    """在候选目录下按各自限定的深度收集 *.zip。"""
    out, seen = [], set()
    for r, depth in roots:
        if not r or not os.path.isdir(r):
            continue
        r = os.path.realpath(r)                 # ★ 按真实路径去重，见下方注释
        for dp, dns, fns in os.walk(r):
            dns[:] = [d for d in dns if d not in _SKIP and not d.startswith(".")]
            if _rel_depth(dp, r) >= depth:
                dns[:] = []                     # 到达深度上限，不再往下
            for f in fns:
                if f.lower().endswith(".zip"):
                    p = os.path.realpath(os.path.join(dp, f))
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
        if len(out) > MAX_ZIPS:                 # 熔断：目录明显选错了
            return sorted(out), True
    return sorted(out), False


# ---------- 只解我们真正要的 zip ----------
# share 目录里同时放着好几个课题组的数据集（肝脏肿瘤 CT、中医诊疗文本、NPU 驱动……），
# 2026-09-24 实测（8 个 zip）：无差别全解会多解 7 个完全无关的包，
# 压缩后合计约 3.9 GB —— 其中「帕金森病患者与健康老年人行走运动参数数据.zip」
# 1.64 GB 属于另一份数据，按要求不使用，而且它里面的表名跟二期那张**完全同名**，
# 解出来会跟二期内部那张撞车。
# 所以解压前先看 zip 的「文件清单」（只读中央目录，不解压任何内容，5.6 GB 的包也秒回），
# 里面没有我们要的数据集目录就跳过。
WANT_DIRS = ("year2帕金森病诊断步态特征数据",              # DIR_FEAT
             "year2帕金森病患者与健康老年人行走运动参数数据")   # DIR_RAW


def zip_dirs(zf_path):
    """zip 里出现了 WANT_DIRS 里的哪几个目录名。只读中央目录，不解压。"""
    found = set()
    try:
        with zipfile.ZipFile(zf_path) as zf:
            for info in zf.infolist():
                n = _decoded_name(info)
                for w in WANT_DIRS:
                    if w in n:
                        found.add(w)
    except Exception:
        return set()
    return found


def pick_zips(zips):
    """挑出该解压的 zip，返回 (要解的, [(跳过的, 原因)])。

    ★ 判定标准是「两个数据集目录都在」——那才是二期那个包。
      只按一个关键词命中就解，会误伤另一份包：它的文件名里就带着
      「帕金森病患者与健康老年人行走运动参数数据」，一旦它内部的目录也叫这个名字，
      就会被一起解出来（1.64 GB，而且里面的表跟二期**完全同名**，解出来互相覆盖）。
      但也不能一刀切「必须两个都在」就把二期漏掉（万一原始数据在嵌套 zip 里），
      所以：有「两个都在」的就只解那些；一个都没有时才退回解「只匹配到一个」的。
    """
    full, partial, none = [], [], []
    for z in zips:
        d = zip_dirs(z)
        (full if d == set(WANT_DIRS) else partial if d else none).append(z)
    if full:
        # ★ none 也要一起报出来。上面那个分支只报告 partial 的话，
        #   「共 N 个跳过」的 N 会漏掉完全无关的那些，用户会以为平台上看错目录了。
        return full, ([(z, "只含一个数据集目录，疑似同名的那份，不用") for z in partial]
                      + [(z, "不含目标数据集") for z in none])
    # 没有完整包 —— 退回解「只匹配到一个」的，并说清楚为什么
    if partial:
        print("\n  ⚠️ 没有哪个 zip 同时含两套数据集，改为解「只含其中一个」的包：")
        for z in partial:
            print(f"     {os.path.basename(z)}  含 {sorted(zip_dirs(z))}")
    return partial, [(z, "不含目标数据集") for z in none]


def looks_like_data(d):
    try:
        names = os.listdir(d)
    except OSError:
        return False
    return (any(n.startswith("year") for n in names)
            or any(n.startswith("表1_") for n in names))


def find_data_dirs(base, maxdepth=5):
    """从 base 往下找出所有「直接装着 year* 或 表1_」的目录（由浅到深）。
    ★ 两个 zip 各装一个数据集，所以这里会返回两个平级目录，不是一个。"""
    if not base or not os.path.isdir(base):
        return []
    from collections import deque
    q, seen = deque([(os.path.abspath(base), 0)]), set()
    hits = []
    while q:
        d, depth = q.popleft()
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        if looks_like_data(d):
            hits.append(d)
            continue                 # 命中就不再往下，免得把数据集内部也算成一层
        if depth < maxdepth:
            try:
                for n in os.listdir(d):
                    p = os.path.join(d, n)
                    if n in _SKIP or n.startswith("."):
                        continue
                    if os.path.isdir(p):
                        q.append((p, depth + 1))
            except OSError:
                pass
    return hits


SEP = "=" * 70

if __name__ == "__main__":
    print(SEP)
    print("【0】环境")
    print(SEP)
    print(f"  用户        = {os.environ.get('USER', '?')}")
    print(f"  工作目录    = {os.getcwd()}")
    print(f"  解包目标    = {WORK}")
    print(f"  zip 候选目录 = {[r for r, _ in ROOTS if r]}")

    # 1) 已经解过了吗
    done = find_data_dirs(WORK)
    if done and not FORCE:
        print(f"\n[跳过] {WORK} 下已有 {len(done)} 个数据集目录：")
        for d in done:
            print(f"       {d}")
        print(f"       要重解就加参数：python 00_unpack.py --force")
        print(f"\n下一步：python 00_inspect.py")
        sys.exit(0)

    # 2) 找 zip
    print(f"\n{SEP}\n【1】找 zip\n{SEP}")
    zips, blew_up = find_zips(ROOTS)
    if blew_up:
        print(f"  ⚠️ 找到了 {len(zips)} 个 zip，远超预期（>{MAX_ZIPS}），"
              f"说明候选目录选错了，已停止以免乱解压。")
        print(f"  请指定数据目录后重跑：")
        print(f"      export PD_DATA_ROOT='/你的/数据目录'")
        print(f"      python 00_unpack.py")
        sys.exit(1)
    if not zips:
        print("  没找到任何 zip。可能数据已经是解压好的文件夹，那就直接跑：")
        print("      python 00_inspect.py")
        print("  如果确认数据是 zip 但在别处，指定一下再重跑：")
        print("      export PD_DATA_ROOT='/你的/数据目录'")
        print("      python 00_unpack.py")
        sys.exit(0)
    for z in zips:
        print(f"  {z}   ({os.path.getsize(z) / 1e6:.2f} MB)")

    # 2b) 只留下含目标数据集的 zip（除非 --all）
    if not ALL_ZIPS:
        zips, drop = pick_zips(zips)
        if drop:
            print(f"\n  只解含目标数据集的 zip（共 {len(drop)} 个跳过）：")
            for z, why in drop:
                print(f"    [跳过] {os.path.basename(z)}   "
                      f"({os.path.getsize(z) / 1e6:.2f} MB)  —— {why}")
            print(f"    要全部解压就加参数：python 00_unpack.py --all")
        if not zips:
            print(f"\n  没有 zip 含目标数据集 {WANT_DIRS}。")
            print(f"  确认一下数据是不是已经解压好了，先跑：python 00_inspect.py")
            sys.exit(0)

    # 3) 解压
    print(f"\n{SEP}\n【2】解压到 {WORK}\n{SEP}")
    os.makedirs(WORK, exist_ok=True)
    for z in zips:
        stem = os.path.splitext(os.path.basename(z))[0]
        dest = os.path.join(WORK, stem)
        os.makedirs(dest, exist_ok=True)
        try:
            n = unzip_to(z, dest)
            print(f"  [OK] {os.path.basename(z)} -> {dest}  （{n} 个文件）")
        except Exception as e:
            print(f"  [失败] {os.path.basename(z)}: {type(e).__name__}: {e}")

    # 4) zip 里套 zip，再解一层
    nested, _ = find_zips([(WORK, 3)])
    if nested:
        print(f"\n{SEP}\n【3】发现嵌套 zip，再解一层\n{SEP}")
        for z in nested[:MAX_ZIPS]:
            stem = os.path.splitext(os.path.basename(z))[0]
            dest = os.path.join(os.path.dirname(z), stem + "_x")
            os.makedirs(dest, exist_ok=True)
            try:
                n = unzip_to(z, dest)
                print(f"  [OK] {os.path.basename(z)} -> {dest}  （{n} 个文件）")
            except Exception as e:
                print(f"  [失败] {os.path.basename(z)}: {type(e).__name__}: {e}")

    # 5) 数据落到哪了
    print(f"\n{SEP}\n【4】结果\n{SEP}")
    print(f"  解包目录结构（前 3 层）：")
    for dp, dns, fns in os.walk(WORK):
        d = _rel_depth(dp, WORK)
        dns[:] = [x for x in dns if not x.startswith(".")]
        if d > MAX_SHOW_DEPTH:
            dns[:] = []
            continue
        print("   " + "  " * d + (os.path.basename(dp) or dp) + "/")
        for f in sorted(fns)[:8]:
            print("   " + "  " * (d + 1) + f)
        if len(fns) > 8:
            print("   " + "  " * (d + 1) + f"... 共 {len(fns)} 个文件")

    hits = find_data_dirs(WORK)
    if hits:
        print(f"\n  ✅ 认出 {len(hits)} 个数据集目录：")
        for h in hits:
            print(f"       {h}")
            try:
                print(f"         里面: {sorted(os.listdir(h))[:6]}")
            except OSError:
                pass
        print(f"\n  下一步：python 00_inspect.py")
        print(f"  （common.py 会挨个根目录去认，两个数据集分开放也能找齐）")
    else:
        print(f"\n  ⚠️ 解压完了但没在 {WORK} 下认出 year*/表1_ 结构。")
        print(f"     把上面这段目录结构发回给我，我看一眼再调。")
    print(SEP)
