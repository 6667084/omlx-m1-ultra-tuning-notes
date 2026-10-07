"""按 safetensors metadata 的 model_name 统计 ~/.omlx/cache（只读；--delete NAME... 才删除）。"""
import json, struct, sys, os, collections, pathlib
ROOT = pathlib.Path.home() / ".omlx/cache"

def meta(p):
    try:
        with open(p, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n)).get("__metadata__", {}) or {}
    except Exception as e:
        return {"_err": str(e)}

main = {}  # block_hash -> (model, path, size)
for d in ROOT.iterdir():
    if d.is_dir() and len(d.name) == 1:
        for f in d.glob("*.safetensors"):
            m = meta(f)
            main[f.stem] = (m.get("model_name", "?"), f, f.stat().st_size, int(m.get("token_count", 0) or 0))
side = []  # (block_hash, model, size, path)：sidecar 目录按模型签名分组，目录内文件名为块哈希
for d in (ROOT / "_gdn_sidecars").iterdir():
    for x in d.iterdir():
        h = x.name.split(".")[0]
        side.append((h, main.get(h, ("?",))[0], x.stat().st_size, x))
agg = collections.defaultdict(lambda: [0, 0])
for h, (mn, f, s, t) in main.items():
    agg[mn][0] += 1; agg[mn][1] += s
print("主块：")
for mn, (c, s) in sorted(agg.items()):
    print(f"  {mn:45s} {c:4d} 块 {s/2**30:7.2f} GiB")
sagg = collections.defaultdict(lambda: [0, 0, 0])
for h, mn, s, d in side:
    k = mn
    sagg[k][0] += 1; sagg[k][1] += s
    if h not in main: sagg[k][2] += 1
print("GDN sidecar：")
for mn, (c, s, orphan) in sorted(sagg.items()):
    print(f"  {mn:45s} {c:4d} 个 {s/2**30:7.2f} GiB（孤儿 {orphan}）")

targets = set(sys.argv[2:]) if len(sys.argv) > 2 and sys.argv[1] == "--delete" else set()
if targets:
    freed = 0
    for h, (mn, f, s, t) in main.items():
        if mn in targets:
            f.unlink(); freed += s
    import shutil
    for h, mn, s, d in side:
        if mn in targets:
            d.unlink(); freed += s
    print(f"已删除 {sorted(targets)}：{freed/2**30:.2f} GiB")

if len(sys.argv) > 3 and sys.argv[1] == "--delete-since":
    # 用法：--delete-since <epoch> <model> [...]：只删指定模型在该时刻之后创建的主块，以及挂在这些主块上的 sidecar
    since = float(sys.argv[2]); names = set(sys.argv[3:]); gone = set(); freed = 0
    for h, (mn, f, s, t) in list(main.items()):
        if mn in names and f.stat().st_mtime >= since:
            f.unlink(); gone.add(h); freed += s
    for h, mn, s, d in side:
        if h in gone:
            d.unlink(); freed += s
    print(f"按时间精确删除 {sorted(names)} 主块 {len(gone)} 个：{freed/2**30:.2f} GiB")
