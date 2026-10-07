"""把 MLX 模型包的浮点张量转换为 F16（M1/M2 无原生 bf16）。源目录只读，输出新目录。
用法：to_fp16_v2.py <src> <dst> [--f32]   （--f32 同时转换 F32 张量，用于 BGE 等 FP32 包）
不含待转换张量的分片与其他文件用 APFS 克隆（cp -c），不占额外空间。越界（|x|>65504）即中止。"""
import sys, os, json, struct, subprocess
import mlx.core as mx
src, dst = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
want = {"BF16"} | ({"F32"} if "--f32" in sys.argv else set())
os.makedirs(dst, exist_ok=True)
F16_MAX = 65504.0
rep = {"src": src, "cast_dtypes": sorted(want), "tensors_cast": 0, "max_abs": 0.0, "underflow_nonzero_to_zero": 0, "cloned": [], "rewritten": []}
for f in sorted(os.listdir(src)):
    sp, dp = os.path.join(src, f), os.path.join(dst, f)
    if os.path.isdir(sp) or f.startswith("."):
        continue
    hdr = {}
    if f.endswith(".safetensors"):
        with open(sp, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    if not any(isinstance(v, dict) and v.get("dtype") in want for k, v in hdr.items() if k != "__metadata__"):
        subprocess.run(["cp", "-c", sp, dp], check=True); rep["cloned"].append(f); continue
    meta = hdr.get("__metadata__", {}) or {}
    ws = mx.load(sp); out = {}; cast = 0
    for k, v in ws.items():
        if v.dtype in (mx.bfloat16, mx.float32) and hdr[k]["dtype"] in want:
            vf = v.astype(mx.float32); m = mx.max(mx.abs(vf)).item()
            if m > F16_MAX:
                sys.exit(f"ABORT: {k} max|x|={m} 超出 F16 范围")
            rep["max_abs"] = max(rep["max_abs"], m)
            h = v.astype(mx.float16)
            rep["underflow_nonzero_to_zero"] += int(mx.sum((vf != 0) & (h == 0)).item())
            out[k] = h; cast += 1
        else:
            out[k] = v
    mx.eval(out); mx.save_safetensors(dp, out, metadata=meta)
    rep["tensors_cast"] += cast; rep["rewritten"].append([f, cast]); print(f, cast, flush=True)
    del ws, out
cfg_p = os.path.join(dst, "config.json")
if os.path.exists(cfg_p):
    os.remove(cfg_p)  # 克隆件，改写前先断开
    c = json.load(open(os.path.join(src, "config.json")))
    for obj in (c, c.get("text_config", {}), c.get("vision_config", {}), c.get("audio_config", {})):
        if isinstance(obj, dict):
            for key in ("torch_dtype", "dtype"):
                if obj.get(key) in ("bfloat16", "float32"):
                    obj[key] = "float16"
    json.dump(c, open(cfg_p, "w"), indent=2, ensure_ascii=False)
name = os.path.basename(dst)
json.dump(rep, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"fp16-{name}.json"), "w"), indent=1)
print("REPORT", {k: v for k, v in rep.items() if k not in ("cloned", "rewritten")})
