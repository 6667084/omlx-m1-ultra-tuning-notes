"""把已导入的 MTP 头（model-mtp.safetensors）中的线性层按生产版规格量化为 4-bit gs64 affine；norm 保持原精度。原地改写目标目录（目标须为克隆副本）。"""
import sys, json, os
import mlx.core as mx
d = sys.argv[1]
p = os.path.join(d, "model-mtp.safetensors")
w = mx.load(p)
LIN = ("mtp.fc", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
out, qent = {}, {}
for k, v in w.items():
    mod = k[:-len(".weight")] if k.endswith(".weight") else None
    if mod and any(mod.endswith(s) for s in LIN) and v.ndim == 2:
        wq, sc, bi = mx.quantize(v, group_size=64, bits=4)
        out[mod + ".weight"], out[mod + ".scales"], out[mod + ".biases"] = wq, sc.astype(v.dtype), bi.astype(v.dtype)
        qent[mod] = {"group_size": 64, "bits": 4, "mode": "affine"}
    else:
        out[k] = v
mx.eval(*out.values())
os.remove(p)  # 克隆副本：删除后写新文件，不影响源目录的同名文件
mx.save_safetensors(p, out, metadata={"format": "mlx"})
ip = os.path.join(d, "model.safetensors.index.json"); idx = json.load(open(ip))
for k in out: idx["weight_map"][k] = "model-mtp.safetensors"
json.dump(idx, open(ip, "w"), indent=2)
cp = os.path.join(d, "config.json"); c = json.load(open(cp))
for sec in ("quantization", "quantization_config"):
    if isinstance(c.get(sec), dict): c[sec].update(qent)
json.dump(c, open(cp, "w"), indent=2)
print("quantized", len(qent), "linears;", "tensors", len(w), "->", len(out), "; size MB", round(os.path.getsize(p) / 1e6))
