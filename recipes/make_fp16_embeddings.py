#!/usr/bin/env python3
"""BAAI/bge-reranker-v2-m3: store the three embedding tables (word / position / token-type) in FP16 and keep every
other tensor in FP32 -- what YCF-AI/bge-reranker-v2-m3-fp16emb contains (2.27 GB -> 1.74 GB).

The conversion is only performed when it is lossless: each table must survive FP32 -> FP16 -> FP32 unchanged
(true for the upstream checkpoint: all 567,755,777 parameters are exactly representable in FP16). Otherwise it aborts.
Linear weights, LayerNorm and the classifier head stay FP32 on purpose: on oMLX 0.7.0 the XLM-R attention mask is FP32,
so FP16 linear weights are up-cast on every request and the extra work costs +9.9 % latency while a large model is resident.

Usage:  python make_fp16_embeddings.py <upstream_snapshot_dir> <out_dir>
        (upstream_snapshot_dir = BAAI/bge-reranker-v2-m3 with model.safetensors in FP32; config/tokenizer are copied untouched)
Needs: mlx (Apple Silicon). Built and verified with mlx 0.32.2.
"""
import os
import shutil
import sys

import mlx.core as mx

AUX = ["config.json", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "sentencepiece.bpe.model"]


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in AUX:
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    w = mx.load(os.path.join(src, "model.safetensors"))
    out, n_tab, n_par = {}, 0, 0
    for k, v in w.items():
        assert v.dtype == mx.float32, (k, v.dtype)
        if k.startswith("roberta.embeddings.") and k.endswith("_embeddings.weight"):
            h = v.astype(mx.float16)
            if not bool(mx.all(h.astype(mx.float32) == v).item()):
                sys.exit(f"{k} is not exactly representable in FP16 - refusing to convert")
            out[k] = h
            n_tab += 1
            n_par += v.size
        else:
            out[k] = v
    mx.eval(out)
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), out, metadata={"format": "pt"})
    size = os.path.getsize(os.path.join(dst, "model.safetensors"))
    print(f"wrote {dst}/model.safetensors  {size / 1e6:.1f} MB  ({n_tab} embedding tables / {n_par:,} parameters stored in FP16, lossless)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
