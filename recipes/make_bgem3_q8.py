#!/usr/bin/env python3
"""BAAI/bge-m3 (FP32 safetensors)  ->  MLX 8-bit affine, group size 64 (what YCF-AI/bge-m3-8bit-MLX contains).

  * `embeddings.word_embeddings` and the Linear weights of the 24 encoder layers -> 8-bit affine, group size 64
    (scales / biases stored in FP16)
  * everything else (LayerNorm, biases, position / type embeddings, pooler) -> FP16
  * only the dense head is kept; sparse_linear.pt and colbert_linear.pt are not used by oMLX /v1/embeddings

Usage:
    python bin2st.py pytorch_model.bin fp32/model.safetensors        # upstream ships only a .bin
    cp <upstream config/tokenizer files> fp32/                        # config.json, tokenizer.json, 1_Pooling/, ...
    python make_bgem3_q8.py fp32 out
Needs: mlx (Apple Silicon). Built and verified with mlx 0.32.2.
"""
import json
import os
import re
import shutil
import sys

import mlx.core as mx

AUX = ["config_sentence_transformers.json", "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
       "tokenizer_config.json", "sentencepiece.bpe.model", "tokenizer.json"]
ENC_LINEAR = re.compile(r"^encoder\.layer\.\d+\.(attention\.self\.(query|key|value)|attention\.output\.dense|intermediate\.dense|output\.dense)\.weight$")
BITS, GROUP = 8, 64


def main(src, dst):
    os.makedirs(os.path.join(dst, "1_Pooling"), exist_ok=True)
    for f in AUX:
        shutil.copyfile(os.path.join(src, f), os.path.join(dst, f))
    shutil.copyfile(os.path.join(src, "1_Pooling", "config.json"), os.path.join(dst, "1_Pooling", "config.json"))

    w32 = mx.load(os.path.join(src, "model.safetensors"))
    out = {}
    for k, v in w32.items():
        assert float(mx.max(mx.abs(v)).item()) <= 65504, k          # FP16 range guard
        v16 = v.astype(mx.float16)
        if k == "embeddings.word_embeddings.weight" or ENC_LINEAR.match(k):
            q, s, b = mx.quantize(v16, group_size=GROUP, bits=BITS)
            base = k[: -len(".weight")]
            out[k], out[base + ".scales"], out[base + ".biases"] = q, s, b
        else:
            out[k] = v16
    mx.eval(out)
    mx.save_safetensors(os.path.join(dst, "model.safetensors"), out, metadata={"format": "mlx"})

    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["torch_dtype"] = "float16"
    cfg["quantization"] = {"group_size": GROUP, "bits": BITS}
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    n_q = sum(1 for k in out if k.endswith(".scales"))
    size = os.path.getsize(os.path.join(dst, "model.safetensors"))
    print(f"wrote {dst}/model.safetensors  {size / 1e6:.1f} MB  quantized layers: {n_q}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
