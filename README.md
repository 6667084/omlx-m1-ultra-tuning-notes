**English** | [简体中文](README.zh-CN.md)

# Tuning 5 local models on an M1 Ultra (64 GB) with oMLX 0.7.0 — measured notes

> Tested 2026-10-06 / 10-07 (section 9 on community code-model candidates added 10-07 afternoon). Every number comes from one machine. Every adopted change was checked with
> ABBA runs in the same session and a fixed 260-question quality set. Community numbers are labelled.

## TL;DR

| Model | Final setup | Measured effect |
|---|---|---|
| **CyberTiel-Coder-35B-A3B** (oQ6e MoE) | Non-quantized **BF16 tensors cast to FP16** offline + **adaptive MTP depth** | Prefill **+28–50 %** (16K: 1096 → 1644 tok/s); decode at 16K **+22 %**; short-chat decode +5 % (109–116 tok/s); quality 206/260 = 206/260 |
| **Qwen3.8-27B** (MTPLX Optimized-Speed FP16, 4-bit) | **`mtp_fixed_depth: 2`** | 47–48 tok/s, compared with 27.4 with MTP off (**+73 %**) and 44.3 with 0.7.0's default adaptive depth (**+7 %**) |
| **MiniCPM-V-4.6-8bit** | Kept as the dedicated vision model | Same OCR accuracy as the 27B, **3–4× faster**, **~1/9 of the memory** |
| Qwen3-Embedding-0.6B-8bit | Unchanged (already FP16) | — |
| bge-reranker-v2-m3 | Unchanged; **chunk long docs on the client** | Avoids the silent 512-token truncation in 0.7.0 |

## Setup and method

**Hardware and software**
- Mac Studio M1 Ultra (64-core GPU), 64 GB, macOS 15.8.1.
- oMLX 0.7.0 (build 2987), MLX 0.32.2, mlx-lm 0.31.4.dev, mlx-vlm 0.7.1.

**Chat test**
- Three prompts: a long Chinese explanation, a Python module, and a math word problem.
- T=0, thinking off, 512 output tokens.
- A unique nonce in each prompt guarantees `cached_tokens=0`.
- Metric: the server-side `generation_tokens_per_second`, median of 9 requests.

**Built-in benchmark** (same method as the omlx.ai leaderboard)
- pp1K/4K/16K + tg128, with the `code_python` corpus.
- **Heads-up:** the built-in benchmark **uploads results to omlx.ai by default**. Passing `"align_prompt_to_ane": true` skips the upload. It turns PP4096 into PP4097, which is negligible.

**Quality gate**
- 260 fixed questions from oMLX's bundled eval data: MMLU 50, CMMLU 50, TruthfulQA 50, GSM8K 50, HumanEval 20, MBPP 30, LiveCodeBench 10.
- Generated code runs only inside `sandbox-exec`.

**Vision gate**
- A 35-item suite plus a 30-row document OCR (60 fields).
- All test images are synthetic.

## 1. After upgrading to 0.7.0, check your MTP fields

0.7.0 **deprecates `mtp_num_draft_tokens`**. Depth is now set with:
- `mtp_fixed_depth`: a fixed depth.
- `mtp_adaptive_max_depth`: a ceiling for adaptive depth. Leave both empty to get the model's default adaptive depth.

The old key is **ignored silently**. Our documented "d2 / d3" settings had not been applied since the upgrade.

## 2. Sweep the MTP depth per model

![CyberTiel](images/fig1_cybertiel_mtp_depth.png)
![Qwen](images/fig2_qwen38_mtp_depth.png)

**MoE CyberTiel (3B active per token): adaptive depth wins.**

| Setting | tok/s |
|---|---:|
| MTP off | 59.4 |
| d1 | 69.5 |
| d2 | 89.8 |
| d3 | 84.7 |
| **adaptive (default)** | **94–95** |

**Dense Qwen3.8-27B: fixed d2 wins.**

| Setting | tok/s |
|---|---:|
| MTP off | 27.4 |
| d3 | 38.6 |
| d4 | 40.4 |
| adaptive | 44.3 |
| **fixed d2** | **47–48** |

On Qwen3.8-27B, verification costs more per cycle than on the MoE, so deeper drafts stop paying off sooner. Fixed d2 also leads in the built-in benchmark at 4K (42.0 vs 39.9) and at 16K (44.2 vs 43.0).

For Qwen3.8-27B in 0.7.0, `mtp_adaptive_max_depth` cannot go below 4, so use `mtp_fixed_depth`.

DFlash2, with a BF16 draft model, ran at half the speed of MTP d2 on M1 (23.6 tok/s).

## 3. Biggest win on M1/M2: cast BF16 tensors to FP16

M1 and M2 have **no native bfloat16 ALU**, so Metal emulates it. Quantized MLX checkpoints still ship their non-quantized tensors in BF16: scales, biases, norms, embeddings, the MTP head and the vision tower. Activations follow those tensors, so every forward pass pays for the emulation. Prefill pays the most.

The idea comes from oMLX PR [#3277](https://github.com/jundot/omlx/pull/3277), which is not yet merged. It reports +60 % prefill on an M1 Ultra.

**Our mistake.** We had already tried the cast on 10-06, but we only measured **short-context decode with MTP off**. That showed ±3 %, so we rejected it. Measuring prefill and long context told a different story:

![fp16](images/fig3_fp16_cast_prefill_decode.png)

**CyberTiel-35B-A3B** (1569 BF16 tensors, 2.94 GB; mean of 2 ABBA runs per arm):

| | BF16 | FP16 | Change |
|---|---:|---:|---:|
| Prefill 1K | 1085 | 1393 | **+28 %** |
| Prefill 4K | 1406 | 1891 | **+34 %** |
| Prefill 16K | 1096 | 1644 | **+50 %** |
| TTFT at 16K | 14.9 s | 10.0 s | −33 % |
| Decode at 16K context | 96 | 117 | **+22 %** |
| Decode at 4K context | 92 | 89 | flat (noise) |
| Chat decode | 106 | 111 | +5 % |
| 260-question set | 206 | 206 | = |

**Quality noise.** Re-running BF16 alone flipped 4 of the 60 coding questions; LiveCodeBench went from 5/10 to 3/10. MTP makes T=0 output non-deterministic ([#4089](https://github.com/jundot/omlx/issues/4089)). The BF16-vs-FP16 difference stayed inside that noise.

**Where the cast did not help** (also measured):

| Model / part | Result |
|---|---|
| Qwen3.8 MTPLX "-FP16" pack, text weights | Already FP16 |
| Qwen3.8 vision tower (0.86 GB BF16) | No change |
| MiniCPM-V | ~3 % overall: image encode −14 %, decode −2 % |
| bge-reranker (F32) | Same speed; scores differ by < 2e-4 |
| Qwen3-Embedding | Already FP16 |

The cast pays off for **large models with long prompts**, such as coding agents and long documents.

**Conversion script:** [`scripts/to_fp16.py`](scripts/to_fp16.py), or the condensed version below. Only the shards that contain BF16 tensors are rewritten; every other file is APFS-cloned with `cp -c`.

```python
# to_fp16.py <src> <dst>
import sys, os, json, struct, subprocess
import mlx.core as mx
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    sp, dp = os.path.join(src, f), os.path.join(dst, f)
    if os.path.isdir(sp) or f.startswith("."): continue
    hdr = {}
    if f.endswith(".safetensors"):
        with open(sp, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n))
    if not any(isinstance(v, dict) and v.get("dtype") == "BF16" for v in hdr.values()):
        subprocess.run(["cp", "-c", sp, dp], check=True); continue
    ws, out = mx.load(sp), {}
    for k, v in ws.items():
        if v.dtype == mx.bfloat16:
            assert mx.max(mx.abs(v.astype(mx.float32))).item() <= 65504, k
            v = v.astype(mx.float16)
        out[k] = v
    mx.eval(out); mx.save_safetensors(dp, out, metadata=hdr.get("__metadata__") or {})
# then set "dtype"/"torch_dtype" from "bfloat16" to "float16" in config.json (and in text_config / vision_config)
```

After you swap the weights, **purge that model's SSD KV cache blocks and GDN sidecars**. Keep the BF16 copy for rollback.

## 4. A small dedicated VLM still earns its place

![vision](images/fig4_vision_three_models.png)

| Metric | MiniCPM-V-4.6 | Qwen3.8-27B | CyberTiel-35B |
|---|---:|---:|---:|
| OCR | Perfect | Perfect | Misread one phone number |
| Counting | Fails | Correct | Fails |
| 35-item suite | **14.5 s** | 44.3 s | 15.1 s |
| 30-row document | **7.7 s** | 32 s | 14.9 s |
| Resident memory | **2.3 GB** | 20.2 GB | 29.7 GB |

**Routing**
- OCR, charts, image descriptions → MiniCPM-V.
- Counting and spatial reasoning → Qwen3.8-27B.
- A quick look at an image mid-chat → CyberTiel.

## 5. A legacy `soft_threshold` makes models evict each other

In 0.7.0, `memory.soft_threshold` means "let the tier decide" **only when it is exactly 0.85**. Any other value is an explicit override.

Our leftover 0.70 from the 0.6.x days caused two problems:
- The admission target dropped to about 29 GB, which is below the 30 GB default model.
- Every vision or rerank call evicted the default model.

Back at 0.85, four models (34.8 GB in total) stay resident together and switching takes 0.1–0.3 s.

## 6. Never point a different engine at your production model folder

Another MLX engine rewrote our `model.safetensors.index.json`. It dropped 333 `vision_tower.*` entries, and vision was silently disabled for two weeks. oMLX returned `'NoneType' object has no attribute 'patch_embed'` on image requests.

**Fix:** rebuild the index from the safetensors shard headers. No weight bytes change.

**Prevention:**
- Give other engines an APFS clone (`cp -c`), not the production folder.
- Afterwards, check that the index entry count equals the tensor count across the shard headers.

## 7. `/v1/rerank` truncates encoder rerankers at 512 tokens (0.7.0)

![rerank](images/fig6_reranker_truncation.png)

Test: a ~3K-token document with the answer in the middle, plus two short keyword-only distractors.

| | Score of the long doc | Rank |
|---|---:|---:|
| Whole document | **0.000** | 3rd (last) |
| Chunked on the client (~300 chars, max over chunks) | **0.996** | 1st |

The request-level `max_length` and the per-model settings cannot override the limit ([#4161](https://github.com/jundot/omlx/issues/4161)).

## 8. Read leaderboard numbers with the corpus in mind

![corpus](images/fig5_leaderboard_corpus_effect.png)

A community entry from another M1 Ultra 64c / 64 GB machine shows Qwen3.8-27B at **67.7 tok/s at 4K**. The same session reads 41 at 1K and 33 at 8K. It used the **Code (Mixed)** corpus, where the 4K slice is easy for MTP to predict.

On our machine, with the same corpus, fixed d2 gives:

| Context | 1K | 4K | 8K |
|---|---:|---:|---:|
| tok/s | 43.1 | 51.4 | 30.7 |

Fixed d2 still beats adaptive on that corpus. Compare the whole curve from one session, not the best cell.

For comparison, the community **oQ4e** CyberTiel on an M1 Ultra 64c shows 4K PP 1219 / TG 73.6. Our larger **oQ6e** build after the FP16 cast reaches 4K PP 1891 / TG 89, and 16K PP 1644 / TG 117.

## 9. Evaluate code models in the thinking mode you actually use

On the afternoon of 10-07 we checked the oMLX discussions and the Hugging Face trending list, then picked two candidates. Both have the same architecture as our main model (Qwen3.6-35B-A3B MoE) and the same size (oQ6e, about 28 GB):

- **CyberTiel upstream 09-29 requant.** It uses a new code- and security-weighted calibration corpus; 461 of 472 quantized tensors changed.
- **KAT-Coder-V2.5-Dev-VL-oQ6e-mtp.** Recommended in the discussions; Kwaipilot reports 69.4 on SWE-bench Verified.

Both were cast to FP16 (section 3) and compared with production in the same session.

![thinking flip](images/fig8_code_candidates_thinking_flip.png)

| Test | Production CyberTiel | Upstream 09-29 requant | KAT-Coder-V2.5 |
|---|---:|---:|---:|
| Code 324 (HumanEval 164 / MBPP 100 / LCB 60), thinking off, T=0 | 242 (re-run 240) | 247 | **265** (+29/−6, p=0.0001) |
| General 260-question gate | 205 | 206 | 212 |
| **LiveCodeBench 60, thinking on, T=0.6** (how we use it) | **40** | 38 | **26** (0 wins / 14 losses) |
| Median answer length with thinking on | 926 tok | 902 tok | 306 tok |
| Prefill 4K / 16K (tok/s) | 1910 / 1620 | — | 1917 / 1620 |

- **KAT really is better at short code with thinking off.** It was trained with RL to think very briefly, though, so on algorithm problems that need long reasoning it loses clearly to CyberTiel. A thinking-off HumanEval/MBPP comparison alone would have picked the wrong model.
- **The upstream requant** stayed within noise on all three tests (re-running the same config moves the 324-question score by about 2). It also used 27 % more thinking tokens. The card's gains were measured on GGUF with SWE-bench-Live and did not reproduce in MLX.
- Speed is identical (same architecture, same quantization), so we kept production as is.
- Method note: after hours of back-to-back tests, the SSD cache hit its cap and two 28 GB models were swapping in and out. Chat speed dropped to 77–92 tok/s and came back to 112 after a restart and cache cleanup. Only compare models within one ABBA session.

## Rejected or not applicable

| Option | Verdict |
|---|---|
| DFlash2 | Half the speed of MTP d2 |
| MTPLX 2.12.2 | Garbled output on M1 Ultra |
| `Qwen3.8-27B-oQ4e-fp16-mtp` as a replacement | Same speed; Chinese 8–10 % slower |
| oQ A8 INT8 prefill | M5 and newer only |
| ANE prefill | Needs group size 64 or 128; our Qwen pack uses gs32 |
| Idle stall from #4040 | Not reproduced: TTFT is 0.38–0.52 s after 0.3–15 s idle, so we skip `sudo sysctl iogpu.wired_limit_mb` |
| Qwen3.8-Flash-Next | 105 GiB at 4-bit; does not fit in 64 GB |
| CyberTiel upstream 09-29 requant | Within noise on all three quality tests (section 9) |
| KAT-Coder-V2.5-Dev oQ6e-mtp | +23 with thinking off, −14 with thinking on (section 9) |

## Final settings

```jsonc
"CyberTiel-Coder-35B-MLX":               { "mtp_enabled": true, "enable_thinking": true, "temperature": 0.6, "top_p": 0.95, "top_k": 20 },  // FP16-cast weights
"Qwen3.8-27B-MTPLX-Optimized-Speed-oMLX": { "mtp_enabled": true, "mtp_fixed_depth": 2, "thinking_budget_enabled": true, "thinking_budget_tokens": 16384 },
"MiniCPM-V-4.6-8bit":                     { "temperature": 0.0, "top_p": 1.0 },
// settings.json
"memory": { "memory_guard_tier": "balanced", "soft_threshold": 0.85, "hard_threshold": 0.95 }
```

**Caveats**
- MTP makes T=0 output non-deterministic, so measure a noise floor before you compare quality.
- CyberTiel is an abliterated model; run it in an OS-level sandbox.
- The 0.7.0 macOS 15 build is missing `chat.html`, so `/admin/chat` returns 500. The API is unaffected.

**Links:** [oMLX 0.7.0](https://github.com/jundot/omlx/releases/tag/v0.7.0) · [PR #3277](https://github.com/jundot/omlx/pull/3277) ·
[#4161](https://github.com/jundot/omlx/issues/4161) · [#4089](https://github.com/jundot/omlx/issues/4089) · [#4040](https://github.com/jundot/omlx/issues/4040) ·
[omlx.ai benchmarks](https://omlx.ai/benchmarks) · [MTPLX on M1 Ultra (blog)](https://rbeckner.com/articles/mtplx-m1-ultra-serving-reality/)

## Data, scripts and license

- `data/2026-10-06/`: MTP depth sweeps. `summary.jsonl` holds per-setting medians, `results.jsonl` per-request results, `builtin.jsonl` built-in benchmark runs, and `vision_compare.jsonl` the three-model vision comparison.
- `data/2026-10-07/` covers the second round:
  - BF16-vs-FP16 ABBA runs: `builtin.jsonl` and `summary.jsonl`.
  - The 260-question quality set: `q260-*.jsonl`, one row per question, plus `q260-summary.jsonl`.
  - Vision ABBA: `vision_compare.jsonl`.
  - Reranker truncation test: `rerank.jsonl`.
  - Idle-TTFT probe: `idle_ttft.jsonl`.
- `data/2026-10-07-candidates/` (section 9): per-question results for the 324-question code set (`q260-code324-*`), the 260-question gate (`q260-g260-*`) and thinking-on LiveCodeBench 60 (`q260-lcb60think-*`); same-session speed ABBA in `builtin.jsonl`, `results.jsonl`, `summary.jsonl`.
  - Conversion reports: `fp16-*.json`.
- `scripts/to_fp16.py`: the offline BF16→FP16 cast. It only rewrites shards that contain BF16 tensors and APFS-clones everything else. Usage: `python to_fp16.py <src_model_dir> <dst_dir> [--f32]`.
- Text, figures and data: CC BY 4.0. Script: MIT. All test images are synthetic; no personal data is included.
