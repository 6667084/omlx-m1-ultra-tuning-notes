# Restoring the oMLX stack on a new Mac · 在新 Mac 上还原 oMLX 配置

Files · 文件
- `settings.redacted.json` — global oMLX `~/.omlx/settings.json` with secrets removed. Replace `<DIR>`, set your own API key in the admin UI, add a proxy only if you need one. 全局设置（已脱敏）。
- `model_settings.json` — per-model settings for the 5 models (aliases, context, max tokens, sampling, MTP, TTL, pinning). Key = model **directory name**. 五个模型的逐模型设置，键名=模型目录名。

Order · 顺序
1. Install oMLX (we run 0.7.0 build 2987). Start it once, then **quit** so it does not overwrite files. 安装并先启动一次再退出。
2. Download the models (see the three model repos; for embedding/reranker use the upstream repos below) into `<DIR>/production-models` (symlinks are fine). 下载模型。
   - `mlx-community/Qwen3-Embedding-0.6B-8bit` (rev 407ad23…)
   - `BAAI/bge-reranker-v2-m3` (rev 953dc6f…)
3. Copy the two JSON files into `~/.omlx/` (chmod 600), fix `<DIR>` and the key fields, start oMLX. 拷入、改路径与密钥、启动。
4. Check: `GET /v1/models/status` lists 5 models; `~/.omlx/logs` has no crash; run a 3-prompt chat decode, 1K/4K/16K prefill, and a short vision test. 验证。
5. Create one sub-key per client and give each its own env var; never export the master key to agents. 每个客户端一个 sub-key。

Do not · 不要
- Re-use stale `mtp_num_draft_tokens` (ignored by 0.7.0); leave MTP depth empty for the MoE model and `mtp_fixed_depth: 2` for the 27B.
- Turn on hot cache on a 64 GB Mac; set `model_type_override=llm` on vision-capable models; point another engine at the production folder.
