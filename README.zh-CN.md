[English](README.md) | **简体中文**

# M1 Ultra 64 GB 跑 oMLX 0.7.0：五个本地模型的调优实测笔记

> 关键词：oMLX、MLX、Apple Silicon、M1 Ultra、Lightning MTP、投机解码、BF16→FP16、Qwen3.8-27B、
> CyberTiel-35B-A3B、MiniCPM-V-4.6、Reranker
>
> 测试时间：2026-10-06 ~ 10-07（10-07 追加第 10 节：社区热门代码模型候选实测；第 11 节：hot cache 与并发调优；10-10 追加“补充二”：检索栈重做）。所有数字都在同一台机器上实测，配置可以直接照抄复现。
> 文中的“社区数据”均注明了来源。
>
> **2026-10-07 晚更新：** Qwen3.8 与 MiniCPM-V 已换成去审查版（速度与能力无损），三个日常模型连同确切设置已上传 Hub——见“补充”一节、`config/` 与 `recipes/`。
>
> **2026-10-10 更新：** 嵌入模型已更换（Qwen3-Embedding-0.6B 下线 → **BGE-M3 8-bit**），重排器做了无损瘦身；两者都已上传 Hub，并对整套栈做了复检——见“补充二”。

![test samples](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig7_vision_test_samples.png)

---

## TL;DR

| 模型 | 最终配置 | 实测效果（对比调优前） |
|---|---|---|
| **CyberTiel-Coder-35B-A3B**（oQ6e，MoE） | 把 BF16 浮点张量离线转成 **FP16** + **自适应 MTP 深度** | prefill **+28~50%**（16K：1096 → 1644 tok/s），16K 长上下文解码 **+22%**；对话口径 109–116 tok/s；260 题能力集 206 = 206 |
| **Qwen3.8-27B**（MTPLX Optimized-Speed FP16，4-bit） | **固定 MTP 深度 d2** | 对话口径 47–48 tok/s，较不开 MTP（27.4）**+73%**，较 0.7.0 默认自适应 **+7%** |
| **MiniCPM-V-4.6-8bit** | 不改，作为专职视觉模型保留 | OCR 与 27B 同为满分，**快 3–4 倍，内存只有约 1/9** |
| **BGE-M3**（10-10 取代 Qwen3-Embedding-0.6B） | **8-bit（gs64）**、上下文上限 2048、`embedding_batch_size` 16；oMLX 0.7.0 上 FP16 权重反而更慢 | 612 MB（上游 2.27 GB）；单条 11.8 ms（FP32 13.1）；对 FP32 余弦 ≥ 0.999（见补充二） |
| **bge-reranker-v2-m3** | **只把嵌入表存 FP16**（无损，2.27 → 1.74 GB）；**客户端分块** | 延迟与 FP32 持平（+1.6 %）；全 FP16 在大模型常驻时慢 9.9 %；分块让长文 top-1 命中 0.23 → 0.50（见补充二） |

有 10 条经验值得分享，前三条收益最大（第 9 条是选型时最容易踩的坑，第 10 条让 64 GB 机器的长上下文真正可用）：

1. **M1/M2 没有硬件 BF16。** 把模型里没量化的 BF16 张量转成 FP16，prefill 能快 30–50%。只测短上下文解码会看不出来，我们第一次就因此误判过。
2. **升级 oMLX 0.7.0 之后，旧的 MTP 深度字段会静默失效。** 而且最优深度因模型而异，必须逐档测。
3. **社区榜单的 TG 数字高度依赖测试语料。** 同一模型同一台机器，4K 一档可以虚高 50%。
4. **评测代码模型必须用你实际的思考模式。** 社区热推的 KAT-Coder-V2.5 关思考时比我们的主力多对 23 题，开思考却少对 14 题。

---

## 1. 测试环境与口径

| 项目 | 值 |
|---|---|
| 硬件 | Mac Studio，M1 Ultra（20 核 CPU / **64 核 GPU**），**64 GB** 统一内存 |
| 系统 | macOS 15.8.1 |
| 推理服务 | **oMLX 0.7.0**（build 2987）；MLX 0.32.2、mlx-lm 0.31.4.dev、mlx-vlm 0.7.1 |
| 内存守卫 | tier `balanced`，`soft_threshold 0.85`（见第 6 节） |

**测速口径**

- **对话口径**（贴近真实使用）：
  - 三类提示词：中文长解释、写 Python 模块、数学推理，各跑 3 轮。
  - `temperature=0`，关闭思考，输出 512 token。
  - 每个请求开头加唯一随机串，保证 `cached_tokens=0`。
  - 取服务端返回的 `generation_tokens_per_second` 中位数，不用客户端 SSE 计时（会被分块缓冲扭曲）。
- **内置基准口径**（与 omlx.ai 社区榜相同）：
  - 用 admin 的 `/admin/api/bench/start` 跑 pp1K/4K/16K + tg128，语料是 `code_python`。
  - **注意**：内置基准默认会把结果**自动上传**到 omlx.ai 社区榜。我们加了 `"align_prompt_to_ane": true`：prompt 变成 4097 token（只差 1 个，可忽略），服务端据此跳过上传，结果只留本地。
- **对照方法**：所有“采纳”结论都经过 **ABBA** 交替测试（A、B、B、A），并且在**同一时段**完成。不同日期的绝对值不能直接比较，同一台机器隔天也可能差 10%。
- **质量门禁**：
  - **260 题固定能力集**：MMLU 50、CMMLU 50、TruthfulQA 50、GSM8K 50、HumanEval 20、MBPP 30、LiveCodeBench 10。
  - 题目来自 oMLX 自带 eval 数据，按固定种子抽样，T=0。
  - 生成的代码只在 `sandbox-exec` 沙箱里执行。
- **视觉门禁**：
  - 35 项视觉套件：OCR、图表读数、计数、多图对比、形状场景。
  - 一张 30 行大图逐行转写，共 60 个字段。
  - 测试图全部是合成数据，不含真实个人信息。

---

## 2. 经验一：升级到 0.7.0 后，先检查 MTP 字段

oMLX 0.7.0 **废弃了 `mtp_num_draft_tokens`**，MTP 深度改由两个新字段控制：
- `mtp_fixed_depth`：固定深度。
- `mtp_adaptive_max_depth`：自适应深度的上限；两者都为空时，使用模型默认的自适应深度。

旧字段**不会报错，只是被忽略**。我们的文档里一直写着 "CyberTiel d2 / Qwen d3"，升级后其实都在跑自适应深度，直到重新测速才发现。

**做法**：每次升级 oMLX，先对照 `~/.omlx/model_settings.json` 检查字段有没有被废弃或改名，再重新扫一遍 MTP 深度。

---

## 3. 经验二：最优 MTP 深度因模型而异，必须逐档测

### CyberTiel-35B-A3B（MoE）：自适应最快

![CyberTiel MTP depth](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig1_cybertiel_mtp_depth.png)

| 配置 | 中文 | 代码 | 推理 | 中位 | 接受率 |
|---|---:|---:|---:|---:|---:|
| MTP 关 | 59.4 | 59.7 | 59.3 | 59.4 | — |
| 固定 d1 | 65.9 | 69.5 | 76.1 | 69.5 | 86% |
| 固定 d2 | 77.1 | 90.1 | 92.4 | 89.8 | 83% |
| 固定 d3 | 68.4 | 86.0 | 90.2 | 84.7 | 78% |
| 自适应，上限 4 | 74.7 | 88.4 | 85.5 | 84.7 | 81% |
| **自适应（默认）** | 78–82 | 94–98 | 94–100 | **94.1 / 95.0 / 94.2** | 82–84% |

### Qwen3.8-27B（稠密）：固定 d2 最快，反而比默认自适应快

![Qwen3.8 MTP depth](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig2_qwen38_mtp_depth.png)

| 配置 | 中文 | 代码 | 推理 | 中位 |
|---|---:|---:|---:|---:|
| MTP 关 | 27.3 | 27.2 | 27.5 | 27.4 |
| 固定 d3 | 33.0 | 39.5 | 38.8 | 38.6 |
| 固定 d4 | 32.9 | 40.5 | 41.3 | 40.4 |
| 自适应（默认，3 轮） | 35–42 | 44–46 | 45–47 | 44.2 / 44.8 / 44.3 |
| **固定 d2（4 轮 ABBA）** | 43.6–44.3 | 47.2–48.7 | 47.5–48.7 | **47.2 / 47.2 / 48.2 / 47.7** |
| DFlash2（BF16 草稿模型） | 17.9 | 24.4 | 23.6 | 23.6 |

内置基准下 d2 同样领先（单位 tok/s）：

| 上下文 | 固定 d2 | 自适应 |
|---|---:|---:|
| 1K | 46.8 | — |
| 4K | 42.0 | 39.9 |
| 16K | 44.2 | 43.0 |

**为什么会这样？** 稠密 27B 每次 verify 的代价比 MoE（每 token 只激活 3B）高得多，在 M1 上深一层草稿的边际收益很快被 verify 成本吃掉。

另外两点：
- DFlash2 草稿模型在 M1 上只有 MTP d2 的一半速度，两次独立测试结论相同。
- 0.7.0 中 Qwen3.8-27B 的 `mtp_adaptive_max_depth` 不能低于 4，想要 d2 只能用 `mtp_fixed_depth`。

```json
"Qwen3.8-27B-MTPLX-Optimized-Speed-oMLX": { "mtp_enabled": true, "mtp_fixed_depth": 2 }
"CyberTiel-Coder-35B-MLX":                { "mtp_enabled": true }
```

---

## 4. 经验三（最大收益）：M1/M2 上把 BF16 张量转成 FP16

### 原理

量化模型的权重主体是打包好的整数（`U32`），但 scales、biases、norm、embedding、MTP 头、视觉塔等**非量化张量**通常以 **BF16** 发布。前向计算的激活精度跟随这些张量走。

**M1/M2 没有原生 bfloat16 运算单元**，Metal 只能软件模拟；FP16 却是原生支持的。这些 BF16 张量会拖慢每一次前向计算，而矩阵乘占比最大的 prefill 受影响最明显。

这个思路来自 oMLX 社区 PR [#3277](https://github.com/jundot/omlx/pull/3277)（尚未合入 0.7.0）。作者在 M1 Ultra 128 GB 上测 Qwen3.8-Flash-Next：prefill **+60%**，decode +5~14%。

### 我们的教训

10-06 我们已经给 CyberTiel 做过一次 FP16 转换，但**只测了短上下文、关闭 MTP 的纯解码**（58.0 vs 56.4–59.4），结论是“无收益”，于是否决了。
10-07 看到 #3277 后补测 prefill 和长上下文，结果完全不同：

![FP16 cast](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig3_fp16_cast_prefill_decode.png)

**CyberTiel-35B-A3B**（转换 1569 个 BF16 张量，共 2.94 GB；ABBA，各跑 2 次取均值）：

| | BF16 原版 | FP16 转换后 | 变化 |
|---|---:|---:|---:|
| prefill 1K | 994 / 1176 | 1353 / 1434 | **+28%** |
| prefill 4K | 1410 / 1403 | 1842 / 1940 | **+34%** |
| prefill 16K | 1096 / 1097 | 1644 / 1643 | **+50%** |
| 16K 首字时间 | 14.9 s | 10.0 s | −33% |
| 16K 上下文解码 | 97.2 / 94.4 | 115.5 / 117.7 | **+22%** |
| 4K 上下文解码 | 92.5 / 91.5 | 87.4 / 90.7 | 持平（噪声内） |
| 对话口径解码（ABBA） | 102.5 / 108.7 | 109.1 / 112.3 | +5% |
| 260 题能力集 | 206/260 | 206/260 | 持平 |
| 显存占用峰值 | 32.5 / 34.7 / 35.6 GB | 相同 | 0 |

质量判定细节：
- BF16 原版自己重跑一次，代码类题目就有 4 题翻转，LiveCodeBench 5/10 → 3/10。这是开启 MTP 时 T=0 的已知不确定性，见上游 [#4089](https://github.com/jundot/omlx/issues/4089)。
- BF16 和 FP16 之间的差异小于这个噪声底，FP16 版两次重跑完全一致。判定：**无退化**。

### 什么时候没用？（同样实测过）

| 模型 / 部件 | BF16 体量 | 结果 | 原因判断 |
|---|---|---|---|
| Qwen3.8-27B 文本部分 | 0 | 不需要 | 该打包本来就是 FP16 张量（“-FP16”版） |
| Qwen3.8-27B 视觉塔 | 0.86 GB | 无变化 | 视觉编码不是瓶颈，耗时主要在解码 |
| MiniCPM-V-4.6-8bit | 1.87 GB | 整体只快约 3%：图像编码 −14%，解码 −2% | 小模型以解码为主；没达到 5% 的上线门槛 |
| bge-reranker-v2-m3（F32） | 2.12 GB | 全 FP16：单独运行速度相同，但大模型常驻时慢 9.9 %；只转嵌入表则无代价（见补充二） | 升精度的临时缓冲在大模型常驻时代价被放大 |
| Qwen3-Embedding-0.6B-8bit | 0 | 不需要 | 已是 FP16（10-10 已下线，见补充二） |

**结论**：收益集中在**大模型 + 长 prompt** 的场景，例如代码 agent 和长文档。

### 怎么做（离线转换，不改 oMLX）

```python
# to_fp16.py <src> <dst>  —— 源目录只读；只改写含 BF16 张量的分片，其他文件用 APFS 克隆（cp -c，不占空间）
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
            m = mx.max(mx.abs(v.astype(mx.float32))).item()
            assert m <= 65504, f"{k} 超出 FP16 范围 ({m})"   # 越界即中止
            v = v.astype(mx.float16)
        out[k] = v
    mx.eval(out); mx.save_safetensors(dp, out, metadata=hdr.get("__metadata__") or {})
# 最后把 config.json（含 text_config / vision_config）里的 "dtype"/"torch_dtype": "bfloat16" 改为 "float16"
```

CyberTiel 的转换报告：
- 最大绝对值 25.4，远小于 FP16 上限 65504。
- 26103 个极小值下溢为 0（占 BF16 元素的极小比例）。
- `model.safetensors.index.json` 不需要改。

**上线注意**：
1. 先放进隔离目录测 prefill、长上下文解码和能力集，过关后再替换。
2. 替换后**按 model_name 清掉旧的 SSD KV 缓存**（包括 GDN sidecar），否则会命中 BF16 时代的缓存块。
3. 把 BF16 原版留作回滚。
4. 上游 PR 合入后可以改用运行时开关，不再需要离线转换。

---

## 5. 经验四：专职小视觉模型 + 大模型分工

CyberTiel 和 Qwen3.8 都是多模态模型。MiniCPM-V-4.6 还有必要保留吗？三个模型跑同一套题：

![vision](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig4_vision_three_models.png)

| 指标 | MiniCPM-V-4.6 8-bit | Qwen3.8-27B | CyberTiel-35B |
|---|---:|---:|---:|
| 35 项套件 | 35/35 | 35/35 | 33/35 |
| 计数题（严格 JSON 复核） | ❌ 会陷入重复计数 | ✅ | ❌ |
| OCR：密集票据 / 小字 / 图表 / 30 行大图 60 字段 | 全对 | 全对 | 读错一个电话号码 |
| 套件总耗时 | **14.5 s** | 44.3 s | 15.1 s |
| 30 行大图转写 | **7.7 s** | 32.0 s | 14.9 s |
| 解码速度 | **~180 tok/s** | ~52 tok/s | ~85 tok/s |
| 常驻内存 | **~2.3 GB** | ~20.2 GB | ~29.7 GB |

**路由建议**：

| 任务 | 模型 |
|---|---|
| OCR、票据、图表读数、看图描述、多图对比 | MiniCPM-V |
| 计数、空间关系推理、需要思考的视觉任务 | Qwen3.8-27B |
| 对话中顺手看一眼图 | CyberTiel（不用于需要精确数字的场景） |

---

## 6. 经验五：内存守卫的遗留参数会让模型互相驱逐

`settings.json` 里的 `memory.soft_threshold` 在 0.7.0 中的含义是：
- **只有等于 0.85 时**才表示“不覆盖、交给档位管理”（`balanced` 档实际是 0.90）。
- 其他任何值都被当成**显式覆盖**。

我们在 0.6.x 时代手工设成了 0.70，带来两个问题：
- 准入软目标只剩约 29 GB，连默认的 CyberTiel（30 GB）自己都超标。
- 一调用视觉或重排序模型，就会把默认模型挤出内存，切回来要重新加载约 10 s。

改回 0.85 之后，CyberTiel、MiniCPM、Reranker、Embedding 四个模型（共 34.8 GB）可以同时常驻，切换只要 0.1–0.3 s，swap 没有增长。Qwen3.8 加载时仍会正确地与 CyberTiel 互斥。

---

## 7. 经验六：不要让其他引擎直接加载生产模型目录

09-21 我们用另一个 MLX 引擎直接加载了 Qwen3.8 的生产目录做对比，它**改写了 `model.safetensors.index.json`**，丢掉了：
- 31 个 `mtp.*` 条目（当时发现并补回了）；
- **333 个 `vision_tower.*` 条目**（当时没发现）。

0.7.0 按 index 加载权重，于是视觉塔被静默禁用。图像请求返回 500（`'NoneType' object has no attribute 'patch_embed'`），这个问题持续了两周才被发现。

**修复**：读取每个分片的 safetensors 头，重建 index。权重文件一个字节都不用动。

**预防**：
- 需要对照测试时，先用 `cp -c` 做 APFS 克隆，再交给其他引擎。
- 测完核对生产目录 index 的哈希，以及条目数是否和分片头一致。

```python
# 核对：index 条目数应等于所有分片头里的张量总数
import json, struct, glob
idx = json.load(open("model.safetensors.index.json"))["weight_map"]
n = 0
for f in glob.glob("*.safetensors"):
    with open(f, "rb") as fh:
        h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0])); n += len(h) - ("__metadata__" in h)
print(len(idx), n)   # 两者必须相等
```

---

## 8. 经验七：0.7.0 的 Reranker 会把长文档静默截断到 512 token

`/v1/rerank` 遇到编码器类重排序模型（如 bge-reranker-v2-m3，模型本身支持 8192）时：
- 每个 query-文档对都按 **512 token** 截断；
- 请求体里的 `max_length` 和模型设置都不起作用（上游 [#4161](https://github.com/jundot/omlx/issues/4161)）。

我们构造了一个约 3K token 的长文，把真正的答案放在中部，再放两个只含关键词的短干扰文：

![rerank](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig6_reranker_truncation.png)

| | 长文 | 干扰 A | 干扰 B | 长文排名 |
|---|---:|---:|---:|---:|
| 整篇送入 | **0.000** | 0.543 | 0.680 | 第 3（末位） |
| 客户端按约 300 字切块，取最大分 | **0.996** | 0.543 | 0.680 | **第 1** |

**对策**：在客户端按段落或约 300 字切块（相邻块留重叠），每篇文档取各块的最高分。在上游修复之前，这是唯一可靠的办法。

**追加（2026-10-10）**：96 组「大海捞针」实测与推荐做法（约 192 token 窗口、不重叠、取最大分；重叠 25 % 无益）见“补充二”的补2.5。

---

## 9. 经验八：社区榜单的数字要看语料

omlx.ai 社区榜上，另一台同为 M1 Ultra 64 核 GPU / 64 GB 的机器跑 Qwen3.8-27B，4K TG 能到 66–72 tok/s，是我们的 1.6 倍。
详情页显示他们用的是 **Code (Mixed)** 语料，而且同一会话里各档差异巨大：

| 上下文 | 社区那台机器的 TG（tok/s） |
|---|---:|
| 1K | 41 |
| **4K** | **68** |
| 8K | 33 |
| 16K | 42 |

我们用同一语料复测（不上传）：

![corpus effect](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig5_leaderboard_corpus_effect.png)

| 本机 Qwen3.8-27B | 1K | 4K | 8K | 16K |
|---|---:|---:|---:|---:|
| Code (Python)，d2 | 46.8 | 42.0 | — | 44.2 |
| Code (Mixed)，d2 | 43.1 | **51.4** | 30.7 | — |
| Code (Mixed)，自适应 | 42.7 | 48.7 | 28.6 | — |

**结论**：
- Code (Mixed) 的 4K 那段文本很容易被 MTP 猜中（续写内容和 prompt 高度重复），所以 4K 一档普遍虚高。1K 和 8K 两档就没有这个现象。
- 换成这个语料后，d2 仍然胜过自适应，不需要改配置。
- **看榜单时，先看语料和同一会话的各档曲线，别只看最高的那一格。**

其他社区数据对照（同为 M1 Ultra 64 核 GPU，omlx.ai，2026-10）：

| 模型 | 来源 | 4K PP | 4K TG | 16K PP | 16K TG |
|---|---|---:|---:|---:|---:|
| Cyber-Tiel-35B-A3B **oQ4e**-MTP | 社区，128 GB | 1219 | 73.6 | 1144 | 92.0 |
| CyberTiel-35B-A3B **oQ6e**，BF16 原版 | 本机 | 1406 | 92 | 1096 | 96 |
| CyberTiel-35B-A3B **oQ6e**，FP16 转换 | 本机 | **1891** | 89 | **1644** | **117** |

我们的 6-bit 版本更大，转成 FP16 后反而全面快于社区的 4-bit 版本。

---

## 10. 经验九：评测代码模型，必须用你实际使用的思考模式

10-07 下午我们复查了 oMLX 讨论区和 Hugging Face 趋势榜，挑了两个与主力 CyberTiel 同架构（Qwen3.6-35B-A3B MoE）、
体积相同（oQ6e 约 28 GB）的候选。两者都先按第 4 节转成 FP16，再与生产版同时段对比：

- **CyberTiel 上游 09-29 重量化版**：作者换了代码 + 安全加权的校准语料，472 个量化张量里有 461 个变了。
- **KAT-Coder-V2.5-Dev-VL-oQ6e-mtp**：讨论区有人推荐；快手官方 SWE-bench Verified 69.4，带护栏。

![thinking flip](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig8_code_candidates_thinking_flip.png)

| 测试 | 生产 CyberTiel | 上游 09-29 重量化 | KAT-Coder-V2.5 |
|---|---:|---:|---:|
| 代码 324 题，关思考，T=0（HumanEval 164 / MBPP 100 / LCB 60） | 242（重跑 240） | 247 | **265**（+29/−6，p=0.0001） |
| 通用 260 题门禁 | 205 | 206 | 212 |
| **LiveCodeBench 60，开思考，T=0.6**（我们实际的用法） | **40** | 38 | **26**（0 胜 14 负） |
| 开思考回答长度中位数 | 926 token | 902 token | 306 token |
| 速度（prefill 4K / 16K，对话口径） | 1910 / 1620，77–92 | — | 1917 / 1620，91–92 |

结论：

- **KAT 关思考写短代码确实更强**，但它经过 RL 训练，思考非常短。到了需要长推理的算法题上，反而明显输给 CyberTiel。如果只跑关思考的 HumanEval/MBPP，就会得出完全相反的选型结论。
- **上游重量化版**三项测试都在噪声内（同一配置重跑，324 题会相差 2 题），思考 token 反而多了 27%。模型卡上的提升来自 GGUF + SWE-bench-Live，在 MLX 口径下没有复现。
- 两者速度一致（同架构、同量化）。所以这次不换模型。

> 方法提示：长时间压测后，SSD 缓存会涨到上限，几个 28 GB 模型轮换也会拖慢整机。我们测试时段的对话速度掉到 77–92 tok/s，重启并清理缓存后回到 112 tok/s。跨模型对比只看同时段 ABBA 数据。

---

## 11. 经验十：64 GB 机器上关掉 hot cache，长上下文才真正可用

oMLX 0.7.0 的 hot cache（内存缓存层，本机原设 6 GB）存放在 oMLX 进程的 CPU 内存里。
空闲时这块内存会被 macOS 压缩，但仍然算在内存守卫的 oMLX 占用里。结果是：只要加载了 CyberTiel（约 30 GB），
64K 的 prompt 就会被守卫拒绝（"Prefill would require ~44.5 GB … dynamic ceiling 43.1 GB"）。

| hot cache | oMLX 空载常驻 | 12K 多轮对话命中缓存的首字 | CyberTiel 64K | CyberTiel 128K |
|---|---:|---:|---|---|
| 6 GB | 7.8 GB | 0.47–0.68 s | ❌ 被拒绝 | — |
| **0（只用 SSD 缓存）** | **0.8 GB** | **0.43–0.46 s** | ✅ prefill 957 tok/s，解码 71 | ✅ prefill 614，解码 60，峰值 36.8 GB |

本机 SSD 恢复缓存块比读被压缩的内存还快，所以关掉 hot cache 没有任何代价。上游 main 的 #4252 修了同一个问题（拒绝前先释放 hot cache），但还没进正式版。

同一轮还把 `max_concurrent_requests` 从 4 调回上游默认的 8。8 路并发时总吞吐 +18%，最差首字从 15.3 s 降到 3.4 s（上限为 4 时，多出的请求只能排队）。

要不要把 Metal 上限提高（`sudo sysctl iogpu.wired_limit_mb=57344`，48 → 56 GB），让 CyberTiel（约 26 GB）和 Qwen3.8（约 19.5 GB）同时常驻？
要看使用方式。我们日常只用 CyberTiel，Qwen 是按需手动调用。切换的代价只是偶尔多等一次加载：Qwen 约 6 s，回切 CyberTiel 约 10 s。
为此长期把系统和 KV 的余量压到约 8 GB 并不划算，所以保持默认 48 GB；CyberTiel 单独跑 128K 时峰值 36.8 GB，完全够用。
如果你的两个模型需要频繁交替，才值得提高上限。

这一轮否决的方案：
- Burst Decode `aggressive`：在噪声内。
- ANE prefill：CyberTiel 反而更慢；Qwen 每个 ANE 实例要 14 GiB，64 GB 内存装不下。
- 把 Qwen3.8 换成 gs64 打包来启用 Q4 prefill 内核：prefill 完全一致。27B 稠密模型在 M1 上约 290 tok/s，已经是算力上限。
- TurboQuant 4-bit KV：两个模型的解码都慢约 20%。

---

## 补充：这些笔记背后的模型已公开发布，并附完整设置（2026-10-07 晚）

![本机栈](images/fig11_local_stack_overview.png)

我们每天在用的三个模型——最终权重、中英文模型卡、配图和逐模型 oMLX 设置——已上传到 Hub，机器出意外或换新 Mac 时可以快速还原：

| 模型 | 仓库 | 角色 | 实测（M1 Ultra 64 GB） |
|---|---|---|---|
| CyberTiel-Coder-35B-A3B（FP16 张量、fixed MTP 头） | [YCF-AI/CyberTiel-Coder-35B-MLX](https://huggingface.co/YCF-AI/CyberTiel-Coder-35B-MLX) | 日常主力、代码/智能体、去审查 | 107–116 tok/s，prefill 4K ≈ 1890 |
| Qwen3.8-27B Huihui 去审查 oQ4e + MTP（FP16 张量） | [YCF-AI/Qwen3.8-27B-Huihui-abliterated-oQ4e-MTP-FP16-MLX](https://huggingface.co/YCF-AI/Qwen3.8-27B-Huihui-abliterated-oQ4e-MTP-FP16-MLX) | 手动调用的推理 + 最强视觉 | 48–49 tok/s，视觉 35/35 |
| MiniCPM-V-4.6 Huihui 去审查 8-bit、downsample 4x | [YCF-AI/MiniCPM-V-4.6-Huihui-abliterated-8bit-MLX](https://huggingface.co/YCF-AI/MiniCPM-V-4.6-Huihui-abliterated-8bit-MLX) | OCR / 图表 | 约 140–180 tok/s，2.3 GB |

嵌入与重排模型后来（2026-10-10）做了改动，也已公开——见下一节“补充二”。（早期版本里用到的 Qwen3-Embedding-0.6B-8bit 已下线；分块建议见第 8 节与补2.5。）

**把审查模型换成去审查版而不牺牲速度与能力。** 门禁：ABBA 对话测速、260 题能力集（McNemar）、35 项视觉 + 大图 OCR、16K 上下文、以及同时检查"史实是否客观陈述"的敏感探针集。

![Qwen 替换](images/fig9_qwen38_uncensored_swap.png)
![MiniCPM 替换](images/fig10_minicpm_uncensored_swap.png)

- Qwen3.8-27B：采用 Huihui abliterated oQ4e——速度 49.3 vs 48.7 tok/s，260 题 215 vs 221（p=0.307），视觉 35/35，探针 10/10 且史实客观。PocketAiHub 版 10/10 但粉饰史实、16K 慢 4 %，否决。
- MiniCPM-V-4.6：Huihui abliterated 8-bit + 4x——35/35，OCR 120/120，144 vs 145 tok/s。Heretic 版空间描述退化（33/35），否决。
- 同一模型 ID 下换权重，所有客户端配置无需改动，**但必须按 model_name 清理该模型的 SSD KV 缓存**，否则会复用旧权重算出的缓存。

**还原工具包：** [`config/`](config/)（脱敏全局设置、全部逐模型设置、`RESTORE.md`）、[`recipes/`](recipes/)（FP16 转换、MTP 头量化、对话测速、260 题脚本、缓存清理）、原始数据 `data/2026-10-07-uncensor/`。

## 补充二：检索栈重做并公开——BGE-M3 8-bit + 嵌入表无损 FP16 的重排器（2026-10-10）

![本机栈](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig11_local_stack_overview.png)

2026-10-10 重做了本机检索模型并下线旧嵌入模型（Qwen3-Embedding-0.6B）：现在的栈是 CyberTiel、Qwen3.8-27B、MiniCPM-V、**BGE-M3**、**bge-reranker-v2-m3**，仍是五个模型。两个新模型都已上传 Hub，附中英文模型卡、配图、可逐字节复现的转换脚本与 oMLX 设置条目：

| 模型 | 仓库 | 做了什么 | 实测（M1 Ultra 64 GB，oMLX 0.7.0） |
|---|---|---|---|
| BAAI/bge-m3（稠密头） | [YCF-AI/bge-m3-8bit-MLX](https://huggingface.co/YCF-AI/bge-m3-8bit-MLX) | `.bin` 不经 torch、不执行 pickle 转 safetensors，**8-bit（gs64）：612 MB（上游 2.27 GB）**，上下文上限 2048，`embedding_batch_size` 16 | 单条 11.8 ms（FP32 13.1），HTTP 20 ms；索引批 13K tok/s；向量对 FP32 余弦 ≥ 0.999 |
| BAAI/bge-reranker-v2-m3 | [YCF-AI/bge-reranker-v2-m3-fp16emb](https://huggingface.co/YCF-AI/bge-reranker-v2-m3-fp16emb) | **嵌入表存 FP16**（无损，2.27 → 1.74 GB）；客户端分块、请求大小、候选数的用法规则 | 24 篇 412 ms，与 FP32 持平（+1.6 %，在其自身波动内）；排序完全相同 |

用 `recipes/` 里的脚本重建这两个文件，得到的 `model.safetensors` 与已发布的**逐字节一致**（2026-10-10 对照上游 revision 验证）。

### 补2.1 oMLX 0.7.0 上 FP16 权重反而慢——元凶是注意力掩码

![FP16 陷阱](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig13_bgem3_fp16_mask.png)

原生 XLM-R 路径的注意力掩码是 FP32，会把 FP16 激活升成 FP32；而嵌入/重排引擎每个请求后清空 MLX 缓冲池，所以每个请求都要重新为这些临时缓冲付费。BGE-M3 单条：FP32 13.3 ms、**FP16 44.5 ms（×3.3）**、8-bit 12.4 ms。运行时仅把掩码改成 FP16 的对照实验（磁盘上什么都没改）：FP16 降到 11.5 ms，批吞吐比 FP32 高 16 %。这就是上游 PR #4168，含于 v0.7.1.dev1（开发版；稳定版仍是 0.7.0）。稳定版发布之前，这两个模型不要给 oMLX 存成 FP16。

### 补2.2 8-bit 可以、4-bit 不行——以及为什么小规模 MRR 测试不够

![变体](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig12_bgem3_variants.png)
![质量](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig15_bgem3_retrieval_quality.png)

8-bit（gs64）：体积小 3.7 倍，单条延迟低于 FP32，最小余弦 0.99924，48 条查询 MRR@10 0.589 vs 0.582（配对 bootstrap Δ +0.007，CI [−0.010, +0.032]）。4-bit 的 MRR 一样（0.585），但最小余弦 0.93、top-10 近邻重叠仅 81 %——48 条查询的排序测试看不出来。8-bit 的向量与 FP32 的 BGE-M3 足够接近，我们拿它给托管的 BGE-M3 做本地兜底（请在自己的数据上复核）。

### 补2.3 批大小、上下文上限与内存

![批大小与内存](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig14_bgem3_batch_memory.png)

- `embedding_batch_size` **32 → 16**：索引吞吐 −3～4 %，但索引批在跑时到来的单条查询等待 256 ms（原 467 ms），最坏长文批峰值 +7.9 GB（原 +15.6 GB）。
- **必须设 `max_context_window`（2048）**：不设时 oMLX 取 `max_position_embeddings` = 8194，越过 8192 个可用位置——不报错、不出 NaN，但向量已不同。注意力是朴素 O(L²)：单条 8192 token 约 14–16 GB。
- 与 28.5 GB 大模型同驻：索引批 +2.0 GB 无压力事件；最坏情形（24 × 2048 token）+8.0 GB，越过软阈值约 1 秒，无驱逐。

### 补2.4 重排器选哪个精度？必须在大模型常驻下测

![重排器精度](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig16_reranker_precision.png)

上游 FP32 检查点恰好能用 FP16 精确表示（567,755,777 个参数全部），所以磁盘上转换无损。24 篇、CyberTiel 常驻：FP32 446.5 ms，**嵌入表 FP16 453.5 ms（+1.6 %）**，全 FP16 490.5 ms（**+9.9 %**；单独运行只慢 1.5 %）。我们最初仅凭单独运行的数据上线了全 FP16，同驻测试后 15 分钟撤回。量化（8/4-bit）的重排器权重在 0.7.0 上加载失败（严格加载）。纯 `transformers` 加载新文件时按 FP32 读入，分数与原件一致（最大 |Δp| 6e-7）。

### 补2.5 0.7.0 上用好重排器：分块、请求大小、停顿、K、阈值

![长文](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig17_reranker_longdoc.png)
![请求形状](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig18_reranker_request_shape.png)

- **长文必须分块**（第 8 节的量化版）：96 组「大海捞针」里，截断 512 token 找到答案块的比例 22.9 %；约 192 token 窗口、不重叠、取最大分 → **50.0 %**；所有分块方案都优于截断，彼此无差异，重叠只多花时间。客户端：`recipes/chunked_rerank.py`。
- **单请求 ≤ 32 篇**（≈ 27 ms/对，线性），**聊天流在跑时 ≤ 16 篇**：重排与 LLM 解码共用一个 MLX 线程，请求多长，流式输出就停多久（8/24/64/128 篇 → 0.18/0.64/1.5/3.0 s；128 篇拆成 16×8 → 最长停顿 0.5 s，总耗时 +19 %）。请求被完全串行化，并行发送只会排队；按长度分桶反而慢 2 %；0.7.0 没有篇数上限（1000 篇 ×512 token 约 +45 GB）。
- **一阶段 K = 16** 召回 100 %，重排后 MRR 与 K = 24 相同、耗时少 ⅓（48 条技术文档查询，请在自己的语料上复核）。
- **绝对分数门槛会丢答案**：48 条里有 8 条的金标分 < 0.2，「分数 ≥ 0.2」的过滤会丢掉 17 % 查询的答案。

### 补2.6 上游修复预演（PR #4168）

![预演](https://huggingface.co/datasets/YCF-AI/omlx-m1-ultra-tuning-notes/resolve/main/images/fig19_reranker_4168_preview.png)

在隔离的 oMLX 包副本里应用该 PR（生产应用未动）：全 FP16 成为最快方案——24 篇 414 → 306 ms（−26 %），footprint −40 %，长文可整篇评分（R@1 0.417，截断为 0.229）。全 FP16 在 1874 对里有 1 对越过 0.05 阈值，所以首个稳定版发布后先重做门禁再切换。

### 补2.7 一路上抓到的测量缺陷

- 评测集 chunk id 用文件名，发生重名（315 块里 94 块被遮蔽）；靠「HTTP 与进程内向量一致性校验」才发现，随后所有检索数字都已重算。
- 基准里自设 `mx.set_cache_limit(4 GB)` 造出了 2.7–2.9 倍的假「断崖」（oMLX 把缓存上限设为整机内存）；直接调用模型类绕过了引擎的逐请求清池，看起来像 33 GB 泄漏。基准请走引擎类或 HTTP，并用生产设置。
- 单独运行的数字误导过我们一次（全 FP16）；内存类改动的速度门禁必须在大模型常驻下测。

### 补2.8 下线旧嵌入模型后的复检（2026-10-10，`data/2026-10-10-final-check/`）

逐模型 oMLX 设置与「补充」一节发布的完全一致；全局设置只有新的 `embedding_batch_size` 16，以及机主 10-08 调大的 SSD 缓存上限（92 → 185 GB；缓存占用仅约 10 GB，对速度没有影响）。权重与 `config.json` 哈希、index 条目（vision 333、MTP 42/29）、10 项功能门禁（两个大模型的对话/思考/工具调用/视觉；BGE-M3；重排器）全部通过，MiniCPM-V 合成图 OCR 4/4 字段。解码（同一对话口径）：MiniCPM-V 179.8 tok/s（≈ 180），Qwen3.8-27B 48.4（10-07 为 49.2–49.4），BGE-M3 单条 20.8 ms、HTTP 索引批 ≈ 13K tok/s，重排器 24 篇 415.6 ms。**CyberTiel 在五个时间窗里测得 91–98 tok/s（含重启 oMLX 之后），10-07 为 98.5–116**；同时 MLX 微基准正常（≈ 590 GB/s、FP16 18 TFLOPS），Qwen3.8 与基准相差不到 2 %。我们怀疑是后台进程的 CPU 侧抖动先影响最快的解码循环，但没有证实。请把解码速度当作随时段变化的数字，对比配置只在同一时段用 ABBA。

## 12. 试过但否决的方案

| 方案 | 结论 | 依据 |
|---|---|---|
| DFlash2 投机解码（Qwen3.8） | ❌ | 23.6 tok/s，只有 MTP d2 的一半；两次独立测试结论一致 |
| MTPLX 2.12.2 原生引擎 | ❌ | M1 Ultra 上开不开 MTP 都输出乱码。另一位 M1 Ultra 用户的博客也显示：调优器宣称 25 tok/s，实际服务只有 16.5 tok/s |
| 官方 `Qwen3.8-27B-oQ4e-fp16-mtp` 替代 | ❌ | 同口径下打平，中文慢 8–10% |
| Qwen INT8 激活 prefill（oQ A8） | 不适用 | 只在 M5 及更新芯片（原生 INT8 张量运算）上生效；而且只支持 4/5-bit、group size 64 |
| ANE prefill（Qwen3.8） | 不适用 | 要求 group size 为 64 或 128，我们用的 Qwen 打包是 gs32。CyberTiel 在 0.6.x 实测“生效但无增益” |
| 空闲 2 s 后首次前向变慢约 900 ms（上游 #4040） | 本机未复现 | CyberTiel 空闲 0.3 s 到 15 s，首字时间都在 0.38–0.52 s，无需 `sudo sysctl iogpu.wired_limit_mb` |
| Qwen3.8-Flash-Next（180B MoE） | ❌ | 4-bit 就要 105 GiB，64 GB 装不下 |
| MiniCPM / Reranker / Qwen 视觉塔转 FP16 | ❌ | 见第 4 节，收益低于 5% 的门槛 |
| CyberTiel 上游 09-29 重量化版 | ❌ | 见第 10 节：三项能力测试都在噪声内 |
| KAT-Coder-V2.5-Dev oQ6e-mtp | ❌（生产口径） | 见第 10 节：关思考 +23 题，开思考 −14 题 |
| ANE prefill / gs64 Qwen / TurboQuant KV / Burst aggressive | ❌ | 见第 11 节 |
| BGE-M3 在 oMLX 0.7.0 上存 FP16 | ❌（待上游修复） | 单条慢 ×3.3（FP32 注意力掩码），见补2.1 |
| BGE-M3 4-bit | ❌ | 48 查询 MRR 不变，但余弦 0.93、top-10 近邻重叠 81 %，见补2.2 |
| bge-reranker-v2-m3 全 FP16（0.7.0） | ❌ | 大模型常驻时延迟 +9.9 %，见补2.4 |
| 8/4-bit 量化的重排器权重 | ❌ | 0.7.0 加载失败（严格加载） |

---

## 13. 最终配置（照抄即可）

```jsonc
// ~/.omlx/model_settings.json（节选；建议通过 admin API 或 GUI 修改，不要在服务运行时手改文件）
"CyberTiel-Coder-35B-MLX": {            // 权重已离线转换为 FP16 张量
  "temperature": 0.6, "top_p": 0.95, "top_k": 20,
  "enable_thinking": true, "mtp_enabled": true,   // 自适应深度（两个深度字段都留空）
  "max_context_window": 131072, "max_tokens": 32768, "ttl_seconds": 1800, "is_default": true
},
"Qwen3.8-27B-MTPLX-Optimized-Speed-oMLX": {
  "mtp_enabled": true, "mtp_fixed_depth": 2,
  "thinking_budget_enabled": true, "thinking_budget_tokens": 16384,
  "max_context_window": 131072, "max_tokens": 32768, "ttl_seconds": 600
},
"MiniCPM-V-4.6-8bit": { "temperature": 0.0, "top_p": 1.0, "max_context_window": 65536, "ttl_seconds": 600 },
"bge-m3-8bit": { "max_context_window": 2048, "ttl_seconds": 1800, "model_alias": "BGE-M3" },   // 补充二；旧嵌入模型 Qwen3-Embedding 已于 10-10 下线
"bge-reranker-v2-m3": { "ttl_seconds": 3600 }
```

```jsonc
// ~/.omlx/settings.json（节选）
"memory": { "memory_guard_tier": "balanced", "soft_threshold": 0.85, "hard_threshold": 0.95 }   // 0.85 = 交给档位管理
"scheduler": { "max_concurrent_requests": 8, "chunked_prefill": true, "embedding_batch_size": 16 }   // 前两项为上游默认值；批大小见补2.3
"cache": { "hot_cache_max_size": "0" }   // 64 GB 机器上关闭 hot cache，长上下文才进得去（见第 11 节）
```

---

## 14. 注意事项

- **开启 MTP 后，T=0 的输出不保证逐字一致**（上游 #4089）。做质量对比时先测出噪声底，例如同一配置重跑一次。
- **内置基准会自动上传到 omlx.ai**，上传内容包括芯片、内存、模型名、模型设置和成绩。不想公开可以加 `align_prompt_to_ane: true`；或者在 GUI 里确认上传选项。
- **CyberTiel 是 abliterated（去除拒答）模型**，请只在 OS 级沙箱里使用，限制网络与执行权限，并防范提示注入。
- 0.7.0 的 macOS 15 安装包缺少 `chat.html`，`/admin/chat` 页面会报 500，但不影响 API。
- GUI 保存设置会卸载全部模型并中断进行中的请求。改配置优先走 admin API：`PUT /admin/api/models/{id}/settings`。

---

## 参考

- oMLX 0.7.0 Release：<https://github.com/jundot/omlx/releases/tag/v0.7.0>
- PR #3277：M1/M2 上使用 float16 激活：<https://github.com/jundot/omlx/pull/3277>
- Issue #4161：rerank 512 截断：<https://github.com/jundot/omlx/issues/4161>
- Issue #4089：MTP 下贪心解码非确定：<https://github.com/jundot/omlx/issues/4089>
- Issue #4040：空闲后首次前向停顿：<https://github.com/jundot/omlx/issues/4040>
- omlx.ai 社区榜：<https://omlx.ai/benchmarks>
- MTPLX on M1 Ultra（第三方博客）：<https://rbeckner.com/articles/mtplx-m1-ultra-serving-reality/>
- 模型：
  - [Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-MLX-oQ6e-MTP)
  - [Qwen3.8-27B-MTPLX-Optimized-Speed-FP16](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16)
  - [MiniCPM-V-4.6-8bit](https://huggingface.co/mlx-community/MiniCPM-V-4.6-8bit)
  - [Qwen3-Embedding-0.6B-8bit](https://huggingface.co/mlx-community/Qwen3-Embedding-0.6B-8bit)
  - [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)

*欢迎拿你的机器复现并回帖交流。M1/M2 用户尤其建议试一下第 4 节的 FP16 转换。*

## 数据、脚本与许可

- `data/2026-10-06/`：MTP 深度扫描。`summary.jsonl` 是各配置的中位数，`results.jsonl` 是逐请求结果，`builtin.jsonl` 是内置基准，`vision_compare.jsonl` 是三模型视觉对比。
- `data/2026-10-07/`：第二轮测试数据。
  - FP16 转换的 ABBA 测速：`builtin.jsonl`、`summary.jsonl`。
  - 260 题能力集：`q260-*.jsonl`（逐题结果）、`q260-summary.jsonl`（汇总）。
  - 视觉 ABBA：`vision_compare.jsonl`。
  - Reranker 截断测试：`rerank.jsonl`。
  - 空闲后首字延迟：`idle_ttft.jsonl`。
  - 转换报告：`fp16-*.json`。
- `data/2026-10-07-candidates/`（第 10 节）：代码 324 题（`q260-code324-*`）、260 题门禁（`q260-g260-*`）、开思考 LiveCodeBench 60（`q260-lcb60think-*`）逐题结果；同时段测速 ABBA：`builtin.jsonl`、`results.jsonl`、`summary.jsonl`。
- `data/2026-10-07-perf/`（第 11 节）：hot cache 与并发 ABBA（`multiturn.jsonl`、`concurrent.jsonl`），长上下文、ANE、TurboQuant、Burst 测试（`builtin.jsonl`、`summary.jsonl`）。
- `scripts/to_fp16.py`：离线 BF16→FP16 转换脚本。只改写含 BF16 张量的分片，其余文件用 APFS 克隆。用法：`python to_fp16.py <源模型目录> <输出目录> [--f32]`。
- `data/2026-10-10-bge-m3/`、`data/2026-10-10-reranker/`（补充二）：只含数值结果、不含评测文本——变体与批/长度/内存扫描、队头阻塞、质量汇总与 bootstrap、各变体重排分数（1874 对）及 PyTorch FP32 参考、长文试验、`http_runs.txt` 终端摘录；`*_ARTIFACT.jsonl` 是已撤回的测量（补2.7）。
- `data/2026-10-10-final-check/`（补2.8）：下线旧嵌入模型后的复检数据。
- `recipes/bin2st.py`、`make_bgem3_q8.py`、`make_fp16_embeddings.py`、`chunked_rerank.py`（补充二）：可逐字节复现的转换脚本与重排器分块客户端。
- 许可：文字、图表与数据采用 CC BY 4.0；脚本采用 MIT。测试图全部为合成数据，不含任何个人信息。

*欢迎拿你的机器复现并回帖交流。M1/M2 用户尤其建议试一下第 4 节的 FP16 转换。*
