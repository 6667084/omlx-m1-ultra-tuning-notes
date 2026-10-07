"""对话口径测速：T=0、关思考、固定三类提示词；每次请求带唯一 nonce 前缀保证 cached_tokens=0。
用法：python3 bench_chat.py <alias> <label> [rounds] [max_tokens] [base_url]
结果追加到 results.jsonl；同时抓取窗口内 server.log 的 MTP 接受率行。"""
import sys, json, time, uuid, statistics, os, re
from omlx_api import call
alias, label = sys.argv[1], sys.argv[2]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 512
base = sys.argv[5] if len(sys.argv) > 5 else None
PROMPTS = {
    "zh_explain": "请用中文详细解释 TCP 三次握手和四次挥手的过程，以及为什么需要 TIME_WAIT 状态。分点说明，不少于 600 字。",
    "code": "Write a complete Python module implementing an LRU cache with TTL expiry, thread safety, and unit tests using unittest. Include docstrings.",
    "reason": "一个水池有进水管 A 和出水管 B。单开 A 需 6 小时注满，单开 B 需 9 小时放空。若先开 A 两小时后再同时开 A、B，问还需多少小时注满？请逐步推理并验证答案。",
}
LOG = os.path.expanduser("~/.omlx/logs/server.log")
def body(p):
    return {"model": alias, "messages": [{"role": "user", "content": f"[run {uuid.uuid4().hex[:12]}]\n" + p}],
            "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
kw = {"base": base} if base else {}
# 预热（含可能的冷加载），不计入
for _i in range(10):
    try:
        call("/v1/chat/completions", body("你好，回复 OK。") | {"max_tokens": 16}, **kw); break
    except Exception as e:
        print("warmup retry", e, flush=True); time.sleep(5)
log_pos = os.path.getsize(LOG) if not base else 0
rows = []
for r in range(rounds):
    for name, p in PROMPTS.items():
        res, dt = call("/v1/chat/completions", body(p), **kw)
        u = res["usage"]
        row = {"label": label, "alias": alias, "prompt": name, "round": r, "completion_tokens": u["completion_tokens"],
               "gen_tps": u.get("generation_tokens_per_second"), "ttft": u.get("time_to_first_token"),
               "e2e": round(dt, 2), "cached": u.get("prompt_tokens_details", {}).get("cached_tokens"), "ts": time.time()}
        rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
with open("results.jsonl", "a") as f:
    for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
acc = []
if not base:
    with open(LOG) as fh:
        fh.seek(log_pos)
        for line in fh:
            m = re.search(r"MTP\[\d+\].*tok/cycle=([\d.]+) accept=\d+/\d+ \(([\d.]+)%\)", line)
            if m: acc.append((float(m.group(1)), float(m.group(2))))
by = {}
for row in rows: by.setdefault(row["prompt"], []).append(row["gen_tps"])
summary = {"label": label, "per_prompt_median": {k: round(statistics.median(v), 1) for k, v in by.items()},
           "overall_median": round(statistics.median([x["gen_tps"] for x in rows]), 1),
           "cv_pct": round(100 * statistics.pstdev([x["gen_tps"] for x in rows]) / statistics.mean([x["gen_tps"] for x in rows]), 1),
           "mtp_tok_per_cycle_median": round(statistics.median([a[0] for a in acc]), 2) if acc else None,
           "mtp_accept_median": round(statistics.median([a[1] for a in acc]), 1) if acc else None}
print("SUMMARY", json.dumps(summary, ensure_ascii=False))
with open("summary.jsonl", "a") as f: f.write(json.dumps(summary, ensure_ascii=False) + "\n")
