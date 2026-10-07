"""固定 260 题能力集（oMLX 自带 eval 数据，seed 固定抽样；MMLU50/CMMLU50/TruthfulQA50/GSM8K50/HumanEval20/MBPP30/LiveCodeBench10）。
T=0、top_p 1、top_k 0、关思考；评分复用 oMLX 原始 extract_answer/check_answer；生成代码只在 macOS sandbox-exec 内执行（禁读用户目录、禁网络）。
用法：python3 q260.py <alias> <label> [workers]   结果写 q260-<label>.jsonl（可续跑）"""
import sys, json, asyncio, hashlib, subprocess, tempfile, types, concurrent.futures, os
sys.path.insert(0, "/Applications/oMLX.app/Contents/Resources")
sys.path.insert(1, "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages")
from omlx.eval import BENCHMARKS
from omlx_api import call
SUITES = json.loads(os.environ["Q_SUITES"]) if os.environ.get("Q_SUITES") else {"mmlu": 50, "cmmlu": 50, "truthfulqa": 50, "gsm8k": 50, "humaneval": 20, "mbpp": 30, "livecodebench": 10}
SANDBOX = '(version 1)(deny default)(allow process*)(allow sysctl-read)(allow mach-lookup)(allow file-read*)(deny file-read* (subpath "/Users"))(allow file-write* (subpath "/private/tmp") (subpath "/dev"))'
_run = subprocess.run
def sandbox_run(cmd, **kw):
    cmd = list(cmd); cmd[0] = "/usr/bin/python3"
    kw["env"] = {"PATH": "/usr/bin:/bin", "HOME": "/private/tmp", "LANG": "en_US.UTF-8"}
    return _run(["/usr/bin/sandbox-exec", "-p", SANDBOX] + cmd, **kw)
for name in ("humaneval", "mbpp", "livecodebench"):
    sys.modules["omlx.eval." + name].subprocess = types.SimpleNamespace(run=sandbox_run, TimeoutExpired=subprocess.TimeoutExpired, PIPE=subprocess.PIPE)
tempfile.tempdir = "/private/tmp"
alias, label = sys.argv[1], sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
out = f"q260-{label}.jsonl"
done = set()
if os.path.exists(out):
    done = {json.loads(l)["id"] for l in open(out)}
jobs = []
for name, n in SUITES.items():
    bench = BENCHMARKS[name](); items = asyncio.run(bench.load_dataset(n)); assert len(items) == n
    for i, it in enumerate(items):
        key = f"{name}:{it.get('id', i)}"
        if key not in done: jobs.append((name, bench, key, it))
def run(job):
    name, bench, key, it = job
    body = {"model": alias, "messages": bench.format_prompt(it), "max_tokens": bench.get_max_tokens(), "temperature": 0, "top_p": 1, "top_k": 0,
            "repetition_penalty": 1, "chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("Q_THINK"):  # 生产口径：开思考 + 模型推荐采样
        body.update({"max_tokens": 24576, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}})
    try:
        r, dt = call("/v1/chat/completions", body, timeout=900)
        text = bench._strip_think_tags(r["choices"][0]["message"]["content"] or ""); pred = bench.extract_answer(text, it)
        ok = bool(bench.check_answer(pred, it)); fin = r["choices"][0].get("finish_reason"); u = r["usage"]
    except Exception as e:
        ok, pred, fin, u, dt = False, None, "error:" + type(e).__name__, {}, 0
    return {"id": key, "suite": name, "sha": hashlib.sha256(json.dumps(it, sort_keys=True).encode()).hexdigest()[:16], "correct": ok,
            "pred": str(pred)[:200], "finish": fin, "completion_tokens": u.get("completion_tokens"), "wall": round(dt, 2)}
with concurrent.futures.ThreadPoolExecutor(workers) as ex, open(out, "a") as f:
    for row in ex.map(run, jobs):
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
rows = [json.loads(l) for l in open(out)]
by = {}
for r in rows: by.setdefault(r["suite"], []).append(r["correct"])
summ = {"label": label, "total": f"{sum(r['correct'] for r in rows)}/{len(rows)}", **{k: f"{sum(v)}/{len(v)}" for k, v in by.items()},
        "truncated": sum(1 for r in rows if r["finish"] == "length"), "errors": sum(1 for r in rows if str(r["finish"]).startswith("error"))}
print("SUMMARY", json.dumps(summ, ensure_ascii=False)); open("q260-summary.jsonl", "a").write(json.dumps(summ, ensure_ascii=False) + "\n")
