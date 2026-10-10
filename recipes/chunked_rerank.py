#!/usr/bin/env python3
"""Chunk-and-max reranking client for oMLX's /v1/rerank with bge-reranker-v2-m3.

Why: oMLX 0.7.0 truncates every (query, document) pair at 512 tokens, so an answer that sits deep inside a long
document is never seen (in our 96-trial needle test the top-1 hit rate was 0.229; with the method below it was 0.500).

What it does
  * splits each document into ~192-token windows (no overlap) with the reranker's own tokenizer.json
  * sends at most 32 windows per request (use --max-per-request 16 while a chat model is streaming:
    the reranker and LLM decoding share one MLX thread, so a request pauses the stream for about its own duration)
  * document score = max over its windows; returns documents sorted by that score
Requests are executed one after another on the server, so sending them in parallel only queues them.

Needs: `pip install tokenizers` (HF tokenizers) and a running oMLX server. Standard library for HTTP.

CLI:    python chunked_rerank.py --tokenizer tokenizer.json --query "..." doc1.txt doc2.txt
Module: from chunked_rerank import rerank;  rerank(query, [text1, text2], tokenizer_path="tokenizer.json")
Environment: OMLX_BASE_URL (default http://127.0.0.1:8000/v1), OMLX_API_KEY, OMLX_RERANK_MODEL (default BGE-Reranker-V2-M3)
"""
import argparse
import json
import os
import urllib.request

from tokenizers import Tokenizer


def split_windows(tok, text, window=192):
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) <= window:
        return [text]
    return [tok.decode(ids[i:i + window]) for i in range(0, len(ids), window)]


def _post(url, payload, api_key, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    # local server: bypass any system proxy
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        return json.loads(r.read())


def rerank(query, docs, tokenizer_path="tokenizer.json", base_url=None, api_key=None, model=None,
           window=192, max_per_request=32):
    """Return [(doc_index, score, best_window_text), ...] sorted by score, highest first."""
    base_url = (base_url or os.environ.get("OMLX_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
    api_key = api_key or os.environ.get("OMLX_API_KEY", "")
    model = model or os.environ.get("OMLX_RERANK_MODEL", "BGE-Reranker-V2-M3")
    tok = Tokenizer.from_file(tokenizer_path)
    if len(tok.encode(query, add_special_tokens=False).ids) > 250:
        print("warning: the query is long; query + window + special tokens must stay within 512 tokens")
    windows = []                                   # (doc_index, text)
    for di, d in enumerate(docs):
        windows += [(di, w) for w in split_windows(tok, d, window)]
    scores = [0.0] * len(windows)
    for s in range(0, len(windows), max_per_request):
        batch = windows[s:s + max_per_request]
        res = _post(f"{base_url}/rerank", {"model": model, "query": query, "documents": [w for _, w in batch],
                                            "return_documents": False}, api_key)
        for r in res["results"]:
            scores[s + r["index"]] = r["relevance_score"]
    best = {}
    for (di, text), sc in zip(windows, scores):
        if di not in best or sc > best[di][0]:
            best[di] = (sc, text)
    return sorted(((di, sc, tx) for di, (sc, tx) in best.items()), key=lambda t: -t[1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--query", required=True)
    ap.add_argument("--window", type=int, default=192)
    ap.add_argument("--max-per-request", type=int, default=32)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    texts = [open(f, encoding="utf-8").read() for f in a.files]
    for di, sc, tx in rerank(a.query, texts, a.tokenizer, window=a.window, max_per_request=a.max_per_request):
        print(f"{sc:.3f}  {a.files[di]}  ...{tx[:80].strip()!r}")
