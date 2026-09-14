from __future__ import annotations
import argparse, asyncio, time
import aiohttp
from rich.console import Console
from rich.table import Table
from .metrics import RequestMetrics, BenchmarkResult

async def one(session, url, prompt, max_tokens):
    started = time.perf_counter(); first = None
    async with session.post(url, json={"prompt": prompt, "max_tokens": max_tokens}) as response:
        data = await response.json(); first = time.perf_counter()
        text = data["choices"][0]["text"]
    latency = time.perf_counter() - started
    tokens = max(1, len(text.split()))
    return RequestMetrics(first - started, (latency - (first - started)) / max(tokens, 1), latency, tokens)

async def main(url, requests, concurrency, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def guarded():
            async with sem: return await one(session, url, "Explain paged attention briefly.", max_tokens)
        result = BenchmarkResult(await asyncio.gather(*(guarded() for _ in range(requests))))
    table = Table(title="Serving Benchmark")
    for col in ("Metric", "p50", "p95", "p99"): table.add_column(col)
    for field in ("ttft", "tpot", "latency"):
        table.add_row(field, f"{result.percentile(field,50):.4f}", f"{result.percentile(field,95):.4f}", f"{result.percentile(field,99):.4f}")
    table.add_row("throughput", f"{result.throughput:.2f}", "-", "-")
    Console().print(table)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--url", default="http://localhost:8000/v1/completions"); p.add_argument("--requests", type=int, default=100); p.add_argument("--concurrency", type=int, default=16); p.add_argument("--max-tokens", type=int, default=32)
    a = p.parse_args(); asyncio.run(main(a.url, a.requests, a.concurrency, a.max_tokens))
