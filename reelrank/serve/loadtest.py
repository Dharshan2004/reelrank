"""Lightweight async httpx load test for /recommend.

Prefer Locust for the default path (`make loadtest`). Keep this for a
no-spawn, fixed-N smoke check: `make loadtest-httpx`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import time

import httpx
import pandas as pd

from reelrank.data.paths import ARTIFACTS, PROCESSED, RESULTS
from reelrank.eval.runlog import write_result
from reelrank.serve.validation import validate_recommendations


async def run_load(base_url: str, n_requests: int = 200, concurrency: int = 8) -> dict:
    users = pd.read_parquet(PROCESSED / "users.parquet")["user_id"].astype(int).tolist()
    users = users[: max(n_requests, 1)]
    latencies = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        async def one(uid: int):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.get("/recommend", params={"user_id": uid, "k": 10})
                    r.raise_for_status()
                    validate_recommendations(r.json())
                    latencies.append((time.perf_counter() - t0) * 1000)
                except Exception:
                    errors += 1

        t_wall0 = time.perf_counter()
        await asyncio.gather(*[one(users[i % len(users)]) for i in range(n_requests)])
        wall = time.perf_counter() - t_wall0

    latencies.sort()
    def pct(p):
        if not latencies:
            return None
        return float(latencies[min(len(latencies) - 1, int(p / 100 * len(latencies)))])

    metrics = {
        "n_requests": n_requests,
        "concurrency": concurrency,
        "errors": errors,
        "qps": float(n_requests / wall) if wall > 0 else 0.0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mean_ms": float(statistics.mean(latencies)) if latencies else None,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    write_result("serving_loadtest_httpx", "serving", metrics, config={
        "n_requests": n_requests, "concurrency": concurrency, "tool": "httpx",
        "response_validation": "nonempty_unique_finite_probabilities_and_timings",
    })
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:43123")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run_load(args.base_url, args.n, args.concurrency))


if __name__ == "__main__":
    main()
