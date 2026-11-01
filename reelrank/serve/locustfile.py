"""Locust load test for GET /recommend.

Headless (default via `make loadtest`):
  locust -f reelrank/serve/locustfile.py --headless -u 8 -r 4 -t 30s \\
    --host http://127.0.0.1:43123

Web UI:
  locust -f reelrank/serve/locustfile.py --host http://127.0.0.1:43123
"""

from __future__ import annotations

import json
import platform
import random

from locust import HttpUser, between, events, task

from reelrank.data.paths import PROCESSED, RESULTS
from reelrank.eval.runlog import write_result
from reelrank.serve.validation import validate_recommendations

_USER_IDS: list[int] = []


def _load_user_ids() -> list[int]:
    path = PROCESSED / "users.parquet"
    if not path.exists():
        return list(range(1, 501))
    import pandas as pd

    return pd.read_parquet(path)["user_id"].astype(int).tolist()


@events.init.add_listener
def _on_init(environment, **_kwargs) -> None:
    global _USER_IDS
    _USER_IDS = _load_user_ids()


@events.quitting.add_listener
def _write_results(environment, **_kwargs) -> None:
    """Persist p50/p95/p99/QPS + hardware so `make_readme_table` can pick it up."""
    total = environment.stats.total
    if total.num_requests == 0:
        return

    def pct(p: float) -> float | None:
        try:
            return float(total.get_response_time_percentile(p / 100.0))
        except Exception:
            return None

    duration = max(total.last_request_timestamp - total.start_time, 1e-9)
    metrics = {
        "tool": "locust",
        "n_requests": int(total.num_requests),
        "concurrency": int(getattr(environment.parsed_options, "num_users", 0) or 0),
        "errors": int(total.num_failures),
        "qps": float(total.num_requests / duration),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mean_ms": float(total.avg_response_time) if total.num_requests else None,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
    }
    out = write_result("serving_loadtest", "serving", metrics, config={
        "concurrency": metrics["concurrency"], "observed_duration_seconds": duration,
        "think_time_seconds": [0.05, 0.2], "throughput_claim": "achieved_workload_not_capacity",
        "response_validation": "nonempty_unique_finite_probabilities_and_timings",
    })
    print(f"wrote {out}")
    print(metrics)


class RecommendUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task
    def recommend(self) -> None:
        uid = random.choice(_USER_IDS) if _USER_IDS else 1
        with self.client.get(
            "/recommend",
            params={"user_id": uid, "k": 10},
            name="/recommend",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status={resp.status_code}")
                return
            try:
                body = resp.json()
            except Exception as exc:
                resp.failure(f"bad json: {exc}")
                return
            try:
                validate_recommendations(body)
            except (ValueError, KeyError, TypeError) as exc:
                resp.failure(f"invalid response: {exc}")
                return
            resp.success()
