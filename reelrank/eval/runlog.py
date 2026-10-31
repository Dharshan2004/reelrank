"""Write reproducible run result JSON files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from reelrank.data.paths import RESULTS


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def write_result(
    run_id: str,
    stage: str,
    metrics: dict[str, Any],
    config: dict[str, Any] | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "stage": stage,
        "metrics": metrics,
        "config": config or {},
        "config_hash": config_hash(config or {}),
        "seed": seed,
        "git_sha": _git_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload["extra"] = extra
    path = RESULTS / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
