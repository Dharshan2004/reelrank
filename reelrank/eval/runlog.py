"""Write reproducible run result JSON files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
import zipfile
import shutil
from pathlib import Path
from typing import Any

from reelrank.data.paths import RESULTS, ROOT, ARTIFACTS, PROCESSED


def _record_path(path: Path) -> str:
    """Keep checkout-relative paths when possible; support external storage."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance() -> dict:
    """Fingerprint actual source and inputs, including uncommitted Python files."""
    source_paths = sorted((ROOT / "reelrank").rglob("*.py"))
    source_paths += [p for p in [ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "Makefile"] if p.exists()]
    source_paths += sorted((ROOT / "configs").glob("*.yaml"))
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in source_paths}
    inputs = {}
    for directory in [PROCESSED, ARTIFACTS]:
        for p in sorted(directory.glob("*")):
            if p.is_file() and p.suffix in {".parquet", ".json", ".pt", ".npy", ".faiss", ".joblib"}:
                inputs[_record_path(p)] = file_hash(p)
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {"dirty_worktree": dirty, "source_hash": config_hash(sources), "source_files": sources, "input_files": inputs}


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL)
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
        "provenance": _provenance(),
    }
    if extra:
        payload["extra"] = extra
    # Keep immutable records while preserving the familiar latest-result aliases.
    archive = RESULTS / "runs" / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{run_id}-{uuid.uuid4().hex[:8]}"
    archive.mkdir(parents=True)
    payload["record_path"] = _record_path(archive / "result.json")
    with zipfile.ZipFile(archive / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as snapshot:
        for relative in payload["provenance"]["source_files"]:
            snapshot.write(ROOT / relative, arcname=relative)
    if extra and extra.get("per_user_scores"):
        scores = Path(extra["per_user_scores"])
        if scores.is_file():
            shutil.copy2(scores, archive / "per_user_scores.parquet")
            payload["extra"] = {**extra, "archived_per_user_scores": _record_path(archive / "per_user_scores.parquet")}
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    (archive / "result.json").write_text(serialized)
    path = RESULTS / f"{run_id}.json"
    path.write_text(serialized)
    return path
