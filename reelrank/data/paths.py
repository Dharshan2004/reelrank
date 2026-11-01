"""Shared paths; storage overrides are read once when this module is imported."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# An installed wheel must not write datasets into site-packages.
_STORAGE_ROOT = ROOT if (ROOT / "pyproject.toml").is_file() else Path.cwd()


def _directory(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


DATA = _directory("REELRANK_DATA_DIR", _STORAGE_ROOT / "data")
RAW = DATA / "raw"
PURE = RAW / "KuaiRand-Pure" / "data"
PROCESSED = DATA / "processed"
ARTIFACTS = _directory("REELRANK_ARTIFACTS_DIR", _STORAGE_ROOT / "artifacts")
RESULTS = _directory("REELRANK_RESULTS_DIR", _STORAGE_ROOT / "results")
