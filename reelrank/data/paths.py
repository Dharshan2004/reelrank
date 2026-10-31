"""Shared filesystem paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PURE = RAW / "KuaiRand-Pure" / "data"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"
