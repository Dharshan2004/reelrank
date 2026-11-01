"""FAISS index build + recall/latency benchmark."""

from __future__ import annotations

import time
from typing import Any

import faiss
import numpy as np

from reelrank.data.paths import ARTIFACTS


def build_and_benchmark(
    embeddings: np.ndarray,
    n_queries: int = 500,
    k: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    xb = np.ascontiguousarray(embeddings.astype(np.float32))
    d = xb.shape[1]

    # Single-thread FAISS on macOS tends to avoid rare native segfaults during HNSW build.
    faiss.omp_set_num_threads(1)

    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    faiss.write_index(flat, str(ARTIFACTS / "index_flat.faiss"))

    rng = np.random.default_rng(seed)
    q_idx = rng.choice(len(xb), size=min(n_queries, len(xb)), replace=False)
    xq = xb[q_idx]

    _, gt = flat.search(xq, k)

    def _latency(index: faiss.Index, reps: int = 3) -> dict[str, float]:
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            index.search(xq, k)
            times.append((time.perf_counter() - t0) * 1000.0 / len(xq))
        arr = np.array(times)
        # Sample per-query latency (avoids a fragile full per-query loop on some FAISS builds).
        sample = xq[: min(64, len(xq))]
        per = []
        for i in range(len(sample)):
            t0 = time.perf_counter()
            index.search(sample[i : i + 1], k)
            per.append((time.perf_counter() - t0) * 1000.0)
        per_arr = np.array(per)
        return {
            "p50_ms": float(np.percentile(per_arr, 50)),
            "p99_ms": float(np.percentile(per_arr, 99)),
            "mean_ms": float(per_arr.mean()),
            "batch_mean_ms": float(arr.mean()),
        }

    metrics: dict[str, Any] = {
        "flat": _latency(flat),
        "n_items": int(len(xb)),
        "n_queries": int(len(xq)),
        "k": k,
    }

    try:
        hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = 200
        hnsw.add(xb)
        hnsw.hnsw.efSearch = 64
        faiss.write_index(hnsw, str(ARTIFACTS / "index_hnsw.faiss"))
        _, hnsw_idx = hnsw.search(xq, k)
        recalls = []
        for i in range(len(xq)):
            gt_set = set(gt[i].tolist())
            hit = len(gt_set.intersection(set(hnsw_idx[i].tolist())))
            recalls.append(hit / k)
        metrics["hnsw_recall_vs_flat"] = float(np.mean(recalls))
        metrics["hnsw"] = _latency(hnsw)
    except Exception as exc:  # noqa: BLE001 - native FAISS failures vary by platform
        metrics["hnsw_error"] = f"{type(exc).__name__}: {exc}"
        metrics["hnsw_recall_vs_flat"] = float("nan")

    return metrics
