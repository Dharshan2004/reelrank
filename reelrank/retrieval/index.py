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

    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    faiss.write_index(flat, str(ARTIFACTS / "index_flat.faiss"))

    hnsw = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(xb)
    hnsw.hnsw.efSearch = 64
    faiss.write_index(hnsw, str(ARTIFACTS / "index_hnsw.faiss"))

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
        # per-query ms approx
        # recompute distribution on one pass
        per = []
        for i in range(len(xq)):
            t0 = time.perf_counter()
            index.search(xq[i : i + 1], k)
            per.append((time.perf_counter() - t0) * 1000.0)
        per_arr = np.array(per)
        return {
            "p50_ms": float(np.percentile(per_arr, 50)),
            "p99_ms": float(np.percentile(per_arr, 99)),
            "mean_ms": float(per_arr.mean()),
            "batch_mean_ms": float(arr.mean()),
        }

    _, hnsw_idx = hnsw.search(xq, k)
    # recall vs exact
    recalls = []
    for i in range(len(xq)):
        gt_set = set(gt[i].tolist())
        hit = len(gt_set.intersection(set(hnsw_idx[i].tolist())))
        recalls.append(hit / k)
    metrics = {
        "hnsw_recall_vs_flat": float(np.mean(recalls)),
        "flat": _latency(flat),
        "hnsw": _latency(hnsw),
        "n_items": int(len(xb)),
        "n_queries": int(len(xq)),
        "k": k,
    }
    return metrics
