"""Summarize seed variation and paired user uncertainty without conflating them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reelrank.data.paths import ROOT
from reelrank.eval.runlog import file_hash, write_result

CONTRACT = ("cohort_hash", "relevance_hash", "candidate_pool_hash", "candidate_count",
            "n_users", "cohort_seed", "evaluation_label", "seen_item_policy", "training_label")


def paired_interval(difference, resamples=5000, seed=42):
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 1 or not len(difference) or not np.isfinite(difference).all():
        raise ValueError("expected finite nonempty per-user differences")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 100):
        indices = rng.integers(0, len(difference), size=(min(100, resamples-start), len(difference)))
        means[start:start+len(indices)] = difference[indices].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def load_record(path):
    record = json.loads(path.read_text())
    scores = Path(record["extra"]["archived_per_user_scores"])
    if not scores.is_absolute():
        scores = ROOT / scores
    if file_hash(scores) != record["extra"]["per_user_sha256"]:
        raise ValueError(f"Per-user checksum mismatch: {path}")
    frame = pd.read_parquet(scores).sort_values("user_id").set_index("user_id")
    if frame.index.has_duplicates or frame.empty or not np.isfinite(frame["recall@200"]).all():
        raise ValueError(f"Invalid per-user scores: {path}")
    if not frame["recall@200"].between(0, 1).all():
        raise ValueError("Recall must be between zero and one")
    if not np.isclose(frame["recall@200"].mean(), record["metrics"]["recall@200"], atol=1e-12, rtol=0):
        raise ValueError("Per-user scores disagree with reported mean")
    return record, frame["recall@200"]


def validate_pair(reference, other, reference_scores, other_scores):
    if any(reference["config"][k] != other["config"][k] for k in CONTRACT):
        raise ValueError("Evaluation contracts differ")
    if not reference_scores.index.equals(other_scores.index):
        raise ValueError("Evaluation users differ")
    # Processed data must match; model artifacts deliberately differ per seed.
    def inputs(record):
        return {k: v for k, v in record["provenance"]["input_files"].items() if "/processed/" in k}
    if not inputs(reference) or inputs(reference) != inputs(other):
        raise ValueError("Processed input fingerprints differ or are missing")


def summarize(directory):
    protocol = json.loads((directory / "protocol.json").read_text())
    seeds = protocol["seeds"]
    records, scores, sources = [], [], {}
    for seed in seeds:
        path = directory / f"seed{seed}" / "retrieval_two_tower.json"
        record, values = load_record(path)
        if record["seed"] != seed:
            raise ValueError("Training seed mismatch")
        for field in ("epochs", "batch_size", "lr", "emb_dim", "temperature", "cohort_seed", "max_eval_users"):
            if record["config"][field] != protocol[field]:
                raise ValueError(f"Hyperparameter differs from predeclared protocol: {field}")
        if records:
            validate_pair(records[0], record, scores[0], values)
            if record["provenance"]["source_hash"] != records[0]["provenance"]["source_hash"]:
                raise ValueError("Training source changed across seeds")
        records.append(record); scores.append(values)
        sources[str(path.relative_to(ROOT))] = file_hash(path)
    means = np.array([v.mean() for v in scores])
    averaged = np.stack([v.to_numpy() for v in scores]).mean(axis=0)
    comparisons = {}
    for method in ("popularity", "itemcf"):
        path = directory / "baselines" / f"retrieval_{method}.json"
        baseline, values = load_record(path)
        validate_pair(records[0], baseline, scores[0], values)
        kwargs = {"resamples": protocol["bootstrap_resamples"], "seed": protocol["bootstrap_seed"]}
        comparisons[method] = {
            "recall@200": float(values.mean()),
            "relative_lift_of_seed_mean_pct": float((means.mean()/values.mean()-1)*100),
            "mean_difference": float((averaged-values.to_numpy()).mean()),
            "seed_averaged_user_difference_ci95": paired_interval(averaged-values.to_numpy(), **kwargs),
            "per_seed_difference_ci95": {str(s): paired_interval(v.to_numpy()-values.to_numpy(), **kwargs)
                                           for s, v in zip(seeds, scores)},
        }
        sources[str(path.relative_to(ROOT))] = file_hash(path)
    metrics = {"recall@200_mean": float(means.mean()), "recall@200_std": float(means.std(ddof=1)),
               "per_seed_recall@200": dict(zip(map(str, seeds), means.tolist())), "comparisons": comparisons}
    write_result("retrieval_seed_sweep", "retrieval_seed_sweep", metrics,
                 config={"seeds": seeds, "cohort_seed": 42, "n_users": len(averaged),
                         "bootstrap_resamples": protocol["bootstrap_resamples"], "bootstrap_seed": protocol["bootstrap_seed"],
                         "interval_scope": "paired users conditional on these fitted models; not full training uncertainty",
                         "standard_deviation": "sample SD across training seeds (ddof=1)"},
                 seed=42, extra={"source_records_sha256": sources,
                                 "protocol_sha256": file_hash(directory / "protocol.json")})
    lines = ["# Retrieval seed sweep", "", "Fixed cohort; fixed hyperparameters; no best-seed selection.", "",
             "| Training seed | Recall@200 |", "|---|---:|"]
    lines += [f"| {s} | {m:.6f} |" for s, m in zip(seeds, means)]
    lines += ["", f"Mean ± sample SD: **{means.mean():.6f} ± {means.std(ddof=1):.6f}**.", "",
              "| Baseline | Recall@200 | Mean relative lift | Paired absolute-difference 95% CI* |", "|---|---:|---:|---|"]
    for method, comparison in comparisons.items():
        lo, hi = comparison['seed_averaged_user_difference_ci95']
        lines.append(f"| {method} | {comparison['recall@200']:.6f} | {comparison['relative_lift_of_seed_mean_pct']:+.2f}% | [{lo:.6f}, {hi:.6f}] |")
    lines += ["", "*Bootstrap of seed-averaged per-user recall differences, 5,000 user resamples. Conditional on these fitted models; not an interval for a newly trained model or online utility. Per-seed intervals are in the JSON summary. An interval crossing zero is inconclusive, not evidence of equivalence.",
              "", "Random-exposure evaluation remains exploratory; this does not repair prior holdout exposure. The original seed-42 serving model is retained. End-to-end ranking was not rerun and its prior NDCG interval still crosses zero."]
    (directory / "SUMMARY.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory.resolve())
