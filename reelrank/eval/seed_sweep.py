"""Run a predeclared retrieval sweep without touching the serving bundle.

Each child process receives its own artifact/results roots before imports.
An existing sweep directory is refused to prevent accidental overwrites.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reelrank.data.paths import ARTIFACTS, RESULTS, ROOT
from reelrank.eval.runlog import file_hash


def child_environment(artifacts: Path, results: Path) -> dict[str, str]:
    return {**os.environ, "REELRANK_ARTIFACTS_DIR": str(artifacts.resolve()),
            "REELRANK_RESULTS_DIR": str(results.resolve()),
            "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="retrieval_seed_sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a directory basename")
    if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds) or args.seeds[0] != 42:
        parser.error("provide at least three distinct seeds, starting with 42")
    dest = RESULTS / args.name
    artifact_root = ARTIFACTS / args.name
    if dest.exists() or artifact_root.exists():
        parser.error("sweep directory exists; inspect prior work before choosing a new name")
    protected = {str(p): file_hash(p) for directory in (ARTIFACTS, RESULTS)
                 for p in directory.iterdir() if p.is_file()}
    reference = json.loads((RESULTS / "retrieval_two_tower.json").read_text())["metrics"]["recall@200"]
    protocol = {"seeds": args.seeds, "epochs": 8, "batch_size": 1024, "lr": 0.001,
                "emb_dim": 64, "temperature": 0.05, "cohort_seed": 42,
                "max_eval_users": 2000, "bootstrap_resamples": 5000, "bootstrap_seed": 42,
                "reference_seed42_recall": reference, "reproduction_tolerance": 1e-8,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "protected_files": protected}
    dest.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    (dest / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    def run(label, module, arguments):
        env = child_environment(artifact_root / label, dest / label)
        start = time.monotonic()
        with (dest / f"{label}.log").open("w") as log:
            subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{label} complete in {time.monotonic()-start:.1f}s", flush=True)
    for seed in args.seeds:
        run(f"seed{seed}", "reelrank.retrieval.train",
            ["--epochs", "8", "--batch-size", "1024", "--lr", "0.001", "--emb-dim", "64",
             "--temperature", "0.05", "--seed", str(seed), "--cohort-seed", "42", "--max-eval-users", "2000"])
        if seed == 42:
            measured = json.loads((dest / "seed42/retrieval_two_tower.json").read_text())["metrics"]["recall@200"]
            if abs(measured-reference) > protocol["reproduction_tolerance"]:
                raise RuntimeError(f"Seed42 reproduction gate failed: {measured} vs {reference}; investigate before continuing")
    run("baselines", "reelrank.retrieval.baselines", ["--seed", "42", "--max-eval-users", "2000"])
    changed = [p for p, digest in protected.items() if not Path(p).is_file() or file_hash(Path(p)) != digest]
    if changed:
        raise RuntimeError(f"Protected serving/result files changed: {changed}")
    (dest / "integrity.json").write_text(json.dumps({"protected_files_unchanged": True, "count": len(protected)}, indent=2)+"\n")
    subprocess.run([sys.executable, "-m", "reelrank.eval.seed_summary", str(dest)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
