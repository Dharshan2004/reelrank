"""Cohen's kappa between LLM judge labels and human spot-check labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from reelrank.data.paths import ARTIFACTS, RESULTS
from reelrank.eval.runlog import write_result


def _norm(x: object) -> str | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().lower()
    if s in {"", "nan", "none"}:
        return None
    if s in {"safe", "0", "s"}:
        return "safe"
    if s in {"unsafe", "1", "u", "unsafe ", "spam", "bad"}:
        return "unsafe"
    return s


def compute_kappa(csv_path: Path | None = None) -> dict:
    path = csv_path or (ARTIFACTS / "safety_human_spotcheck.csv")
    if not path.exists():
        raise FileNotFoundError(f"missing spot-check CSV: {path}")
    df = pd.read_csv(path)
    if "label" not in df.columns or "human_label" not in df.columns:
        raise ValueError("CSV must have columns: label, human_label")

    judge = df["label"].map(_norm)
    human = df["human_label"].map(_norm)
    mask = judge.notna() & human.notna() & judge.isin(["safe", "unsafe"]) & human.isin(
        ["safe", "unsafe"]
    )
    paired = df.loc[mask].copy()
    if len(paired) == 0:
        raise RuntimeError(
            f"no filled human labels in {path}; set human_label to safe/unsafe"
        )

    y_j = paired["label"].map(_norm).tolist()
    y_h = paired["human_label"].map(_norm).tolist()
    labels = ["safe", "unsafe"]
    cm = confusion_matrix(y_j, y_h, labels=labels)
    kappa = float(cohen_kappa_score(y_j, y_h, labels=labels))
    agree = float(sum(a == b for a, b in zip(y_j, y_h)) / len(y_j))
    metrics = {
        "n_spotcheck_rows": int(len(df)),
        "n_labeled": int(len(paired)),
        "n_unlabeled": int((~mask).sum()),
        "agreement": agree,
        "cohen_kappa": kappa,
        "judge_unsafe": int(sum(y == "unsafe" for y in y_j)),
        "human_unsafe": int(sum(y == "unsafe" for y in y_h)),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),  # rows=judge, cols=human
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_result(
        "safety_human_kappa",
        "safety",
        metrics,
        config={"csv": str(path), "labels": labels},
    )
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=ARTIFACTS / "safety_human_spotcheck.csv",
        help="spot-check CSV with human_label filled",
    )
    args = parser.parse_args()
    compute_kappa(args.csv)


if __name__ == "__main__":
    main()
