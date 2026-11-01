"""Generate results markdown table from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

from reelrank.data.paths import RESULTS


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.stem in {"e2e_no_safety", "e2e_with_safety"}:
            # Retained historical files predate the corrected feature contract.
            continue
        data = json.loads(path.read_text())
        metrics = data.get("metrics", {})
        # Put the primary task before auxiliary click/like metrics.
        if data.get("stage") == "ranking":
            metrics = {k: metrics[k] for k in ["long_view", "is_click", "is_like", "fusion"] if k in metrics}
        flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        nested = []
        for k, v in metrics.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    nested.append(f"{k}.{kk}={vv}")
        rows.append(
            {
                "run_id": data.get("run_id", path.stem),
                "stage": data.get("stage", ""),
                "split": data.get("config", {}).get("evaluation_split", ""),
                "metrics": flat,
                "nested": nested[:8],
            }
        )

    lines = [
        "| Run | Stage / evaluation | Key metrics |",
        "|---|---|---|",
    ]
    for r in rows:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(r["metrics"].items())[:6]]
        parts += r["nested"][:4]
        lines.append(f"| `{r['run_id']}` | {r['stage']} {r['split']} | {', '.join(parts)} |")

    out = RESULTS / "RESULTS_TABLE.md"
    lines += ["", "Legacy e2e_no_safety/e2e_with_safety JSON files are retained for history and excluded from this current table. Safety records predate the correctness fixes; labels were not regenerated.",
              "", "Random-exposure results are exploratory reuse. Positive lift requires its paired interval and scope; no online benefit is established."]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
