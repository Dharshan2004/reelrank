"""Generate results markdown table from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

from reelrank.data.paths import RESULTS, ROOT


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        metrics = data.get("metrics", {})
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
                "metrics": flat,
                "nested": nested[:8],
            }
        )

    lines = [
        "| Run | Stage | Key metrics |",
        "|---|---|---|",
    ]
    for r in rows:
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(r["metrics"].items())[:6]]
        parts += r["nested"][:4]
        lines.append(f"| `{r['run_id']}` | {r['stage']} | {', '.join(parts)} |")

    out = ROOT / "results" / "RESULTS_TABLE.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
