"""Render a short docs/demo.gif from live /recommend responses."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

from reelrank.data.paths import ROOT


def _font(size: int):
    for name in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        p = Path(name)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def _frame(title: str, body: str, size=(960, 540)) -> Image.Image:
    img = Image.new("RGB", size, (18, 22, 28))
    draw = ImageDraw.Draw(img)
    title_font = _font(28)
    body_font = _font(18)
    draw.rectangle((24, 24, size[0] - 24, size[1] - 24), outline=(70, 120, 100), width=2)
    draw.text((48, 48), title, fill=(220, 230, 220), font=title_font)
    y = 100
    for line in body.splitlines():
        for wrapped in textwrap.wrap(line, width=78) or [""]:
            draw.text((48, y), wrapped, fill=(190, 200, 195), font=body_font)
            y += 24
            if y > size[1] - 48:
                return img
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:43123")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    parser.add_argument("--users", nargs="*", type=int, default=[1, 42, 100])
    args = parser.parse_args()

    frames: list[Image.Image] = []
    frames.append(
        _frame(
            "ReelRank /recommend",
            "Cascade demo\nretrieve → caption safety → ranking → top-k\n+ per-stage timings",
        )
    )

    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        frames.append(_frame("GET /health", json.dumps(health.json(), indent=2)))

        for uid in args.users:
            r = client.get("/recommend", params={"user_id": uid, "k": 5})
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])[:5]
            lines = [
                f"user_id={data.get('user_id')}  k={len(items)}  ranker={data.get('ranker')}",
                f"timings_ms={json.dumps(data.get('timings_ms', {}), indent=None)}",
                f"diagnostics={json.dumps(data.get('diagnostics', {}), indent=None)}",
                "",
                "top items:",
            ]
            for it in items:
                cap = (it.get("caption") or "")[:60].replace("\n", " ")
                lines.append(
                    f"- {it.get('video_id')}  score={it.get('score'):.3f}  "
                    f"p_long={it.get('p_long_view'):.3f}  {cap}"
                )
            frames.append(_frame(f"GET /recommend?user_id={uid}&k=5", "\n".join(lines)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=1800,
        loop=0,
        optimize=False,
    )
    print(f"wrote {args.out} frames={len(frames)}")


if __name__ == "__main__":
    main()
