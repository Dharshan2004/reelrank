"""LLM judge for caption safety labels (optional; falls back to heuristics)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from reelrank.data.paths import ARTIFACTS, PROCESSED
from reelrank.safety.rules import rule_hit, rule_score

POLICY = """
Label short-video captions for a feed safety filter.
Unsafe if ANY apply: spam/ads/solicitation (WeChat/QQ adds, sales funnels), adult/sexual,
violence/gore encouragement, scams/misleading money schemes.
Otherwise Safe.
Reply with JSON only: {"label":"safe"|"unsafe","reason":"..."}.
""".strip()


def _cache_path() -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / "safety_judge_cache.json"


def _load_cache() -> dict:
    p = _cache_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_cache(cache: dict) -> None:
    _cache_path().write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def _heuristic_label(text: str) -> dict:
    hit = rule_hit(text)
    return {
        "label": "unsafe" if hit else "safe",
        "reason": "rule_hit" if hit else "no_rule_hit",
        "score": rule_score(text),
        "source": "heuristic",
    }


def _openai_label(text: str, model: str = "gpt-4o-mini") -> dict:
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": POLICY},
            {"role": "user", "content": text[:1000]},
        ],
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    latency_ms = (time.perf_counter() - t0) * 1000
    content = payload["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    usage = payload.get("usage", {})
    return {
        "label": "unsafe" if str(parsed.get("label", "")).lower() == "unsafe" else "safe",
        "reason": parsed.get("reason", ""),
        "score": 1.0 if str(parsed.get("label", "")).lower() == "unsafe" else 0.0,
        "source": "openai",
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def label_captions(n: int = 1500, seed: int = 42, prefer_llm: bool = True) -> Path:
    videos = pd.read_parquet(PROCESSED / "videos.parquet")
    caps = videos[["video_id", "caption"]].fillna({"caption": ""})
    # Prefer non-empty captions; oversample likely-spam via rules for class balance
    nonempty = caps[caps["caption"].astype(str).str.len() > 0].copy()
    # Weak positives: rule hits + captions for videos with elevated is_hate in train
    train = pd.read_parquet(PROCESSED / "interactions_train.parquet")
    hate_rate = train.groupby("video_id")["is_hate"].mean()
    hate_vids = set(hate_rate[hate_rate > 0].index.astype(int).tolist())
    rng = nonempty.sample(frac=1.0, random_state=seed)
    rule_pos = rng[rng["caption"].map(rule_hit)]
    hate_pos = rng[rng["video_id"].isin(hate_vids) & ~rng["caption"].map(rule_hit)]
    rest = rng[~rng["caption"].map(rule_hit) & ~rng["video_id"].isin(hate_vids)]
    n_pos = min(len(rule_pos) + len(hate_pos), max(150, n // 4))
    take_rule = min(len(rule_pos), max(len(rule_pos), n_pos // 3))
    take_hate = min(len(hate_pos), n_pos - take_rule)
    n_neg = min(len(rest), n - take_rule - take_hate)
    sample = pd.concat(
        [rule_pos.head(take_rule), hate_pos.head(take_hate), rest.head(n_neg)],
        ignore_index=True,
    )

    cache = _load_cache()
    use_llm = prefer_llm and bool(os.environ.get("OPENAI_API_KEY"))
    rows = []
    cost_log = []
    for _, row in sample.iterrows():
        text = str(row["caption"])
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in cache:
            lab = cache[key]
        else:
            if use_llm:
                try:
                    lab = _openai_label(text)
                except Exception as e:
                    lab = _heuristic_label(text)
                    lab["error"] = str(e)
            else:
                lab = _heuristic_label(text)
                # Weak label: treat elevated is_hate videos as unsafe when no LLM
                if lab["label"] == "safe" and int(row["video_id"]) in hate_vids:
                    lab = {
                        "label": "unsafe",
                        "reason": "weak_is_hate",
                        "score": 0.6,
                        "source": "weak_hate",
                    }
            cache[key] = lab
        rows.append(
            {
                "video_id": int(row["video_id"]),
                "caption": text,
                "label": lab["label"],
                "y": 1 if lab["label"] == "unsafe" else 0,
                "source": lab.get("source"),
                "reason": lab.get("reason", ""),
            }
        )
        if lab.get("source") == "openai":
            cost_log.append(
                {
                    "latency_ms": lab.get("latency_ms"),
                    "prompt_tokens": lab.get("prompt_tokens"),
                    "completion_tokens": lab.get("completion_tokens"),
                }
            )

    _save_cache(cache)
    out = ARTIFACTS / "safety_labels.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    (ARTIFACTS / "safety_judge_cost.json").write_text(
        json.dumps(
            {
                "n_labeled": len(rows),
                "llm_calls": len(cost_log),
                "mode": "openai" if use_llm else "heuristic",
                "cost_samples": cost_log[:20],
                "note": (
                    "OPENAI_API_KEY missing; heuristic/rule labels used"
                    if not use_llm
                    else "LLM judge used with cache"
                ),
            },
            indent=2,
        )
    )
    # Spot-check sample for human review (fill offline)
    spot = pd.DataFrame(rows).sample(n=min(200, len(rows)), random_state=seed)
    spot_path = ARTIFACTS / "safety_human_spotcheck.csv"
    spot[["video_id", "caption", "label"]].assign(human_label="").to_csv(spot_path, index=False)
    print({"labeled": len(rows), "unsafe": int(sum(r["y"] for r in rows)), "mode": "openai" if use_llm else "heuristic"})
    return out


if __name__ == "__main__":
    label_captions()
