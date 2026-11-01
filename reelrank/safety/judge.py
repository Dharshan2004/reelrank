"""LLM judge for caption safety labels (optional; falls back to heuristics)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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


DEFAULT_MODEL = "gpt-5.6-luna"


def cache_key(text: str, source: str, model: str | None = None) -> str:
    """Do not reuse labels from another policy, model, or heuristic fallback."""
    payload = {"version": 2, "text": text, "policy": POLICY, "source": source, "model": model}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _openai_label(text: str, model: str = DEFAULT_MODEL) -> dict:
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    # Luna is a reasoning model; effort=none keeps high-volume labeling cheap/fast.
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": POLICY},
            {"role": "user", "content": text[:1000]},
        ],
        "response_format": {"type": "json_object"},
        "reasoning_effort": "none",
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
        "model": model,
        "latency_ms": latency_ms,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def label_captions(
    n: int = 1500,
    seed: int = 42,
    prefer_llm: bool = True,
    model: str | None = None,
) -> Path:
    model = model or os.environ.get("OPENAI_SAFETY_MODEL", DEFAULT_MODEL)

    videos = pd.read_parquet(PROCESSED / "videos.parquet")
    caps = videos[["video_id", "caption"]].fillna({"caption": ""})
    # Prefer non-empty captions; oversample likely-spam via rules for class balance
    nonempty = caps[caps["caption"].astype(str).str.len() > 0].copy()
    # Rule-enriched sampling is for annotation efficiency, not prevalence estimation.
    # Dislike feedback is not a content-policy label or annotation target.
    rng = nonempty.sample(frac=1.0, random_state=seed)
    rule_pos = rng[rng["caption"].map(rule_hit)]
    rest = rng[~rng["caption"].map(rule_hit)]
    take_rule = min(len(rule_pos), max(0, n // 4))
    n_neg = min(len(rest), n - take_rule)
    sample = pd.concat(
        [rule_pos.head(take_rule), rest.head(n_neg)],
        ignore_index=True,
    )

    cache = _load_cache()
    use_llm = prefer_llm and bool(os.environ.get("OPENAI_API_KEY"))
    source = "openai" if use_llm else "heuristic"
    requested_model = model if use_llm else None
    n_cached = sum(
        1
        for _, row in sample.iterrows()
        if cache_key(str(row["caption"]), source, requested_model) in cache
    )
    print(
        f"safety judge: n={len(sample)} cached={n_cached} "
        f"mode={'openai/' + model if use_llm else 'heuristic'} "
        f"(~{len(sample) - n_cached} API calls left)",
        flush=True,
    )
    rows = []
    cost_log = []
    llm_ok = 0
    llm_fail = 0
    for i, (_, row) in enumerate(tqdm(sample.iterrows(), total=len(sample), desc="label captions")):
        text = str(row["caption"])
        key = cache_key(text, source, requested_model)
        fresh_call = False
        if key in cache:
            lab = cache[key]
        else:
            if use_llm:
                try:
                    lab = _openai_label(text, model=model)
                    llm_ok += 1
                    fresh_call = True
                except Exception as e:
                    llm_fail += 1
                    if llm_fail <= 3:
                        print(f"\nLLM call failed ({llm_fail}): {e}", flush=True)
                    lab = _heuristic_label(text)
                    lab["error"] = str(e)
            else:
                lab = _heuristic_label(text)
            # Failed API calls may be used in this run, but must not poison the
            # successful-LLM cache; the next LLM run retries these captions.
            actual_source = lab["source"]
            actual_model = lab.get("model") if actual_source == "openai" else None
            cache[cache_key(text, actual_source, actual_model)] = lab
            # Persist often so Ctrl+C / crashes keep progress.
            if (llm_ok + llm_fail) % 25 == 0:
                _save_cache(cache)
        rows.append(
            {
                "video_id": int(row["video_id"]),
                "caption": text,
                "label": lab["label"],
                "y": 1 if lab["label"] == "unsafe" else 0,
                "source": lab.get("source"),
                "model": lab.get("model"),
                "policy_hash": hashlib.sha256(POLICY.encode()).hexdigest(),
                "reason": lab.get("reason", ""),
            }
        )
        if fresh_call:
            cost_log.append(
                {
                    "latency_ms": lab.get("latency_ms"),
                    "prompt_tokens": lab.get("prompt_tokens"),
                    "completion_tokens": lab.get("completion_tokens"),
                }
            )

    _save_cache(cache)
    if use_llm:
        print(f"LLM calls ok={llm_ok} fail_fallback={llm_fail}", flush=True)
    out = ARTIFACTS / "safety_labels.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    (ARTIFACTS / "safety_judge_cost.json").write_text(
        json.dumps(
            {
                "n_labeled": len(rows),
                "llm_calls": len(cost_log),
                "llm_failures": llm_fail,
                "cached_labels": n_cached,
                "source_counts": pd.Series([r["source"] for r in rows]).value_counts().to_dict(),
                "sampling": "rule_enriched_not_prevalence_representative",
                "mode": "openai" if use_llm else "heuristic",
                "model": model if use_llm else None,
                "cost_samples": cost_log[:20],
                "note": (
                    "OPENAI_API_KEY missing; heuristic/rule labels used"
                    if not use_llm
                    else f"LLM judge ({model}) used with cache"
                ),
            },
            indent=2,
        )
    )
    # Spot-check sample for human review (fill offline)
    spot = pd.DataFrame(rows).sample(n=min(200, len(rows)), random_state=seed)
    spot_path = ARTIFACTS / "safety_human_spotcheck.csv"
    # Never erase completed human review during a retrain.
    if spot_path.exists():
        spot_path = ARTIFACTS / f"safety_human_spotcheck_pending_{time.time_ns()}.csv"
    spot[["video_id", "caption", "label"]].assign(human_label="").to_csv(spot_path, index=False)
    print({"labeled": len(rows), "unsafe": int(sum(r["y"] for r in rows)), "mode": "openai" if use_llm else "heuristic"})
    return out


if __name__ == "__main__":
    label_captions()
