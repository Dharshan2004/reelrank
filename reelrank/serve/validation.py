"""Response checks shared by load generators; HTTP 200 alone is insufficient."""
import math


def validate_recommendations(body: dict, k: int = 10) -> None:
    items = body.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= k:
        raise ValueError("Expected between one and k recommendations")
    ids = [item["video_id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate recommendations")
    for item in items:
        for key in ["score", "p_click", "p_long_view", "p_like"]:
            value = float(item[key])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {key}")
    times = body.get("timings_ms", {})
    if "total_ms" not in times or any(not math.isfinite(float(v)) or float(v) < 0 for v in times.values()):
        raise ValueError("Invalid timings")
