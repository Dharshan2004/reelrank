import pandas as pd

from reelrank.safety import judge


def test_failed_llm_is_retried_and_human_annotations_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(judge, "PROCESSED", tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-no-network")
    pd.DataFrame({"video_id": [1], "caption": ["A peaceful mountain"]}).to_parquet(tmp_path / "videos.parquet")
    reviewed = tmp_path / "safety_human_spotcheck.csv"
    reviewed.write_text("video_id,human_label\n1,safe\n")
    calls = []

    def label(text, model):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("temporary test failure")
        return {"label": "safe", "source": "openai", "model": model, "reason": "test"}

    monkeypatch.setattr(judge, "_openai_label", label)
    judge.label_captions(n=1)
    assert pd.read_parquet(tmp_path / "safety_labels.parquet").source.tolist() == ["heuristic"]
    judge.label_captions(n=1)
    assert pd.read_parquet(tmp_path / "safety_labels.parquet").source.tolist() == ["openai"]
    judge.label_captions(n=1)
    assert len(calls) == 2  # Only the successful model label is reusable.
    assert reviewed.read_text() == "video_id,human_label\n1,safe\n"


def test_policy_and_model_changes_invalidate_label_cache(monkeypatch):
    original = judge.cache_key("caption", "openai", "model-a")
    assert original != judge.cache_key("caption", "heuristic")
    assert original != judge.cache_key("caption", "openai", "model-b")
    monkeypatch.setattr(judge, "POLICY", "a different policy")
    assert original != judge.cache_key("caption", "openai", "model-a")
