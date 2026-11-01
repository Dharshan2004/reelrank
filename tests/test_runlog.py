import json
import zipfile

from reelrank.eval import runlog


def test_runs_preserve_old_results_and_actual_source(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "ROOT", tmp_path)
    monkeypatch.setattr(runlog, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(runlog, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(runlog, "PROCESSED", tmp_path / "processed")
    monkeypatch.setattr(runlog, "_git_sha", lambda: "test-sha")
    source = tmp_path / "reelrank" / "model.py"
    source.parent.mkdir()
    source.write_text("version = 1\n")
    p = runlog.write_result("same-name", "test", {"score": 1})
    first = json.loads(p.read_text())
    source.write_text("version = 2\n")
    runlog.write_result("same-name", "test", {"score": 2})
    second = json.loads(p.read_text())
    assert first['provenance']['source_hash'] != second['provenance']['source_hash']
    archived = tmp_path / first['record_path']
    assert json.loads(archived.read_text())['metrics']['score'] == 1
    with zipfile.ZipFile(archived.parent / "source.zip") as snapshot:
        assert snapshot.read("reelrank/model.py") == b"version = 1\n"


def test_run_records_support_external_storage(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "input.json").write_text('{"version": 1}')
    monkeypatch.setattr(runlog, "ROOT", root)
    monkeypatch.setattr(runlog, "RESULTS", external / "results")
    monkeypatch.setattr(runlog, "ARTIFACTS", external)
    monkeypatch.setattr(runlog, "PROCESSED", root / "data")
    monkeypatch.setattr(runlog, "_git_sha", lambda: "test-sha")
    result = json.loads(runlog.write_result("external", "test", {}).read_text())
    from pathlib import Path
    assert Path(result["record_path"]).is_file()
    assert str(external / "input.json") in result["provenance"]["input_files"]
