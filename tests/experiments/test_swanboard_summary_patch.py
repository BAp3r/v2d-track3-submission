from __future__ import annotations

import hashlib
import json

import pytest

from experiments.nvidia_v2d.runtime import patch_swanboard_summary as patcher


def test_summary_uses_storage_folder_instead_of_display_order(tmp_path, monkeypatch):
    source = ("\n".join(before for before, _after in patcher.REPLACEMENTS) + "\n").encode()
    monkeypatch.setattr(patcher, "ORIGINAL_SHA256", hashlib.sha256(source).hexdigest())
    target, receipt = tmp_path / "project.py", tmp_path / "receipt.json"
    target.write_bytes(source)
    result = patcher.apply(target, receipt)
    assert result["status"] == "patched"
    assert result["metric_data_modified"] is False
    assert json.loads(receipt.read_text())["patched_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    tags = [{"name": "reward", "sort": 0, "folder": "1"},
            {"name": "loss", "sort": 1, "folder": "2"},
            {"name": "legacy", "sort": 7, "folder": None}]
    expression = patcher.REPLACEMENTS[1][1].split("=", 1)[1].strip()
    for tag, expected in zip(tags, ["1", "2", "7"], strict=True):
        calls = []
        eval(expression, {"tag": tag, "expr": {"run_id": "run"},
                          "get_tag_dir": lambda run, folder, calls=calls: calls.append((run, folder))})
        assert calls == [("run", expected)]
    before_repeat = target.read_bytes()
    assert patcher.apply(target, receipt)["status"] == "already_patched"
    assert target.read_bytes() == before_repeat


def test_unknown_source_is_not_modified(tmp_path):
    target = tmp_path / "project.py"
    target.write_text("unknown release")
    receipt = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="Unsupported"):
        patcher.apply(target, receipt)
    assert target.read_text() == "unknown release"
    assert not receipt.exists()


def test_ambiguous_source_is_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        patcher.transform(b"missing expected code")
