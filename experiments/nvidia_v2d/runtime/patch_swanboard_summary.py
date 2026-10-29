"""Repair swanboard 0.1.10b2 project summaries in an isolated SDK environment.

The SDK stores tags under ``folder``; ``sort`` is display order and starts at 0.
This patch changes only the dashboard reader, never stored metric data.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import json
import os
import sys
import sysconfig
from pathlib import Path

ORIGINAL_SHA256 = "b0d94c724bee6f36bdc3a0840577c86c71645fc98b96aa8b28302c959acc222c"
REPLACEMENTS = (
    ('"tags": [{"name": tag["name"], "sort": tag["sort"]} for tag in __to_list(experiment.tags)],',
     '"tags": [{"name": tag["name"], "sort": tag["sort"], "folder": tag["folder"]} for tag in __to_list(experiment.tags)],'),
    ('tag_path = get_tag_dir(expr["run_id"], str(tag["sort"]))',
     'tag_path = get_tag_dir(expr["run_id"], str(tag["folder"] if tag["folder"] is not None else tag["sort"]))'),
)


def transform(source: bytes) -> bytes:
    text = source.decode("utf-8")
    for before, after in REPLACEMENTS:
        if text.count(before) != 1:
            raise ValueError("Unexpected swanboard source; refusing an ambiguous patch")
        text = text.replace(before, after)
    return text.encode("utf-8")


def apply(target: Path, receipt: Path) -> dict:
    source = target.read_bytes()
    before_sha = hashlib.sha256(source).hexdigest()
    if before_sha != ORIGINAL_SHA256:
        # A repeat is accepted only if undoing exactly this patch matches the pinned original.
        original = source.decode("utf-8")
        for before, after in REPLACEMENTS:
            if original.count(after) != 1:
                raise ValueError("Unsupported or modified swanboard source SHA256")
            original = original.replace(after, before)
        if hashlib.sha256(original.encode("utf-8")).hexdigest() != ORIGINAL_SHA256:
            raise ValueError("Unsupported or modified swanboard source SHA256")
        return {"status": "already_patched", "target": str(target), "sha256": before_sha}
    updated = transform(source)
    result = {"status": "prepared", "target": str(target),
              "original_sha256": before_sha, "patched_sha256": hashlib.sha256(updated).hexdigest(),
              "diff": "".join(difflib.unified_diff(source.decode().splitlines(True),
                                                  updated.decode().splitlines(True),
                                                  fromfile="project.py.original", tofile="project.py.patched")),
              "metric_data_modified": False}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    temporary = target.with_suffix(".py.patch-tmp")
    with temporary.open("xb") as stream:
        stream.write(updated)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(target.stat().st_mode)
    temporary.replace(target)
    result["status"] = "patched"
    receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        parser.error("Run with the isolated observability virtualenv Python")
    if importlib.metadata.version("swanboard") != "0.1.10b2":
        parser.error("This patch is pinned to swanboard 0.1.10b2")
    target = Path(sysconfig.get_path("purelib")) / "swanboard/controller/project.py"
    result = apply(target, args.receipt.resolve())
    print(json.dumps({key: value for key, value in result.items() if key != "diff"}))


if __name__ == "__main__":
    main()
