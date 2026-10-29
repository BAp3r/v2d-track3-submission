"""本地追加式训练指标入口；不依赖外部账号，不补造历史 loss。"""

from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.nvidia_v2d.policy_metric_curves import export_curves


class JsonMetricLogger:
    def __init__(
        self, path: Path, *, resume: bool = False, delegate: Any = None,
        context: Mapping[str, Any] | None = None,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if resume and path.exists():
            # A partial last write needs explicit recovery; never silently erase evidence.
            for line in path.read_text().splitlines():
                json.loads(line)
            if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                raise ValueError("指标末行不完整，恢复前需保留并修复尾行")
        self.path = path
        self.stream = path.open("a" if resume else "x", encoding="utf-8")
        self.csv_stream = path.with_suffix(".csv").open("a" if resume else "x", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_stream)
        if self.csv_stream.tell() == 0:
            self.csv_writer.writerow(["session_id", "wall_time_unix", "environment_steps", "metric", "value"])
        self.session_id = uuid.uuid4().hex
        self.context = dict(context or {})
        self.metadata_path = path.with_name(f"{path.stem}.session-{self.session_id}.json")
        with self.metadata_path.open("x", encoding="utf-8") as metadata_stream:
            json.dump({"session_id": self.session_id, "context": self.context}, metadata_stream, ensure_ascii=False)
            metadata_stream.write("\n")
            metadata_stream.flush()
            os.fsync(metadata_stream.fileno())
        self.delegate = delegate
        self.last_step = -1
        self.count = 0
        self.closed = False

    def log(self, metrics: Mapping[str, float], step: int) -> None:
        if step <= self.last_step:
            raise ValueError("指标步数必须严格递增")
        invalid = [name for name, value in metrics.items() if not math.isfinite(float(value))]
        row = {
            "schema_version": "nvidia_v2d.policy_metrics.v2",
            "session_id": self.session_id,
            "wall_time_unix": time.time(),
            "environment_steps": int(step),
            "metrics": {k: float(v) if math.isfinite(float(v)) else None for k, v in metrics.items()},
            "nonfinite_metric_names": invalid,
            "nonfinite_metric_values": {name: str(float(metrics[name])) for name in invalid},
            "context": {key: value for key, value in self.context.items() if key != "overrides"},
            "metadata_path": str(self.metadata_path),
        }
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        for name, value in metrics.items():
            self.csv_writer.writerow([self.session_id, row["wall_time_unix"], step, name, float(value)])
        self.csv_stream.flush()
        os.fsync(self.csv_stream.fileno())
        self.last_step = step
        self.count += 1
        if self.count == 1:
            export_curves(self.path)
        if self.delegate is not None:
            self.delegate.log(metrics, step=step)

    def log_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        export_curves(self.path)
        if self.delegate is not None and hasattr(self.delegate, "log_checkpoint"):
            self.delegate.log_checkpoint(*args, **kwargs)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.stream.close()
            self.csv_stream.close()
            try:
                export_curves(self.path)
            finally:
                if self.delegate is not None:
                    self.delegate.close()


def main() -> None:
    # This driver runs inside the fixed FlashCHORD image. Hydra arguments follow
    # the two positional paths; the upstream script and algorithm stay unchanged.
    script, metrics_path, *overrides = sys.argv[1:]
    spec = importlib.util.spec_from_file_location("local_logged_flash_training", script)
    if spec is None or spec.loader is None:
        raise ValueError("无法加载官方训练入口")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = module.run_training

    def logged_training(*args, **kwargs):
        logger = JsonMetricLogger(
            Path(metrics_path), delegate=kwargs.get("logger"),
            context={
                "script": script,
                "script_sha256": hashlib.sha256(Path(script).read_bytes()).hexdigest(),
                "overrides": overrides,
                "aggregation": "one_collection_all_optimizer_updates_mean",
                "log_interval": 1,
            },
        )
        kwargs["logger"] = logger
        kwargs["log_interval"] = 1
        try:
            return original(*args, **kwargs)
        finally:
            logger.close()

    module.run_training = logged_training
    sys.argv = [script, *overrides]
    module.main()


if __name__ == "__main__":
    main()
