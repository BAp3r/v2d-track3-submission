"""本地实验作业：独立输出、输入哈希、进程组时限和失败凭据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from experiments.nvidia_v2d.policy_gate import verify_gate


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    phase: Literal["validation", "retarget", "training", "evaluation", "render"]
    episode_id: str
    backend: Literal["official", "dexpilot", "multitip", "fixture"]
    argv: list[str] = Field(min_length=1)
    cwd: Path
    inputs: list[Path] = Field(default_factory=list)
    gate: Path | None = None
    retarget_parquet: Path | None = None
    required_outputs: list[str] = Field(default_factory=list)
    wall_limit_s: float = Field(default=7200, gt=0, le=7200, allow_inf_nan=False)

    @field_validator("required_outputs")
    @classmethod
    def local_outputs(cls, value: list[str]) -> list[str]:
        if any(not p or Path(p).is_absolute() or ".." in Path(p).parts for p in value):
            raise ValueError("输出只能是作业目录内的相对路径")
        return value


class FileStamp(BaseModel):
    path: str
    size: int
    mtime_ns: int
    sha256: str


class RunRecord(BaseModel):
    schema_version: Literal["nvidia_v2d.policy_run.v1"] = "nvidia_v2d.policy_run.v1"
    status: Literal["running", "succeeded", "failed", "timed_out", "interrupted"]
    spec: RunSpec
    input_manifest_path: str
    output_manifest_path: str
    log_path: str
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    failure_reason: str | None = None
    input_unchanged: bool | None = None
    outputs: list[FileStamp] = Field(default_factory=list)


def stamp(path: Path) -> FileStamp:
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"输入/输出必须是无符号链接的普通文件: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"读取时文件变化: {path}")
    return FileStamp(
        path=str(path.absolute()),
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest.hexdigest(),
    )


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_job(spec: RunSpec, output: Path) -> RunRecord:
    """仅结束本作业进程组；成功退出也必须通过产物及输入不变检查。"""
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("输出路径不能经过符号链接")
    paths = list(spec.inputs)
    if spec.phase in ("training", "evaluation"):
        if spec.gate is None:
            raise ValueError("training/evaluation requires an explicit input gate")
        if spec.retarget_parquet is None or spec.backend == "fixture":
            raise ValueError("训练gate必须绑定当前官方或自有重定向文件；fixture使用独立短测入口")
        gate = verify_gate(
            spec.gate, backend=spec.backend, parquet=spec.retarget_parquet, episode_id=spec.episode_id
        )
        paths.extend(Path(p) for p in gate["verified_files"])
        paths = list(dict.fromkeys(paths))
    inputs = [stamp(path) for path in paths]
    output.mkdir(parents=True, exist_ok=False)
    record = RunRecord(
        status="running",
        spec=spec,
        input_manifest_path=str(output / "input-manifest.json"),
        output_manifest_path=str(output / "output-manifest.json"),
        log_path=str(output / "run.log"),
        started_at=datetime.now(UTC).isoformat(),
    )
    _write(Path(record.input_manifest_path), [x.model_dump() for x in inputs])
    _write(output / "run.json", record.model_dump(mode="json"))
    process = None
    try:
        env = {**os.environ, "FMC3_RUN_OUTPUT": str(output.resolve())}
        with Path(record.log_path).open("wb") as log:
            process = subprocess.Popen(
                spec.argv,
                cwd=spec.cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record.exit_code = process.wait(timeout=spec.wall_limit_s)
        if record.exit_code:
            raise RuntimeError(f"子进程退出码 {record.exit_code}")
        record.outputs = [stamp(output / p) for p in spec.required_outputs]
        record.status = "succeeded"
    except subprocess.TimeoutExpired:
        if process is not None:
            _stop(process)
        record.status, record.failure_reason = "timed_out", "达到本轮作业时限，已停止所属进程组"
    except KeyboardInterrupt:
        if process is not None:
            _stop(process)
        record.status, record.failure_reason = "interrupted", "作业被中断，保留日志和部分结果"
    except Exception as error:
        if process is not None and process.poll() is None:
            _stop(process)
        record.status, record.failure_reason = "failed", str(error)
    finally:
        record.finished_at = datetime.now(UTC).isoformat()
        if process is not None:
            record.exit_code = process.returncode
        try:
            record.input_unchanged = [stamp(path) for path in paths] == inputs
        except (OSError, ValueError):
            record.input_unchanged = False
        if not record.input_unchanged:
            record.status, record.failure_reason = "failed", "输入文件发生变化或无法复核"
        _write(Path(record.output_manifest_path), [x.model_dump() for x in record.outputs])
        _write(output / "run.json", record.model_dump(mode="json"))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = RunSpec.model_validate_json(args.spec.read_text())
    result = run_job(spec, args.output)
    print(result.model_dump_json(indent=2))
    raise SystemExit(0 if result.status == "succeeded" else 1)


if __name__ == "__main__":
    main()
