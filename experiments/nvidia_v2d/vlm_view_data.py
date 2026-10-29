"""实验分支的只读标注对照服务；不启动 worker，不操作产品数据库。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from experiments.nvidia_v2d.vlm_compare_audit import timing


class ModelSource(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    name: str
    batch: Path


class ViewerConfig(BaseModel):
    plan: Path
    media_batch: Path
    models: list[ModelSource]
    policy_index: Path | None = None


class Segment(BaseModel):
    label: str
    start_time_s: float
    end_time_s: float
    review_status: str = "unreviewed"
    confidence: float | None = None


class CandidateView(BaseModel):
    segments: list[Segment]
    coverage: float
    overlap_pairs: int
    out_of_bounds: int
    gaps: list[tuple[float, float]]
    source: Literal["runtime_log"] = "runtime_log"
    log_rounding_s: float = 0.1


class ModelResult(BaseModel):
    id: str
    name: str
    status: str
    job_id: str | None = None
    failure_reason: str | None = None
    candidates: CandidateView | None = None
    segments: list[Segment] = []
    coverage: float = 0
    overlap_pairs: int = 0
    out_of_bounds: int = 0
    gaps: list[tuple[float, float]] = []


class EpisodeView(BaseModel):
    id: str
    task: str
    group: Literal["ego", "third"]
    dataset_root: str
    duration_s: float
    fps: float
    models: list[ModelResult]


class ModelProgress(BaseModel):
    id: str
    name: str
    succeeded: int
    failed: int
    pending: int
    segments: int


class RawCallView(BaseModel):
    id: str
    kind: Literal["video", "text", "frames"]
    response: str | None
    error: str | None
    started: float
    elapsed_s: float


class Snapshot(BaseModel):
    episodes: list[EpisodeView]
    models: list[ModelProgress]
    total: int


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return dict(json.load(stream))


def confined_file(path: Path, root: Path) -> Path:
    if not path.is_file() or any(p.is_symlink() for p in [path, *path.parents]):
        raise FileNotFoundError(path.name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise FileNotFoundError(path.name)
    return path


def uncovered(segments: list[Segment], duration: float) -> list[tuple[float, float]]:
    cursor, gaps = 0.0, []
    for segment in sorted(segments, key=lambda s: s.start_time_s):
        a, b = max(0, segment.start_time_s), min(duration, segment.end_time_s)
        if b <= a:
            continue
        if a > cursor:
            gaps.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        gaps.append((cursor, duration))
    return gaps


_CANDIDATE_LOG_LIMIT = 2 * 1024**2
_SEGMENT_LOG_PREFIX = (
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
    r"video_ingestion_agent\.ingestion\.segmentation\.segmenter - INFO - "
)
_SEGMENT_LOG_ROW = re.compile(
    _SEGMENT_LOG_PREFIX + r"\s{4,}video_clip_\d+: \[(-?\d+(?:\.\d+)?)s-(-?\d+(?:\.\d+)?)s\] (.+)$"
)
_SEGMENT_LOG_END = re.compile(_SEGMENT_LOG_PREFIX + r"Total segments before dedup: (\d+)\s*$")
_LOG_RECORD_START = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - ")


def candidate_log_segments(path: Path) -> list[Segment] | None:
    """只提取完整候选阶段；摘要与条数不符时不冒充已重建的候选层。"""
    with path.open("rb") as stream:
        raw = stream.read(_CANDIDATE_LOG_LIMIT + 1)
    if len(raw) > _CANDIDATE_LOG_LIMIT:
        raise ValueError("候选日志超出显示上限")
    segments: list[Segment] = []
    candidate_record = False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    for line in text.splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        summary = _SEGMENT_LOG_END.match(line)
        if summary is not None:
            return segments if len(segments) == int(summary[1]) else None
        if candidate_record and line.strip() and not _LOG_RECORD_START.match(line):
            # A newline in an action/object makes its first log row an incomplete label.
            return None
        match = _SEGMENT_LOG_ROW.match(line.rstrip("\r\n"))
        candidate_record = match is not None
        if match is not None:
            segments.append(
                Segment(
                    label=match[3],
                    start_time_s=float(match[1]),
                    end_time_s=float(match[2]),
                )
            )
    return None


class ComparisonStore:
    def __init__(self, config: ViewerConfig) -> None:
        self.config = config
        self._candidate_cache: dict[Path, tuple[int, int, list[Segment] | None]] = {}
        self.episodes = {e["run_key"]: e for e in read_json(config.plan)["episodes"]}
        if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", key) for key in self.episodes):
            raise ValueError("视频ID格式无效")
        if len({model.id for model in config.models}) != len(config.models):
            raise ValueError("模型ID重复")

    def states(self) -> dict[str, dict[str, Any]]:
        return {
            model.id: read_json(model.batch / "run.json")
            if (model.batch / "run.json").exists()
            else {"jobs": []}
            for model in self.config.models
        }

    def snapshot(self) -> Snapshot:
        states = self.states()
        jobs = {key: {j["run_key"]: j for j in state["jobs"]} for key, state in states.items()}
        episodes = []
        for key, source in self.episodes.items():
            results = []
            for model in self.config.models:
                job = jobs[model.id].get(key, {})
                status = job.get("status", "pending")
                segments = [Segment.model_validate(s) for s in job.get("annotations", [])]
                stats = timing(
                    [s.model_dump() for s in segments], source["duration_seconds"], source["fps"]
                )
                results.append(
                    ModelResult(
                        id=model.id,
                        name=model.name,
                        status=status,
                        job_id=job.get("job_id"),
                        failure_reason=job.get("failure_reason"),
                        candidates=self.candidates(
                            model, job, source["duration_seconds"], source["fps"]
                        ),
                        segments=segments,
                        coverage=stats["coverage"],
                        overlap_pairs=stats["overlap_pairs"],
                        out_of_bounds=stats["out_of_bounds"],
                        gaps=uncovered(segments, source["duration_seconds"]),
                    )
                )
            episodes.append(
                EpisodeView(
                    id=key,
                    task=source.get("task") or "",
                    dataset_root=source["dataset_root"],
                    group="ego" if source["dataset_root"].startswith("track_3") else "third",
                    duration_s=source["duration_seconds"],
                    fps=source["fps"],
                    models=results,
                )
            )
        progress = []
        for model in self.config.models:
            rows = [next(r for r in e.models if r.id == model.id) for e in episodes]
            good = sum(r.status == "succeeded" for r in rows)
            bad = sum(r.status == "failed" for r in rows)
            progress.append(
                ModelProgress(
                    id=model.id,
                    name=model.name,
                    succeeded=good,
                    failed=bad,
                    pending=len(rows) - good - bad,
                    segments=sum(len(r.segments) for r in rows),
                )
            )
        return Snapshot(episodes=episodes, models=progress, total=len(episodes))

    def candidates(
        self, model: ModelSource, job: dict[str, Any], duration: float, fps: float
    ) -> CandidateView | None:
        job_id = job.get("job_id")
        if job_id is None:
            return None
        if not isinstance(job_id, str) or not re.fullmatch(r"[a-zA-Z0-9-]+", job_id):
            raise ValueError("任务ID格式无效")
        parent = model.batch / ("preserved" if job.get("status") == "failed" else "workspace/jobs")
        path = parent / job_id / "runtime.log"
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError("候选日志路径不安全")
        if not path.exists():
            return None
        confined_file(path, model.batch)
        before = path.stat()
        if before.st_size > _CANDIDATE_LOG_LIMIT:
            raise ValueError("候选日志超出显示上限")
        cached = self._candidate_cache.get(path)
        if cached is not None and cached[:2] == (before.st_mtime_ns, before.st_size):
            segments = cached[2]
        else:
            segments = candidate_log_segments(path)
            after = path.stat()
            if (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size):
                self._candidate_cache[path] = (after.st_mtime_ns, after.st_size, segments)
        if segments is None:
            return None
        stats = timing([segment.model_dump() for segment in segments], duration, fps)
        return CandidateView(
            segments=segments,
            coverage=stats["coverage"],
            overlap_pairs=stats["overlap_pairs"],
            out_of_bounds=stats["out_of_bounds"],
            gaps=uncovered(segments, duration),
        )

    def media(self, key: str) -> Path:
        if key not in self.episodes:
            raise KeyError(key)
        root = self.config.media_batch / "recordings"
        return confined_file(root / key / "color.mp4", root)

    def calls(self, key: str, model_id: str) -> list[RawCallView]:
        if key not in self.episodes:
            raise KeyError(key)
        model = next((m for m in self.config.models if m.id == model_id), None)
        if model is None:
            raise KeyError(model_id)
        job = next((j for j in self.states()[model_id]["jobs"] if j["run_key"] == key), None)
        if job is None:
            return []
        if not re.fullmatch(r"[a-zA-Z0-9-]+", job["job_id"]):
            raise ValueError("任务ID格式无效")
        parent = model.batch / ("preserved" if job["status"] == "failed" else "workspace/jobs")
        root = parent / job["job_id"] / "model-calls"
        calls = []
        for path in sorted(root.glob("*.json"))[:100]:
            confined_file(path, model.batch)
            if path.stat().st_size > 1024**2:
                raise ValueError("响应文件超出显示上限")
            call = read_json(path)
            calls.append(
                RawCallView(
                    id=path.stem,
                    kind=call["request"]["kind"],
                    response=call.get("response", {}).get("text"),
                    error=call.get("error_type"),
                    started=call["started_unix"],
                    elapsed_s=call["finished_unix"] - call["started_unix"],
                )
            )
        return sorted(calls, key=lambda c: c.started)
