"""只读比较多模型subtask；时间覆盖和启发式线索不冒充准确率。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(data)
    return result.hexdigest()


def timing(segments: list[dict[str, Any]], duration: float, fps: float) -> dict[str, Any]:
    intervals = []
    invalid = outside = over_frame = 0
    for segment in segments:
        a, b = segment["start_time_s"], segment["end_time_s"]
        if not math.isfinite(a) or not math.isfinite(b) or b <= a:
            invalid += 1
            continue
        outside += a < -1e-8 or b > duration + 1e-8
        over_frame += a < -1 / fps - 1e-8 or b > duration + 1 / fps + 1e-8
        if b > 0 and a < duration:
            intervals.append((max(a, 0), min(b, duration)))
    intervals.sort()
    merged: list[list[float]] = []
    for a, b in intervals:
        if not merged or a > merged[-1][1] + 1e-8:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    covered = sum(b - a for a, b in merged)
    overlap_pairs = sum(
        min(b, d) - max(a, c) > 1e-8
        for index, (a, b) in enumerate(intervals)
        for c, d in intervals[index + 1 :]
    )
    return dict(
        covered_s=covered,
        coverage=covered / duration,
        uncovered_s=duration - covered,
        invalid=invalid,
        out_of_bounds=outside,
        beyond_one_frame=over_frame,
        overlap_pairs=overlap_pairs,
        overlap_extra_s=sum(b - a for a, b in intervals) - covered,
    )


def raw_call_metrics(root: Path) -> dict[str, Any]:
    """统计JSON语法有效性；不把解析成功当作语义或动作边界正确。"""
    calls = sorted((root / "model-calls").glob("*.json"))
    counts: Counter[str] = Counter()
    failures = []
    request_failures = []
    for path in calls:
        call = json.loads(path.read_text())
        kind = call["request"]["kind"]
        counts[kind + "_calls"] += 1
        if kind != "video":
            continue
        if "error_type" in call:
            counts["video_request_failed"] += 1
            request_failures.append(str(path))
            continue
        text = call.get("response", {}).get("text", "")
        match = re.search(r"\[[\s\S]*\]", text)
        try:
            proposals = json.loads(match.group(0)) if match else None
        except json.JSONDecodeError:
            proposals = None
        if isinstance(proposals, list):
            counts["video_json_arrays"] += 1
            counts["raw_proposal_items"] += len(proposals)
        else:
            counts["video_json_array_invalid"] += 1
            failures.append(str(path))
    return {
        "available": bool(calls),
        **dict(counts),
        "invalid_array_traces": failures,
        "request_failure_traces": request_failures,
    }


def audit(batch: Path, episodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    state = json.loads((batch / "run.json").read_text())
    if state["status"] != "completed":
        raise ValueError("只能审计已完成的整批任务")
    keys = [row["run_key"] for row in state["jobs"]]
    if len(keys) != len(episodes) or set(keys) != set(episodes):
        raise ValueError("实际视频范围与批准清单不符")
    db = sqlite3.connect(f"file:{batch / 'dashboard.sqlite3'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    totals: Counter[str] = Counter()
    rows = []
    for job in state["jobs"]:
        source = episodes[job["run_key"]]
        if sha(Path(source["source_video"])) != source["source_video_sha256"]:
            raise ValueError("原视频与批准SHA不符")
        db_job = db.execute("SELECT * FROM jobs WHERE id=?", (job["job_id"],)).fetchone()
        if db_job is None or db_job["status"] != job["status"]:
            raise ValueError("SQLite终态与清单不符")
        row = dict(
            run_key=job["run_key"],
            status=job["status"],
            job_id=job["job_id"],
            duration_s=source["duration_seconds"],
            dataset_root=source["dataset_root"],
        )
        root = batch / "workspace/jobs" / job["job_id"]
        trace_root = root if job["status"] == "succeeded" else batch / "preserved" / job["job_id"]
        row["raw_calls"] = raw_call_metrics(trace_root)
        totals.update(
            dict(
                duration_s=source["duration_seconds"],
                **{
                    k: v
                    for k, v in row["raw_calls"].items()
                    if isinstance(v, int) and k != "available"
                },
            )
        )
        if job["status"] != "succeeded":
            totals["uncovered_s"] += source["duration_seconds"]
            row["failure_reason"] = job.get("failure_reason")
            rows.append(row)
            continue
        manifest = root / "result_manifest.json"
        if (
            str(manifest) != job["output_manifest_path"]
            or sha(manifest) != job["output_manifest_sha256"]
        ):
            raise ValueError("发布清单路径或SHA不符")
        result = json.loads(manifest.read_text())
        if result["source_video_sha256"] != source["source_video_sha256"]:
            raise ValueError("发布结果源视频不符")
        for artifact in result["artifacts"]:
            path = root / artifact["path"]
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("产物路径不安全")
            if path.stat().st_size != artifact["size_bytes"] or sha(path) != artifact["sha256"]:
                raise ValueError("发布产物校验不符")
        subtasks = json.loads((root / "subtasks.json").read_text())["subtasks"]
        stored = db.execute(
            "SELECT s.* FROM segments s JOIN annotation_runs r ON r.id=s.annotation_run_id "
            "WHERE r.job_id=? AND s.deleted_at IS NULL ORDER BY s.sequence_index",
            (job["job_id"],),
        ).fetchall()
        if len(stored) != len(subtasks) or any(
            x["review_status"] != "unreviewed" or x["confidence"] is not None for x in stored
        ):
            raise ValueError("发布标注数量/待审核状态不符")
        for segment, saved in zip(subtasks, stored, strict=True):
            if any(
                segment[key] != saved[key]
                for key in ["label", "start_time_s", "end_time_s", "confidence"]
            ):
                raise ValueError("已发布标注内容与SQLite不符")
        stats = timing(subtasks, source["duration_seconds"], source["fps"])
        row.update(
            segments=subtasks,
            timing=stats,
            model_evidence=json.loads((root / "subtasks.json").read_text())["ingestion"],
        )
        totals.update(
            dict(
                segments=len(subtasks),
                verified_videos=1,
                **{k: v for k, v in stats.items() if k != "coverage"},
            )
        )
        rows.append(row)
    db.close()
    duration = totals["duration_s"]
    return dict(
        status_counts=dict(Counter(j["status"] for j in state["jobs"])),
        totals={
            **dict(totals),
            "weighted_coverage": totals["covered_s"] / duration if duration else None,
        },
        episodes=rows,
        scope_exact=True,
        raw_source_and_artifact_checks_passed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch", action="append", required=True, help="模型名称=绝对批次目录")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("报告目录必须全新")
    episodes = {x["run_key"]: x for x in json.loads(args.plan.read_text())["episodes"]}
    models = {}
    for item in args.batch:
        name, path = item.split("=", 1)
        models[name] = audit(Path(path), episodes)
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "plan_sha256": sha(args.plan),
        "models": models,
        "说明": "相同70原视频和官方分块/提示/合并流程；模型原生处理器不同。覆盖不是召回率，标注数量不是正确率；均未复核，无仿真训练。",
    }
    (args.output / "多模型subtask对照.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    lines = [
        "# MOSS ACE 与 Qwen 的 subtask 实测对照",
        "",
        report["说明"],
        "",
        "| 模型 | 成功/失败 | 建议数 | 标注时间覆盖 | 超过一帧越界 | 重叠对数 | 无效视频JSON/请求失败 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, model in models.items():
        t, s = model["totals"], model["status_counts"]
        coverage = "—" if t["weighted_coverage"] is None else f"{t['weighted_coverage']:.2%}"
        raw_errors = (
            f"{t.get('video_json_array_invalid', 0)}/{t.get('video_request_failed', 0)}"
            if "video_calls" in t
            else "未逐调用记录"
        )
        lines.append(
            f"| {name} | {s.get('succeeded', 0)}/{s.get('failed', 0)} | {t.get('segments', 0)} | {coverage} | {t.get('beyond_one_frame', 0)} | {t.get('overlap_pairs', 0)} | {raw_errors} |"
        )
    lines += [
        "",
        "覆盖率以批准的全部70段时长为分母，失败视频按零覆盖计入。三套结果分别保存在独立数据库，不覆盖已有标注。原始提议与模型合并后的发布结果应分开看；更多片段或更高覆盖不等于更好的动作边界。",
        "",
        "新模型使用2 FPS、15秒块、1.5秒重叠；MOSS每帧视觉预算上限262144像素，ACE使用原生处理器整块8388608像素预算、最多128帧；处理器采用不同帧数、缩放和时间标记。本次MOSS/ACE固定随机种子42，实际分段/文本调用温度0、输出上限4096 token。原Qwen基线使用默认像素预算，未按本次常驻服务方式重跑；该表是部署流程对照，不是纯权重控制变量评测。",
        "",
        "所有建议仍是待审核、置信度为空；没有人工动作时间真值，不能据此报准确率、F1或宣称某模型全面胜出。具体片段、输入来源和模型证据见同目录JSON。",
    ]
    (args.output / "中文多模型subtask对照.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
