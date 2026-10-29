import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from experiments.nvidia_v2d.vlm_compare_viewer import create_viewer
from experiments.nvidia_v2d.vlm_view_data import ComparisonStore, ViewerConfig


@pytest.fixture
def viewer(tmp_path: Path):
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "run_key": "episode-0",
                        "duration_seconds": 10,
                        "fps": 20,
                        "dataset_root": "track_3/public",
                        "task": "test",
                    }
                ]
            }
        )
    )
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "run.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "run_key": "episode-0",
                        "job_id": "job-1",
                        "status": "succeeded",
                        "annotations": [
                            {"label": "lift", "start_time_s": 1, "end_time_s": 5, "confidence": None},
                            {
                                "label": "place",
                                "start_time_s": 4,
                                "end_time_s": 8,
                                "confidence": None,
                            },
                        ],
                    }
                ]
            }
        )
    )
    media = batch / "recordings/episode-0/color.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"0123456789abcdef")
    config = ViewerConfig.model_validate(
        {
            "plan": plan,
            "media_batch": batch,
            "models": [
                {"id": "moss", "name": "MOSS", "batch": batch},
                {"id": "ace", "name": "ACE", "batch": tmp_path / "not-started"},
            ],
        }
    )
    return config, media


def test_live_results_keep_pending_distinct_and_calculate_union_gaps(viewer):
    config, media = viewer
    before = (media.read_bytes(), media.stat().st_mtime_ns)
    store = ComparisonStore(config)
    snapshot = store.snapshot()
    moss, ace = snapshot.episodes[0].models
    assert moss.coverage == 0.7 and moss.overlap_pairs == 1
    assert moss.gaps == [(0, 1), (8, 10)]
    assert ace.status == "pending" and ace.segments == []
    assert snapshot.models[1].pending == 1
    (config.models[1].batch).mkdir()
    (config.models[1].batch / "run.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "run_key": "episode-0",
                        "job_id": "failed-1",
                        "status": "failed",
                        "failure_reason": "no clips",
                    }
                ]
            }
        )
    )
    assert store.snapshot().episodes[0].models[1].status == "failed"
    assert (media.read_bytes(), media.stat().st_mtime_ns) == before


def test_media_supports_range_without_transcoding_or_writes(viewer):
    config, media = viewer
    before = (media.read_bytes(), media.stat().st_mtime_ns)
    with TestClient(create_viewer(config)) as client:
        assert client.get("/api/compare").status_code == 200
        response = client.get("/api/compare/media/episode-0", headers={"Range": "bytes=3-6"})
        assert response.status_code == 206 and response.content == b"3456"
        assert client.post("/api/compare").status_code == 405
        assert client.get("/api/compare/media/unknown").status_code == 404
        assert client.get("/api/compare/calls/episode-0/unknown").status_code == 404
    assert (media.read_bytes(), media.stat().st_mtime_ns) == before


def test_media_rejects_symlink_to_outside_file(viewer, tmp_path):
    config, media = viewer
    external = tmp_path / "external.mp4"
    external.write_bytes(b"do not expose")
    media.unlink()
    media.symlink_to(external)
    with TestClient(create_viewer(config)) as client:
        assert client.get("/api/compare/media/episode-0").status_code == 404
    assert external.read_bytes() == b"do not expose"


def test_raw_calls_are_job_bound_and_only_read_known_model(viewer):
    config, _ = viewer
    folder = config.models[0].batch / "workspace/jobs/job-1/model-calls"
    folder.mkdir(parents=True)
    (folder / "call.json").write_text(
        json.dumps(
            {
                "request": {"kind": "video"},
                "response": {"text": "raw original"},
                "started_unix": 1,
                "finished_unix": 2,
            }
        )
    )
    with TestClient(create_viewer(config)) as client:
        calls = client.get("/api/compare/calls/episode-0/moss").json()
        assert calls[0]["response"] == "raw original" and calls[0]["elapsed_s"] == 1
        assert client.get("/api/compare/calls/unknown/moss").status_code == 404


def test_partial_raw_response_returns_retryable_error(viewer):
    config, _ = viewer
    folder = config.models[0].batch / "workspace/jobs/job-1/model-calls"
    folder.mkdir(parents=True)
    (folder / "being-written.json").write_text("{")
    with TestClient(create_viewer(config)) as client:
        assert client.get("/api/compare/calls/episode-0/moss").status_code == 503


def segment_log(start: float, end: float, label: str, index: int = 1) -> str:
    return (
        "2026-10-26 12:00:00,000 - video_ingestion_agent.ingestion.segmentation.segmenter"
        f" - INFO -     video_clip_{index:04d}: [{start:.1f}s-{end:.1f}s] {label}\n"
    )


def summary_log(count: int) -> str:
    return (
        "2026-10-26 12:00:00,000 - video_ingestion_agent.ingestion.segmentation.segmenter"
        f" - INFO - Total segments before dedup: {count}\n"
    )


def test_candidate_layer_keeps_six_log_proposals_separate_from_one_published_segment(viewer):
    config, _ = viewer
    batch = config.models[0].batch
    state = json.loads((batch / "run.json").read_text())
    state["jobs"][0]["annotations"] = [
        {"label": "combined action", "start_time_s": 0, "end_time_s": 10, "confidence": None}
    ]
    (batch / "run.json").write_text(json.dumps(state))
    path = batch / "workspace/jobs/job-1/runtime.log"
    path.parent.mkdir(parents=True)
    text = "".join(segment_log(i, i + 1, f"action-{i}", i + 1) for i in range(6))
    text += (
        "2026-10-26 12:00:00,000 - video_ingestion_agent.ingestion.segmentation.dedup"
        " - INFO - LLM merged video_clip_0001 + video_clip_0002 -> [0.0s-10.0s] combined action\n"
        "2026-10-26 12:00:00,000 - video_ingestion_agent.ingestion.segmentation.segmenter"
        " - INFO - Total segments before dedup: 6\n"
    )
    text += segment_log(0, 10, "not a new proposal", 99)
    path.write_text(text)
    result = ComparisonStore(config).snapshot().episodes[0].models[0]
    assert len(result.segments) == 1 and result.coverage == 1
    assert result.candidates is not None
    assert [s.label for s in result.candidates.segments] == [f"action-{i}" for i in range(6)]
    assert result.candidates.coverage == 0.6
    assert result.candidates.gaps == [(6, 10)]
    assert result.candidates.source == "runtime_log"
    assert result.candidates.log_rounding_s == 0.1
    assert all(
        s.review_status == "unreviewed" and s.confidence is None for s in result.candidates.segments
    )


def test_candidate_cache_avoids_rereads_and_refreshes_for_appended_active_log(viewer, monkeypatch):
    config, _ = viewer
    path = config.models[0].batch / "workspace/jobs/job-1/runtime.log"
    path.parent.mkdir(parents=True)
    path.write_text(segment_log(1, 4, "first"))
    reads = 0
    original = Path.open

    def counted_open(self, *args, **kwargs):
        nonlocal reads
        if self == path:
            reads += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    store = ComparisonStore(config)
    assert store.snapshot().episodes[0].models[0].candidates is None
    assert store.snapshot().episodes[0].models[0].candidates is None and reads == 1
    with original(path, "a") as stream:
        stream.write(summary_log(1))
    updated = store.snapshot().episodes[0].models[0].candidates
    assert updated is not None and len(updated.segments) == 1 and reads == 2
    assert store.snapshot().episodes[0].models[0].candidates is not None and reads == 2


def test_candidate_missing_and_empty_logs_differ_and_failed_job_uses_preserved_root(viewer):
    config, _ = viewer
    store = ComparisonStore(config)
    assert store.snapshot().episodes[0].models[0].candidates is None
    batch = config.models[0].batch
    path = batch / "workspace/jobs/job-1/runtime.log"
    path.parent.mkdir(parents=True)
    path.write_text("no accepted segmentation proposals yet\n" + summary_log(0))
    result = store.snapshot().episodes[0].models[0].candidates
    assert result is not None and result.segments == [] and result.coverage == 0
    state = json.loads((batch / "run.json").read_text())
    state["jobs"][0]["status"] = "failed"
    state["jobs"][0]["annotations"] = []
    (batch / "run.json").write_text(json.dumps(state))
    preserved = batch / "preserved/job-1/runtime.log"
    preserved.parent.mkdir(parents=True)
    preserved.write_text(segment_log(2, 3, "preserved proposal") + summary_log(1))
    result = store.snapshot().episodes[0].models[0].candidates
    assert result is not None and result.segments[0].label == "preserved proposal"


@pytest.mark.parametrize("violation", ["job_id", "symlink", "oversize"])
def test_candidate_log_rejects_unsafe_paths_and_unbounded_reads(viewer, tmp_path, violation):
    config, _ = viewer
    batch = config.models[0].batch
    state = json.loads((batch / "run.json").read_text())
    if violation == "job_id":
        state["jobs"][0]["job_id"] = "../../outside"
        (batch / "run.json").write_text(json.dumps(state))
    else:
        path = batch / "workspace/jobs/job-1/runtime.log"
        path.parent.mkdir(parents=True)
        if violation == "symlink":
            outside = tmp_path / "outside.log"
            outside.write_text(segment_log(0, 5, "private"))
            path.symlink_to(outside)
        else:
            path.write_bytes(b"x" * (2 * 1024**2 + 1))
    with TestClient(create_viewer(config)) as client:
        assert client.get("/api/compare").status_code == 503


def test_candidate_layer_ignores_an_unfinished_last_log_line(viewer):
    config, _ = viewer
    path = config.models[0].batch / "workspace/jobs/job-1/runtime.log"
    path.parent.mkdir(parents=True)
    path.write_text(segment_log(0, 2, "finished") + segment_log(2, 3, "being-written").rstrip("\n"))
    result = ComparisonStore(config).snapshot().episodes[0].models[0].candidates
    assert result is None


@pytest.mark.parametrize(
    "text",
    [
        segment_log(0, 1, "only one row") + summary_log(2),
        segment_log(0, 1, "first line\ncontinuation") + summary_log(1),
    ],
)
def test_candidate_layer_does_not_present_incomplete_or_multiline_log_as_complete(viewer, text):
    config, _ = viewer
    path = config.models[0].batch / "workspace/jobs/job-1/runtime.log"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    assert ComparisonStore(config).snapshot().episodes[0].models[0].candidates is None
