import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.nvidia_v2d.policy_budget import TrainingBudget
from experiments.nvidia_v2d.policy_experiment import ExperimentSpec, training_argv
from experiments.nvidia_v2d.policy_gate import verify_gate
from experiments.nvidia_v2d.policy_publish import (
    EPISODE,
    LEGACY_EXPERIMENT_SHA256,
    RESET_OVERRIDE,
    publish,
)
from experiments.nvidia_v2d.policy_run import RunRecord, RunSpec, stamp
from experiments.nvidia_v2d.policy_view_data import Scene


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_gate(root, backend):
    source = root / "source"
    target = root / f"retarget/{backend}"
    alias = {"official": "official-ik", "multitip": "unidex-inspired-ik-v2"}[backend]
    loaded = source / "loaded.parquet"
    calibration = source / "calibration.json"
    parquet = target / "data.parquet"
    loaded_sha = write(loaded, {})
    calibration_sha = write(calibration, {})
    parquet_sha = write(parquet, {})
    episode = source / "input-gate.json"
    episode_sha = write(
        episode,
        {
            "source": {"sequence_id": EPISODE, "frame_count": 249},
            "input_contract_passed": True,
            "blockers": [],
        },
    )
    retarget = target / "manifest.json"
    retarget_sha = write(
        retarget,
        {
            "backend": alias,
            "status": "completed",
            "source_unchanged": True,
            "frame_count": 249,
            "input_parquet": str(loaded),
            "input_sha256": loaded_sha,
            "input_manifest_path": str(calibration),
            "calibration_sha256": calibration_sha,
            "result_sha256": parquet_sha,
        },
    )
    audit = target / "audit.json"
    audit_sha = write(
        audit,
        {
            "backend": alias,
            "status": "succeeded",
            "source_unchanged": True,
            "frame_count": 249,
            "retarget_dir": str(target),
            "input_sha256": {str(retarget): retarget_sha},
        },
    )
    gate = target / "gate.json"
    write(
        gate,
        {
            "schema_version": "nvidia_v2d.sharpa_training_gate.v1",
            "training_ready": True,
            "input_contract_passed": True,
            "purpose": "mechanism_validation",
            "mechanism_validation": True,
            "frame_count": 249,
            "retarget_dir": str(target),
            "episode_gate_path": str(episode),
            "episode_gate_sha256": episode_sha,
            "geometry_audit_path": str(audit),
            "geometry_audit_sha256": audit_sha,
            "retarget_manifest_path": str(retarget),
            "retarget_manifest_sha256": retarget_sha,
        },
    )
    return gate, parquet


def fixture(tmp_path):
    runs = tmp_path / "policy-runs"
    viewer = tmp_path / "viewer"
    sweep = tmp_path / "policy-sweep-001"
    repository = tmp_path / "code"
    runs.mkdir()
    viewer.mkdir()
    code = {}
    for name in ["policy_experiment.py", "policy_experiment_evaluation.py"]:
        path = repository / "experiments/nvidia_v2d" / name
        code[str(path)] = write(path, {})
    budget = TrainingBudget(total_steps=2097152, world_count=64, wall_limit_s=5400)
    gates = {backend: make_gate(tmp_path, backend) for backend in ["official", "multitip"]}
    plan_runs = []
    specs = {}
    for seed in [42, 43, 44]:
        for backend in ["official", "multitip"]:
            gate, parquet = gates[backend]
            spec = ExperimentSpec(
                backend=backend,
                parquet=parquet,
                gate=gate,
                budget=budget,
                seed=seed,
                replay_capacity=65536,
                replay_minimum=1024,
                replay_batch=512,
                evaluation_worlds=2,
                image="sha256:" + "a" * 64,
            )
            spec_path = sweep / "specs" / f"{backend}-{seed}.json"
            spec_sha = write(spec_path, spec.model_dump(mode="json"))
            output = runs / f"ab-{backend}-seed{seed}-001"
            plan_runs.append(
                {
                    "backend": backend,
                    "seed": seed,
                    "spec_path": str(spec_path),
                    "spec_sha256": spec_sha,
                    "output": str(output),
                    "frozen_training_argv": training_argv(spec),
                }
            )
            specs[(backend, seed)] = spec
    plan = {
        "schema_version": "nvidia_v2d.policy_sweep.v1",
        "created_at": "2026-10-27T00:00:00Z",
        "purpose": "mechanism_validation",
        "budget": budget.model_dump(mode="json"),
        "image": "sha256:" + "a" * 64,
        "repository": str(repository),
        "input_root": str(tmp_path),
        "cache": str(tmp_path / "cache"),
        "source_sha256": code,
        "runs": plan_runs,
    }
    plan_sha = write(sweep / "frozen-plan.json", plan)
    state = {
        "schema": "nvidia_v2d.policy_sweep_state.v1",
        "plan_sha256": plan_sha,
        "status": "running",
        "runs": [],
    }
    write(sweep / "sweep-state.json", state)
    scene = Scene(
        kind="kinematic_reference",
        timestamps_s=[i / 20 for i in range(249)],
        hands=[],
        objects=[
            {
                "name": "white_pot",
                "vertices": [[0, 0, 0]],
                "faces": [],
                "poses": [np.eye(4).reshape(-1).tolist()] * 249,
            }
        ],
    )
    variants = []
    for backend in ["official", "multitip"]:
        path = viewer / "scenes-002" / f"{backend}.json"
        sha = write(path, scene.model_dump())
        variants.append(
            {
                "id": backend,
                "backend": backend,
                "status": "succeeded",
                "kind": "kinematic_reference",
                "scene_path": f"scenes-002/{backend}.json",
                "sha256": sha,
                "metrics": {"left_tip_p95_mm": 12.0},
            }
        )
    write(
        viewer / "policy-index.json",
        {
            "episodes": {
                EPISODE: {
                    "variants": variants,
                    "notes": ["保留几何限制说明", "当前执行视图为1024步短测"],
                },
                "unrelated": {"variants": [], "notes": ["保留其它片段"]},
            }
        },
    )
    return runs, viewer, sweep, plan, state, specs


def complete_run(f, backend="official", seed=42):
    runs, viewer, sweep, plan, state, specs = f
    spec = specs[(backend, seed)]
    run = next(r for r in plan["runs"] if r["backend"] == backend and r["seed"] == seed)
    output = Path(run["output"])
    output.mkdir()
    checkpoint = output / "train/policy_2097152.safetensors"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic-test-policy")
    checkpoint_sha = stamp(checkpoint).sha256
    code = {Path(path).name: sha for path, sha in plan["source_sha256"].items()}
    evidence = {
        "assistance": "none",
        "voc_scale_max": 0.0,
        "applied_control_scale_max": 0.0,
        "action_step_count": 249,
        "start_frame": 0,
        "motion_end_frame": -1,
        "checkpoint_sha256": checkpoint_sha,
        "eval_script_sha256": code["policy_experiment_evaluation.py"],
        "eval_code_sha256": "b" * 64,
        "observed_control_scales": [
            {"voc_scale_max_abs": 0.0, "applied_object_control_scale_max_abs": 0.0}
        ]
        * 249,
    }
    rows = []
    for step in range(249):
        for env in range(2):
            row = {
                "environment_id": env,
                "step": step,
                "reference_frame": step,
                "time_s": step / 20,
                "object_body_name": "white_pot",
                "object_id": 0,
                "object_body_id": 0,
                "environment_valid": True,
            }
            for source in ["reference", "achieved"]:
                row.update({f"{source}_position_{a}": 0.0 for a in "xyz"})
                row.update({f"{source}_quaternion_{a}": 0.0 for a in "xyz"})
                row[f"{source}_quaternion_w"] = 1.0
            row["achieved_position_x"] = float(2 - env)
            rows.append(row)
    metadata = {
        "schema": "flash_chord_object_trajectories_v1",
        "position_unit": "meter",
        "quaternion_order": "xyzw",
        "start_frame": "0",
        "step_count": "249",
        "control_fps": "20",
        "world_count": "2",
        "object_body_count": "1",
        "pose_width": "7",
        "row_order": "step,environment_id,object_body_id",
        "checkpoint_sha256": checkpoint_sha,
        "eval_code_sha256": "b" * 64,
        "eval_script_sha256": code["policy_experiment_evaluation.py"],
    }
    trajectory = output / "unassisted.object_trajectories.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows).replace_schema_metadata(
            {f"flash_chord.{k}": v for k, v in metadata.items()}
        ),
        trajectory,
    )
    trajectory_sha = stamp(trajectory).sha256
    evidence["trajectory_sha256"] = trajectory_sha
    metrics = {
        "environment_steps": 2097152,
        "parquet": str(spec.parquet),
        "checkpoint": "/run/train/policy_2097152.safetensors",
        "checkpoint_sha256": checkpoint_sha,
        "step_count": 249,
        "world_count": 2,
        "sim_control_fps": 20.0,
        "start_frame": 0,
        "motion_start_frame": 0,
        "motion_end_frame": -1,
        "metrics": {
            "mppe_cm": 9.0,
            "object_position_error_m": 0.09,
            "chord_sr": 0.0,
            "world_count": 2,
        },
        "reference_tracking": {"success_rate": 0.0},
    }
    experiment = {
        "schema": "nvidia_v2d.policy_experiment.v1",
        "status": "succeeded",
        "adapter_source_unchanged": True,
        "spec": spec.model_dump(mode="json"),
        "adapter_code_sha256": code,
        "reference": {"frames": 249, "fps": 20.0, "object_names": ["white_pot"]},
        "stages": [
            {
                "name": "train",
                "status": "succeeded",
                "exit_code": 0,
                "argv": ["python", "train.py", *run["frozen_training_argv"]],
            },
            {"name": "evaluate", "status": "succeeded", "exit_code": 0},
        ],
        "train_checkpoint": {
            "environment_steps": "2097152",
            "global_update_step": "100",
            "voc_scale": "0.0",
        },
        "metrics": metrics,
        "trajectory_validation": {"sha256": trajectory_sha},
        "evaluation_evidence": evidence,
    }
    write(output / "unassisted.metrics.json", metrics)
    write(output / "experiment.json", experiment)
    gate = verify_gate(spec.gate, backend=backend, parquet=spec.parquet, episode_id=EPISODE)
    inputs = list(
        dict.fromkeys(
            [
                Path(run["spec_path"]),
                spec.parquet,
                *map(Path, plan["source_sha256"]),
                *map(Path, gate["verified_files"]),
            ]
        )
    )
    write(output / "input-manifest.json", [stamp(p).model_dump() for p in inputs])
    names = ["experiment.json", "unassisted.metrics.json", "unassisted.object_trajectories.parquet"]
    outputs = [stamp(output / name) for name in names]
    write(output / "output-manifest.json", [row.model_dump() for row in outputs])
    run_spec = RunSpec(
        phase="training",
        episode_id=EPISODE,
        backend=backend,
        argv=["test-fixture"],
        cwd=Path(plan["repository"]),
        inputs=inputs,
        gate=spec.gate,
        retarget_parquet=spec.parquet,
        required_outputs=names,
        wall_limit_s=5430,
    )
    record = RunRecord(
        status="succeeded",
        spec=run_spec,
        input_manifest_path=str(output / "input-manifest.json"),
        output_manifest_path=str(output / "output-manifest.json"),
        log_path=str(output / "run.log"),
        started_at="2026-10-27T00:00:00Z",
        finished_at="2026-10-27T00:01:00Z",
        exit_code=0,
        input_unchanged=True,
        outputs=outputs,
    )
    write(output / "run.json", record.model_dump(mode="json"))
    state["runs"].append(
        {"backend": backend, "seed": seed, "output": str(output), "status": "succeeded"}
    )
    write(sweep / "sweep-state.json", state)
    return output


def test_pending_publish_includes_all_six_and_preserves_reference_and_other_episode(tmp_path):
    f = fixture(tmp_path)
    runs, viewer, sweep, *_ = f
    result = publish(runs, viewer, sweep)
    assert result["completed_runs"] == 0
    index = json.loads((viewer / "policy-index.json").read_text())
    entry = index["episodes"][EPISODE]
    assert len(entry["variants"]) == 8
    assert entry["variants"][0]["metrics"] == {"left_tip_p95_mm": 12.0}
    assert [r["seed"] for r in entry["variants"][2:]] == [42, 42, 43, 43, 44, 44]
    assert "210万" in entry["notes"][0] and any("尚未完成" in n for n in entry["notes"])
    assert "unrelated" in index["episodes"] and not any("短测" in n for n in entry["notes"])
    assert not list(viewer.glob("rollout-ab-*"))
    before = (viewer / "policy-index.json").stat().st_mtime_ns
    assert publish(runs, viewer, sweep)["status"] == "unchanged"
    assert (viewer / "policy-index.json").stat().st_mtime_ns == before


def test_success_publishes_actual_environment_zero_and_idempotently_reuses(tmp_path):
    f = fixture(tmp_path)
    complete_run(f)
    runs, viewer, sweep, *_ = f
    assert publish(runs, viewer, sweep)["completed_runs"] == 1
    entry = json.loads((viewer / "policy-index.json").read_text())["episodes"][EPISODE]
    variant = entry["variants"][2]
    assert variant["metrics"]["mppe_cm"] == 9.0 and variant["assistance"] == "none"
    scene = json.loads((viewer / variant["scene_path"]).read_text())
    assert scene["objects"][0]["poses"][0][3] == 2.0
    files = {str(p): stamp(p) for p in viewer.glob("rollout-ab-*/*")}
    assert publish(runs, viewer, sweep)["status"] == "unchanged"
    assert files == {str(p): stamp(p) for p in viewer.glob("rollout-ab-*/*")}


@pytest.mark.parametrize(
    "changed",
    [
        "input-manifest.json",
        "output-manifest.json",
        "experiment.json",
        "unassisted.metrics.json",
        "unassisted.object_trajectories.parquet",
        "train/policy_2097152.safetensors",
    ],
)
def test_stale_binding_is_blocked_without_export(tmp_path, changed):
    f = fixture(tmp_path)
    output = complete_run(f)
    (output / changed).write_bytes(b"{}")
    result = publish(f[0], f[1], f[2])
    assert result["variants"][0]["status"] == "blocked"
    assert not list(f[1].glob("rollout-ab-*"))


def test_existing_export_is_never_overwritten_when_corrupt(tmp_path):
    f = fixture(tmp_path)
    complete_run(f)
    publish(f[0], f[1], f[2])
    path = f[1] / "rollout-ab-official-42-001/scene.json"
    path.write_bytes(b"corrupt")
    result = publish(f[0], f[1], f[2])
    assert result["variants"][0]["status"] == "blocked"
    assert path.read_bytes() == b"corrupt"


def test_failed_and_running_runs_are_kept_without_cherry_picking(tmp_path):
    f = fixture(tmp_path)
    output = complete_run(f)
    run = json.loads((output / "run.json").read_text())
    run.update(status="timed_out", failure_reason="达到时限")
    write(output / "run.json", run)
    f[4]["runs"][0]["status"] = "failed"
    f[4]["runs"].append(
        {
            "backend": "multitip",
            "seed": 42,
            "output": str(f[0] / "ab-multitip-seed42-001"),
            "status": "running",
        }
    )
    write(f[2] / "sweep-state.json", f[4])
    result = publish(f[0], f[1], f[2])
    assert [r["status"] for r in result["variants"]] == [
        "timed_out",
        "running",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert result["completed_runs"] == 0


def test_state_plan_sha_mismatch_preserves_existing_index(tmp_path):
    f = fixture(tmp_path)
    before = (f[1] / "policy-index.json").read_bytes()
    f[4]["plan_sha256"] = "0" * 64
    write(f[2] / "sweep-state.json", f[4])
    with pytest.raises(ValueError, match="冻结计划"):
        publish(f[0], f[1], f[2])
    assert (f[1] / "policy-index.json").read_bytes() == before


def restamp_outputs(output):
    record = json.loads((output / "run.json").read_text())
    record["outputs"] = [
        stamp(output / name).model_dump()
        for name in [
            "experiment.json",
            "unassisted.metrics.json",
            "unassisted.object_trajectories.parquet",
        ]
    ]
    write(output / "output-manifest.json", record["outputs"])
    write(output / "run.json", record)


def test_all_six_seeds_are_published_without_filtering_by_success_metric(tmp_path):
    f = fixture(tmp_path)
    for seed in [42, 43, 44]:
        for backend in ["official", "multitip"]:
            complete_run(f, backend, seed)
    result = publish(f[0], f[1], f[2])
    assert result["completed_runs"] == 6
    entry = json.loads((f[1] / "policy-index.json").read_text())["episodes"][EPISODE]
    assert all(
        r["status"] == "succeeded" and r["metrics"]["chord_sr"] == 0 for r in entry["variants"][2:]
    )
    assert not any("尚未完成" in n for n in entry["notes"])
    assert len(list(f[1].glob("rollout-ab-*"))) == 6


def test_success_with_bound_but_incomplete_budget_cannot_publish(tmp_path):
    f = fixture(tmp_path)
    output = complete_run(f)
    experiment = json.loads((output / "experiment.json").read_text())
    experiment["train_checkpoint"]["environment_steps"] = "16384"
    write(output / "experiment.json", experiment)
    restamp_outputs(output)
    result = publish(f[0], f[1], f[2])
    assert result["variants"][0]["status"] == "blocked"
    assert not list(f[1].glob("rollout-ab-*"))


def test_wrong_episode_run_cannot_borrow_public44_scene(tmp_path):
    f = fixture(tmp_path)
    output = complete_run(f)
    record = json.loads((output / "run.json").read_text())
    record["spec"]["episode_id"] = "track_3-public-episode-000031"
    write(output / "run.json", record)
    result = publish(f[0], f[1], f[2])
    assert result["variants"][0]["status"] == "blocked"


def bind_plan(f):
    f[4]["plan_sha256"] = write(f[2] / "frozen-plan.json", f[3])
    write(f[2] / "sweep-state.json", f[4])


def legacy_fixture(tmp_path):
    f = fixture(tmp_path)
    # Exact immutable module from commit 72b4043; never imported/executed.
    content = (Path(__file__).parent / "fixtures/policy_experiment_v1_0.txt").read_bytes()
    assert hashlib.sha256(content).hexdigest() == LEGACY_EXPERIMENT_SHA256
    source = Path(f[3]["repository"]) / "experiments/nvidia_v2d/policy_experiment.py"
    source.write_bytes(content)
    f[3]["source_sha256"][str(source)] = stamp(source).sha256
    for run in f[3]["runs"]:
        run["frozen_training_argv"].remove(RESET_OVERRIDE)
    bind_plan(f)
    return f


def test_known_v1_0_source_publishes_full_chain_with_explicit_training_limitation(tmp_path):
    f = legacy_fixture(tmp_path)
    output = complete_run(f)
    originals = {
        str(path): stamp(path)
        for path in [*f[2].rglob("*.json"), *output.rglob("*.*")]
        if path.is_file()
    }
    result = publish(f[0], f[1], f[2])
    assert result["completed_runs"] == 1
    entry = json.loads((f[1] / "policy-index.json").read_text())["episodes"][EPISODE]
    assert entry["publication"]["training_argv_contract"] == "frozen-v1.0-default-reset"
    assert entry["publication"]["training_source_sha256"] == LEGACY_EXPERIMENT_SHA256
    assert any("旧版1.0" in note and "辅助回升" in note for note in entry["notes"])
    assert entry["variants"][2]["assistance"] == "none"
    assert originals == {name: stamp(Path(name)) for name in originals}
    assert publish(f[0], f[1], f[2])["status"] == "unchanged"


@pytest.mark.parametrize(
    "override",
    [
        "reset.voc_decay_steps=1000",
        RESET_OVERRIDE,
        "reset.seed=43",
        "training.total_environment_steps=2097216",
        "training.replay.batch_size=256",
        "scene.support=true",
        "reward.object_weight=0",
    ],
)
def test_known_legacy_source_does_not_allow_other_fully_rebound_argv_changes(tmp_path, override):
    f = legacy_fixture(tmp_path)
    argv = f[3]["runs"][0]["frozen_training_argv"]
    key = override.split("=", 1)[0] + "="
    argv[:] = [arg for arg in argv if not arg.startswith(key)]
    argv.append(override)
    bind_plan(f)
    complete_run(f)  # All input/output stamps and actual train argv agree with the tampered plan.
    before = (f[1] / "policy-index.json").read_bytes()
    with pytest.raises(ValueError, match="冻结配置/预算/种子"):
        publish(f[0], f[1], f[2])
    assert (f[1] / "policy-index.json").read_bytes() == before
    assert not list(f[1].glob("rollout-ab-*"))


def test_unregistered_v1_0_source_cannot_borrow_the_legacy_exception(tmp_path):
    f = legacy_fixture(tmp_path)
    source = Path(f[3]["repository"]) / "experiments/nvidia_v2d/policy_experiment.py"
    source.write_bytes(source.read_bytes() + b"\n# Different source, still claims version 1.0.\n")
    f[3]["source_sha256"][str(source)] = stamp(source).sha256
    bind_plan(f)
    complete_run(f)
    with pytest.raises(ValueError, match="冻结配置/预算/种子"):
        publish(f[0], f[1], f[2])
    assert not list(f[1].glob("rollout-ab-*"))


def test_claimed_legacy_hash_must_match_actual_frozen_source_bytes(tmp_path):
    f = legacy_fixture(tmp_path)
    source = Path(f[3]["repository"]) / "experiments/nvidia_v2d/policy_experiment.py"
    source.write_bytes(b"different code claiming the registered digest")
    before = (f[1] / "policy-index.json").read_bytes()
    with pytest.raises(ValueError, match="SHA不一致"):
        publish(f[0], f[1], f[2])
    assert (f[1] / "policy-index.json").read_bytes() == before


@pytest.mark.parametrize("override", ["reset.seed=99", "reset.voc_decay_steps=1000"])
def test_legacy_actual_train_argv_still_must_equal_the_frozen_plan(tmp_path, override):
    f = legacy_fixture(tmp_path)
    output = complete_run(f)
    experiment = json.loads((output / "experiment.json").read_text())
    experiment["stages"][0]["argv"].append(override)
    write(output / "experiment.json", experiment)
    restamp_outputs(output)
    result = publish(f[0], f[1], f[2])
    assert result["variants"][0]["status"] == "blocked"
    entry = json.loads((f[1] / "policy-index.json").read_text())["episodes"][EPISODE]
    assert "实际训练参数与冻结参数不同" in entry["variants"][2]["failure_reason"]
    assert not list(f[1].glob("rollout-ab-*"))


def test_current_source_cannot_omit_reset_disable_override(tmp_path):
    f = fixture(tmp_path)
    source = Path(f[3]["repository"]) / "experiments/nvidia_v2d/policy_experiment.py"
    current = Path(__file__).parents[2] / "experiments/nvidia_v2d/policy_experiment.py"
    source.write_bytes(current.read_bytes())
    f[3]["source_sha256"][str(source)] = stamp(source).sha256
    assert stamp(source).sha256 != LEGACY_EXPERIMENT_SHA256
    f[3]["runs"][0]["frozen_training_argv"].remove(RESET_OVERRIDE)
    bind_plan(f)
    complete_run(f)
    with pytest.raises(ValueError, match="冻结配置/预算/种子"):
        publish(f[0], f[1], f[2])
    assert not list(f[1].glob("rollout-ab-*"))
