import hashlib
import json

import pytest

from experiments.nvidia_v2d.policy_gate import verify_gate


def fixture_gate(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()

    def write(path, data):
        path.write_text(json.dumps(data))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    loaded, calibration, parquet = (
        source / "loaded.parquet",
        source / "calibration.json",
        target / "data.parquet",
    )
    input_sha, calibration_sha, output_sha = (
        write(path, {}) for path in (loaded, calibration, parquet)
    )
    episode = source / "input-gate.json"
    episode_sha = write(
        episode,
        {
            "source": {"sequence_id": "public-44", "frame_count": 2},
            "input_contract_passed": True,
            "blockers": [],
        },
    )
    retarget = target / "manifest.json"
    retarget_sha = write(
        retarget,
        {
            "backend": "official-ik",
            "status": "completed",
            "source_unchanged": True,
            "frame_count": 2,
            "input_parquet": str(loaded),
            "input_sha256": input_sha,
            "input_manifest_path": str(calibration),
            "calibration_sha256": calibration_sha,
            "result_sha256": output_sha,
        },
    )
    audit = tmp_path / "audit.json"
    audit_sha = write(
        audit,
        {
            "backend": "official-ik",
            "status": "succeeded",
            "source_unchanged": True,
            "frame_count": 2,
            "retarget_dir": str(target),
            "input_sha256": {str(retarget): retarget_sha},
        },
    )
    gate = tmp_path / "gate.json"
    write(
        gate,
        {
            "schema_version": "nvidia_v2d.sharpa_training_gate.v1",
            "training_ready": True,
            "input_contract_passed": True,
            "purpose": "mechanism_validation",
            "mechanism_validation": True,
            "frame_count": 2,
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


def test_gate_binds_every_input_and_backend(tmp_path):
    gate, parquet = fixture_gate(tmp_path)
    result = verify_gate(gate, backend="official", parquet=parquet, episode_id="public-44")
    assert len(result["verified_files"]) == 7
    with pytest.raises(ValueError, match="路线"):
        verify_gate(gate, backend="multitip", parquet=parquet)
    with pytest.raises(ValueError, match="视频编号"):
        verify_gate(gate, backend="official", parquet=parquet, episode_id="public-31")


@pytest.mark.parametrize(
    "changed",
    [
        "source/loaded.parquet",
        "source/calibration.json",
        "source/input-gate.json",
        "target/data.parquet",
        "audit.json",
    ],
)
def test_rejects_stale_evidence(tmp_path, changed):
    gate, parquet = fixture_gate(tmp_path)
    (tmp_path / changed).write_text("changed")
    with pytest.raises(ValueError, match="变化"):
        verify_gate(gate, backend="official", parquet=parquet)


def test_boolean_without_provenance_is_not_a_gate(tmp_path):
    gate, parquet = fixture_gate(tmp_path)
    gate.write_text('{"training_ready": true}')
    with pytest.raises(ValueError):
        verify_gate(gate, backend="official", parquet=parquet)


def test_null_hash_cannot_disable_binding(tmp_path):
    gate, parquet = fixture_gate(tmp_path)
    raw = json.loads(gate.read_text())
    raw["geometry_audit_sha256"] = None
    gate.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="SHA256"):
        verify_gate(gate, backend="official", parquet=parquet)


def repaired_gate(tmp_path):
    original_gate, original = fixture_gate(tmp_path)
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    def write(p, data):
        p.write_text(json.dumps(data))
        return digest(p)
    candidate = tmp_path / "repaired.parquet"
    candidate.write_bytes(b"distinct repaired reference")
    audit = tmp_path / "repair-audit.json"
    checks = dict.fromkeys(("finite_arrays", "frame_count", "contact_normals", "object_support",
                           "robot_contact_mapping", "release_clearance", "unchanged_auxiliary_and_physics"), True)
    audit_sha = write(audit, {"candidate_sha256": digest(candidate), "checks": checks})
    manifest = tmp_path / "repair-manifest.json"
    manifest_sha = write(manifest, {"status": "completed", "training_allowed": True,
        "input_parquet": str(original), "output_parquet": str(candidate), "frame_count": 2,
        "source_unchanged": True, "input_sha256": {str(original): digest(original)},
        "original_sha256": digest(original), "output_sha256": digest(candidate),
        "audit_path": str(audit), "audit_sha256": audit_sha})
    gate = tmp_path / "repair-gate.json"
    write(gate, {"schema_version": "nvidia_v2d.repaired_reference_gate.v1", "training_ready": True,
        "purpose": "mechanism_validation", "geometry_quality_accepted": False,
        "parent_gate_path": str(original_gate), "parent_gate_sha256": digest(original_gate),
        "original_parquet": str(original), "repair_manifest_path": str(manifest),
        "repair_manifest_sha256": manifest_sha})
    return gate, candidate, audit


def test_repaired_gate_keeps_parent_chain_and_binds_changed_reference(tmp_path):
    gate, candidate, _ = repaired_gate(tmp_path)
    result = verify_gate(gate, backend="official", parquet=candidate)
    assert str(candidate) in result["verified_files"]
    assert str(tmp_path / "target/data.parquet") in result["verified_files"]
    candidate.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="变化"):
        verify_gate(gate, backend="official", parquet=candidate)


def test_repaired_gate_rejects_unaccepted_contact_mapping_even_with_fresh_hashes(tmp_path):
    gate, candidate, audit = repaired_gate(tmp_path)
    record = json.loads(audit.read_text())
    record["checks"]["robot_contact_mapping"] = False
    audit.write_text(json.dumps(record))
    manifest = tmp_path / "repair-manifest.json"
    record = json.loads(manifest.read_text())
    record["audit_sha256"] = hashlib.sha256(audit.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(record))
    record = json.loads(gate.read_text())
    record["repair_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    gate.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="acceptance"):
        verify_gate(gate, backend="official", parquet=candidate)


@pytest.mark.parametrize("missing", [None, "actual_object_collision_union", "full_hand_collision_union",
                                    "temporal_continuity", "source_contact_evidence_preserved"])
def test_collision_gate_requires_real_union_and_complete_audit(tmp_path, missing):
    gate, candidate, audit = repaired_gate(tmp_path)
    union = tmp_path / "collision.npz"
    union.write_bytes(b"frozen actual convex shapes")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    record = json.loads(audit.read_text())
    for key in ("actual_object_collision_union", "full_hand_collision_union", "temporal_continuity",
                "source_contact_evidence_preserved"):
        record["checks"][key] = True
    if missing:
        del record["checks"][missing]
    record["collision_union_sha256"] = digest(union)
    audit.write_text(json.dumps(record))
    manifest = tmp_path / "repair-manifest.json"
    record = json.loads(manifest.read_text())
    record.update(schema="nvidia_v2d.collision_reference_repair.v1", collision_union_path=str(union),
                  collision_union_sha256=digest(union), audit_sha256=digest(audit))
    manifest.write_text(json.dumps(record))
    record = json.loads(gate.read_text())
    record.update(schema_version="nvidia_v2d.repaired_reference_gate.v2", repair_manifest_sha256=digest(manifest))
    gate.write_text(json.dumps(record))
    if missing:
        with pytest.raises(ValueError, match="acceptance"):
            verify_gate(gate, backend="official", parquet=candidate)
    else:
        assert str(union) in verify_gate(gate, backend="official", parquet=candidate)["verified_files"]
        union.write_bytes(b"changed actual convex shapes")
        with pytest.raises(ValueError, match="变化"):
            verify_gate(gate, backend="official", parquet=candidate)
