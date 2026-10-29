"""Physical zero-residual/reference tracking diagnostic, never learned-policy success.

Run inside the frozen FlashCHORD image. Inputs/checkpoint remain read-only.
The optional robot-ground collision filter is an explicitly labeled causal ablation,
not an accepted physical repair or task-success claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--action-mode", choices=("zero", "policy", "scripted-grasp"), default="zero")
    parser.add_argument("--ground-mode", choices=("original", "robot-filtered"), default="original")
    parser.add_argument("--support", type=Path, help="Explicit finite physical support; removes infinite plane")
    parser.add_argument("--motion-speed", type=float, default=None)
    parser.add_argument("--wrist-kp", type=float, default=None)
    parser.add_argument("--wrist-kd", type=float, default=None)
    parser.add_argument("--grasp-offset-max", type=float, default=0.04)
    parser.add_argument("--grasp-center-feedback-max", type=float, default=0.03)
    parser.add_argument("--grasp-phases", type=float, nargs=4, default=(2.5, 3.3, 8.5, 9.5),
                        metavar=("START", "FULL", "RELEASE", "END"), help="source-time seconds")
    args = parser.parse_args()
    if (args.wrist_kp is None) != (args.wrist_kd is None):
        parser.error("both --wrist-kp and --wrist-kd are required for the gain ablation")
    if args.wrist_kp is not None and args.support is None:
        parser.error("gain ablation requires an explicit finite support")
    if args.action_mode == "scripted-grasp" and (args.support is None or args.motion_speed is None):
        parser.error("scripted-grasp requires explicit finite --support and --motion-speed")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import jax.numpy as jnp
    import newton
    from flash_chord.evaluation import harness
    from flash_chord.runtime.contact import read_contacts_w
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.utils.quat import quat_rotate_xyzw

    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    checkpoint_config = json.loads(read_checkpoint_metadata(args.checkpoint)["resolved_config_json"])
    tracking_criteria = asdict(harness.reference_tracking_criteria(checkpoint_config))
    original_record = harness.record_object_pose_rollout
    original_ground = newton.ModelBuilder.add_ground_plane
    original_setup = harness.setup_scene
    setup_module = importlib.import_module("flash_chord.scene.setup")
    original_support_resolver = setup_module.support_usda_for_reference
    if args.support is not None:
        if not args.support.is_file() or args.ground_mode != "original":
            raise ValueError("finite support must exist and cannot be mixed with collision filtering")
        setup_module.support_usda_for_reference = lambda reference: args.support

        def finite_setup(**kwargs):
            kwargs["include_support"] = True
            kwargs["collision"] = replace(kwargs["collision"], ground=False, robot_support_collision=True)
            if args.wrist_kp is not None:
                from experiments.nvidia_v2d.physical_wrist_config import configure_wrist
                configure_wrist(kwargs["embodiment"].config, kp=args.wrist_kp, kd=args.wrist_kd)
            if args.motion_speed is not None:
                if not 0 < args.motion_speed <= 1:
                    raise ValueError("motion speed must be in (0,1]")
                kwargs["motion_speed"] = args.motion_speed
            return original_setup(**kwargs)

        harness.setup_scene = finite_setup
    filtered_pairs = []

    def add_ground(builder, *positional, **kwargs):
        first = len(builder.shape_body)
        result = original_ground(builder, *positional, **kwargs)
        if args.ground_mode == "robot-filtered":
            for ground in range(first, len(builder.shape_body)):
                for shape in range(first):
                    body = builder.shape_body[shape]
                    if body < 0:
                        continue
                    label = str(builder.body_label[body])
                    if "sharpa" in label or "_wrist_" in label:
                        builder.add_shape_collision_filter_pair(shape, ground)
                        filtered_pairs.append([shape, ground])
        return result

    bodies, diagnostics, slot_contacts, scales, ground_contacts = [], [], [], [], []
    metadata = {}
    reference_wrists = {}
    initial = {}
    rollout_data = {}
    feedback_data = {}

    def recording(env, jax_env, policy_action, *positional, **kwargs):
        model = env.scene.model
        count = model.body_count // env.world_count
        labels = list(model.body_label[:count])
        reference = kwargs["reference"]
        metadata.update(body_labels=labels, error_names=list(env.termination.error_names),
                        diagnostic_names=list(env.termination.packed_diagnostic_names),
                        body_count=count, world_count=env.world_count,
                        fps=float(env.config.sim.fps), action_dimension=env.action.action_dim)
        initial["body_pose"] = env.state_0.body_q.numpy().reshape(env.world_count, count, 7).copy()
        for side in ("left", "right"):
            reference_wrists[side] = np.asarray(reference.wrist_pos_w(side)).copy()
        # Visible object vertices use the same finalized shape transforms as evaluation.
        shape_bodies = model.shape_body.numpy()
        types = model.shape_type.numpy()
        transforms = model.shape_transform.numpy()
        scale = model.shape_scale.numpy()
        initial_geometry = []
        for binding in env.scene.objects:
            for body in binding.bodies:
                pose = initial["body_pose"][0, body.body_id]
                pieces = []
                for shape in range(binding.shapes.start, binding.shapes.stop):
                    source = model.shape_source[shape]
                    if shape_bodies[shape] != body.body_id or int(types[shape]) != int(newton.GeoType.MESH):
                        continue
                    if source is None or not hasattr(source, "vertices"):
                        continue
                    local = np.asarray(source.vertices) * scale[shape]
                    local = quat_rotate_xyzw(transforms[shape, 3:7], local) + transforms[shape, :3]
                    pieces.append(quat_rotate_xyzw(pose[3:7], local) + pose[:3])
                if pieces:
                    points = np.concatenate(pieces)
                    initial_geometry.append({"body_id": body.body_id, "name": binding.name,
                                             "world_min_xyz": points.min(axis=0).tolist(),
                                             "world_max_xyz": points.max(axis=0).tolist()})
        metadata["initial_object_geometry"] = initial_geometry
        feedback = None
        if args.action_mode == "scripted-grasp":
            from experiments.nvidia_v2d.scripted_grasp_feedback import (
                ScriptedGraspFeedback,
            )

            feedback = ScriptedGraspFeedback(
                env, reference, max_offset=args.grasp_offset_max,
                center_feedback_max=args.grasp_center_feedback_max, phases=tuple(args.grasp_phases),
            )
            metadata["scripted_feedback"] = feedback.metadata()

        class Recorder:
            def __getattr__(self, name):
                return getattr(jax_env, name)

            def step(self, action):
                voc = np.asarray(env.voc_scale.numpy())
                applied = np.asarray(env.object_control_scale.numpy())
                if not np.isfinite(voc).all() or not np.isfinite(applied).all() or np.any(voc) or np.any(applied):
                    raise RuntimeError("nonzero or nonfinite auxiliary object control")
                scales.append([float(np.max(np.abs(voc))), float(np.max(np.abs(applied)))])
                result = jax_env.step(action)
                bodies.append(env.state_0.body_q.numpy().reshape(env.world_count, count, 7).copy())
                diagnostics.append(env.termination.packed_diagnostics.numpy().copy())
                slot_contacts.append(env.contact.raw_contact_count.numpy().copy())
                contacts = read_contacts_w(model, env.state_0, env.contacts)
                ground_contacts.append(int(np.count_nonzero((contacts.body0_ids < 0) | (contacts.body1_ids < 0))))
                return result

        def action(actor, observation, key):
            if args.action_mode == "policy":
                return policy_action(actor, observation, key)
            if feedback is not None:
                return jnp.asarray(feedback.action(motion_speed=args.motion_speed)), key
            return jnp.zeros((env.world_count, env.action.action_dim), dtype=jnp.float32), key

        rollout = original_record(env, Recorder(), action, *positional, **kwargs)
        rollout_data.update(achieved_pose_w=rollout.achieved_pose_w, reference_pose_w=rollout.reference_pose_w,
                            tracking_error=rollout.tracking_error,
                            completion_step=rollout.completion.completion_step,
                            completion_terminated=rollout.completion.terminated,
                            completion_truncated=rollout.completion.truncated,
                            completion_reference_progress=rollout.completion.reference_progress)
        if feedback is not None:
            feedback_data.update(feedback.traces())
        return rollout

    newton.ModelBuilder.add_ground_plane = add_ground
    harness.record_object_pose_rollout = recording
    try:
        outcome = harness.evaluate_checkpoint(args.checkpoint, world_count=args.worlds, start_frame=0,
                    reset_mode="explicit", motion_start_frame=0, motion_end_frame=-1,
                    object_trajectories_output=args.output / "diagnostic.object_trajectories.parquet",
                    eval_script=Path(__file__))
    finally:
        harness.record_object_pose_rollout = original_record
        newton.ModelBuilder.add_ground_plane = original_ground
        harness.setup_scene = original_setup
        setup_module.support_usda_for_reference = original_support_resolver
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != checkpoint_sha:
        raise RuntimeError("checkpoint changed")
    harness.write_evaluation_report(args.output / "diagnostic.metrics.json", outcome)
    np.savez_compressed(args.output / "diagnostic.npz", actual_bodies=np.asarray(bodies),
                        initial_bodies=initial["body_pose"], diagnostics=np.asarray(diagnostics),
                        hand_object_contact_counts=np.asarray(slot_contacts),
                        global_ground_contact_counts=np.asarray(ground_contacts), auxiliary_scales=np.asarray(scales),
                        reference_left_wrist=reference_wrists["left"], reference_right_wrist=reference_wrists["right"],
                        **rollout_data, **feedback_data)
    report = {"kind": {"zero": "physical_reference_diagnostic", "policy": "policy_causal_ablation",
                       "scripted-grasp": "scripted_feedback_feasibility"}[args.action_mode],
              "claim": "diagnostic only; reference/scripted control is not learned-policy task success",
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": checkpoint_sha,
              "action_mode": args.action_mode, "ground_mode": args.ground_mode,
              "reference_tracking_criteria": tracking_criteria,
              "support": str(args.support) if args.support else None,
              "support_sha256": hashlib.sha256(args.support.read_bytes()).hexdigest() if args.support else None,
              "motion_speed_override": args.motion_speed,
              "wrist_gain_ablation": None if args.wrist_kp is None else {
                  "kp": args.wrist_kp, "kd": args.wrist_kd, "effort_limit_N": 45.0,
                  "applies_to": "robot translational wrists only; orientation and fingers unchanged"},
              "filtered_robot_ground_pairs": len(filtered_pairs), "steps": len(bodies),
              "auxiliary_scale_max": np.max(np.asarray(scales), axis=0).tolist(),
              "contact_frames": int(np.count_nonzero(np.asarray(slot_contacts).sum(axis=1))),
              "contact_frames_by_world": np.count_nonzero(
                  np.asarray(slot_contacts).reshape(len(slot_contacts), args.worlds, -1).sum(axis=2), axis=0
              ).tolist(),
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir()}, **metadata}
    (args.output / "diagnostic.manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"steps": len(bodies), "metrics": harness.evaluation_scalars(outcome),
                      "contact_frames": report["contact_frames"], "ground_mode": args.ground_mode}))


if __name__ == "__main__":
    main()
