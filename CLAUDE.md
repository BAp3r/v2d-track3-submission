# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

A fork of NVIDIA's Video to Data (V2D) monorepo, being used for the **V2D Challenge Track 3**:
from an egocentric video, deliver a reconstructed object mesh plus the object trajectory produced
by a **learned policy actually executing in physics simulation** (Sharpa hands, FlashCHORD).

**Read [`AGENTS.md`](AGENTS.md) before starting any work.** It holds binding project rules (in
Chinese): the priority order (submission validity → official metrics → unassisted reliability →
hand naturalness), the rule that reference playback / kinematic replay / assisted execution must
never be passed off as a learned-policy rollout, the requirement to pre-state hypothesis, baseline,
budget and stop condition before any training, and the method-level review after two failed fix
rounds. [`progress.md`](progress.md) is the running experiment log; append a dated entry every
round (config, baseline, results, failures, time, decision) and never delete earlier entries.

## Layout

| Path | Role | Own guide |
|---|---|---|
| `reconstruction/` | Containerized vision modules (`v2d_*`): video → depth, SLAM, masks, mesh, object poses, MANO hands. Host wrappers `docker run` each module. | [`reconstruction/CLAUDE.md`](reconstruction/CLAUDE.md) |
| `robotic_grounding/` | Retargeting (Pink/Pinocchio IK to Sharpa) and Isaac Lab RL; `flash_chord/` is the Newton/Warp/JAX trainer (FlashSAC/PPO) used for Track 3. | [`robotic_grounding/CLAUDE.md`](robotic_grounding/CLAUDE.md), [`flash_chord/README.md`](robotic_grounding/flash_chord/README.md) |
| `video_ingestion_agent/` | LangGraph clip segmentation/retrieval. Not used by Track 3 work. | [`video_ingestion_agent/CLAUDE.md`](video_ingestion_agent/CLAUDE.md) |
| `experiments/nvidia_v2d/` | **Our Track 3 code**: bridge, retargeting, training/eval jobs, rollout export, submission packing, plus dated Chinese reports (`*_2026MMDD.md`). | [`experiments/nvidia_v2d/README.md`](experiments/nvidia_v2d/README.md) |
| `tests/experiments/` | CPU tests for `experiments/nvidia_v2d`. | |
| `tools/e2e/`, `run_e2e.sh` | Upstream iPhone→GR00T reference workflow (`python -m tools.e2e.cli`). | `docs/e2e_workflow.md` |
| `.e2e/` | Git-ignored. All run outputs, local venv, inputs, checkpoints and submission evidence (`.e2e/zero_success_20261027/`). | |

Companion skills live in `.claude/skills/` (reconstruction setup/run, robotic_grounding onboard/run/doctor, etc.).

## Commands

Experiment tests run in a standalone CPU venv (requirements in
`experiments/nvidia_v2d/runtime/requirements-standalone.txt`). Disable plugin autoload so host ROS
plugins aren't injected:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .e2e/zero_success_20261027/venv/bin/python -m pytest tests/experiments -q
```

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .e2e/zero_success_20261027/venv/bin/python -m pytest tests/experiments/test_nvidia_v2d_sharpa_episode.py::test_interpolation_is_counted_not_blocking -q
```

Tests that need GPU/FlashCHORD, pyarrow or Pinocchio assets are skipped via `importorskip`; CPU
tests passing says nothing about manipulation success. Training runs in the pinned FlashCHORD image
(`fmc3-flashchord:20261027-001`, built from `experiments/nvidia_v2d/runtime/Dockerfile.legacy-builder`);
retargeting runs in the separate env at `/var/tmp/fmc3-sharpa-validation-20261027-001/retarget-runtime-standalone-001`.
See [`runtime/README.md`](experiments/nvidia_v2d/runtime/README.md). Subproject build/lint commands
are in their own CLAUDE.md files.

## Track 3 pipeline (big picture)

Each stage writes files to a fresh output directory; the next stage reads them by path and checks hashes.

1. **Reconstruction** (`reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py`, `--hand_tracking {hamer,wilor,ace,hawor,dynhamr}`, `--run_droid_slam`).
   One job per (episode, object). Output: `reconstruction/result_slam/result.npz` + `mesh.obj`, with
   camera poses from DROID-SLAM and hands from `hamer_aligned_filled/`. The ACE backend (`v2d_ace_hand`)
   must add the MANO hand mean when exporting; see its README.
2. **Submission inputs** (`experiments.nvidia_v2d.submission_inputs prepare|bridge`).
   `prepare` collects the 20 evaluation episodes and 36 episode-objects from a reconstruction run.
   `bridge` turns them into a `ManoSharpaData` `loaded.parquet` + `calibration.json`. Two-object
   episodes go through `sharpa_multi_object.assemble`: the anchor object's hand trajectory is used
   for the scene, and one scene transform is applied to every body. **Never PCA-align objects
   separately and then stitch them together**; that was measured to introduce about 33 cm of
   relative error.
3. **Retargeting** (`experiments.nvidia_v2d.sharpa_multitip --backend official-ik|unidex-inspired-ik-v2`). Keep the three methods distinct: official IK, DexPilot-chain (`sharpa_dexpilot*`) and multi-tip.
4. **Training / evaluation** (`policy_experiment.py`, `policy_single_sample.py`, `policy_run.py`).
   These jobs run in Docker. They bind the input gate with `policy_gate.verify_gate`, log through
   `policy_metric_logging` (SwanLab + TensorBoard, see `OBSERVABILITY.md`), audit that Adam
   actually applied updates, and evaluate every saved checkpoint unassisted.
5. **Rollout + packing** (`submission_batch.py` → `submission_rollout.py`, per-episode `diagnostic.npz`).
   Rollouts are fixed to world0 and require `auxiliary_scales == 0`. The official kit's reader
   round-trips every exported trajectory. The kit source is `docs/v2d_challenge/assets/v2d_submission_kit.zip`.

## Gates: validity vs quality

Input/training checks are split on purpose:
- **Blocking (validity):** identity/video hash, non-finite or non-rigid poses, missing frames, shared camera gauge across objects, preserved relative object transform, inputs unchanged during a run, real optimizer updates, zero assistance.
- **Recorded only (quality):** interpolated/extrapolated hand frames (`source_code` 2/3), shape provenance, geometry/penetration quality, contact/support/release repair checks. `geometry_quality_accepted` is always `False`, which means "not certified", not "rejected".

Don't add new quality checks as blockers unless the user explicitly asks.

## Conventions

- Outputs must be new absolute directories outside the input roots (`require_new_output`). Writers open files with `"x"`. Original NAS videos, reconstruction jobs and checkpoints are read-only.
- Never commit generated assets, videos, models or venvs; they go under `.e2e/` or `/var/tmp`.
- Units are meters; submission quaternions are `xyzw`, while the public dataset parquet uses `wxyz`. Mesh and pose must share the body frame.
- Local metrics (CHORD, object-only, ADD-AUC against our reconstructed reference) are proxies. Report them separately from official hidden-GT scores (AUC, SP-SR, MP-SR, RPE, MPPE, CD-O). `TRACK3_PROGRESS_REPORT_20261030.md` §6 explains the official scoring.
- Public GT meshes must not be used for Track 3 registration or mesh scoring.
- This repo has no git identity configured. Earlier commits are authored as `shenjiahao <2152312121>`.
