# Track 3 actual-rollout submission baseline

Submission-producing adapters for the NVIDIA Video-to-Data Track 3 challenge. This is a minimal source snapshot, not a copy of unrelated workspace data or local Git history. `SOURCE_MANIFEST.json` records the exact adapter source hashes.

## Method

1. Reuse the reconstructed mesh and camera/object trajectories belonging to each evaluation video. Recover completed raw outputs from preserved failed packaging jobs as needed. Never source object poses from the submission template or hidden labels.
2. Assemble every episode's objects in one physical scene using a common camera-to-scene transform. Preserve original metric mesh body coordinates and object identities.
3. Use the existing official IK adapter and learned FlashCHORD actors. This full-coverage baseline uses the two-body episode03 actor (131072 training transitions) for two-body scenes and the frozen one-body actor (2097152 transitions) for single-body scenes. It performs zero new training and makes no generalization/success claim.
4. Execute the learned policy in physics with object assistance exactly zero at every step. Capture every source frame at native20Hz in fixed world0. Export achieved poses, not the reconstructed reference poses.
5. Pack all20 episodes and36 meshes using the official submission kit. Keep success/quality metrics separate from numerical and submission validity.

Missing hand reference poses are explicitly marked inferred (nearest observed pose); an entirely unobserved hand uses a neutral shape parked outside the task. These are reference-input assumptions, not measured hand reconstruction. Strict research validation remains the default; delivery uses an explicit inference allowance. All original inputs remain unchanged.

## Entry points

- `experiments.nvidia_v2d.submission_inputs prepare`: derive numerically usable input bundles with provenance from an original batch manifest.
- `experiments.nvidia_v2d.submission_inputs bridge`: shared scene / MANO geometry export, including multi-object scenes.
- `experiments.nvidia_v2d.sharpa_multitip --backend official-ik`: robot reference export.
- `experiments.nvidia_v2d.submission_rollout`: native-speed learned-policy physical rollout using an explicit reference.
- `experiments.nvidia_v2d.submission_batch`: execute ready episodes, validate complete actual trajectories, copy meshes, and export official-format per-episode results.
- `experiments.nvidia_v2d.policy_logged_launch`: supervised SwanLab/TensorBoard metric persistence and readback verification. Its historical generic launch status uses the word training even when the wrapped command is evaluation-only; this baseline has zero new training.

The original policies were produced with `policy_experiment` and its saved configuration/checkpoint metadata. Checkpoints and evaluation-video assets are supplied separately under their own terms. They are not embedded in this source repository.

## Runtime

Public upstream: https://github.com/nvidia-isaac/video_to_data
Official submission kit runtime revision: `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.
This run uses the existing frozen FlashCHORD image digest `sha256:7fdce3ad594fbad8b5983fde7846908be2ed6179ba4ebcfd1edefbcbcf4c628c`; metadata retains evaluator/checkpoint SHA256 values. MANO bridge image: `sha256:9243c47c5d1bf75625cc684d140e89f106a33613ad5ab3f7ba8c6e3688581d64`.

The host adapters need Python3.11, NumPy, SciPy, PyArrow, Pydantic2, Trimesh and Pillow. The official IK additionally uses Pinocchio/NLopt and the upstream Sharpa assets. MANO data must be obtained under its own license. The upstream simulation runtime supplies Newton, MuJoCo-Warp, JAX and FlashCHORD. SwanLab/TensorBoard run in a separate host environment.

`submission_inputs` expects the original reconstruction batch `run.json` and a plan containing video hashes, episode IDs, object names, frame counts and fps. Each prepared episode has immutable source/provenance sidecars. Batch CPU preparation may be scheduled independently; `submission_batch` consumes `episode-XXXXXX/cpu-status.json` records and official IK outputs beneath its `--root`.

## Evaluation interpretation

Local metrics track reconstructed references and are not the competition's hidden-ground-truth scores. Failed rollouts are retained and submitted without selection. Neither the inferred hand inputs nor low local success are represented as verified natural grasping.
