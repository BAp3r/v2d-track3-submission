# Official ACE-Ego-Hand as the WiLoR component

`--hand_tracking ace` uses the official ACE entrypoint and exports its predictions
into the existing hand matching, depth alignment and interpolation path. Use
`--hand_tracking wilor` for the corresponding WiLoR baseline. In the HaMeR or
HaWoR paths, `--hand_detector ace` replaces only the prompting component.

The inference image pins ACE and VideoX-Fun source revisions and installs its
Python dependencies with uv. Host orchestration uses the locked project in
`reconstruction/environments/ego`.

```bash
uv run --project reconstruction/environments/ego python reconstruction/modules/v2d_ace_hand/docker/build.py
```

From `reconstruction/`, add these flags to an ego command with a fresh output:

```bash
--hand_tracking ace \
--ace_weights /path/to/ACE-Ego-Hand \
--ace_backbone /path/to/Wan2.2-Fun-5B-Control \
--ace_mano /path/to/models-containing-MANO_LEFT-and-MANO_RIGHT \
--ace_cache /path/to/writable-local-cache \
--ace_input_intrinsics /path/to/calibrated-camera.json
```

The optional camera override accepts flat `{fx,fy,cx,cy,width,height}` JSON or
the official camera JSON. Without it, ACE uses the pipeline's stable intrinsics.
Use the recorded calibration when available. Native fisheye input needs prior
undistortion to a pinhole view.

Inference invokes the unchanged official `infer_video.py`: one continuous VAE
encoding of the complete video, followed by official tiled inference with its
native temporal anchors. RGB resizing uses official `INTER_AREA`. For long clips,
a disk-backed decoder preserves exactly the same contiguous float32 NCHW values
without holding multiple full video copies in RAM. The decoder has a byte-exact
regression against the official implementation. `--chunk_frames` remains a
compatibility argument and no longer resets temporal context.

The native official pickle is retained under `wilor_raw/native_predictions/`.
`--ace_native_predictions /path/to/color.pkl` explicitly imports an official
K-given pickle for CPU export; its frame count and calibrated intrinsics are
validated and its source is recorded. Per-frame JSON outputs keep original frame
indices, gate presence using official `claim`, and use physical MANO projection
for bboxes.

## MANO pose contract and the October 23 correction

Official ACE uses `smplx(flat_hand_mean=False)`: its finger axis-angle values are
residuals relative to the MANO hand mean. Existing WiLoR consumers use flat-mean
`manotorch` and expect absolute axis-angle. The export therefore adds the
corresponding physical left/right MANO mean before converting left rotations to
the mirrored-right convention. Camera-space translation stays unchanged.

The original adapter omitted this mean. On actual frame 2400, right-hand mesh
error against the official smplx forward was approximately 31.1 mm mean and
90.3 mm maximum per vertex. Corrected export matches the real manotorch consumer
to numerical precision. Tests cover the actual frame and both hand conventions;
left/right MANO asset shape asymmetry is measured separately.

Every record and completion marker carry the pose-convention version. Completion
also records the official inference implementation, video fingerprint and camera
hash. Legacy output caches are rejected. Use a fresh output directory after this
correction: hand prompts, masks, matching, alignment and interpolation all need
regeneration. The October 23 native `ace_*.pkl` files remain unaffected by the
export bug; the old derived ACE results are invalid for quality comparison.

`--skip_hand_overlays` skips diagnostic hand mesh/mask MP4s while preserving all
predictions and portable result/Three.js exports. Model checkpoints and MANO
retain their upstream licenses; ACE code is MIT.
