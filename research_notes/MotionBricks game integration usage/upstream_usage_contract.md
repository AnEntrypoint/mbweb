# MotionBricks (G1) upstream runtime usage contract

Scope: NVIDIA MotionBricks, G1 humanoid variant, real-time/interactive use. Offline batch generation excluded.

Legend: **[Doc]** = stated in a cited source. **[Inf]** = my inference from cited material. **[Gap]** = not stated in any source I could read.

## Sources

- S1 paper, arXiv abstract: https://arxiv.org/abs/2604.24833 (v1, 27 Apr 2026)
- S2 paper, arXiv HTML: https://arxiv.org/html/2604.24833v1 (text is 141k chars; I read 0-100k and 100k-end; appendix text was partially truncated/paraphrased by the fetch tool, so appendix quotes below are from a summarizing fetch, not verbatim)
- S3 project page: https://nvlabs.github.io/motionbricks (links and high-level claims only)
- S4 code location: https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks (README, "preview release")
- S5 `docs/motion_representation.md`: https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/motionbricks/docs/motion_representation.md
- S6 upstream issue list (titles only): https://github.com/NVlabs/GR00T-WholeBodyControl/issues?q=motionbricks
- S7 secondary press (unverified details): https://www.gigazine.net/gsc_news/en/20260615-nvidia-motionbricks/

Not reached: the paper PDF (https://research.nvidia.com/labs/gear/motionbricks/pdfs/motionbricks_siggraph_2026.pdf) and the raw `README.md` of the motionbricks folder (the rendered page was used). The root-motion handling in the demo script (`scripts/interactive_demo_g1.py`) was not read. Source code was not inspected; everything is from docs and paper text.

## Status and release

- [Doc, S4] Code is a "preview release" inside GR00T-WholeBodyControl. It ships "two components": an interactive G1 demo and a synthetic training pipeline. "A full release with the complete training pipeline is targeted for approximately one month out." (as of the fetch date; treat as stale)
- [Doc, S6] A UE5 demo/plugin is repeatedly requested in issues (#131, #170, #249) with no release shown. #207 asks how to integrate MotionBricks into Sonic; still open at last view.
- [Doc, S7] SIGGRAPH 2026 / ACM TOG venue: secondary source only. [Gap] Not confirmed from NVIDIA or the paper header I read.
- [Doc, S7] Licensing (Apache-2.0 code, NVIDIA Open Model License for weights): one secondary source only. [Gap] Unverified; check the repo LICENSE.
- [Doc, S4] Checkpoints: `motionbricks/out/` (G1-clip.ckpt ~7.5 MB; motionbricks_vqvae ~273 MB; motionbricks_pose ~1.6 GB; motionbricks_root ~391 MB). Requires Git LFS. Requirements: Python 3.10+, CUDA GPU, Git LFS. Install: `pip install -e .` in `motionbricks/`.
- [Doc, S4] G1 demo entry: `DISPLAY=:1 python scripts/interactive_demo_g1.py` (MuJoCo viewer). Keyboard control: WASD relative to camera; style keys V (slow walk), Z (hand crawl), X (walk boxing), F (zombie walk). Crawling "do not support side-only directions."
- [Doc, S4] Linux keyboard capture needs `pynput python-xlib` (X11). Wayland/macOS/Windows: shortcut conflicts; README is ambiguous on macOS.

## 1. Per-frame inputs

### 1a. Control commands (game-side)
- [Doc, S2 Sec 3, Alg. 1] The loop reads "game event or user commands C". It replans when C changes or the motion buffer B runs low. Stage 0 builds keyframe constraints T from character state S and C via smart primitives.
- [Doc, S2 Sec 6.1] Locomotion: "arbitrary velocity, heading, and style commands" (S4 wording). Naive root target = commanded velocity and heading extrapolated over a **1.0 s horizon**, then smoothed by a **critically damped spring** from current root r0 and velocity v0 (damping gamma; value not given).
- [Doc, S2 Sec 6.1] Style is set by choosing keyframes sampled from short reference clips or hand-authored poses placed on the spring-smoothed trajectory. No separate style vector is described.
- [Gap] Units of the velocity and heading command (m/s? normalized?) are not stated. Heading units are radians by inference only (sampled Uniform(0, 2pi) in training, S5).

### 1b. Keyframes
- [Doc, S2 Sec 3, 4.3] Keyframe slots 0-3 = context (recent character states, "always available"). Slots 4-7 = target keyframes; decoder takes 1 to 4 target keyframes at inference. Missing constraints are replaced by a learnable mask embedding.
- [Doc, S2 Sec 6.2] Smart objects: each keyframe set has drop-frame attribute tau. tau = 0 is a hard constraint; tau > 0 is soft guidance that may transition up to tau frames early. Each keyframe has flag omega that rotates it about the interaction pivot at runtime.
- [Doc, S2 App D] Keyframes are blended by interpolating individual keyframes in joint space from reference poses, then fed to the backbone. Not a 2D blendspace. No numeric weights or interpolation formula given in the text I could read. [Gap]
- [Doc, S2 App G] UE5: a "task actor" stores sparse target poses sampled from clip assets. Box traces find interactables; an activated smart object becomes the keyframe provider and interrupts the current motion. A falling state places a landing keyframe from a projectile estimate.

### 1c. Timing and goal
- [Doc, S2 Sec 3] Data and in-betweening segments: 12-64 frames at **30 FPS** ("For all four datasets, we use 30 FPS").
- [Doc, S2 Sec 5.1] Root module predicts frame count T2 at 4-frame resolution, or uses ground-truth T1 if given. This is a training-time/offline option; whether the runtime uses it is [Gap].
- [Doc, S2 App C] Replan trigger: "a replan is triggered when fewer than tau frames remain in the motion buffer," plus on any command change. Evaluated replan intervals 1-8 frames; recommended default **3 to 9 frames** with instant replan on command change. (Paper's own recommendation, not S4 code.)
- [Doc, S2 Sec 7.4] Robot deployment: "Replanning is triggered at 10 Hz or whenever commands change." Rationale: lazy replanning avoids competing for resources with the latency-sensitive low-level controller.

### 1d. Previous-frame state and history
- [Doc, S2 Alg. 1] "Pop next frame from B; update S." Buffer length and exact history contents are not stated. [Gap]
- [Doc, S2 Sec 3, 4.3] Context = first 4 frames of the decoder input, drawn from prior output/state.
- [Doc, S2 Sec 7.3] Pose tokens are initialized to masked-token embeddings at inference. There is no documented warm-start from previous output.

### 1e. Per-frame feature vector (model input, S5)
- [Doc, S5] "Full feature vector is 418-dimensional per frame"; local subset is 413 dims. Batch key `"motion"` is [T, 414] normalized global representation. Dtypes not stated. [Gap]
- Body features (409 dims, shared):
  - `ric_data` 99: 33 non-root joint positions minus per-frame root XZ, world frame, not heading-canonicalized.
  - `global_rot_data` 204: world-frame 6D continuous rotations for all 34 joints.
  - `local_vel` 102: finite-difference velocity, world frame despite the name.
  - `foot_contacts` 4: binary, left ankle, left toe, right ankle, right toe. Thresholds not given.
- Root features:
  - `global_root_pos` 3: XYZ, world frame.
  - `global_root_heading` 2: (cos, sin) of the Y-axis rotation angle.
  - `local_root_rot_vel` 1: angular velocity about Y.
  - `local_root_vel` 2: XZ translational velocity in heading-aligned frame.
  - `global_root_y` 1: root height.
- Normalization: z-score, `(x - mean) / sqrt(std^2 + 1e-5)`, stats in `stats/motion/`.
- [Doc, S5] Inference convention: `first_heading_angle = 0` (deterministic). `change_first_heading` puts the first frame's root XZ at the origin and keeps Y.

## 2. Coordinate frame, handedness, units

- [Doc, S5] Motion space: **Y-up, Z-forward, right-handed**.
- [Doc, S5] MuJoCo space: **Z-up, X-forward, right-handed**. Mapping: motion X = MuJoCo Y; motion Y = MuJoCo Z; motion Z = MuJoCo X.
- [Doc, S2 Sec 4] Global root = projected global positions plus cos/sin of pelvis heading. Local root = projected positional velocity and angular velocity of pelvis in global coordinates.
- [Doc, S2 Sec 4] Joint positions are global, relative to the root position. Rotations are global.
- [Doc, S2 Sec 4] "we do not apply heading canonicalization during training"; random rotation augmentation used.
- [Doc, S2 Sec 7.2] Units: positions and errors in meters; jitter in m/s^2; foot skate in m/frame; penetration in m; reach success 5 cm / 15 deg.
- [Doc, S5] Quaternion order: the motion representation uses 6D rotations, not quaternions. MuJoCo qpos root quaternion is (w, x, y, z) at indices 3-6.
- [Gap] Quaternion order / format used when the model's output is converted to engine transforms (e.g., UE5, Unity, Godot) is not stated.

## 3. Output

- [Doc, S2 Eq. 3] Decoder output per frame: {r_l, p, q, v, c} = local root trajectory, joint positions, rotations, velocities, contact labels, for T frames.
- [Doc, S2 Sec 5.1] Root module emits global root r_g; converted to local root for the pose module and decoder. The decoder can "refine the predicted root trajectory" (r_g,4 in progressive refinement, Table 1).
- [Doc, S2 Sec 7.3] Root override is supported: "we keep the predicted pose tokens fixed but override the root values with interpolated trajectories."
- [Doc, S5] Dual representation is lossless: `dual_rep.global_to_local` and `dual_rep.local_to_global`.
- [Doc, S2 App G] UE5 animation graph updates the context pose buffer each frame; it invokes the backbone only when new target keyframes arrive, otherwise samples from the buffered motion.
- [Gap] Whether the output root is integrated into world transform by the model's root or ignored in favour of gameplay: not stated in paper or docs. The demo's handling of root motion was not read.
- [Gap] Local vs global for final joint rotations handed to a game engine's skeleton: not stated beyond "global" in the feature definition.

## 4. Smoothing, blending, warm-start, reset

- [Doc, S2 Sec 6.1] Critically damped spring on the root target before the root module and decoder (two refinement stages).
- [Doc, S2 App D] Keyframes blended in joint space, not in a blendspace; drop-frame replanning means blended keyframes are not enforced rigidly.
- [Doc, S2 Sec 3] Replan on command change or low buffer.
- [Doc, S2 Sec 7.3] Pose tokens start as masked embeddings at inference (no warm start).
- [Gap] Explicit reset procedure (on teleport, respawn, state reset) is not documented in paper or docs. Only the heading reset `first_heading_angle = 0` and the first-frame XZ origin in S5 are documented.
- [Gap] Recommended game-loop tick rate, sub-step, or interpolation between replans is not documented.

## 5. Foot sliding, floating, jitter

- [Doc, S2 Sec 7.2] Metrics reported: Joint Jitter and Root Jitter (m/s^2), Foot Skate (m/frame), Penetration (m), Foot Contact Accuracy (%).
- [Doc, S2 Sec 7.2/Fig 14] Authors report foot skate stays low under significant root manipulation.
- [Doc, S2 Sec 7.4] Training uses standard foot-sliding and velocity losses in the tokenizer.
- [Doc, S2 Sec 7.4] Accurate foot contacts "helpful for postprocessing in downstream applications when necessary." So the authors expect downstream post-processing (IK/contact) is an option; no specific recommended fix given.
- [Doc, S2 Sec 8.1] Limitation: kinematic planners "may generate physically implausible motions (self-collisions)."
- [Doc, S2 App D] Foot sliding and floating are said to be handled implicitly by tolerating imperfect keyframes.
- [Doc, S6] No open or closed issues in the titles list mention jitter, foot sliding, floating, root motion, or reset (14 issues checked by title only; bodies not read). [Gap] Not verified against bodies.
- [Inf] Given the explicit "lazy replanning" rationale, mid-replan discontinuities are plausible sources of visible pops, but no source says so.

## 6. Performance

- [Doc, S1/S2/S3] 2 ms latency and 15,000 FPS throughput on desktop RTX 5090 (author reported, paper and project page).
- [Doc, S2 Sec 7.4] Jetson Orin: about 5 ms per inference.
- [Doc, S2 Sec 7.4 and S2 App G] Deployment path: ONNX export loaded with TensorRT in a native plugin. Stated for UE5 app; for the G1 robot, Jetson Orin + Unitree SDK.
- [Gap] The S4 release's own inference path (PyTorch vs ONNX/TensorRT) is not stated in the docs read.

## 7. G1 hardware handoff

- [Doc, S2 Sec 7.4] Real robot: "we drive real robots by supplying the generated kinematic motions as tracking targets," using the tracking controller from Luo et al. (2025).
- [Doc, S2 Sec 7.4] G1 skeleton: 34 joints, = 32 active G1 joints plus 2 dummy toe joints. Proprietary data retargeted to G1 via GMR.
- [Gap] The state/command signal format between MotionBricks and the tracking controller is not stated.

## 8. Recommended integration pattern (as documented)

- [Doc, S2 Sec 3, 7.4, App C, App G] Event-driven replan: call the backbone when (a) commands change, (b) buffer has fewer than tau (or 3-9) frames left, (c) on a fixed 10 Hz cadence for robot deployment. Between replans, sample from the buffered motion at 30 FPS.
- [Doc, S2 App G] Engine shape: a per-character "task actor" holds keyframe providers; an animation graph owns the pose buffer; the backbone is invoked only on new target keyframes. A StateTree handles locomotion, keyframe sequence, and falling.
- [Inf] A game loop should therefore decouple render tick from the 30 FPS motion clock and replan asynchronously on command changes. This is inferred from the above, not stated as a recommendation by NVIDIA.
- [Gap] No Unity or Godot integration documented. No Unreal plugin release found (S6 repeated requests).

## 9. Inconsistencies and flags

- Joint count: skeleton has 32 active joints; MuJoCo model has 29 hinge joints (qpos 7-35). Unexplained in S5. [Gap]
- `local_vel` is named "local" but is world frame (S5 says so explicitly).
- Units of the velocity/heading command are missing from both paper and docs.
- Performance numbers are author-reported on RTX 5090; not independently verified.
- Release timing text in S4 ("one month out") is undated in the fetch and likely stale.
- S7 (Gigazine) is the only source for venue and licensing claims; S3 project page is aggregator-heavy.
- Appendix text (D, G, C) came through a summarizing fetch with truncation. Quotes there are paraphrase-grade. Verify against the PDF before relying on them.

## 10. Items to verify next (ordered)

1. Read `motionbricks/scripts/interactive_demo_g1.py` and the inference module in the S4 repo: input tensor names, dtypes, output keys, root application, buffer length, replan logic.
2. Read the paper PDF appendices (C, D, G) verbatim for interpolation method and UE5 root application.
3. Read the body of issues #252, #207, #127, #132 and #170 for real integration problems and workarounds.
4. Confirm license and venue from the repo LICENSE and NVIDIA's own pages.
