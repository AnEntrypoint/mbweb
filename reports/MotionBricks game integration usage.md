# Clip seams and side projection drive the MotionBricks artifacts

The spazzing and floating come from how the page uses MotionBricks, not from a model fault the notes can show. Upstream MotionBricks is a stateful, keyframe-conditioned generator that expects the game to own root travel, ground contact, and continuity between replans, and the notes document no game-side recipe for any of these. The page instead generates every clip from the same standing boundary with no hand-over, rescales the figure for each clip, pins the pelvis to a screen point while walking along the screen's vertical axis, and discards the model's root height and horizontal travel. **Clip seams** (every swap and loop point is a hard cut) and the **side-view projection** are the two highest-impact causes, and both are **confirmed by reading the code**. Their magnitudes are **suspected**: the notes contain no run, measurement, or screenshot. The ranking below is therefore a judgement from code paths, and the first fix is a measurement pass that sizes them.

## What upstream expects from a game loop

Upstream treats generation as a loop rather than a one-shot call. The decoder reads **context slots 0-3** (recent character states, "always available") and **target slots 4-7**, where the decoder takes **1 to 4 target keyframes** and replaces missing constraints with a learnable mask embedding ([S2 paper](https://arxiv.org/html/2604.24833v1), Sec. 3, 4.3). Replanning is event-driven: the paper's appendix recommends replanning when fewer than **3 to 9 frames** remain in the motion buffer, and immediately on any command change ([S2 paper](https://arxiv.org/html/2604.24833v1), App. C). The **10 Hz** cadence the notes cite is specific to robot deployment, where lazy replanning keeps compute away from the low-level controller ([S2 paper](https://arxiv.org/html/2604.24833v1), Sec. 7.4). It is not a game-loop rule, and the failure-modes note applies it to games too broadly. Between replans the game samples the buffered motion at **30 FPS** ([S2 paper](https://arxiv.org/html/2604.24833v1), Sec. 3). Pose tokens start as masked embeddings with **no documented warm start** from the previous output (Sec. 7.3), so continuity across plans is the game's responsibility. The engine pattern in the paper's appendix G keeps a pose buffer that updates every frame and invokes the backbone only when new target keyframes arrive, and the notes read it as a sign that rendering should decouple from the 30 FPS motion clock ([S2 paper](https://arxiv.org/html/2604.24833v1), App. G; inference, not an NVIDIA recommendation).

Units and frames are documented in the code release. Motion space is **Y-up, Z-forward, right-handed**, positions are in metres, and the global root is projected position plus cos/sin of pelvis heading ([S5 motion representation](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/motionbricks/docs/motion_representation.md); [S2 paper](https://arxiv.org/html/2604.24833v1), Sec. 4). The units of the velocity and heading **command** are not stated anywhere the notes read, and the explicit reset procedure, the root-to-world integration, and the recommended game tick rate are all **gaps** in the notes.

## Root and ground handling are undocumented, so the page had to choose

The notes do not establish how the upstream demo applies the root. The demo's root handling in `scripts/interactive_demo_g1.py` was not read, and the paper says the decoder can "refine the predicted root trajectory" and that root override is supported by keeping pose tokens fixed and replacing root values with interpolated trajectories ([S2 paper](https://arxiv.org/html/2604.24833v1), Sec. 5.1, 7.3). Whether the output root is integrated into world transform or replaced by gameplay is a **gap**. The paper also says foot skate is "handled implicitly by tolerating imperfect keyframes" (App. D) and that accurate contact labels are "helpful for postprocessing in downstream applications" (Sec. 7.4). Nothing in the notes documents an IK, foot-lock, or ground-snap step inside MotionBricks, so the game must supply one. Shipped motion-matching systems get foot consistency from root-motion clips and foot-bone matching, and Unreal states that "animations used in a Motion Matching setup for locomotion must contain root motion" ([Unreal docs](https://dev.epicgames.com/documentation/en-us/unreal-engine/motion-matching-in-unreal-engine)). Foot skate is a standard metric for this gap, defined as "undesirable sliding of feet during ground contact" in m/frame ([S2 paper](https://arxiv.org/html/2604.24833v1), Sec. 7.2). Post-process footskate cleanup has an established method that "exactly satisfies footplant constraints" ([Kovar et al., SCA 2002](https://research.cs.wisc.edu/graphics/Gallery/kovar.vol/Cleanup)).

## The page breaks the contract at each clip seam and in the projection

The local audit (read only, cited to `/workspace/mbweb/web/app.mjs` and `/workspace/mbweb/component/motionbricks.c`) and my spot-check of those files agree on the core path. The page sends one stateless `infer` call per clip. Every call uses the same global-root and pose conditioning, **eight slots of the standing pose with unit heading**, and **all-zero local-root velocities**, with the seed incremented per clip (`web/app.mjs:297-307`; `component/motionbricks.c:54-85`). No commands, keyframes, masks, sampling, or duration settings are ever set. The previous clip never feeds the next (`web/app.mjs:307-315`), and the old clip loops with a modulo wrap while the next is generated (`web/app.mjs:366`). The paper's continuity mechanism, context from recent states, is thus unused. Each clip begins from a fresh sample at standing, so the page swaps them at a visible seam. Clips run **24 to 64 frames** under library defaults, which is 0.8 to 2.1 s per swap (the length range is cited to the local audit's reading of the upstream API document, not re-read here).

The projection adds three problems. First, `prepareClip` recomputes `span` per clip, and `drawFrame` scales every frame by it, so **the figure's size pops at every swap** (`web/app.mjs:105`, `:231`). Second, the side view uses the model's joint Z as the horizontal axis and never removes the root yaw, so any yaw in a clip turns the silhouette relative to the camera (`web/app.mjs:264`; `web/lib/skeleton.mjs:45-59`). Third, the facing mirror is `cos(heading) < 0 ? -1 : 1` (`web/app.mjs:251`). The heading wanders with a random turn rate, so it crosses ±90 degrees routinely, and each crossing flips the figure instantly. The model's own yaw does not drive the mirror.

The floating has two code-level causes. The pelvis is pinned at `(body.x, body.y)` on the screen, and the model's root height and horizontal travel are discarded (`web/app.mjs:233-234`, `:264-265`). Worse, `body.y` is used both as the walking coordinate (`web/app.mjs:206-208`, `:219`) and as the pelvis screen height (`web/app.mjs:265`). Walking up or down the stage therefore moves the figure vertically with no ground line drawn, which reads as floating. This coupling is **a finding that the local audit does not state**, and I confirmed it by reading those lines. The walk speed is a constant, `WALK_PACE = 0.12` of the smaller canvas side per second (`web/app.mjs:145`, `:206`), while the clip's own pelvis travel is computed into `state.clip.speed` and never used, so the feet slide (the code comment at `web/app.mjs:143-144` admits this). The step size also depends on the frame-change gate, so `dt` alternates between about 33 and 50 ms at 60 Hz (`web/app.mjs:176`, `:247`); the magnitude of that jitter is **suspected**.

## Ranking the causes by likely impact

Spazzing is most likely from the clip seams and the per-clip rescale. Each swap is a discontinuity in both pose and size, and the figure's silhouette changes again when the yaw or the mirror flips. Floating is most likely from the screen-y coupling and the discarded root, since the figure has no ground reference and moves vertically on screen as it walks. The table ranks the causes; the status column distinguishes what the code shows from what needs a run to size.

| Rank | Cause | Symptom | Status |
|---|---|---|---|
| 1 | Hard clip swap and loop wrap, standing boundary, no context carry (`web/app.mjs:297-315`, `:366`) | Spazz | Confirmed in code; magnitude suspected |
| 2 | Walk coordinate shares screen y with pelvis height (`web/app.mjs:206-208`, `:265`) | Float, vertical drift | Confirmed in code by this review; magnitude is visual |
| 3 | Per-clip rescale from per-clip span (`web/app.mjs:105`, `:231`) | Size pop every 0.8-2.1 s | Confirmed in code; magnitude depends on clip variance |
| 4 | Standing targets in slots 4-7 if masks are on (upstream slot semantics per the paper; mask state not in notes) | Stiff, stationary clips, snap back at seams | Suspected; could outrank all others if confirmed |
| 5 | Root height and XZ discarded, no contact or ground snap (`web/app.mjs:233-234`, `:264-265`) | Float | Confirmed in code; upstream root handling is a gap |
| 6 | Facing mirror flips at cos(heading)=0 (`web/app.mjs:251`) | Spazz | Confirmed in code |
| 7 | Fixed walk pace, unused clip speed (`web/app.mjs:145`, `:206`, `:107-113`) | Foot slide | Confirmed in code |
| 8 | Model yaw not removed before projection (`web/app.mjs:264`) | Silhouette turns between clips | Code confirmed; magnitude suspected |
| 9 | Frame-gated `dt` and per-draw random heading (`web/app.mjs:176`, `:189`) | Body jitter | Code confirmed; magnitude suspected |
| 10 | Library-default sampling and random durations (`component/motionbricks.c`, no setters) | Clips look unrelated | Suspected |
| 11 | Ternary bundle, no fidelity check in audited path | Unknown quality | Suspected |
| 12 | Virtual joints drawn, duplicate rAF loops, dead code, doc drift | Visual noise, CPU | Confirmed for the code; low impact |

Rank 4 needs a precise note. The paper says slots 4-7 are target keyframes and missing ones are masked ([S2 paper](https://arxiv.org/html/2604.24833v1), Sec. 3, 4.3). The page fills all eight slots with the standing pose, and the component's comments say "enabled global-root slots need unit cos/sin" (`component/motionbricks.c:52-53`). If those slots are enabled, the model is being asked to hit standing targets at every step, which would explain the stationary clips the page's own comment describes. The notes do not establish the mask state, so this stays a hypothesis that one run can settle.

## Recommended fixes, ranked by traceable finding

The first step is the measurement pass, because several fixes depend on facts the notes lack. The fixes below are ordered by expected impact and cite the ranked cause they address.

| Fix | Addresses rank | Verification |
|---|---|---|
| 1. Read the upstream demo's root handling (`scripts/interactive_demo_g1.py`) and dump, for N seeds: root travel, root yaw, root height, minimum foot height, clip span, and the output with slots 4-7 masked versus enabled | 4, 5, 8, 10 | Sizes every suspected magnitude and settles the mask question before other fixes land |
| 2. Walk on a ground-plane coordinate and project the side view to a fixed ground line at a fixed screen y, with a drawn ground | 2, 5 | Figure stays on one baseline while walking up or down the stage |
| 3. Carry continuity across seams: feed the outgoing clip's last frames as context slots 0-3, or inertialize the outgoing pose into the incoming clip over a short window; replace the loop wrap with a blend from last frame to first | 1 | Frame-to-frame pose delta at each swap is bounded, measured by the Q4 checks |
| 4. Freeze or smooth the scale, computing span once per session or from the standing pose | 3 | Figure height is constant across swaps |
| 5. Replace the mirror with a continuous rule: derive facing from a smoothed walk direction with a hysteresis band, or drop mirroring | 6 | No instant flip at ±90 degrees |
| 6. Ground the figure per frame: shift the pelvis so the lowest joint on a contact frame sits on the ground line | 5, 7 | Foot height at contact stays near zero per the Q4 check |
| 7. Drive the step from `state.clip.speed` and clip pelvis travel instead of `WALK_PACE` | 7 | Foot slide metric drops against the clip's own speed |
| 8. Remove root yaw from pelvis-relative positions before projection, only after fix 1 confirms the yaw magnitude | 8 | Silhouette stays fixed across seeds |
| 9. Use a fixed-step accumulator with interpolation for body motion, and draw the heading noise from the frame delta | 9 | Body step size is constant across draw rates |
| 10. Set masks or real targets for slots 4-7, and set sampling or duration explicitly if fix 1 shows them to matter | 4, 10 | Clip root travel matches the intended command |
| 11. Remove dead code (`steerHeading`, `speedPx`, `wrapUnit`), draw only the 30 joints the upstream calls physical, use one rAF loop, and fix the doc drift listed in the local audit | 12 | Lint and visual check |

Fixes 2, 6, and 8 depend on fix 1, since the notes never establish how the upstream demo applies root travel. Fix 3 is the strongest single change under the current evidence, but it is an engineering choice, not a documented upstream recommendation.

## Conclusion

The page treats each MotionBricks clip as a standalone image, and the upstream contract treats generation as a continuous stream conditioned on the previous state. Most of the artifacts follow from that mismatch, which means the model is probably not the first place to look. The notes leave three things unsettled: how the upstream demo applies root travel, whether slots 4-7 constrain the output, and how large each suspected magnitude is on screen. A measurement pass answers all three before any rendering change, and the screen-y coupling is the one finding that was not in the local audit and can be fixed without knowing any of them.

## Update: the upstream demo and the chained-clip change

The upstream G1 demo (`motionbricks/scripts/interactive_demo_g1.py`) was read after the first pass. It runs a streaming loop: each step takes `get_next_frame()` and writes the model's own qpos into the simulated robot, feeds the robot's current qpos back as context (`use_qpos=1`), and calls `generate_new_frames` again. Root height and travel are never discarded. Continuity comes from the model's own previous output, not from a standing reset.

The inference contract (`docs/API-INFERENCE.md`) allows the same thing per clip. Boundary 0 takes four source frames, and the output starts with them. The page filled both boundaries with the standing pose, which is why clips began and ended at rest.

The change in this update:
- The next clip's source is the previous clip's last four frames, canonicalised about the first source root (origin X/Z, yaw zero), as the API doc prescribes. The output is placed back into the session frame with the same transform, so the clip continues from where the last one ended.
- Target slots 4-7 are masked off (`free-target`, in the WIT and the glue), so clips no longer have to end standing. The masks follow the API doc: source slots enabled, local-root slot 3 disabled.
- The chained clip plays from its fifth frame, since its first four repeat the previous clip's last four.

Verification: the coordinate math passes a Node check against a synthetic skeleton (canonical pose rows match forward kinematics, the orientation columns match, and the transform round-trips to about 1e-7). The smoke test passes on the rebuilt site, including a chained clip. Seam smoothness on the real model is not yet measured, and the model's fidelity to the source frames is unverified. A visual check on the page is the next step.

Walk pace now follows the clip's own pelvis speed, per drawn frame, instead of the fixed `WALK_PACE` constant. The feet can still slide where the wander heading differs from the model's travel direction, and the speed magnitude is unmeasured.

### Root cause of the crouch and the float, found by running the component

Headless runs of the real component (`scripts/diagnose-chain.mjs`) found two causes that the earlier reading had missed:

- **The standing source was underground.** The neutral skeleton is pelvis-centred (pelvis y = 0, feet at −0.79 m), and the standing defaults used those heights as they were. A figure whose feet sit 0.79 m below the floor makes the model fold into a crouch: the pelvis-to-foot distance fell from 0.69 m to 0.20 m over one clip. Lifting the standing source by 0.787 m holds the pelvis at 0.80 m, the lowest joint on the floor within 5 mm, and pelvis-to-foot at 0.80 m for the whole clip. The glue now makes that lift.
- **The render floated the feet.** Each pose was drawn from the pelvis, so the lowest joint moved relative to the shadow line from frame to frame. `drawFrame` now pins each frame's lowest joint to the ground line.

Measured after both fixes: a chained clip reproduces its source to 1.1 cm at the root and in joint positions (before the lift it was 8 cm), 1.9° of heading, and 1.7 mm of root height.

## GPU feasibility (measured, not implemented)

- The pinned `ggml-webgpu` backend is 4763 lines against Dawn's C++ WebGPU wrapper. Its browser path is the Emscripten build (`#ifdef __EMSCRIPTEN__`), not the WASI build the threaded core uses, so it cannot be dropped into the current toolchain as is.
- Its shader set has no ternary (TQ1_0) matrix-multiply shader. The ternary matmuls dominate the largest graph, so a GPU backend without one would run them on the CPU.
- This machine's Chromium exposes `navigator.gpu`, but `requestAdapter()` returns null (no usable GPU), so a GPU path cannot be executed or measured here.
- Remaining work, in order: pick the toolchain (emscripten build of the core, or a `webgpu.h` shim under WASI); write a WGSL ternary matmul kernel and check it against the CPU path; run on a machine with a WebGPU adapter and measure the largest graph.
