# Local MotionBricks integration audit (read only)

Scope: how /workspace/mbweb feeds MotionBricks per clip and per frame, what it does with the output, and where that diverges from the upstream contract. Upstream reference: `/workspace/motion-bricks/repo/docs/API-INFERENCE.md` (cited as UP-API), `docs/API.md` (UP-API-MAIN), `docs/FORMATS.md` (UP-FORMATS), `include/motionbricks/inference.h`.

Confidence tags: **confirmed-by-reading** means the code path was read and the behaviour follows from the text. **suspected** means it needs a run or a measurement to size.

## 1. What the project calls

- The page never builds per-frame model inputs. It makes one stateless `infer` call per clip. There are no commands, agent, style, boundary-pose or mask calls anywhere in the project (`component/motionbricks.c` contains no `set_mask`, `set_sampling`, `set_boundary_poses`, `agent` or duration call; `grep` over `component/` returns zero matches).
- Component entry: `exports_mb_motionbricks_inference_infer` at `component/motionbricks.c:243-313`. It sets the three raw feature fields (`motionbricks.c:259-264`), the seed (`:265-270`), calls `mb_model_infer` (`:273`), then packs each frame as root xyz plus 136 local XYZW values (`:297-307`).
- Page to worker: `web/app.mjs:297-306` sends `steeredGlobalRoot(globalRoot)`, a copy of `localRoot`, a copy of `pose`, and `state.seed`. `web/worker.mjs:46-54` converts to `Float32Array` and `BigInt(seed)` and calls `inference.infer`.

## 2. Per-frame and per-clip input construction

| Input | Source | Value sent | Reference |
|---|---|---|---|
| global-root (40 f32) | `standing_from_model` | 8 slots of `[0, y0, 0, 1, 0]` (x=0, height=neutral pelvis y, z=0, cos=1, sin=0) | `component/motionbricks.c:63-69`; `web/app.mjs:297-301` |
| global-root heading | `steeredGlobalRoot` | returns the standing values unchanged unless `state.steerHeading` is set. Nothing ever sets it. **confirmed-by-reading** dead path | `web/app.mjs:29`, `:330-340`; the only occurrences are `:29`, `:332-334` |
| local-root (32 f32) | `standing_from_model` | all zeros: zero yaw velocity, zero x/z velocity, height 0 | `component/motionbricks.c:62` |
| pose (2424 f32) | `standing_from_model` | 8 slots of 303: 33 non-root positions `[x-root.x, y absolute, z-root.z]`, then 34 identity 6D orientations | `component/motionbricks.c:71-82` |
| seed | `state.seed` | starts at 0, +1 after each generated clip | `web/app.mjs:16-24`, `:304`, `:307` |
| sampling, duration mask, feature masks | none | library defaults (Gumbel temperature 1, all durations enabled, local-root slot 3 disabled) | absence in `component/motionbricks.c`; UP-API:89-98 |

Per frame, the page sends nothing to the model. The only per-frame state that reaches the model is none. Source and target boundaries are both the standing pose for every clip. The previous clip's output never feeds the next clip (`web/app.mjs:297-307`).

Checked against the contract, these match: the layout sizes (`motionbricks.c:16-20`, `wit/motionbricks.wit:7-9`), the `Y absolute` rule for pose positions (`motionbricks.c:72-75` against UP-API:67-70), the 6D orientation columns `[R00,R10,R20,R01,R11,R21]` (`motionbricks.c:78-82` against UP-API:70-73), the unit cos/sin pair (`motionbricks.c:68-69`, `app.mjs:336-337` against UP-API:81-84).

## 3. Output consumption

- `motionbricks.c:280-307` checks shape (34 joints, `root_count == frames*3`, `rotation_count == frames*34*4`) and repacks. Frame stride is 139 f32. Not a conversion: values are passed through unchanged. Units are metres, Y-up, 30 FPS (UP-API-MAIN:101-103).
- `web/worker.mjs:53-54` copies the values and transfers the buffer. No further change.
- `web/app.mjs:61-64` resolves the pending promise with `{frames, values}`. `generate()` (`app.mjs:291-323`) calls `prepareClip` once per clip (`app.mjs:308`).
- Forward kinematics: `web/lib/skeleton.mjs:34-68`.
  - Root world position is taken from output root xyz (`skeleton.mjs:45-49`).
  - Child world position is `parent_world + R_parent_global * neutral_bone` (`skeleton.mjs:53-63`).
  - Global rotation is `R_parent_global * R_local` (`skeleton.mjs:59`). Quaternion product checked by hand against the Hamilton formula (`skeleton.mjs:8-17`). `rotate` (`skeleton.mjs:20-30`) matches q v q-conjugate. Both are correct for XYZW.
- Projection: `prepareClip` (`app.mjs:87-129`) subtracts joint 0 (pelvis) from all joints in Z (`:95-98`) and uses Y relative to joint 0 for the extent (`:97`, `:121`). `drawFrame` (`app.mjs:223-289`) draws with `toScreen` (`app.mjs:263-266`):
  - screen x = `body.x + dx + facing * (z_joint - z_pelvis) * scale`
  - screen y = `body.y + dy - (y_joint - y_pelvis) * scale`
- So the output root translation is used only as the FK origin and then cancelled. The pelvis is pinned on screen at `(body.x, body.y)`. Root height and root X/Z travel are both discarded for drawing. (**confirmed-by-reading**, `app.mjs:233-234`, `:264-265`.)
- Output root rotation (joint 0 local XYZW) is applied to the whole skeleton (`skeleton.mjs:59`). Its yaw is never removed before projection. The projection uses Z as the horizontal axis (`app.mjs:264`), so any yaw in the clip turns the figure relative to the camera. The sign mirror `facing` comes from the walking heading, not from the model yaw (`app.mjs:251`). (**confirmed-by-reading** for the code; **suspected** as a visible cause, the clip yaw magnitude was not measured.)
- Scale: `scale = min(w,h)*0.3/span` where `span` is computed from the current clip (`app.mjs:105`, `:231`). Each new clip can change the figure size.
- Draws 34 joints including the four virtual endpoints (`app.mjs:270-282`). Upstream says those four endpoints (indices 7, 14, 25, 33) carry identity conditioning orientation and are omitted in the MuJoCo replay (UP-API:32-36, UP-FORMATS:76-79). **suspected** visual artefact, not a numeric error.

## 4. Frame loop timing

- Frame rate: `FPS = 30` (`app.mjs:4`), matching UP-API-MAIN:101-103 and UP-API:17-18. **confirmed-by-reading**, correct.
- Playback index is wall-clock driven: `position = startFrame + floor(elapsed * FPS)` (`app.mjs:358-361`). Frame = `position % frames` (`app.mjs:366`), so the old clip loops while the next clip is generated.
- `tick` is `requestAnimationFrame` (`app.mjs:349`, `:368`). It draws only when the frame index changes (`app.mjs:367`). No interpolation between clip frames.
- Body integration `updateBody` runs only inside `drawFrame` (`app.mjs:247`), so only when the clip frame index changes. `dt` is `min(0.1, (now-last)/1000)` (`app.mjs:176`). At 60 Hz rAF the frame change lands on a 16.7 ms grid, so `dt` alternates between about 33 ms and 50 ms. Step size varies by about 50%. **confirmed-by-reading**; magnitude of the jitter is **suspected**.
- Heading wander draws a fresh random number per draw (`app.mjs:189`), so turn rate is noisy per draw. **confirmed-by-reading**.
- Body walk step is `WALK_PACE * min(w,h) * dt` (`app.mjs:206`), with `WALK_PACE = 0.12` (`app.mjs:145`). It does not depend on the clip's pelvis travel. The clip speed array is computed (`app.mjs:107-113`) but never used: `speedPx` is computed (`app.mjs:237-239`) and passed to `updateBody` (`app.mjs:247`, parameter at `:175`), but the parameter is never read. **confirmed-by-reading**. The code comments at `app.mjs:143-144` admit the feet slide.
- No smoothing anywhere on joint values, scale, or body position other than the `dt` cap.
- `generate()` is called from `tick` when the clip has played through (`app.mjs:362-365`). Clip hand-over (`app.mjs:307-315`) resets `state.frame = 0`, redraws frame 0 of the new clip, and calls `startPlayback()`.

## 5. Initialisation and reset path

1. `app.mjs:402-405`: fetch `./model/ternary/manifest.json`, resolve `base_url`, post `init` to the worker.
2. `worker.mjs:15-28` (`loadComponent`): check JSPI (`WebAssembly.Suspending`), `fetchModel` (`web/lib/model.mjs:100-132`, SHA-256 per file), `instantiateComponent` (`web/lib/component.mjs:9-17`), `stageBundle` (`web/lib/component.mjs:21-31`), then `inference.load()`.
3. `worker.mjs:38-44`: read `standingFeatures`, `jointParents`, `neutralPositions`, post `ready`.
4. `app.mjs:54-60`: store parents, neutral, standing; call `generate()`.
5. `component/motionbricks.c:126-138`: `load` replaces `g_model` once. There is no unload between clips and no reset of the model.
6. Reset: none. `state.seed` keeps counting, `steerHeading` stays null, body state persists across clips (`app.mjs:148`), and there is no reset to standing between clips. Scrub input stops playback (`app.mjs:393-396`).
7. Note on `rAF` loops: `generate()` calls `startPlayback()` (`app.mjs:315`) while the previous `tick` loop is still scheduled (`app.mjs:368`). Each generation adds one `tick` loop; the extra loops never exit while `state.playing` stays true. **confirmed-by-reading**. Effect is extra work per frame, no visible artefact. Frame draws are deduplicated by `frame !== state.frame` (`app.mjs:367`).

## 6. Divergences from the upstream contract, ranked

Ranked by how plausibly each explains the user's "spazzing" (S) and "floating" (F) reports.

1. **Stateless clip-by-clip generation, standing boundary on every clip, no hand-over** (S, high). **confirmed-by-reading** for the input path (`app.mjs:297-307`, `motionbricks.c:54-85`). Upstream says the API does not blend seams, remove overlap, or pin constraints exactly (UP-API:6-10, UP-API:121-126). Each clip starts at a fresh sample conditioned on standing, and the page swaps clips at a snap (`app.mjs:307-315`). The old clip also wraps modulo (`app.mjs:366`) while the next one generates. Each clip boundary and loop point is a discontinuity. Magnitude **suspected**.

2. **Root height and root X/Z discarded; pelvis pinned on screen; no ground contact** (F, high). **confirmed-by-reading** (`app.mjs:233-234`, `:264-265`, `:98`, `:122`). The model's vertical root motion is removed. Feet are placed relative to a pelvis that is itself fixed, so the figure floats above no ground reference. The only ground is the obstacle rectangles (`app.mjs:134-138`).

3. **Walking pace is a constant unrelated to the gait** (F, medium-high). **confirmed-by-reading** (`app.mjs:145`, `:206`; unused `speed` at `:107-113` and `speedPx` at `:237`, `:175`). The body moves at a fixed pace, so feet slide against the ground. The code admits this at `app.mjs:143-144`.

4. **Facing mirror tied to walking heading, flips through vertical headings** (S, medium-high). **confirmed-by-reading** (`app.mjs:251`: `facing = cos(heading) < 0 ? -1 : 1`). When the walk heading passes through ±90 degrees, `cos` crosses zero and the figure flips instantly. The model's own yaw is not used for the mirror at all.

5. **Model root yaw not removed before side projection** (S, medium-high). **confirmed-by-reading** for the code (`skeleton.mjs:45-59`, `app.mjs:264`). **suspected** for magnitude. Upstream says yaw is about +Y with +Z forward (UP-API:62-63), and the output rotations are in the canonical frame (UP-API:121-126). Any yaw drift in a clip changes what the side view shows, and the figure can look like it turns or spins between clips. Needs a measurement of root yaw over a clip.

6. **Conditioning says the root is stationary** (F and S, medium). **confirmed-by-reading** that local-root velocities are zero (`motionbricks.c:62`) and every global-root slot sits at the origin (`motionbricks.c:64-66`). Upstream defines those velocities as in-clip finite differences of root motion (UP-API:62-65). The model is therefore asked for a clip with no root travel and no turning, while the page invents travel and turning. **suspected** that this produces unnatural gait. The page's own comment (`app.mjs:143-144`) says the clips "carry too little root travel to walk anywhere". That is the symptom this conditioning would produce.

7. **Per-clip scale changes** (S, medium). **confirmed-by-reading** (`app.mjs:105`, `:231`). `span` is recomputed for each clip, so the figure's size pops at each clip swap.

8. **Sampling and duration left at library defaults** (S, medium). **confirmed-by-reading** that no call sets sampling argmax, durations or masks (`component/motionbricks.c`, zero matches). UP-API:96-98 says duration is argmax and pose tokens are Gumbel-sampled at temperature 1 by default, with the seed controlling the stream. Every clip is a new random draw with a random length in 24..64 frames (UP-API:121-124). **suspected** this makes clips look unrelated to each other.

9. **Quantised substitute bundle, not the upstream G1 weights** (S and F, unknown magnitude). **confirmed-by-reading** that the page loads `./model/ternary/manifest.json` (`app.mjs:402`), with pose and root in TQ1_0 (`model/ternary/manifest.json:6`). `patches/motion-bricks/0001-accept-ternary-provenance.patch` only relaxes the provenance hash check (`model.cpp` metadata_source). The page-side smoke check only asserts that frames differ (`docs/prd/motion-bricks-wasip3/RESULTS.md:84-85`, `05-stick-figure-page.md:19-20`). There is no fidelity check in the audited path. Quality evidence may live in `research_notes/Ternary quants for MotionBricks G1/` but was not audited here. **suspected**.

10. **Frame timing: variable integration step, random per-draw turn** (S, low-medium). **confirmed-by-reading** (`app.mjs:176`, `:189`, `:247`). Body step size and heading noise vary per draw at rAF granularity. Magnitude **suspected**.

11. **Virtual endpoints drawn** (visual, low). **suspected**. All 34 joints are drawn (`app.mjs:270-282`). Upstream says indices 7, 14, 25, 33 are virtual with identity conditioning orientation (UP-API:32-36) and omitted in the MuJoCo replay (UP-FORMATS:76-79). This adds hand and toe dots that the upstream does not treat as physical bodies.

12. **Duplicate rAF loops per generation** (CPU, low). **confirmed-by-reading** (`app.mjs:315`, `:349`, `:368`). Not a visual cause.

13. **Documentation and build drift** (no runtime effect). **confirmed-by-reading**:
    - `web/index.html:15-16` says weights come from Hugging Face. The page actually loads them from `./model/ternary/` (`app.mjs:402`, `web/lib/model.mjs:100-132` via `manifest.base_url "./"`).
    - `scripts/build-site.sh:3-5` says GGUF weights are not part of the site. `scripts/build-site.sh:28-29` copies them.
    - `docs/prd/motion-bricks-wasip3/RESULTS.md:55` describes the `g1-f32/` bundle, which is no longer the page's bundle.
    - `web/index.html:36` loads `./app.js` and `web/app.mjs:44` spawns `./worker.js`. The source files are `app.mjs` and `worker.mjs`. The page only works after the esbuild step in `scripts/build-site.sh:19-22`.
    - `web/app.mjs:131` `wrapUnit`, `web/app.mjs:29` `steerHeading`, and the `speedPx` / `speed` values are dead.

## 7. Verified as correct (no divergence found)

- Layouts and counts: `motionbricks.c:16-20`, `wit/motionbricks.wit:7-9`, `skeleton.mjs:4-5`, `app.mjs:94` (34*3 stride).
- Stateless call and no history (UP-API:6-10).
- Pose row Y absolute, X/Z root-relative (`motionbricks.c:72-75` against UP-API:67-70).
- Global root `[x, height, z, cos, sin]` ordering (`motionbricks.c:64-69` against UP-API:58).
- XYZW order and Hamilton product in FK (`skeleton.mjs:8-30`, `:39` against UP-API-MAIN:101-103 and UP-API:121).
- 30 FPS (`app.mjs:4` against UP-API:17-18).
- Y-up, metres (`skeleton.mjs:1` against UP-API-MAIN:101).
- Reset of seed is not required by the contract.

## 8. Gaps

- No run was performed (read only). Magnitudes for items 1, 4, 5, 8, 9 and 10 are not measured.
- Root yaw distribution over generated clips is unknown. Needs a dump of output root rotations for N seeds.
- Quality of the ternary bundle against `g1-f32` is not verified in the audited files.
- `tests/native/standing.c` and `scripts/build-component.sh` were not read in full. They do not change the per-frame path.
