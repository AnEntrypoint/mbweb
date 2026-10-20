# Animation clips

Clips are text-prompted motions made by Kimodo (the G1 model in kimodo.cpp) and played by the page. The pipeline
turns a prompt list into `web/model/clips/*.clip`, which `scripts/build-site.sh` publishes. Only the clips and
their manifest are committed: CI serves them and never needs the weights.

To add an animation class, add an entry to `clips.json` and run the build.

```sh
animation/fetch.sh                       # once: pinned kimodo.cpp, verified G1 + text encoder weights, CPU build
node animation/build-clips.mjs           # generates clips whose entry is new or changed
node animation/build-clips.mjs jump_up   # or only the named clips
```

`clips.json` entries are `{ "name", "prompt", "frames", "seed"?, "steps"? }`. Names are snake_case and become the
file names. Frames are at 30 FPS. Diffusion defaults to 100 steps, the upstream demo's default.

## Files

| Path | What it is |
| --- | --- |
| `pin.env` | The kimodo.cpp commit, the motion model and the text-encoder quantization. Bump them together. |
| `fetch.sh` | Clones and builds the pinned kimodo.cpp (`kmd-generate`, CPU) and runs the weight fetch. |
| `fetch-weights.mjs` | Downloads the weights into `.kimodo/weights`, resuming partial files and checking each against its manifest. |
| `clips.json` | The clip catalogue: the prompts the page plays. |
| `build-clips.mjs` | Runs `kmd-generate --server` for the catalogue and packs the output into `web/model/clips/`. |
| `../web/model/clips/` | Committed output: one `.clip` per entry and `manifest.json` with the prompt, seed, steps, size, SHA-256 and checks. |

Everything under `.kimodo/` (sources, weights, raw output) is git-ignored. Weights are about 6 GB with the default
q4_k_m encoder (1.1 GB motion model and 5.1 GB encoder).

## Clip format

`.clip` is little-endian float32, one frame after another, 139 values per frame: root X, Y, Z in metres (Y up), then
the 34 G1 joint rotations as XYZW, in the same joint order as the model (`mb_joint_parents`). Rotations are unit
quaternions. Kimodo's native output needs no retargeting: `build-clips.mjs` only packs it and normalises the
quaternions. The manifest records `max_root_step_m` and `max_rotation_norm_error` per clip as a sanity check.

## Playing a clip

The page merges a clip into its reference timeline the way the upstream demo plays Kimodo animations: the clip starts
at the frame on screen, turned to that frame's heading and moved onto its root, and blended in over eight frames. A
plan is not requested while the clip plays. When the clip ends the walk resumes from the clip's last four frames
with a fresh agent. The **Clip** control plays the chosen clip; **J** plays `jump_forward`.

## Limits

- The text encoder is q4_k_m. Q8_0 (the upstream default) is 8.1 GB and was not used here; the pin file selects it.
- Only G1 is supported. SOMA and SMPL-X outputs have different skeletons and need their own conversion.
- Clips are open-loop. Nothing checks that a generated jump clears an obstacle; look at the output.
