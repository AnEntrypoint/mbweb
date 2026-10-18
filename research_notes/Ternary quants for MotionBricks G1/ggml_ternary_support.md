# Ternary quant support in GGML / llama.cpp for MotionBricks G1

## Which ternary types exist

### Takeaway
GGML has two ternary types, `GGML_TYPE_TQ1_0` (1.6875 bpw) and `GGML_TYPE_TQ2_0` (2.0625 bpw). Both are in the upstream enum and have CPU and Vulkan kernels. The motion-bricks pinned ggml commit could not be checked, because GitHub does not serve it.

### Cited Findings
- Enum: `GGML_TYPE_TQ1_0 = 34`, `GGML_TYPE_TQ2_0 = 35`, `GGML_TYPE_COUNT = 43` in ggml.h, ggml-org/ggml default branch (commit a05c9922, 2026-10-15) — https://github.com/ggml-org/ggml/blob/master/include/ggml.h (lines 424-425, 433)
- Block layouts: `block_tq1_0` (qs + qh + ggml_half d, 54 bytes per 256 weights, "1.6875 bpw") and `block_tq2_0` (qs[64] + ggml_half d, 66 bytes per 256 weights, "2.0625 bpw") — ggml-common.h, ggml-org/ggml (~lines 272-288)
- CPU type traits: `[GGML_TYPE_TQ1_0]` and `[GGML_TYPE_TQ2_0]` map to `quantize_row_tq*` and `ggml_vec_dot_tq*_q8_K`, vec_dot_type Q8_K — ggml-cpu/ggml-cpu.c, ggml-org/llama.cpp (lines 401-411)
- Quantizers: `quantize_tq1_0` and `quantize_tq2_0` in ggml-quants.c (lines 2415, 2422) — https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
- Vulkan: pipelines exist for `mul_mat_vec_tq1_0/tq2_0` (f32 and f16 activations), `mul_mat_vec_id`, `dequant_tq*`, and `get_rows_tq*` (ggml-vulkan.cpp lines 2944-2945, 2973-2974, 3030-3031, 3098-3099, 3128-3129) — https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp
- llama.cpp wiring: `LLAMA_FTYPE_MOSTLY_TQ1_0/TQ2_0` map to the ggml types (llama-quant.cpp lines 934-935, 565). Latest release is b11714, 2026-10-17.
- Vulkan support-op switch (ggml-vulkan.cpp ~5542) lists TQ types; whether this is `supports_op` for all ops is unverified.

### Gaps
- Pinned revision `0a249dd4d2f4dfcd962123a3d881f73f93588ccf` (submodule `ggml` in /workspace/motion-bricks/repo, `git ls-tree HEAD ggml`) returns 404/422 on ggml-org/ggml. The revision is not served, so whether its enum has TQ types is unverified. The nearest checked commit is ggml-org/ggml default branch a05c9922.
- The `upstream_revision` string `a0732b64...` in model.cpp:28 is also not on GitHub. It is a MotionBricks metadata value, not the ggml pin.
- The gm codesearch dispatch stalled in the daemon (queued, never claimed), so the local search was done with direct reads.

## Loader constraints in motion-bricks

### Takeaway
The loader rejects any non-F32 tensor it checks, except one BF16 alternate. Upstream ggml can run TQ types, but motion-bricks never routes weights through a non-F32 path.

### Cited Findings
- `tensor_shape` requires `actual_type == type` or the alternate type (/workspace/motion-bricks/repo/src/model.cpp:108-120; check at line 117).
- Only BF16 alternate: `_proj_pose_output_logit.weight` (model.cpp:186-187).
- The VQ codebook is copied with `neural_copy_f32`, which rejects non-F32 (model.cpp:285-287; neural_runtime.cpp:288).
- Checked shapes (model.cpp:184-202): `_pose_token_emb.weight {32,88}`, `_position_emb.embed {1024,1,16}` and `{512,1,16}`, `_proj_pose_output_logit.weight {1024,80}`, `_proj_num_token_output_logit.weight {512,12}`, `_conv_output.model.6.weight {3,512,5}`, `decoder.model.6.weight {3,512,413}`, `quantizer.vq._codebook.embed {32,10,8}`. The `support` component (neutral_joints, motion_mean, motion_std) is F32 and joint_parents is I32.
- Eligibility: TQ blocks are 256-element super-blocks, so ne[0] must be a multiple of 256. Eligible: `_proj_pose_output_logit` (1024) and `_proj_num_token_output_logit` (512). Not eligible: `_pose_token_emb` (32), `_conv_output` (3), `decoder.model.6` (3). Whether ggml asserts this is unverified.

### Inferences
- A TQ tensor would fail the loader check at model.cpp:117 unless the check is relaxed. Graph execution for other tensors, and whether `ggml_mul_mat` takes TQ weights through the runtime, is unverified.

## Size estimate (183,148,382 params)

### Cited Findings (arithmetic)
- F32 total: 183,148,382 x 4 B = 732.6 MB (698.7 MiB).
- All params TQ2_0: 183,148,382 / 256 x 66 B = 47.2 MB (45.0 MiB). Upper bound, not reachable.
- All params TQ1_0: 183,148,382 / 256 x 54 B = 38.6 MB (36.8 MiB). Upper bound, not reachable.
- Realistic mix (unverified): size = eligible_params x bpw / 8 + kept_F32_params x 4. Example with 80% eligible in TQ2_0: 146.5M x 2.0625 / 8 + 36.6M x 4 = 37.8 MB + 146.5 MB = ~184 MB.

### Gaps
- No local GGUF bundles exist, so the per-tensor inventory and the eligible fraction are unverified.
