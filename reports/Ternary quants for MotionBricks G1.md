# Ternary MotionBricks G1 is unpublished and loader-blocked

No ternary (TQ1_0, TQ2_0, or 1.58-bit) MotionBricks G1 file exists in any source checked. Every MotionBricks G1 GGUF found is F32, including the **LocalAI-io** bundle at a pinned revision ([tree API](https://huggingface.co/api/models/LocalAI-io/MotionBricks-G1-GGML/tree/cc2a47603dbc203a4f18f35dd06ed3611833f506?recursive=true)) and the chibifire copy ([tree API](https://huggingface.co/api/models/chibifire/MotionBricks-G1-GGML/tree/main?recursive=true)). The upstream checkpoints are raw PyTorch FP files. A ternary G1 could be produced, because ggml defines both ternary types with CPU and Vulkan kernels. It cannot run in motion-bricks.cpp as the loader stands, and among the checked tensors the ternary path would save very little.

## ggml has ternary types, but the loader accepts only F32

ggml defines **`GGML_TYPE_TQ1_0` (1.6875 bpw)** and **`GGML_TYPE_TQ2_0` (2.0625 bpw)** in its enum ([ggml.h](https://github.com/ggml-org/ggml/blob/master/include/ggml.h)), with CPU kernels in ggml-cpu.c and Vulkan pipelines in ggml-vulkan.cpp. Mainline llama.cpp gained them in [PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151). The motion-bricks pinned ggml revision could not be checked because GitHub does not serve it.

The motion-bricks loader does not accept these types. `tensor_shape` requires `actual_type == type` or an alternate type (`/workspace/motion-bricks/repo/src/model.cpp:108-120`, check at line 117). The only alternate is BF16, for `_proj_pose_output_logit.weight` (model.cpp:186-187). The VQ codebook goes through `neural_copy_f32`, which rejects non-F32 (model.cpp:285-287; `neural_runtime.cpp:288`). Upstream ggml can run TQ weights, but motion-bricks never routes weights through a non-F32 path.

## Published ternary GGUFs are LLMs, and none measures a motion network

Nearly all public ternary GGUFs are LLMs. Examples include **GeorgyGUF/MAI-DS-R1-tq1_0** (about 143 GB over three shards) and a Llama3-8B TQ1_0 repo ([HF search](https://huggingface.co/api/models?search=TQ1_0)). `ewchampion/Ternary-Bonsai-8B-TQ2_0-GGUF` ships a 2.66 GB TQ2_0 file ([tree API](https://huggingface.co/api/models/ewchampion/Ternary-Bonsai-8B-TQ2_0-GGUF/tree/main)). Prism ML's Bonsai models use a Q2_0 format that its README says is "not yet in mainline `llama.cpp`" ([README](https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/raw/main/README.md)).

The one non-LLM ternary GGUF checked, a FLUX.2 Klein diffusion model, ships **Q2_K** instead. Its README says tq1_0 and tq2_0 "produce degraded output" ([README](https://huggingface.co/biali/bonsai-image-ternary-4B-FLUX2-klein-GGUF/raw/main/README.md)), but gives no metrics. The TerDiT paper's abstract reports no FID numbers ([arXiv 2405.14854](https://arxiv.org/abs/2405.14854)). No quality measurement for a motion or regression network was found, so the quality cost of a ternary G1 is unknown.


> **Correction (2026-10-17):** the eligibility figures below are superseded. A Kaggle probe instantiated the upstream pose model and found 59 `nn.Linear` layers totalling 86,699,520 weights, of which 85,180,416 (98.3%) have an input width that is a multiple of 256, so they are ternary-eligible. The backbone transformer linears (1024 and 2048 input widths) are the bulk. The 0.05% figure came from a handful of shapes checked by hand and was not representative. Root and VQ-VAE layers were not yet inventoried.

## Size: even all-ternary saves little, and few tensors qualify

The F32 bundle holds **183,148,382 parameters, or 732.6 MB** (698.7 MiB) ([README](https://huggingface.co/LocalAI-io/MotionBricks-G1-GGML/resolve/cc2a47603dbc203a4f18f35dd06ed3611833f506/README.md)). Ternary blocks are 256-weight super-blocks, so a tensor's first dimension must be a multiple of 256.

Correction to the framing: the all-ternary figures are a floor, not an upper bound. If every parameter were ternary, the size would be **47.2 MB** at TQ2_0 (66 B per block: 183,148,382 / 256 × 66) and **38.6 MB** at TQ1_0 (54 B per block). Any real mixed file is larger, and 732.6 MB is the F32 ceiling.

Only two checked tensors meet the 256 rule: `_proj_pose_output_logit` {1024, 80} and `_proj_num_token_output_logit` {512, 12} (model.cpp:184-202). They hold 81,920 and 6,144 parameters, about **0.05%** of the total. The other checked shapes fail: `_pose_token_emb` (32), `_conv_output` (3), `decoder.model.6` (3). The notes' example of 80% eligible parameters (about 184 MB) is not supported by these shapes, and the 546 MB `pose.gguf` was not inventoried. Whether ggml asserts the 256 rule is unverified.

## Recommendation: inventory the bundle before building ternary

Producing a ternary G1 is feasible, but the checked evidence says it gains little. A TQ tensor fails the check at model.cpp:117 until that check is relaxed, and only the two projection tensors would shrink. Settling the open questions comes first: a per-tensor inventory of a converted bundle fixes the eligible fraction, which decides whether ternary is worth building. A parity test of the TQ build against F32 outputs then measures the quality cost. Whether motion-bricks graph code runs TQ weights through `ggml_mul_mat` is the last gate.

## Unverified

- The pinned ggml revision `0a249dd4...` is not served, so its enum is unknown.
- Whether ggml asserts the 256-element rule.
- Whether motion-bricks graph code executes TQ weights.
- The per-tensor inventory of `pose.gguf`, `root.gguf`, and `vq-decoder.gguf`, and so the eligible fraction.
- Whether `llama-quantize` handles non-LLM tensor layouts.
- Vulkan `supports_op` coverage for TQ types.
- The DS dataset `yolkarian/motionbricks-checkpoints` was not inspected. Private or unindexed repos are invisible, and GitHub code search may miss LFS files.

## Corrected results (held-out, 100 BONES-SEED clips, full archive QAT, 3000 steps)

Earlier numbers in this report came from a fast-path evaluation bug: PyTorch's fused transformer
path bypassed the ternary forward during held-out evaluation. Those numbers are withdrawn. The
figures below use the real forward, and the addition-only form matches the dense ternary model
exactly.

| Config | Held-out loss | vs full precision (0.743) |
|---|---|---|
| All linears and attention in_proj ternary | 0.780 | +5% |
| + first and last blocks and head kept (int8) | 0.702 | -6% |
| + KL distillation from the FP teacher, kept blocks int8 (shipped) | 0.692 | -7% |
| same, kept blocks fp16 (for reference) | 0.672 | -10% |

Shipped artifact: backbone only, 41.1 MiB (ternary 22.4 MiB at 1.6 bits per weight; int8 18.1 MiB
for kept blocks and other linears; fp16 0.4 MiB), packed codes verified against training.
Compression gains are small (~5%): packed codes are near entropy. The VQ-VAE and root model needed
for the demo are not yet exported or measured.

## Root model: ternary QAT against a full-precision control (two seeds)

The root model is trained without a VQ-VAE (its loss is root regression plus a token-count loss).
The pretrained root scores 3.19 on the held-out set before fine-tuning on BONES, so the control is
the same training run kept full precision, not the pretrained model.

| Seed | Full-precision control | Ternary QAT (shipped) |
|---|---|---|
| 42 | 2.073 | 2.046 |
| 43 | 2.084 | 2.035 |

Ternary matches or slightly beats the full-precision control on both seeds (one run per seed,
100 held-out clips). The 3-D convolution kernels stay F32; the other root linears are ternary
(TQ1_0) or Q8_0 for kept first and last blocks and the output projection.

## Root conv kernels stored F16: precision and cost (measured)

- Held-out loss (100 clips, trained root): F32 conv kernels 2.04628; F16 round-trip of the same
  kernels 2.04628 (difference below the 5-decimal resolution). The round-trip changes the kernels by at
  most 9.5e-4 absolute and 2.1e-4 relative per kernel.
- Cast cost: converting all 21 root conv kernels from F16 to F32 takes about 34 ms per inference on one
  CPU thread (measured in Python; the C runtime's cast is not measured here). The cast runs once per
  conv call, not once per model load.
