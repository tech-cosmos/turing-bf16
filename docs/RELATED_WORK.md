# Related work

Survey of projects and techniques related to tbf16, as of 2026-11-02. Links were checked on that
date (GitHub via `gh`, other pages via HTTP). "Overlap" says how close each item is to tbf16.

## Positioning

**Nothing we found does what tbf16 does as a general library.** That means exact per-row/per-channel
power-of-two scaling of bf16 tensors into fp16, for every dense linear and SDPA call, in any
unmodified bf16 model, with no calibration.

The core observation is not new, though. Several projects from the last few weeks state it
independently:

- **WanGP** (deepbeepmeep, 11k stars), from 2026-10-21. Its Turing-only kernels run bf16
  convolutions on fp16 tensor cores using the same exactness argument and power-of-two scales. Its
  fp8 GEMM writes bf16 activations as fp16 with one power-of-two scale per row and 128-element
  block. This is the closest prior art; it predates our commits.
- **Strata PR #593** (2026-10-03, closed). It routes bf16 GEMMs through fp16 on sm_70/75 and notes
  the conversion is exact inside fp16's range, but it does no scaling.
- **fp16-fixer, MOSS-Image-FP16-fixed and minimax-h3-fp16-fix** (all October 2026). They make
  specific models fp16-safe with static power-of-two rescaling, found by calibration or by hand.

The power-of-two per-row/per-block scale itself is standard in low-precision training and
inference (DeepSeek-V3/DeepGEMM UE8M0, OCP MX E8M0, Transformer Engine block scaling, torchao
`round_scales_to_power_of_2`). The Q·2^a / K·2^-a attention fold is a small trick we did not find
written down elsewhere.

What is genuinely distinct in tbf16:

1. **Dynamic, per-call scales.** No calibration, no per-model plan, no clamping, no weight edits.
   It works for activations that calibration did not see.
2. **Coverage as a drop-in.**
   - Dense `nn.Linear` through cuBLAS fp16.
   - SDPA routed to torch's fp16 memory-efficient kernel, which takes torch's bf16 attention on
     sm_75 off the math path that materializes S x S.
   - `convert(model)` for Hugging Face models.
3. **Measured end-to-end LLM speed and accuracy**, including Gemma 3 4B. vLLM refuses fp16 for that
   model, and plain `.half()` gives NaN.

Who benefits:

- Users with a T4 (Colab, Kaggle, cloud), RTX 20xx, TITAN RTX or Quadro RTX who run bf16-only
  checkpoints. Today those land on fp32 (vLLM for Gemma 2/3, ComfyUI "manual cast: float32"), on
  slow emulated bf16 (plain PyTorch), or on lossy clamp hacks.
- Probably also V100 (sm_70), which has the same fp16-only tensor cores. Untested by us; WanGP,
  Strata and fp16-fixer all target it.

## 1. Direct prior art: bf16 compute on fp16 tensor cores

| Project | What it does | Overlap / difference |
|---|---|---|
| **WanGP** `shared/kernels/scaled_fp16.py`, `fp8_block_triton.py`, `bf16_bits.py` ([repo](https://github.com/deepbeepmeep/WanGP), first commit of these files 2026-10-21, very active, "WanGP Community License") | Video-generation app. On sm_75 only:<br>- bf16 Conv1d/2d/3d run through cuDNN fp16. The weight is scaled to max in [0.5, 1); the input gets a power of two chosen from max\|x\| x max channel L1 norm, so no output overflows. Reports 4.8-6.8x on the LTX-2.5 VAE.<br>- The fp8 W8A16 Triton GEMM writes bf16 activations as fp16 with a power-of-two scale per row and 128-element K block.<br>- bf16-as-bits Triton norms, and a paged Q8 attention on fp16 MMAs. | **Same idea and the same exactness argument.** Their docstring also notes the double rounding (fp16 output, then bf16).<br>Differences: tied to the app; covers convs, quantized linears and their own attention. We found no dense bf16 linear or SDPA path there. |
| **Strata PR #593** ([link](https://github.com/Niko1221/Strata/pull/593), 2026-10-03, closed) | C++ LLM engine. Converts W and X to fp16 for bf16 GEMMs on sm_70/75; "the conversion is exact for every BF16 value inside FP16's range". V100 prefill +13-15%. | Same swap **without scaling**: it only counts overflows under a debug flag. |
| **unsloth-zoo** `fp16_emulation.py` ([file](https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/fp16_emulation.py), 2026-09, AGPL, off by default, unused) | Emulates **fp32** matmul with split fp16 terms (3 GEMMs) and one power-of-two scale per tensor. Its comment says per-row/per-block scaling would be the fix for wide-range tensors. | Different target (fp32 accuracy at about 3x cost). Same power-of-two-into-fp16 building block. |
| **Unsloth, Gemma 3 on T4** ([docs](https://unsloth.ai/docs/basics/gemma-3-how-to-run-and-fine-tune), 2025-03 onward) | Keeps activations in bf16/fp32, runs matmuls in fp16 "without autocast", and norms in fp32. In code, weights are cast straight to fp16 and RMSNorm outputs are clamped to +-65504. | Same goal (fp16 tensor cores, bf16 storage). Unsloth uses no scaling and clamps instead; the patches are per model. |

Not overlapping, but the same niche: [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
(about 1.2k stars, active in 2026) is a Turing-specific vLLM fork focused on quantized weights
(FP8/NVFP4/INT4 W*A16, Marlin, FlashInfer).

## 2. How bf16-only models run on Turing today

| Stack | Behaviour on sm_75 | Source |
|---|---|---|
| PyTorch | `cublasGemmEx` with `CUDA_R_16BF` and no Turing special case. cuBLAS uses non-tensor-core kernels ("emulate it via float32 compute", ptrblck). `torch.cuda.is_bf16_supported()` returns **True** by default (`including_emulation=True`). bf16 SDPA falls to the math backend because the mem-efficient kernel only allows fp16/fp32 below sm80. We found no open PyTorch issue asking for fast bf16 on Turing. | [forum](https://discuss.pytorch.org/t/bfloat16-on-nvidia-v100-gpu/201629), [PR #115924](https://github.com/pytorch/pytorch/pull/115924), [PR #129288](https://github.com/pytorch/pytorch/pull/129288), [PR #116026](https://github.com/pytorch/pytorch/pull/116026) |
| vLLM | `supported_dtypes` is fp16/fp32 below sm80. With `dtype=auto`, bf16 models fall back to fp16. `_FLOAT16_NOT_SUPPORTED_MODELS` (gemma2, gemma3, gemma3_text, glm4) removes fp16, so **Gemma 3 on a T4 runs in fp32**. | [config/model.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py) |
| llama.cpp | bf16 weights below Ampere are converted to fp32 for cuBLAS SGEMM when the batch is over 8; smaller batches use the slow bf16 path. There is no fp16 route for bf16. Its FlashAttention kernels on Turing cast K/V to fp16 without scaling. | `ggml/src/ggml-cuda/` (source read, not linked) |
| ComfyUI | Models without fp16 in `supported_inference_dtypes` (HunyuanVideo, LTXV, Mochi, Qwen-Image, Lumina2, ...) use "manual cast: float32" on Turing. `--force-fp16` gave 3.37x on a 2080 Ti for Z-Image. Flux/Lumina/Qwen paths use `nan_to_num`/clamp hacks. | [repo](https://github.com/Comfy-Org/ComfyUI) `comfy/model_management.py`; [ai-muninn blog, 2026-07](https://ai-muninn.com/en/blog/2080ti-turing-fp32-fallback) |
| Forge / Forge Neo | `fp16_fix(x) = x.clip(-32768, 32768)` for Flux. A Turing extension forces fp16 and warns of "black or burnt" outputs. | [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), [forge-neo-turing-fp16](https://github.com/eduardoabreu81/forge-neo-turing-fp16) |
| XLA | Upcasts bf16 dots to f32 on T4 (bf16 there "is *incredibly* slow"). | [openxla/xla#12429](https://github.com/openxla/xla/issues/12429) |
| TileLang | PR routes sm_75 bf16 to a CUDA-core FMA fallback. | [PR #2811](https://github.com/tile-ai/tilelang/pull/2811) |
| TRL / transformers / accelerate / DeepSpeed | All trust `is_bf16_supported()`, so `bf16=True` is accepted on Turing and runs emulated. TRL's default `bf16=True` breaks on T4/V100/RTX 20xx. | [trl#7889](https://github.com/huggingface/trl/issues/7889) (open, 2026-10) |

**fp16 overflow fixes per model.** Each of these is lossy or model-specific:

- **Gemma.**
  - Gemma 1/2/3 all overflow in fp16 ([gemma_pytorch#10](https://github.com/google/gemma_pytorch/issues/10), [transformers#36822](https://github.com/huggingface/transformers/issues/36822)).
  - The clamp fix [PR #36832](https://github.com/huggingface/transformers/pull/36832) (not merged) measured Gemma-3-12B lm-eval falling from 0.752 (bf16) to 0.657 (fp16 + clip), with GSM8K at 0.874 → 0.647. This is a useful number to compare against.
  - [RFC PR #37226](https://github.com/huggingface/transformers/pull/37226) (closed) folds 1/64 into the embedding and post-norm weights, which is static and exact for Gemma only.
- **T5.** Clamp hidden states to `finfo.max - 1000` ([PR #9487](https://github.com/huggingface/transformers/pull/9487), 2021, still in `modeling_t5.py`), and keep `wo` in fp32 ([PR #20683](https://github.com/huggingface/transformers/pull/20683)).
- **diffusers.**
  - `force_upcast` runs the VAE in fp32; `upcast_attention`/`upcast_softmax` handle attention.
  - Flux fp16 differences come from the text encoders' range ([PR #9097](https://github.com/huggingface/diffusers/pull/9097)).
  - Flux Fill gives black images in fp16 ([#10533](https://github.com/huggingface/diffusers/issues/10533)).
- **Why models overflow.** Outlier features ([LLM.int8()](https://arxiv.org/abs/2208.07339)) and "massive activations" about 10^5 times the median ([Sun et al. 2024](https://arxiv.org/abs/2402.17762)).

## 3. Static rescaling to make a model fp16-safe

These change the model or need calibration. tbf16 scales every call at run time and changes no weights.

| Project | What it does |
|---|---|
| [fp16-fixer](https://github.com/christopherrobertbrooks-tech/fp16-fixer) (2026-10-20, 0 stars, V100 only) | Records fp32 and fp16 runs, then plans power-of-two rescaling across scale-free boundaries (residual stream / 2^k with norm eps / 4^k, attention scale folded into q_norm) or fp32 islands. Applied as hooks.<br>Results: Gemma 3 4B with stream / 32 gives 100% top-1 agreement and 0.13-0.19% logit error; it also fixes umT5-XXL and ACE-Step.<br>Limits: only covers values seen in calibration, and is tied to one attention implementation. |
| [KBlueLeaf/MOSS-Image-FP16-fixed](https://huggingface.co/KBlueLeaf/MOSS-Image-FP16-fixed) (2026-10) | Residual stream scaled by 1/16, baked into the weights. |
| [trothe/minimax-h3-fp16-fix](https://github.com/trothe/minimax-h3-fp16-fix) (2026-10-25)<br>[ComfyUI#16507](https://github.com/Comfy-Org/ComfyUI/issues/16507) (open)<br>[fengyuan-zhu.com blog](https://fengyuan-zhu.com/v100/minimax-h3/) (2026-08) | MiniMax H3 in fp16 on V100/T4:<br>- fp32 residual stream.<br>- `f(x/K)*K` with fixed K = 32-256 around the projections.<br>- Per-row power-of-two scaling for adaLN.<br>About 11x faster than fp32 fallback. |
| [Draw Things, "BF16 and image generation models"](https://engineering.drawthings.ai/p/bf16-and-image-generation-models-803cf0515bee) (2025-04) | Runs bf16-trained FLUX, Hunyuan and HiDream in fp16. MLP down-projections are scaled by 1/4-1/8 and attention is pre-scaled, with factors hand-tuned per layer. |
| [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) (2023) | Finetuned the SDXL VAE so internal activations fit fp16. A different checkpoint with slightly different outputs. |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | Per-channel scale migrated into LayerNorm and weights. Same algebra, but the target is int8, not fp16 range. |

## 4. Power-of-two and fine-grained scaling (training/quantization)

These share the structure "exponent-only scale per row/block, undone after the GEMM". In all of them
the low-precision format is lossy and the scale minimizes the loss. In tbf16 the scale only fixes
range; there is no rounding, because bf16's 8 significand bits fit in fp16's 11.

| Item | Note |
|---|---|
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (2024-12), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | FP8 with 1x128 activation tiles and 128x128 weight blocks, promoting to fp32 every 128 K. V3.1 uses UE8M0 (power-of-two) scales ([model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)). Their scales vary along K, so they are applied inside the K loop; ours are constant along K and fold into the epilogue. |
| [OCP MX spec v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [Rouhani et al. 2023](https://arxiv.org/abs/2310.10537) | E8M0 shared scale per 32 elements (MXFP8/6/4). The same scale type as ours, but along K and with hardware support only on sm_100+. |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Recipes:<br>- `DelayedScaling` and `Float8CurrentScaling`, with optional power-of-two scales (`NVTE_FP8_CURRENT_SCALING_POWER_2_SCALES`).<br>- `MXFP8BlockScaling`.<br>- `Float8BlockScaling`, power of two by default; Blackwell requires it. |
| [torchao float8](https://github.com/pytorch/ao) | Tensorwise or rowwise scaling. `Float8LinearConfig.round_scales_to_power_of_2` exists "to reduce quantization error by avoiding rounding errors when multiplying/dividing by the scaling factor". This is the same reason tbf16 uses powers of two. |
| [Unit Scaling](https://arxiv.org/abs/2303.11257) (2023), [u-μP](https://arxiv.org/abs/2407.17465) (2024), [library](https://github.com/graphcore-research/unit-scaling) | Static per-op scales, chosen at training time, so fp16/fp8 training needs no loss scaling. Requires training the model that way; tbf16 works on any pretrained checkpoint. |
| Loss scaling: [Micikevicius et al. 2017](https://arxiv.org/abs/1710.03740), [adaptive per-layer](https://arxiv.org/abs/1910.12385), [S2FP8](https://arxiv.org/abs/2001.05674) | Standard fp16 training on Turing. Scales gradients per tensor or per layer for range. We found no "bf16 training on Turing" project or per-row scaled fp16 training framework. |

**Running a format on a narrower unit.** These split values into several limbs, because the source
mantissa is wider than the unit's:

- fp32 on fp16 tensor cores ([Markidis 2018](https://arxiv.org/abs/1803.04014), [Ootomo & Yokota 2022](https://arxiv.org/abs/2203.03341)).
- fp32 as 3 bf16 values ([Henry et al. 2019](https://arxiv.org/abs/1904.06376)).
- The Ozaki scheme, and cuBLAS fp32/fp64 emulation ([NVIDIA blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)). cuBLAS's fixed-point fp64 emulation uses "a shared power-of-2 scaling factor" per row of A and column of B, which is structurally the same as tbf16.
- XLA's `ALG_DOT_BF16_BF16_F32_X3/X6/X9`.

tbf16 is the degenerate case: bf16 fits inside fp16's significand, so one GEMM and one scale per
row is enough.

## 5. Attention and kernels on Turing

| Item | Status on sm_75 | Relevance |
|---|---|---|
| FlashAttention 1 | Supported Turing in fp16 only ("bf16 requires Ampere"). | Superseded. |
| FlashAttention 2/3 | Ampere+ only. [Issue #542](https://github.com/Dao-AILab/flash-attention/issues/542) "Turing architecture support" (2023) is still open. The README now points to ssiu's port. | n/a |
| [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) | FA2 port, 132 stars, last push 2026-09-14. Forward and backward, head dims 64/96/128, causal, GQA, varlen. Claims up to about 2x torch SDPA forward on T4. | **fp16 only, and it does not check dtype** (bf16 input would be misread). No LICENSE file. A candidate to replace torch's mem-efficient kernel behind our scaling. |
| torch SDPA | Mem-efficient: fp16/fp32 only below sm80. Flash and cuDNN: sm80+. | What tbf16 uses today. |
| xformers | Kernels now come from [MSLK](https://github.com/meta-pytorch/MSLK) and wrap torch's mem-efficient kernel; bf16 is rejected below sm80. | Nothing beyond torch SDPA. |
| [SageAttention](https://github.com/thu-ml/SageAttention) | Sage2 and SpargeAttn: sm80+ only. Sage1 (Triton, INT8 QKᵀ) works on sm_75 only with Triton ≤ 3.2, about 68-74 vs 27-30 TFLOPS on a 2080 Ti ([#197](https://github.com/thu-ml/SageAttention/issues/197)). Sage1 casts bf16 V straight to fp16 without scaling. | Lossy but fast. Needs old Triton. |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | Lists SM 7.5. Uses `mma.m16n8k8` f16 on sm75; bf16 MMA only for sm80+. Apache-2.0. | fp16 prefill/decode with a paged KV cache. Reusable behind our scaling. |
| llama.cpp `fattn-mma-f16` | Turing fp16 MMA attention; casts K/V to fp16. MIT. | Tied to ggml. |
| Triton | [PR #5066](https://github.com/triton-lang/triton/pull/5066) (merged 2024-11-05) "Deprecate MMAv1; fallback to FMA for computeCapability < 80". Triton 3.3+ (torch 2.7+) has no tensor-core `tl.dot` on sm_75. [PR #9970](https://github.com/triton-lang/triton/pull/9970) "Restore MMA support for Turing GPUs" was closed unmerged on 2026-04-09 (no sm75 CI or maintainer). [triton-windows](https://github.com/woct0rdho/triton-windows) says Turing is supported by Triton ≤ 3.2. | Confirms our Triton dead end. Inductor GEMM templates and FlexAttention also hit the FMA path on Turing. |

**Prior art for the attention fold.** Megatron's `apply_query_key_layer_scaling`, GPT-2's
`reorder_and_upcast_attn` and GPT-BigCode's `attention_softmax_in_fp32` all avoid fp16 overflow in
attention, but none is an exact power-of-two counter-scaling of Q and K.

## 6. Complementary: GeForce half-rate unlock

GeForce Turing runs fp16 tensor-core math **with fp32 accumulation at half rate**. That is the mode
tbf16 uses, so our "fp16 ceiling" of about 56 TFLOPS on the 2080 Ti is the capped rate. T4, TITAN
RTX and Quadro RTX run it at full rate.

- [nlzy/turing-unlock](https://github.com/nlzy/turing-unlock) (2026-10-24, 340 stars, GPL-2.0) is
  a Linux kernel module that lifts the cap on TU102. It reports cuBLAS fp16/fp32-acc going from
  52.8 to 77.9 TFLOPS at 8192³ on a 2080 Ti at 250 W
  ([independent benchmark](https://ai-muninn.com/en/blog/turing-unlock-2080ti-fp16-tensor-core):
  50.6 to 69.9 at 200 W).
- [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) (14k stars) does the same for RTX
  20/30/40; Turing support is marked experimental.

tbf16's GEMMs should benefit directly. Native torch bf16 on Turing does not, because it runs no
tensor-core path.

## Implications for the README

- **Qualify "lossless".** The bf16 → fp16 conversion is exact only for elements within 2^14 of their
  row's max in the linear path, or within about 2^28 for attention, where Q, K and V are scaled up
  to 2^14. Smaller ones become fp16 subnormals and lose bits, and the smallest flush to zero. The linear output is also rounded to fp16 (11 bits) before bf16, which
  is double rounding. The measured error matches native bf16; say that, rather than "bit-exact".
- **Don't claim the observation as new.** Credit WanGP, Strata #593 and fp16-fixer. The claim that
  holds up is "a general, calibration-free drop-in for linear + SDPA".
- **Cite the concrete gap.** vLLM runs Gemma 2/3 in fp32 on T4; ComfyUI uses "manual cast: float32";
  `is_bf16_supported()` returns True on Turing and hides the slow path.
- **Add a baseline row** for a clamp fix (PR #36832-style) or for fp16-fixer's Gemma 3 4B numbers,
  and an lm-eval run, since PR #36832 shows clamping costs 9 points.
- **Explain the 2080 Ti ceiling** (GeForce half rate) and mention turing-unlock. Get T4 numbers,
  which run at full rate but are power-limited to 70 W.

## Ideas worth adopting

1. **Fused epilogue on sm_75.**
   - Option A: vLLM's CUTLASS 2.x `scaled_mm_c2x` sm75 dispatch (int8, `Sm80EVT` with row/column
     broadcast visitors). Retargeted to fp16 inputs (16x8x8 MMA), it gives row scale x column scale
     + bias → bf16 in one kernel, removing our separate epilogue pass. That is part of the
     remaining ~10% gap.
   - Option B: cuBLASLt `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO` folds one of the two
     scale vectors into the GEMM. Whether sm_75 fp16 heuristics support it is untested.
2. **Avoid fp16 GEMM output.** `torch.mm(..., out_dtype=torch.float32)` (torch ≥ 2.8, fp16 inputs)
   or `cublasGemmEx` with fp32 C removes both the K·4^t output range constraint and the double
   rounding, at the cost of epilogue bandwidth. A CUTLASS epilogue would get both benefits for free.
3. **Faster attention.** Put ssiu/flash-attention-turing (about 2x mem-efficient on T4; needs a
   licence and a dtype guard) or FlashInfer's sm75 fp16 kernels behind the same Q/K/V scaling.
   FlashInfer would also give a paged KV cache for serving.
4. **Store scales as int8 exponents** (E8M0-style, as in DeepGEMM `ceil_to_ue8m0`) and apply them
   with `ldexp`. This halves scale memory and keeps the scaling exact by construction.
5. **Per-K-block scaling** (DeepSeek/WanGP-style 128-element blocks) as a fallback for rows whose
   dynamic range exceeds 2^14, if subnormal flushing ever shows up in a model.
6. **Convolutions** following WanGP's `scaled_fp16.py` bound (input scale from max|x| x the max
   channel L1 norm). This would extend tbf16 to diffusion VAEs and video models, which are a large
   part of the Turing/ComfyUI audience.
7. **Volta (sm_70)** should work with the same code: fp16 tensor cores, and the same no-bf16 cuBLAS
   path. It is cheap to add to the JIT targets and test.
