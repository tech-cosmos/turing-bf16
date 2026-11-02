# Results

All numbers on one NVIDIA GeForce RTX 2080 Ti (Turing, sm_75, 11 GB), torch 2.15 + CUDA 13.2,
on pentek5. Timings are steady state (warm-up excluded).

## Why: bf16 on Turing is slower than fp32

`bench/baseline.py`, dense `a @ b`, TFLOPS:

| (M, K, N) | fp32 | fp16 (tensor cores) | bf16 |
|---|---|---|---|
| 4096, 4096, 4096 | 12.2 | 56.9 | 7.8 |
| 8192, 3072, 12288 | 13.4 | 55.4 | 7.6 |
| 16384, 5120, 5120 | 13.3 | 55.8 | 7.6 |

bf16 also loses its GEMV path at M = 2-32 (~4x slower than M = 1). Attention is worse:
torch has no fused bf16 attention kernel for sm_75, so SDPA runs the math path, which
materializes the S x S matrix (5 TFLOPS, 4-7 GB at 4-8k tokens, OOM at 16k).

## How

bf16 has 8 significand bits and fp16 has 11, so a bf16 value converts to fp16 exactly
if it is inside fp16's range. Each row (activations) and each output channel (weights)
gets an exact power-of-two scale that puts its max in [0.5, 1), the GEMM runs on fp16
tensor cores with fp32 accumulation, and the scales are undone in the epilogue. For
attention, Q is scaled by 2^a and K by 2^-a (logits unchanged) and V per head.

## Linear (`bench/linear.py`)

| (M, K, N) | torch bf16 TFLOPS | tbf16 TFLOPS | speedup | fp16 ceiling |
|---|---|---|---|---|
| 4096, 4096, 4096 | 6.6 | 51.4 | 7.8x | 57.2 |
| 8192, 3072, 12288 | 8.1 | 52.1 | 6.4x | 57.3 |
| 16384, 5120, 5120 | 8.1 | 53.2 | 6.6x | 58.1 |
| 512, 4096, 4096 | 7.6 | 52.1 | 6.9x | 55.0 |
| 64, 4096, 11008 | 6.8 | 31.6 | 4.6x | 29.8 |
| 1, 4096, 4096 | 0.48 | 0.44 | ~1x (memory bound) | 0.44 |

The remaining ~10% to the fp16 ceiling is the separate quantize and epilogue passes.

## Attention (`tests/test_attention.py`)

| (B, H, S, D) | torch bf16 | tbf16 | fp16 ceiling |
|---|---|---|---|
| 1, 24, 4096, 128 | 5.1 TFLOPS, 3.9 GB | 26.6 TFLOPS, 0.5 GB | 31.3 |
| 1, 12, 8192, 128 | 4.1 TFLOPS, 7.3 GB | 29.5 TFLOPS, 0.5 GB | 32.3 |
| 1, 24, 16384, 64 | OOM | 31.9 TFLOPS, 0.7 GB | 33.4 |

## Accuracy

Relative error vs an fp64 reference from the same bf16 inputs (`tests/test_accuracy.py`):

| case | torch bf16 | naive `.half()` | tbf16 |
|---|---|---|---|
| gaussian | 1.66e-3 | 2.1e-4 | 1.70e-3 |
| outlier tokens x3e4 | 1.65e-3 | NaN/inf | 1.69e-3 |
| all x1e6 | 1.66e-3 | NaN/inf | 1.70e-3 |
| all x1e-8 | 1.66e-3 | 1.0 (underflow) | 1.70e-3 |

The error is the bf16 rounding of the output; tbf16 adds no measurable error of its own.

## Unmodified Hugging Face models (`bench/e2e_llm.py`)

`tbf16.convert(model)`, single 2080 Ti, prefill = one forward over the prompt.

**Gemma 3 4B** (`unsloth/gemma-3-4b-it`), the case where fp16 breaks:

| | prefill 512 | prefill 2048 | prefill 4096 | decode | KL vs fp32 (top-1 agree) |
|---|---|---|---|---|---|
| `model.half()` | | | | | NaN/inf logits |
| native bf16 | 478 ms | 2084 ms | OOM | 11.5 tok/s | 4.7e-4 (99.80%) |
| tbf16 | 144 ms | 484 ms | 1120 ms | 12.1 tok/s | 4.0e-4 (99.51%) |

Greedy generation identical to native bf16 over 128 tokens.

**Qwen3-1.7B**:

| | prefill 512 | 2048 | 4096 | 8192 | decode | peak @4096 |
|---|---|---|---|---|---|---|
| native bf16 | 214 ms | 1002 ms | 2455 ms | OOM | 17.9 tok/s | 6.61 GB |
| tbf16 | 87 ms | 202 ms | 423 ms | 982 ms | 18.7 tok/s | 4.30 GB |

**Gemma 3 1B**: prefill 1.3x / 3.1x / 3.3x / 4.0x at 512 / 2k / 4k / 8k tokens, decode
15.3 vs 14.9 tok/s.

KL vs fp32 for tbf16 is within run-to-run spread of native bf16 on all three models.

## Convolutions (`bench/conv.py`)

cuDNN has no fast bf16 convolution on sm_75 either. `tbf16/conv.py` uses the same trick as
the linear layers: a pow2 scale per output channel for the weight, one per sample for the
input, an fp16 cuDNN conv with fp32 accumulation, and one fused multiply-add back to bf16.
1x1 convs go through `scaled_linear`. Time in ms; error vs fp32 conv of the same bf16 inputs.

| conv (input) | bf16 | fp16 | fp32 | tbf16 | rel err tbf16 / bf16 |
|---|---|---|---|---|---|
| 2d 3x3 512ch, 128^2 (VAE) | 13.2 | 1.65 | 3.88 | 1.86 | 1.68e-3 / 1.66e-3 |
| 2d 3x3 256ch, 256^2 (VAE) | 11.0 | 1.92 | 4.03 | 2.27 | 1.67e-3 / 1.66e-3 |
| 2d 3x3 128ch, 512^2 (VAE) | 11.9 | 2.38 | 4.56 | 3.02 | 1.68e-3 / 1.66e-3 |
| 2d 1x1 2240->11200, 2x32^2 (Sana FFN) | 13.2 | 2.02 | 8.74 | 2.14 | 1.68e-3 / 1.66e-3 |
| 2d depthwise 3x3 11200ch, 2x32^2 (Sana FFN) | 0.62 | 0.42 | 0.60 | 0.97 | 1.69e-3 / 1.66e-3 |
| 3d 3x3x3 384ch, 6x60x104 (Wan VAE) | 41.3 | 5.73 | 24.2 | 6.03 | 1.68e-3 / 1.66e-3 |
| 3d 3x3x3 192ch, 6x240x416 (Wan VAE) | OOM | 26.1 | 138 | 39.7 | 1.67e-3 / - |
| 3d 3x3x3 96ch, 6x480x832 (Wan VAE) | OOM | 39.6 | 193 | 48.0 | 1.67e-3 / - |

bf16 convs are 6-8x slower than fp16, large bf16 Conv3d OOMs (5.8 GB requested at 96ch
6x480x832), and some bf16 convs have no cuDNN engine at all: Sana's DC-AE VAE fails with
"GET was unable to find an engine" on a 5x5 depthwise conv over a channels-last input.
`tests/test_conv.py` checks 1d/2d/3d, groups, strides, padding modes, channels-last,
subclasses with their own `forward` (Wan's causal conv), and inputs that overflow fp16.
Depthwise convs are memory bound and get slower (0.97 vs 0.62 ms). They are still
converted because the native bf16 path can fail outright.

## Diffusion models (`bench/e2e_diffusion.py`)

diffusers 0.42.1, unmodified pipelines, `tbf16.convert(pipe)` (transformer and VAE). Same
prompt, seed and initial noise for every variant, 20 steps with CFG. The prompt is encoded
once and cached, so the text encoder never shares the GPU (UMT5-XXL runs on the CPU).
s/step is the median step after a warm-up step, decode time is after a warm-up decode,
and peak memory is `max_memory_allocated`. `fp16` loads the pipeline with
`torch_dtype=float16` (diffusers keeps its fp32 modules in fp32). The fp32 reference runs
on a second 2080 Ti, and its timings varied by up to 1.5x between runs.

**Wan2.1-T2V-1.3B** (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, CFG 5 = two transformer passes per step):

| frames x H x W (tokens) | native bf16 | tbf16 | fp16 | fp32 | tbf16 vs native |
|---|---|---|---|---|---|
| 17x320x512 (3,200) | 3.72 s/step | 0.74 | 0.79 | 3.17 | 5.0x |
| 17x480x832 (7,800) | 15.2 | 2.71 | 2.51 | 10.9 | 5.6x |
| 33x480x832 (14,040) | OOM | 6.58 | 6.08 | 22.0 | native OOM |
| 81x480x832 (32,760, the default) | OOM | 28.9 | 27.9 | not run | native OOM |

Total for 20 steps + VAE decode, and peak memory:

| | native bf16 | tbf16 | fp16 | fp32 |
|---|---|---|---|---|
| 17x320x512 | 72.8 + 5.9 s, 7.5 GB | 15.2 + 1.3 s, 4.8 GB | 16.0 + 1.3 s, 4.6 GB | 60.5 + 4.4 s, 8.8 GB |
| 17x480x832 | 304 s + VAE OOM, 9.3 GB | 54.1 + 3.4 s, 7.6 GB | 50.4 + 3.2 s, 7.1 GB | 218 s + VAE OOM |
| 33x480x832 | OOM | 131 + 6.8 s, 7.6 GB | 122 + 6.1 s, 7.2 GB | 467 s + VAE OOM |
| 81x480x832 | OOM | 569 + 16.5 s, 7.7 GB | 563 + 14.8 s, 7.3 GB | |

The peaks are set by the VAE decode. Denoising alone peaks at 3.1 / 3.3 / 3.7 / 4.6 GB
with tbf16 (about fp16's) and 4.1 / 9.3 GB natively. Native bf16 attention is the S x S
math path, so it OOMs from 14k tokens. The native bf16 VAE (bf16 Conv3d) and the fp32 VAE
both OOM at 480x832. Those latents were decoded with an fp32 VAE on the CPU (~90 s for 17
frames) so they could still be compared.

Accuracy vs fp32. "One forward" is the transformer on identical inputs (captured from the
fp32 run at step 10), which the sampler does not amplify. PSNR/SSIM are over all decoded
frames after 20 steps:

| | one forward, rel err | 17x320x512 | 17x480x832 | 33x480x832 |
|---|---|---|---|---|
| native bf16 | 8.8e-3 | 26.5 dB / 0.953 | 29.6 dB / 0.967 | OOM |
| tbf16 | 8.9e-3 | 27.0 dB / 0.955 | 29.8 dB / 0.966 | 23.3 dB / 0.903 |
| fp16 | 1.1e-3 | 47.2 dB / 0.998 | 40.7 dB / 0.997 | 39.0 dB / 0.994 |
| tbf16 vs native bf16 | | 33.1 dB / 0.981 | 37.5 dB / 0.987 | |

tbf16 reproduces native bf16, not fp32. It has the same single-forward error, and where
both run it sits as far from fp32 as native bf16 does (closer to native than either is to
fp32). The 20 sampler steps amplify bf16's 8-bit rounding into a visibly different (equally
plausible) sample. fp16 keeps 11 bits, so its trajectory stays near fp32. At 33 frames
there is no native bf16 to compare against. At 81 frames the middle frames of tbf16 and
fp16 are 30.2 dB apart.

![Wan2.1 17x480x832, middle frame](docs/images/wan2.1_17x480x832.png)
![Wan2.1 81x480x832, middle frame](docs/images/wan2.1_81x480x832.png)

**Sana 1600M** (`Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers`, the checkpoint
trained in bf16, DC-AE VAE, CFG 4.5 batched). This run is incomplete: GPU 5 fell off the bus
(Xid 79) during the full 1024/2048 run, so only these numbers exist:

| | native bf16 | tbf16 | fp16 | fp32 |
|---|---|---|---|---|
| 512x512, s/step (4-step smoke run) | 0.258 | 0.110 | 0.057 | 0.175 |
| 1024x1024, s/step | 0.85 | 0.20 (profiled forward) | 0.17 (profiled forward) | 0.60 |
| 2048x2048, s/step | | | | 3.08 |
| VAE decode 1024x1024 | fails (no cuDNN engine) | 0.40 s (profiled) | 0.34 s (profiled) | 0.83 s |
| one forward vs fp32 (512x512) | 2.7e-2 | 2.6e-2 | 5.6e-3 | |

Native bf16 cannot decode at all, because the DC-AE VAE needs a bf16 conv that cuDNN does
not have on sm_75. tbf16 is about 4x faster per step than native at 1024px (wall time vs a profiled
forward) but 17% slower than fp16, and 2x slower at 512px. With 256 tokens per image the GEMMs are small and the
quantize/epilogue/scaling passes dominate. fp16 gave finite output at 512px. 1024px in fp16
was not verified before the GPU failed.

![Sana 1600M, 512px, 4 steps (smoke run)](docs/images/sana1600m_512px_4steps.png)

**`.half()`**: neither model broke in fp16. Wan2.1 1.3B and Sana 1600M (at 512px) give
finite, normal-looking output in fp16, closer to fp32 than bf16. For these two models, fp16
is the faster and more accurate choice when it is available. tbf16's job here is to run a
bf16 checkpoint as published, at near-fp16 speed, without having to validate fp16 range
first. A model that actually overflows fp16 is still to be found and tested (SD3.5 is gated,
and Flux does not fit on 11 GB).

**What broke and was fixed** (tests in `tests/test_conv.py`, `tests/test_sdpa_api.py`):

- diffusers calls `F.scaled_dot_product_attention(query=..., key=..., value=...)` with
  keywords. tbf16's replacement only took positional `q, k, v` (TypeError), and `patch()`
  indexed `args[:3]`. Both now accept torch's signature.
- fp32 additive masks with bf16 q failed in the fp16 kernel. Any non-fp16 float mask is now
  cast.
- bf16 convs: slow, OOM or missing (see above). Now `tbf16.conv`, converted by `convert()`.
- `tbf16.convert(pipe)` takes a diffusers pipeline and converts every `nn.Module` component.

These were checked and worked as they were: permuted (non-contiguous) q/k/v, head dims 64,
112 (Sana cross-attention) and 128, and bool or additive masks broadcast over heads and
queries with key length 300. Wan's LayerNorms already run in fp32 in diffusers, and Sana's
ReLU linear self-attention is computed in fp32 by diffusers, so neither is affected.
Norms, GELU and RoPE are memory bound and were not bottlenecks.

**Open problems**:

- Sana at 1024/2048 for 20 steps with tbf16 and fp16, and PSNR at those sizes (interrupted
  by the GPU failure). `python -m bench.e2e_diffusion --model sana --sizes 1024x1024,2048x2048 --png 1024x1024`.
- Test a DiT that really overflows in fp16.
- The overhead vs fp16 at small token counts (Sana 512px: 2x). The separate
  quantize/epilogue/scaling passes would have to be fused into neighbouring ops.
- `convert()` only converts modules that are already on the GPU, so it does not combine
  with `enable_model_cpu_offload()`.

## Dead ends and notes

- Triton 3.9 emits no `mma` for sm_75 (`tl.dot` falls back to FMAs, ~0.2 TFLOPS), so the
  GEMM itself is cuBLAS fp16. Triton is fine for the elementwise kernels.
- Triton's Python launcher costs ~45 us per call on pentek5's 2.2 GHz Xeons, which made
  decoding 30% slower than native. The memory-bound kernels are plain CUDA now.
- `patch()` (TorchFunctionMode) is convenient but caches an fp16 copy of every weight
  and adds per-op Python overhead; `convert()` is the real path.
