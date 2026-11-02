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

Forward, with the fused per-head amax and scale-cast kernels:

| (B, H, S, D) | torch bf16 | tbf16 | fp16 ceiling |
|---|---|---|---|
| 1, 24, 4096, 128 | 5.1 TFLOPS, 3.9 GB | 30.6 TFLOPS, 0.5 GB | 31.4 |
| 1, 12, 8192, 128 | 4.1 TFLOPS, 7.3 GB | 31.8 TFLOPS, 0.5 GB | 32.4 |
| 1, 24, 16384, 64 | OOM | 33.0 TFLOPS, 0.7 GB | 33.2 |
| 4, 32, 2048, 128 | 5.0 TFLOPS, 5.4 GB | 28.4 TFLOPS, 0.8 GB | 29.5 |

Forward + backward, causal (`tests/test_autograd.py`):

| (H, S, D) | torch bf16 | tbf16 |
|---|---|---|
| 16, 2048, 128 | 17.3 ms, 1.78 GB | 5.5 ms, 0.82 GB (3.2x) |
| 16, 4096, 64 | 52.6 ms, 4.78 GB | 7.6 ms, 0.82 GB (6.9x) |

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

## Training (`bench/train_llm.py`, `tests/test_autograd.py`)

`tbf16.convert(model, train=True)`. Gradients (dX, dW, db, dQ, dK, dV, with GQA and
bool/float masks) match torch's bf16 autograd against an fp64 reference, including
gradients scaled by 1e-12 (fp16 would flush them to zero) and by 1e8. Linear fwd+bwd runs
at 48-50 TFLOPS against torch bf16's 8 (6.0-6.4x).

Recipes, same init and same batches, two runs at a time on the 2080 Tis
(raw data: `results/train_final.jsonl`):

- **bf16**: fp32 master weights + `torch.autocast(bf16)`, the usual bf16 mixed precision.
- **bf16+tbf16**: the same, after `tbf16.convert(model, train=True)`.
- **fp16-amp**: fp32 master + `autocast(fp16)` + `GradScaler`, the usual Turing recipe.

| task | bf16 | bf16+tbf16 | fp16-amp | fp32 | eval loss (bf16 / tbf16 / fp16) |
|---|---|---|---|---|---|
| 75M byte-level LM from scratch, 4 x 1024 tokens | 527 ms, 6.16 GB | 218 ms (2.4x), 3.35 GB | 142 ms, 3.27 GB | 347 ms | 1.8183 / 1.8167 / 1.8168 |
| LoRA r=16 all-linear, Qwen3-1.7B, 2 x 512, grad ckpt | 2251 ms | 862 ms (2.6x) | 530 ms | | 1.4524 / 1.4528 / 1.4523 |
| LoRA r=16 all-linear, Gemma 3 1B, 2 x 512, grad ckpt | 1323 ms | 868 ms (1.5x) | 639 ms, 2 skipped steps | | 1.7221 / 1.7223 / 1.7216 |

![from-scratch loss curves](docs/images/train_scratch.png)
![LoRA loss curves](docs/images/train_lora_qwen.png)

Honest reading: tbf16 gives bf16 training semantics (no loss scaling, no skipped steps, no
overflow) at 1.5-2.6x the speed of native bf16, with matching loss curves. When a model
trains fine in fp16 (both models above do at this scale), fp16 AMP is still 1.4-1.6x faster.
The gap is mostly host overhead: at these small batches the step is CPU bound on pentek5's
2.2 GHz Xeons (LoRA: 970 ms wall vs 563 ms of GPU kernels for tbf16, 551 vs 392 for fp16),
and tbf16 launches more kernels per layer. Larger batches shrink the gap.

What made the difference while building this:
- Attention backward must keep its tensors in `save_for_backward` (calling torch's
  efficient-attention ops directly). An earlier version kept an inner autograd graph on
  `ctx`, which activation checkpointing could not drop: ~0.7 GB leaked per step and each
  layer was recomputed twice (1118 -> 862 ms/step once fixed).
- Q/K/V and gate/up share one quantized copy of their input (3x/2x less memory for those
  saved activations).
- LoRA adapters (rank 16) are skinny GEMMs where torch's bf16 path is ~20x slower than it
  should be (`magma_sgemmEx`, ~115 ms/step). They now run as fp32 SGEMM.
- Caching frozen base weights in fp16 saved only ~7% for +3.2 GB, so it was dropped.

## Dead ends and notes

- Triton 3.9 emits no `mma` for sm_75 (`tl.dot` falls back to FMAs, ~0.2 TFLOPS), so the
  GEMM itself is cuBLAS fp16. Triton is fine for the elementwise kernels.
- Triton's Python launcher costs ~45 us per call on pentek5's 2.2 GHz Xeons, which made
  decoding 30% slower than native. The memory-bound kernels are plain CUDA now.
- `patch()` (TorchFunctionMode) is convenient but caches an fp16 copy of every weight
  and adds per-op Python overhead; `convert()` is the real path.
