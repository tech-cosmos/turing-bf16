# Working on turing-bf16 (for coding agents and contributors)

For *using* the library, see README.md. This file covers changing it.

## What the code does
bf16 tensors are multiplied on fp16 tensor cores. Each row or column (or attention head) is
scaled by an exact power of two into fp16's range, the fp16 kernel runs with fp32
accumulation, and the scale is undone afterwards. Every change must keep that exact:
scales are powers of two built from exponent bits, never `exp2`/`log2` approximations.

## Layout
| Path | What |
|---|---|
| `tbf16/ops.py` | `quantize_rows`, `prepare_weight`, `scaled_linear` (inference GEMM), Triton fallbacks, small-M GEMV routing |
| `tbf16/attention.py` | `prepare_qkv`, `head_exp`, `head_scale`, inference `sdpa` |
| `tbf16/autograd.py` | training: `ScaledLinearFn`, `ScaledSDPAFn` (torch efficient-attention ops), `FP32LinearFn`, `quantize_cols` |
| `tbf16/conv.py` | scaled Conv1d/2d/3d on fp16 cuDNN (inference) |
| `tbf16/convert.py` | `convert(model, train=...)`, `Linear` (inference), `TrainLinear`, SDPA routing |
| `tbf16/patch.py` | `patch()` TorchFunctionMode (convenience; caches an fp16 weight copy) |
| `tbf16/csrc/*.cu` | CUDA kernels, JIT-built by `tbf16/_ext.py` for sm_75 |
| `tests/` | accuracy vs fp64 references (run on a GPU) |
| `bench/` | speed benchmarks and end-to-end model runs |

## Invariants
- Any fp16 GEMM output must provably fit fp16: `_target_exp(K)` keeps `|C| < K * 4^t <= 65504`.
- Wrap anything that runs an fp16 matmul in `@torch.autocast("cuda", enabled=False)`.
  Autocast(bf16) silently turns fp16 GEMMs back into bf16; this has bitten us three times.
- Training ops must keep tensors in `save_for_backward`, not on `ctx`, so activation
  checkpointing can drop and recompute them.
- Never call `.item()`/`bool()` on GPU tensors in hot paths. The only allowed host sync is the
  attention-backward overflow check (`TBF16_ATTN_CHECK=0` disables it).
- Host CPU launch overhead matters: Triton launches cost ~45 us on slow Xeons, so hot small
  kernels belong in `csrc/`.

## Testing (needs a Turing GPU)
```bash
python -m tbf16                      # environment + quick correctness + speed
python -m tests.test_ops             # linear: CUDA vs Triton backends, edge shapes, ranges
python -m tests.test_accuracy        # linear vs fp64, outliers, tiny/huge values
python -m tests.test_attention       # inference attention accuracy + speed
python -m tests.test_autograd        # gradients vs fp64 (linear, attention, GQA, masks)
python -m tests.test_conv            # scaled convs vs fp64
python -m tests.test_sdpa_api        # SDPA keyword args, masks, strides, head dims
python -m bench.e2e_diffusion --help # diffusers end to end (Wan2.1, Sana)
python -m bench.linear               # GEMM TFLOPS per shape
python -m bench.e2e_llm --ref        # HF LLM end to end (needs 2 GPUs for the fp32 reference)
python -m bench.train_llm --recipe bf16+tbf16   # training recipe comparison
```
Accuracy target: relative error vs fp64 within ~5% of torch's own bf16 result (about 1.7e-3).
Report speed honestly, including regressions, in RESULTS.md.

## Conventions
- Match the existing style: short docstrings that explain why, no decorative comments.
- Commit messages: what changed and the measured effect.
- Keep `RESULTS.md` the single source of numbers; README only summarizes.
