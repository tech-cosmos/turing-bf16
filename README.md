# turing-bf16

Fast bf16 inference on GPUs without bf16 tensor cores: Turing (RTX 20xx, T4, Quadro RTX,
TITAN RTX).

Turing tensor cores run fp16 but not bf16, so bf16 matmuls fall back to a path that is
slower than fp32 (~7.6 vs ~56 TFLOPS on a 2080 Ti) and bf16 attention materializes the
full S x S matrix. Casting the model to fp16 is fast but breaks models whose activations
leave fp16's range (Gemma 3 4B gives NaN logits).

tbf16 keeps the model in bf16 and runs the math on fp16 tensor cores. Every row is moved
into fp16's range with an exact power-of-two scale, so the conversion is lossless (bf16
has 8 significand bits, fp16 has 11), accumulation is fp32, and the scales are undone
afterwards.

```python
import tbf16
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda()
tbf16.convert(model)  # nn.Linear -> fp16 weights + pow2 scales (same memory), SDPA routed
```

On a 2080 Ti: GEMMs 6-8x faster than torch bf16, attention 5-7x faster with ~10x less
memory, and LLM prefill 3-6x faster with the same accuracy as native bf16. See
[RESULTS.md](RESULTS.md).

Status: early research, inference only. Not published yet.

## Layout

- `tbf16/ops.py`: scaled linear (CUDA quantize/epilogue + cuBLAS fp16), Triton fallbacks
- `tbf16/attention.py`: scaled SDPA on torch's fp16 memory-efficient kernel
- `tbf16/convert.py`: `convert(model)`; `tbf16/patch.py`: `patch()` context manager
- `tbf16/csrc/`: CUDA kernels (JIT-built for sm_75 on first use)
- `bench/`, `tests/`: benchmarks and accuracy checks
