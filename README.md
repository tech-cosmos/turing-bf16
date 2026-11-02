# turing-bf16

Fast bf16 matmuls on GPUs without bf16 tensor cores (Turing: RTX 20xx, T4, Quadro RTX, TITAN RTX).

Turing tensor cores run fp16 but not bf16, so bf16 matmuls fall back to a slow path
(~7.6 TFLOPS on a 2080 Ti, slower than fp32). Casting the model to fp16 is fast but
overflows (bf16 models often have activations above fp16's 65504) or underflows.

This project keeps tensors in bf16 and runs the math on fp16 tensor cores. Each row is
moved into fp16's range with an exact power-of-two scale, so the conversion is lossless
(bf16 has 8 significand bits, fp16 has 11), accumulation is fp32, and the scales are
undone at the end.

Status: early research. Not published yet.
