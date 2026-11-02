"""Speed of y = x @ W^T (+ bias) for bf16 x/W: torch bf16 vs tbf16, with torch fp16 as the ceiling."""
import torch
import triton.testing as tt
from tbf16 import prepare_weight, quantize_rows, scaled_linear

SHAPES = [  # (M, K, N)
    (4096, 4096, 4096),
    (8192, 3072, 3072),
    (8192, 3072, 12288),
    (8192, 12288, 3072),
    (16384, 5120, 5120),
    (32760, 5120, 13824),
    (512, 4096, 4096),
    (64, 4096, 11008),
    (1, 4096, 4096),
]


def main():
    print(torch.cuda.get_device_name())
    print(f"{'(M, K, N)':>22} {'bf16':>8} {'fp16':>8} {'tbf16':>8}  {'speedup':>7}  {'quant x':>8} {'gemm':>8}   (TFLOPS / ms)")
    for M, K, N in SHAPES:
        x = torch.randn(M, K, device="cuda").bfloat16()
        w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16()
        b = torch.randn(N, device="cuda").bfloat16()
        w16, ws = prepare_weight(w)
        x16 = x.half()
        f = 2 * M * K * N / 1e9
        t_bf = tt.do_bench(lambda: torch.nn.functional.linear(x, w, b))
        t_fp = tt.do_bench(lambda: torch.nn.functional.linear(x16, w16, b.half()))
        t_us = tt.do_bench(lambda: scaled_linear(x, w16, ws, b))
        t_q = tt.do_bench(lambda: quantize_rows(x))
        t_g = tt.do_bench(lambda: x16 @ w16.t())
        print(f"{str((M, K, N)):>22} {f / t_bf:8.2f} {f / t_fp:8.2f} {f / t_us:8.2f}  {t_bf / t_us:6.2f}x  {t_q:8.3f} {t_g:8.3f}")


if __name__ == "__main__":
    main()
