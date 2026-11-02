"""Speed of y = x @ W^T (+ bias) for bf16 x/W: torch bf16 vs tbf16 (quantize + cuBLAS + epilogue,
fused GEMM, and the default choice between them), with torch fp16 (cuBLAS) as the ceiling."""
import torch
import triton.testing as tt
import tbf16.ops as ops
from tbf16 import prepare_weight, scaled_linear

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


def bench(fn):
    tt.do_bench(fn, warmup=100, rep=20)  # bring the clocks up first
    return tt.do_bench(fn)


def main():
    print(torch.cuda.get_device_name())
    print(f"{'(M, K, N)':>22} {'bf16':>7} {'cublas':>7} {'fused':>7} {'tbf16':>7} {'fp16':>7}  {'speedup':>7}   (TFLOPS)")
    gemm = ops._GEMM
    for M, K, N in SHAPES:
        x = torch.randn(M, K, device="cuda").bfloat16()
        w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16()
        b = torch.randn(N, device="cuda").bfloat16()
        w16, ws = prepare_weight(w)
        x16 = x.half()
        f = 2 * M * K * N / 1e9
        t_bf = bench(lambda: torch.nn.functional.linear(x, w, b))
        t = {}
        for path in ("cublas", "fused", gemm):
            ops._GEMM = path
            t[path] = bench(lambda: scaled_linear(x, w16, ws, b))
        ops._GEMM = gemm
        t_fp = bench(lambda: torch.nn.functional.linear(x16, w16, b.half()))
        fused = f"{f / t['fused']:7.2f}" if M >= ops.SMALL_M else f"{'-':>7}"
        print(f"{str((M, K, N)):>22} {f / t_bf:7.2f} {f / t['cublas']:7.2f} {fused} {f / t[gemm]:7.2f} {f / t_fp:7.2f}"
              f"  {t_bf / t[gemm]:6.2f}x")


if __name__ == "__main__":
    main()
