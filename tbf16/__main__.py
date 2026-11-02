"""python -m tbf16 : check that tbf16 works here and whether it helps on this GPU."""
import sys
import time

import torch


def main():
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: no CUDA GPU visible"); return 1
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]})")
    import tbf16
    from tbf16._ext import ext
    print(f"tbf16 applies to this GPU: {tbf16.is_supported_gpu()}"
          + ("" if tbf16.is_supported_gpu() else "  (it has native bf16 or no fp16 tensor cores; convert() is a no-op)"))
    t = time.time()
    ext()
    print(f"CUDA kernels built/loaded in {time.time() - t:.1f} s")

    torch.manual_seed(0)
    x = torch.randn(512, 1024, device="cuda").bfloat16() * 1e4  # outside fp16's range after a matmul
    w = (torch.randn(768, 1024, device="cuda") * 0.03).bfloat16()
    ref = x.double() @ w.double().t()
    y = tbf16.scaled_linear(x, *tbf16.prepare_weight(w))
    err_lin = ((y.double() - ref).norm() / ref.norm()).item()
    q, k, v = (torch.randn(1, 4, 512, 64, device="cuda").bfloat16() for _ in range(3))
    o_ref = torch.softmax(q.double() @ k.double().transpose(-1, -2) / 8, -1) @ v.double()
    err_att = ((tbf16.sdpa(q, k, v).double() - o_ref).norm() / o_ref.norm()).item()
    ok = err_lin < 3e-3 and err_att < 3e-3
    print(f"linear rel err {err_lin:.2e}, attention rel err {err_att:.2e} (bf16 rounding is ~1.7e-3): {'OK' if ok else 'FAIL'}")

    a = torch.randn(4096, 4096, device="cuda").bfloat16()
    b16 = tbf16.prepare_weight(a)
    def bench(f):
        f(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(5): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / 5
    t_nat, t_us = bench(lambda: a @ a.t()), bench(lambda: tbf16.scaled_linear(a, *b16))
    print(f"4096^3 GEMM: torch bf16 {2 * 4096**3 / t_nat / 1e12:.1f} TFLOPS, tbf16 {2 * 4096**3 / t_us / 1e12:.1f} TFLOPS "
          f"({t_nat / t_us:.1f}x)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
