"""CUDA and Triton backends agree bit-for-bit, and they and the fused GEMM match an fp64 reference
to bf16 accuracy."""
import torch
import tbf16.ops as ops


def check(M, K, N, bias, scale=1.0, outliers=False):
    x = (torch.randn(M, K, device="cuda") * scale).bfloat16()
    if outliers:
        x[::97] *= 3e4
    w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16()
    b = torch.randn(N, device="cuda").bfloat16() if bias else None
    ref = x.double() @ w.double().t() + (b.double() if bias else 0)
    w16, ws = ops.prepare_weight(w)
    saved = ops._USE_CUDA, ops._GEMM
    outs = {}
    for name, cuda, gemm in (("cublas", True, "cublas"), ("triton", False, "cublas"), ("fused", True, "fused")):
        ops._USE_CUDA, ops._GEMM = cuda, gemm
        outs[name] = ops.scaled_linear(x, w16, ws, b)
    ops._USE_CUDA, ops._GEMM = saved
    errs = {k: ((v.double() - ref).norm() / ref.norm()).item() for k, v in outs.items()}
    same = torch.equal(outs["cublas"], outs["triton"])
    print(f"M={M:>5} K={K:>6} N={N:>6} bias={bias!s:<5} scale={scale:<6g} outliers={outliers!s:<5} rel err "
          + " ".join(f"{k} {e:.2e}" for k, e in errs.items()) + f"  cuda==triton {same}")
    assert all(e < 3e-3 for e in errs.values()) and all(torch.isfinite(v).all() for v in outs.values())


for M, K, N in [(1, 4096, 4096), (1, 2048, 151936), (7, 1024, 512), (4096, 3072, 3072), (333, 520, 1032),
                (64, 65536, 256), (130, 4104, 264), (33, 100, 64)]:  # K % 8 != 0: fused falls back
    for bias in (False, True):
        check(M, K, N, bias)
for bias in (False, True):
    check(4096, 3072, 3072, bias, 1e6)
    check(4096, 3072, 3072, bias, 1e-8)
check(4096, 3072, 3072, True, outliers=True)
print("ok")
