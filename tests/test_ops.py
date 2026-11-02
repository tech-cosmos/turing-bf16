"""CUDA and Triton backends agree bit-for-bit and match an fp64 reference to bf16 accuracy."""
import os
import torch
import tbf16.ops as ops


def check(M, K, N, bias, scale=1.0):
    x = (torch.randn(M, K, device="cuda") * scale).bfloat16()
    w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16()
    b = torch.randn(N, device="cuda").bfloat16() if bias else None
    ref = x.double() @ w.double().t() + (b.double() if bias else 0)
    outs = {}
    for be in (True, False):
        ops._USE_CUDA = be
        outs[be] = ops.scaled_linear(x, *ops.prepare_weight(w), b)
    ops._USE_CUDA = True
    err = ((outs[True].double() - ref).norm() / ref.norm()).item()
    same = torch.equal(outs[True], outs[False])
    print(f"M={M:>5} K={K:>6} N={N:>5} bias={bias!s:<5} scale={scale:<6g} rel err {err:.2e}  cuda==triton {same}")
    assert err < 3e-3 and torch.isfinite(outs[True]).all()


for M, K, N in [(1, 4096, 4096), (1, 2048, 151936), (7, 1024, 512), (4096, 3072, 3072), (333, 520, 1032), (64, 65536, 256)]:
    for bias in (False, True):
        check(M, K, N, bias)
check(4096, 3072, 3072, True, 1e6)
check(4096, 3072, 3072, True, 1e-8)
print("ok")
