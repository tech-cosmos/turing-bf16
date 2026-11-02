"""Gradients vs an fp64 reference, compared with native bf16 autograd."""
import math
import torch
import torch.nn.functional as F
import triton.testing as tt
import tbf16.autograd as ag


def rel(a, b):
    if a is None or not torch.isfinite(a).all():
        return "NaN/inf"
    return f"{((a.double() - b).norm() / b.norm()).item():.2e}"


def grads(fn, inputs, dout):
    ins = [t.detach().clone().requires_grad_() for t in inputs]
    out = fn(*ins)
    out.backward(dout)
    return [out] + [t.grad for t in ins]


def linear_case(name, M, K, N, xs=1.0, gs=1.0):
    x = (torch.randn(M, K, device="cuda") * xs).bfloat16()
    w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16()
    b = torch.randn(N, device="cuda").bfloat16()
    dy = (torch.randn(M, N, device="cuda") * gs).bfloat16()
    ref = grads(lambda x, w, b: F.linear(x, w, b), [x.double(), w.double(), b.double()], dy.double())
    nat = grads(lambda x, w, b: F.linear(x, w, b), [x, w, b], dy)
    ours = grads(ag.linear, [x, w, b], dy)
    names = ["y", "dx", "dw", "db"]
    print(f"linear {name:<22}" + "  ".join(f"{n}: bf16 {rel(a, r)} tbf16 {rel(o, r)}" for n, a, o, r in zip(names, nat, ours, ref)))


def ref_attn(q, k, v, causal):
    if k.shape[1] != q.shape[1]:
        k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
        v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        S = s.shape[-1]
        s = s.masked_fill(torch.ones(S, S, dtype=torch.bool, device=s.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v


def attn_case(name, Hq, Hkv, S, D, causal, qs=1.0, gs=1.0, vs=1.0):
    q = (torch.randn(1, Hq, S, D, device="cuda") * qs).bfloat16()
    k = (torch.randn(1, Hkv, S, D, device="cuda") * qs).bfloat16()
    v = (torch.randn(1, Hkv, S, D, device="cuda") * vs).bfloat16()
    do = (torch.randn(1, Hq, S, D, device="cuda") * gs).bfloat16()
    gqa = Hq != Hkv
    ref = grads(lambda q, k, v: ref_attn(q, k, v, causal), [q.double(), k.double(), v.double()], do.double())
    nat = grads(lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=gqa), [q, k, v], do)
    ours = grads(lambda q, k, v: ag.sdpa(q, k, v, is_causal=causal, enable_gqa=gqa), [q, k, v], do)
    names = ["o", "dq", "dk", "dv"]
    print(f"sdpa {name:<24}" + "  ".join(f"{n}: bf16 {rel(a, r)} tbf16 {rel(o, r)}" for n, a, o, r in zip(names, nat, ours, ref)))


def main():
    torch.manual_seed(0)
    linear_case("gaussian", 2048, 1024, 768)
    linear_case("grads x1e-12", 2048, 1024, 768, gs=1e-12)
    linear_case("grads x1e8, x x1e4", 2048, 1024, 768, xs=1e4, gs=1e8)
    linear_case("M=33 K=520 N=72", 33, 520, 72)
    linear_case("long M=40000", 40000, 256, 256)
    attn_case("gaussian", 8, 8, 1024, 64, False)
    attn_case("causal", 8, 8, 1024, 128, True)
    attn_case("GQA 16/4 causal", 16, 4, 1024, 128, True)
    attn_case("dO x1e-10 (needs scaling)", 8, 8, 1024, 64, True, gs=1e-10)
    attn_case("dO x1e6, v x1e4", 8, 8, 1024, 64, True, gs=1e6, vs=1e4)
    attn_case("peaky q,k x8", 8, 8, 1024, 64, True, qs=8)

    print()
    for M, K, N in [(8192, 2048, 6144), (8192, 6144, 2048)]:
        x = torch.randn(M, K, device="cuda").bfloat16().requires_grad_()
        w = (torch.randn(N, K, device="cuda") * K ** -0.5).bfloat16().requires_grad_()
        dy = torch.randn(M, N, device="cuda").bfloat16()
        fl = 6 * M * K * N / 1e9
        t_nat = tt.do_bench(lambda: F.linear(x, w).backward(dy), grad_to_none=[x, w])
        t_us = tt.do_bench(lambda: ag.linear(x, w).backward(dy), grad_to_none=[x, w])
        print(f"linear fwd+bwd {(M, K, N)}: bf16 {fl / t_nat:5.1f} TFLOPS  tbf16 {fl / t_us:5.1f} TFLOPS  {t_nat / t_us:.2f}x")
    for H, S, D in [(16, 2048, 128), (16, 4096, 64)]:
        q, k, v = (torch.randn(1, H, S, D, device="cuda").bfloat16().requires_grad_() for _ in range(3))
        do = torch.randn(1, H, S, D, device="cuda").bfloat16()
        fl = 3.5 * 4 * H * S * S * D / 2 / 1e9  # causal, fwd + bwd ~ 3.5x fwd
        res = []
        for fn in (lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True), lambda: ag.sdpa(q, k, v, is_causal=True)):
            torch.cuda.reset_peak_memory_stats()
            try:
                res.append((tt.do_bench(lambda: fn().backward(do), grad_to_none=[q, k, v]), torch.cuda.max_memory_allocated() / 2**30))
            except torch.OutOfMemoryError:
                res.append((float("nan"), float("nan")))
        print(f"sdpa fwd+bwd causal H={H} S={S} D={D}: bf16 {res[0][0]:7.1f} ms {res[0][1]:.2f} GB   tbf16 {res[1][0]:7.1f} ms {res[1][1]:.2f} GB   {res[0][0] / res[1][0]:.2f}x")


if __name__ == "__main__":
    main()
