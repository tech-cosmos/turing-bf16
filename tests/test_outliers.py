"""Rows with a few huge channels (T5/UMT5 reach ~4e5): the small channels must keep bf16 precision.

The norm-wise error of y hides this (the outlier channel dominates y), so the error is measured
on y minus the outlier channels' exact contribution, against that part's norm.
"""
import torch
import tbf16
from tbf16 import attention


def linear_case(M, K=4096, N=1024, big=3.8e5):
    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda")
    x[:, 7] = big
    x[:, 900] = -big / 3
    x = x.bfloat16()
    w = torch.randn(N, K, device="cuda") * K ** -0.5
    w[:, [7, 900]] = 0  # outlier channels that this layer ignores, so y is all small-channel signal
    w = w.bfloat16()
    xd, wd = x.double(), w.double()
    ref_small = xd @ wd.t()
    big_part = 0
    errs = {}
    for name, y in {"torch bf16": x.float() @ w.float().t(), "tbf16": tbf16.scaled_linear(x, *tbf16.prepare_weight(w))}.items():
        if name == "torch bf16":
            y = y.bfloat16()  # what torch's bf16 linear returns
        errs[name] = ((y.double() - big_part - ref_small).norm() / ref_small.norm()).item()
    return errs


def attention_case(big=3.8e5):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 8, 512, 64, device="cuda") for _ in range(3))
    v[..., 3] = big  # an outlier channel in V
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    ref = torch.softmax(q.double() @ k.double().transpose(-1, -2) / 8, -1) @ v.double()
    keep = torch.ones(64, dtype=torch.bool, device="cuda")
    keep[3] = False
    errs = {}
    for name, o in {"torch bf16": torch.nn.functional.scaled_dot_product_attention(q, k, v),
                    "tbf16": attention.sdpa(q, k, v)}.items():
        errs[name] = ((o.double() - ref)[..., keep].norm() / ref[..., keep].norm()).item()
    return errs


def main():
    worst = 0.0
    for M in (64, 1024):  # cuBLAS path (M < 256) and fused path
        e = linear_case(M)
        print(f"linear M={M:<5} small-channel rel err: torch bf16 {e['torch bf16']:.2e}  tbf16 {e['tbf16']:.2e}")
        worst = max(worst, e["tbf16"] / e["torch bf16"])
    e = attention_case()
    print(f"attention v outlier   small-channel rel err: torch bf16 {e['torch bf16']:.2e}  tbf16 {e['tbf16']:.2e}")
    worst = max(worst, e["tbf16"] / e["torch bf16"])
    print("ok" if worst < 1.5 else f"FAIL: tbf16 up to {worst:.1f}x worse than torch bf16")
    assert worst < 1.5


if __name__ == "__main__":
    main()
