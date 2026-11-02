"""F.scaled_dot_product_attention as diffusers calls it, after tbf16.convert() / inside tbf16.patch():
keyword q/k/v, permuted (non-contiguous) inputs, broadcast additive and boolean masks with key lengths
that are not a multiple of 8, and the head dims of current DiTs."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tbf16

torch.manual_seed(0)


def ref_attn(q, k, v, mask):
    q, k, v = q.double(), k.double(), v.double()
    s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else s + mask.double()
    return torch.softmax(s, -1) @ v


def make(B, H, Sq, Skv, D, scale=1.0):
    # diffusers builds (B, S, H, D) and permutes to (B, H, S, D): non-contiguous inputs
    return [torch.randn(B, S, H, D, device="cuda").mul(scale).bfloat16().permute(0, 2, 1, 3) for S in (Sq, Skv, Skv)]


def masks(B, H, Skv):
    keep = torch.ones(B, Skv, dtype=torch.bool, device="cuda")
    keep[:, Skv * 2 // 3:] = False  # padded text tokens
    additive = (~keep).to(torch.bfloat16) * -10000.0  # Sana: (1 - mask) * -10000 in the model dtype
    return {"none": None,
            "bool (B,1,1,Skv)": keep[:, None, None, :],
            "bf16 additive (B,H,1,Skv)": additive[:, None, None, :].expand(B, H, 1, Skv),
            "fp32 additive (B,1,1,Skv)": additive.float()[:, None, None, :]}


def check(name, out, r):
    err = ((out.double() - r).norm() / r.norm()).item()
    print(f"  {name:<58} rel err {err:.2e}")
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all() and err < 1e-2, err


tbf16.install_sdpa()
try:
    for (B, H, Sq, Skv, D) in [(2, 24, 1024, 1024, 64), (2, 20, 1024, 300, 112), (1, 12, 3200, 512, 128)]:
        print(f"B={B} H={H} Sq={Sq} Skv={Skv} D={D}")
        q, k, v = make(B, H, Sq, Skv, D)
        for mname, mask in masks(B, H, Skv).items():
            r = ref_attn(q, k, v, mask)
            with torch.no_grad():
                check(f"convert, keywords, mask {mname}",
                      F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask), r)
                check(f"convert, positional, mask {mname}", F.scaled_dot_product_attention(q, k, v, mask), r)
    # activations that overflow fp16
    q, k, v = make(1, 8, 512, 512, 128, scale=3e3)
    with torch.no_grad():
        check("convert, q/k/v x3e3", F.scaled_dot_product_attention(query=q, key=k, value=v), ref_attn(q, k, v, None))
finally:
    tbf16.uninstall_sdpa()

with torch.no_grad(), tbf16.patch() as mode:
    q, k, v = make(2, 20, 1024, 300, 112)
    mask = masks(2, 20, 300)["bf16 additive (B,H,1,Skv)"]
    check("patch(), keywords, bf16 additive mask",
          F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask), ref_attn(q, k, v, mask))
    nn.Linear(112, 64).cuda().bfloat16()(q)
assert mode.stats["sdpa"] == 1 and mode.stats["linear"] == 1, mode.stats
print("ok")
