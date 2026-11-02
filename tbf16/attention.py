"""bf16 scaled_dot_product_attention on fp16 kernels.

Q is multiplied by 2**a and K by 2**-a, which leaves every logit unchanged, with `a`
chosen so both fit fp16's range. V gets one power-of-two scale per (batch, head) that
is undone on the output. All scales are exact powers of two and live on the GPU (no
host sync), and torch's fp16 memory-efficient kernel does the work.

Limit: Q and K must satisfy amax(|q|) * amax(|k|) < 2**28, i.e. raw logits below ~2.7e8 before
the softmax scale. Real models are many orders of magnitude below that.
"""
import torch
import torch.nn.functional as F

from . import ops
from ._ext import ext


def _fast(x: torch.Tensor):
    return (ops._USE_CUDA and ops.available() and x.dim() == 4 and x.stride(-1) == 1 and x.shape[-1] % 8 == 0
            and x.data_ptr() % 16 == 0 and all(s % 8 == 0 for s in x.stride()[:3])
            and x.dtype in (torch.bfloat16, torch.float16))


def head_exp(x: torch.Tensor):
    """int32 (B, H): e with amax(|x[b, h]|) = m * 2**e, m in [0.5, 1) (0 for all-zero heads)."""
    amax = ext().head_amax(x) if _fast(x) else x.abs().amax(dim=(-2, -1)).float()
    return torch.frexp(amax)[1]


def head_scale(x: torch.Tensor, e: torch.Tensor, dtype):
    """x * 2**e[b, h] cast to dtype, as a contiguous (B, H, S, D) tensor. Exact while in range."""
    e = e.to(torch.int32).expand(x.shape[0], x.shape[1]).contiguous()
    if _fast(x):
        return ext().head_scale_cast(x, e, dtype)
    return (x.float() * torch.ldexp(torch.ones_like(e, dtype=torch.float32), e)[..., None, None]).to(dtype)


def prepare_qkv(q, k, v):
    """Scaled fp16 copies of q, k, v and the exponents to undo them: (q16, k16, v16, a, ev)."""
    eq, ek = head_exp(q).amax(), head_exp(k).amax()
    # After scaling, the maxima are 2**(eq+a) and 2**(ek-a); put both at ~2**((eq+ek)/2),
    # keeping them at most 2**14 so q.k products stay inside fp16 inputs.
    a = torch.div(ek - eq, 2, rounding_mode="floor")
    a = torch.maximum(torch.minimum(a, 14 - eq), ek - 14)
    ev = head_exp(v)  # v16 max in [0.5, 1): headroom for the backward pass
    q16, k16, v16 = head_scale(q, a, torch.float16), head_scale(k, -a, torch.float16), head_scale(v, -ev, torch.float16)
    return q16, k16, v16, a, ev


def expand_heads(e, n_heads):
    """Per-KV-head exponents -> per-query-head (GQA)."""
    return e if e.shape[1] == n_heads else e.repeat_interleave(n_heads // e.shape[1], dim=1)


@torch.autocast("cuda", enabled=False)
def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    q16, k16, v16, _, ev = prepare_qkv(q, k, v)
    if attn_mask is not None and attn_mask.is_floating_point() and attn_mask.dtype != torch.float16:
        attn_mask = attn_mask.half()  # additive masks must match q's dtype; large negatives become -inf
    out = F.scaled_dot_product_attention(q16, k16, v16, attn_mask=attn_mask, dropout_p=dropout_p,
                                         is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    return head_scale(out, expand_heads(ev, out.shape[1]), q.dtype)
