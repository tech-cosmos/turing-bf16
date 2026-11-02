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


def _exponent(x: torch.Tensor, dim=None):
    """e with amax(|x|) = m * 2**e, m in [0.5, 1). Zero tensors give 0."""
    amax = x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)
    amax = amax.float()
    return torch.frexp(torch.where(amax > 0, amax, torch.ones_like(amax)))[1]


def _scale(x: torch.Tensor, e: torch.Tensor):
    """x * 2**e as fp16. Exact: a power-of-two multiply in bf16 then a cast that stays in range."""
    return (x * torch.ldexp(torch.ones_like(e, dtype=x.dtype), e)).half()


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    eq, ek = _exponent(q), _exponent(k)
    # After scaling, the maxima are 2**(eq+a) and 2**(ek-a); put both at ~2**((eq+ek)/2),
    # keeping them at most 2**14 so q.k products stay inside fp16 inputs.
    a = torch.div(ek - eq, 2, rounding_mode="floor")
    a = torch.minimum(a, 14 - eq)
    a = torch.maximum(a, ek - 14)
    ev = _exponent(v, dim=(-2, -1))
    if attn_mask is not None and attn_mask.is_floating_point() and attn_mask.dtype != torch.float16:
        attn_mask = attn_mask.half()  # additive masks must match q's dtype; large negatives become -inf
    out = F.scaled_dot_product_attention(
        _scale(q, a), _scale(k, -a), _scale(v, 14 - ev), attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
    if ev.shape[-3] != out.shape[-3]:  # GQA: each K/V head serves several query heads
        ev = ev.repeat_interleave(out.shape[-3] // ev.shape[-3], dim=-3)
    return (out.float() * torch.ldexp(torch.ones_like(ev, dtype=torch.float32), ev - 14)).to(q.dtype)
