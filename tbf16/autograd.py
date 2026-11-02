"""Training: backward passes on fp16 tensor cores.

Linear: dX = dY @ W and dW = dY^T @ X are fp16 GEMMs too. The contraction dimension
changes, so dY, W and X are scaled per column (with exact powers of two) where the
forward pass scaled per row.

Attention: torch's fp16 memory-efficient kernels, with dO multiplied by a power of two
g picked so the internal dS stays in fp16's normal range. If any gradient overflows, the
backward is redone with a smaller g. This is loss scaling per op, decided immediately,
so no step is ever skipped.
"""
import math
import os

import torch
import torch.nn.functional as F

from . import attention, ops
from ._ext import ext

# Check attention gradients for fp16 overflow and redo them at a smaller scale if needed.
# Costs one host sync per attention backward; TBF16_ATTN_CHECK=0 turns it off.
CHECK_OVERFLOW = os.environ.get("TBF16_ATTN_CHECK", "1") != "0"


def quantize_cols(x: torch.Tensor, target_exp: int = 0, row_mul=None):
    """bf16 (R, K) -> (fp16 (R, K), fp32 (K,) inverse scales), one power-of-two scale per column.
    row_mul (R,) powers of two, optional: quantize x * row_mul[:, None] instead (exact)."""
    if ops._cuda_ok(x):
        return tuple(ext().quantize_cols(x, target_exp, row_mul))
    if row_mul is not None:
        x = x.float() * row_mul[:, None]
    amax = x.abs().amax(0).float()
    e = torch.frexp(torch.where(amax > 0, amax, torch.ones_like(amax)))[1] - target_exp
    s = torch.ldexp(torch.ones_like(amax), -e)
    return (x.float() * s).half(), torch.ldexp(torch.ones_like(amax), e)


_ONES = {}


def _ones(n, device):
    key = (n, device)
    if key not in _ONES:
        _ONES[key] = torch.ones(n, device=device)
    return _ONES[key]


@torch.autocast("cuda", enabled=False)
def _scaled_mm(a16, sa, b16, sb):
    """(a16 @ b16) * sa[:, None] * sb[None, :] as bf16."""
    c = a16 @ b16
    if ops._cuda_ok(c):
        return ext().epilogue(c, sa, sb, None)
    return (c.float() * sa[:, None] * sb[None, :]).bfloat16()


class ScaledLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias):
        N, K = w.shape
        x16, sx = ops.quantize_rows_shared(x)  # Q/K/V (and gate/up) share one fp16 copy of x
        w16, sw = ops.prepare_weight(w)
        # Saved instead of x (same size). dW reuses it with the row scales folded into dY.
        ctx.save_for_backward(x16, sx, w)
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        return ops.linear_q(x16, sx, w16, sw, bias).reshape(*x.shape[:-1], N)

    @staticmethod
    def backward(ctx, dy):
        x16, sx, w = ctx.saved_tensors
        N, K = w.shape
        dy2 = dy.reshape(-1, N).contiguous()
        M = dy2.shape[0]
        dx = dw = db = None
        if ctx.needs_input_grad[0]:  # contraction over N
            t = ops._target_exp(N)
            dy16, sdy = ops.quantize_rows(dy2, ops.L1_TARGET, l1=True)  # |w16c| <= 1 bounds dX16 by 2^15
            w16c, swc = quantize_cols(w.contiguous(), t)
            dx = _scaled_mm(dy16, sdy, w16c, swc).reshape(ctx.x_shape)
        if ctx.needs_input_grad[1]:  # contraction over M
            # dW[n, k] = sum_m dy[m, n] * sx[m] * x16[m, k]: fold sx into dY, column-scale per n.
            # |x16| < 1, so |dW16| < M * 2^t; keep that under fp16's max.
            t = min(0, 15 - math.ceil(math.log2(max(M, 1))))
            dy16c, sdyc = quantize_cols(dy2, t, sx)
            dw = _scaled_mm(dy16c.t(), sdyc, x16, _ones(K, x16.device))
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = dy2.float().sum(0).to(dy.dtype)
        return dx, dw, db


class FP32LinearFn(torch.autograd.Function):
    """bf16 in/out, fp32 math. Saves the bf16 input (shared with sibling layers), not an fp32 copy."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, x, w, bias):
        ctx.save_for_backward(x, w)
        ctx.has_bias = bias is not None
        return F.linear(x.float(), w.float(), None if bias is None else bias.float()).to(x.dtype)

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dyf = dy.float()
        dx = (dyf @ w.float()).to(x.dtype) if ctx.needs_input_grad[0] else None
        dw = (dyf.reshape(-1, dy.shape[-1]).t() @ x.reshape(-1, x.shape[-1]).float()).to(w.dtype) \
            if ctx.needs_input_grad[1] else None
        db = dyf.reshape(-1, dy.shape[-1]).sum(0).to(dy.dtype) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db


def linear(x, w, bias=None):
    """Differentiable F.linear for bf16 x and w on fp16 tensor cores."""
    return ScaledLinearFn.apply(x, w, bias)


_eff_fwd = torch.ops.aten._scaled_dot_product_efficient_attention
_eff_bwd = torch.ops.aten._scaled_dot_product_efficient_attention_backward


def _attn_bias(mask, q, k):
    """attn_mask -> the additive fp16 bias the efficient kernel takes: (B, H, Sq, Sk), last dim
    padded to a multiple of 16 for alignment (as torch's SDPA does) and sliced back."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        mask = torch.zeros(mask.shape, dtype=torch.float16, device=mask.device).masked_fill_(~mask, float("-inf"))
    mask = mask.to(torch.float16).expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    sk = mask.shape[-1]
    pad = (-sk) % 16
    return F.pad(mask, (0, pad))[..., :sk] if pad else mask.contiguous()


def _expand_kv(t, n_heads):
    return t if t.shape[1] == n_heads else t.repeat_interleave(n_heads // t.shape[1], dim=1)


def _reduce_kv(g16, n_kv):
    """Sum per-query-head gradients back onto their KV head (GQA), in fp32."""
    if g16.shape[1] == n_kv:
        return g16
    B, H, S, D = g16.shape
    return g16.float().view(B, n_kv, H // n_kv, S, D).sum(2)


class ScaledSDPAFn(torch.autograd.Function):
    """Calls torch's fp16 memory-efficient attention forward/backward ops directly and keeps
    everything in save_for_backward, so activation checkpointing can drop and recompute it."""

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        q16, k16, v16, a, ev = attention.prepare_qkv(q, k, v)
        H = q.shape[1]
        bias = _attn_bias(attn_mask, q, k)
        out16, lse, seed, offset = _eff_fwd(q16, _expand_kv(k16, H), _expand_kv(v16, H), bias, True, dropout_p,
                                            is_causal, scale=scale)
        ev_q = attention.expand_heads(ev, H)
        ctx.save_for_backward(q16, k16, v16, out16, lse, seed, offset, bias, a, ev, ev_q)
        ctx.args = (dropout_p, is_causal, scale, q.shape[-1])
        return attention.head_scale(out16, ev_q, q.dtype)

    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def backward(ctx, dout):
        q16, k16, v16, out16, lse, seed, offset, bias, a, ev, ev_q = ctx.saved_tensors
        dropout_p, is_causal, scale, d = ctx.args
        H, Hkv = q16.shape[1], k16.shape[1]
        k16e, v16e = _expand_kv(k16, H), _expand_kv(v16, H)
        # out = out16 * 2^ev, so dL/dout16 = dout * 2^ev. Scale that by g = 2^G so its max sits at
        # 2^t_do, keeping dS = P * (dO16 . v16 - delta) < 2 * d * 2^t_do comfortably inside fp16.
        e = (attention.head_exp(dout) + ev_q).amax()
        t_do = 12 - math.ceil(math.log2(d))
        for attempt in range(6):
            G = t_do - e
            dq16, dk16, dv16, _ = _eff_bwd(attention.head_scale(dout, ev_q + G, torch.float16), q16, k16e, v16e,
                                           bias, out16, lse, seed, offset, dropout_p, [True, True, True, False],
                                           is_causal, scale=scale)
            if not CHECK_OVERFLOW:
                break
            ok = torch.isfinite(dq16).all() & torch.isfinite(dk16).all() & torch.isfinite(dv16).all()
            if bool(ok):  # host sync, once per attention call
                break
            t_do -= 8
        else:
            raise FloatingPointError("tbf16 attention backward overflowed fp16 even after rescaling")
        dq = attention.head_scale(dq16, a - G, dout.dtype)
        dk = attention.head_scale(_reduce_kv(dk16, Hkv), -a - G, dout.dtype)
        dv = attention.head_scale(_reduce_kv(dv16, Hkv), -ev - G, dout.dtype)
        return dq, dk, dv, None, None, None, None, None


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Differentiable bf16 scaled_dot_product_attention on torch's fp16 kernels."""
    return ScaledSDPAFn.apply(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
