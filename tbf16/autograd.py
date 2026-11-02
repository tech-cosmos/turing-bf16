"""Training: backward passes on fp16 tensor cores.

Linear: dX = dY @ W and dW = dY^T @ X are fp16 GEMMs too. The contraction dimension
changes, so dY, W and X are scaled per column (with exact powers of two) where the
forward pass scaled per row.

Attention: torch's fp16 memory-efficient backward, with dO multiplied by a power of two
g picked so the internal dS stays in fp16's normal range. If any gradient overflows, the
backward is redone with a smaller g. This is loss scaling per op, decided immediately,
so no step is ever skipped.
"""
import math

import torch
import torch.nn.functional as F

from . import ops
from ._ext import ext


def quantize_cols(x: torch.Tensor, target_exp: int = 0):
    """bf16 (R, K) -> (fp16 (R, K), fp32 (K,) inverse scales), one power-of-two scale per column."""
    if ops._cuda_ok(x):
        return tuple(ext().quantize_cols(x, target_exp))
    amax = x.abs().amax(0).float()
    e = torch.frexp(torch.where(amax > 0, amax, torch.ones_like(amax)))[1] - target_exp
    s = torch.ldexp(torch.ones_like(amax), -e)
    return (x.float() * s).half(), torch.ldexp(torch.ones_like(amax), e)


def _scaled_mm(a16, sa, b16, sb):
    """(a16 @ b16) * sa[:, None] * sb[None, :] as bf16."""
    c = a16 @ b16
    if ops._cuda_ok(c):
        return ext().epilogue(c, sa, sb, None)
    return (c.float() * sa[:, None] * sb[None, :]).bfloat16()


class ScaledLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias):
        ctx.save_for_backward(x, w)
        ctx.has_bias = bias is not None
        return ops.scaled_linear(x, *ops.prepare_weight(w), bias)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        N, K = w.shape
        dy2 = dy.reshape(-1, N).contiguous()
        x2 = x.reshape(-1, K).contiguous()
        M = dy2.shape[0]
        dx = dw = db = None
        if ctx.needs_input_grad[0]:  # contraction over N
            t = ops._target_exp(N)
            dy16, sdy = ops.quantize_rows(dy2, t)
            w16c, swc = quantize_cols(w.contiguous(), t)
            dx = _scaled_mm(dy16, sdy, w16c, swc).reshape(x.shape)
        if ctx.needs_input_grad[1]:  # contraction over M
            t = ops._target_exp(M)
            dy16c, sdyc = quantize_cols(dy2, t)
            x16c, sxc = quantize_cols(x2, t)
            dw = _scaled_mm(dy16c.t(), sdyc, x16c, sxc)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            db = dy2.float().sum(0).to(dy.dtype)
        return dx, dw, db


def linear(x, w, bias=None):
    """Differentiable F.linear for bf16 x and w on fp16 tensor cores."""
    return ScaledLinearFn.apply(x, w, bias)


def _exp(x, dim=None):
    amax = (x.abs().amax() if dim is None else x.abs().amax(dim=dim, keepdim=True)).float()
    return torch.frexp(torch.where(amax > 0, amax, torch.ones_like(amax)))[1]


def _pow2(e, dtype):
    return torch.ldexp(torch.ones_like(e, dtype=dtype), e)


class ScaledSDPAFn(torch.autograd.Function):
    @staticmethod
    @torch.autocast("cuda", enabled=False)
    def forward(ctx, q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        eq, ek = _exp(q), _exp(k)
        a = torch.div(ek - eq, 2, rounding_mode="floor")
        a = torch.maximum(torch.minimum(a, 14 - eq), ek - 14)
        ev = _exp(v, dim=(-2, -1))  # v16 max in [0.5, 1): leaves headroom for the backward
        q16 = (q * _pow2(a, q.dtype)).half().requires_grad_()
        k16 = (k * _pow2(-a, k.dtype)).half().requires_grad_()
        v16 = (v * _pow2(-ev, v.dtype)).half().requires_grad_()
        if attn_mask is not None and attn_mask.dtype == torch.bfloat16:
            attn_mask = attn_mask.half()
        with torch.enable_grad():  # records the fp16 graph that backward() replays
            out16 = F.scaled_dot_product_attention(q16, k16, v16, attn_mask=attn_mask, dropout_p=dropout_p,
                                                   is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        if ev.shape[-3] != out16.shape[-3]:  # GQA
            ev_q = ev.repeat_interleave(out16.shape[-3] // ev.shape[-3], dim=-3)
        else:
            ev_q = ev
        ctx.graph = (out16, q16, k16, v16)
        ctx.scales = (a, ev, ev_q)
        ctx.d = q.shape[-1]
        return (out16.detach().float() * _pow2(ev_q, torch.float32)).to(q.dtype)

    @staticmethod
    def backward(ctx, dout):
        out16, q16, k16, v16 = ctx.graph
        a, ev, ev_q = ctx.scales
        # out = out16 * 2^ev, so dL/dout16 = dout * 2^ev. Then put max|dO16| at 2^t_do, keeping
        # dS = P * (dO16 . v16 - delta) < 2 * d * 2^t_do comfortably inside fp16.
        do = dout * _pow2(ev_q, dout.dtype)
        t_do = 12 - math.ceil(math.log2(ctx.d))
        e = _exp(do)
        for attempt in range(6):
            g = _pow2(t_do - e, torch.float32)
            dq16, dk16, dv16 = torch.autograd.grad(out16, (q16, k16, v16), (do * g.to(do.dtype)).half(),
                                                   retain_graph=True)
            ok = torch.isfinite(dq16).all() & torch.isfinite(dk16).all() & torch.isfinite(dv16).all()
            if bool(ok):  # host sync, once per attention call
                break
            t_do -= 8
        else:
            raise FloatingPointError("tbf16 attention backward overflowed fp16 even after rescaling")
        ctx.graph = None
        inv_g = 1.0 / g
        dq = (dq16.float() * (_pow2(a, torch.float32) * inv_g)).to(dout.dtype)
        dk = (dk16.float() * (_pow2(-a, torch.float32) * inv_g)).to(dout.dtype)
        dv = (dv16.float() * (_pow2(-ev, torch.float32) * inv_g)).to(dout.dtype)
        return dq, dk, dv, None, None, None, None, None


def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Differentiable bf16 scaled_dot_product_attention on torch's fp16 kernels."""
    return ScaledSDPAFn.apply(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa)
