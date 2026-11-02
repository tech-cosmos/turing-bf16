"""bf16 matmuls on fp16 tensor cores, using exact power-of-two scaling.

bf16 has an 8-bit significand and fp16 has 11, so any bf16 value in fp16's range
converts exactly. We only have to move each row into that range with a
power-of-two scale (which is exact), run the fp16 GEMM with fp32 accumulation,
and undo the scales afterwards.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pow2_scale(amax):
    # Exact power of two s with amax * s in [0.5, 1). Built from the exponent
    # bits so it is exact (exp2/log2 are approximate on the GPU).
    e = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
    e = tl.minimum(tl.maximum(e, 1), 252)  # zero/denormal rows and inf/nan stay finite
    return ((253 - e) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _quantize_rows_kernel(X, Y, S_INV, K, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = X + row.to(tl.int64) * stride_x
    y_row = Y + row.to(tl.int64) * stride_y
    amax = tl.zeros((BLOCK,), tl.float32)
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(x))
    s = _pow2_scale(tl.max(amax, 0))
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        tl.store(y_row + offs, (x * s).to(tl.float16), mask=offs < K)
    tl.store(S_INV + row, 1.0 / s)  # exact: s is a power of two


def quantize_rows(x: torch.Tensor):
    """bf16 (R, K) -> (fp16 (R, K), fp32 (R,) inverse scales). Lossless for normal bf16 values."""
    assert x.dim() == 2 and x.stride(1) == 1
    R, K = x.shape
    y = torch.empty((R, K), device=x.device, dtype=torch.float16)
    s_inv = torch.empty((R,), device=x.device, dtype=torch.float32)
    _quantize_rows_kernel[(R,)](x, y, s_inv, K, x.stride(0), y.stride(0), BLOCK=1024, num_warps=4)
    return y, s_inv


@triton.jit
def _epilogue_kernel(C, SA, SB, BIAS, OUT, M, N, HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    offs = rm[:, None].to(tl.int64) * N + rn[None, :]
    c = tl.load(C + offs, mask=mask, other=0.0).to(tl.float32)
    sa = tl.load(SA + rm, mask=rm < M, other=0.0)
    sb = tl.load(SB + rn, mask=rn < N, other=0.0)
    y = c * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        y += tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(OUT + offs, y.to(tl.bfloat16), mask=mask)


def prepare_weight(w: torch.Tensor):
    """nn.Linear weight (N, K) bf16 -> (fp16 (N, K), fp32 (N,) inverse scales). Do this once."""
    return quantize_rows(w.contiguous())


def scaled_linear(x: torch.Tensor, w16: torch.Tensor, w_sinv: torch.Tensor, bias=None):
    """y = x @ W^T + b for bf16 x, with W given by prepare_weight. Returns bf16."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    x16, x_sinv = quantize_rows(x2)
    c = x16 @ w16.t()  # cuBLAS fp16 tensor cores, fp32 accumulate; |c| <= K so no overflow
    M, N = c.shape
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _epilogue_kernel[grid](c, x_sinv, w_sinv, bias if bias is not None else c, out, M, N,
                           HAS_BIAS=bias is not None, BM=64, BN=64, num_warps=4)
    return out.reshape(*shape[:-1], N)
