"""bf16 matmuls on fp16 tensor cores, using exact power-of-two scaling.

bf16 has an 8-bit significand and fp16 has 11, so any bf16 value in fp16's range
converts exactly. We only have to move each row into that range with a
power-of-two scale (which is exact), run the fp16 GEMM with fp32 accumulation,
and undo the scales afterwards.
"""
import math
import os
import weakref

import torch
import triton
import triton.language as tl

from ._ext import available, ext, gemm_available, gemm_ext

# TBF16_BACKEND=triton forces the Triton kernels (slower to launch; used for comparison).
_USE_CUDA = os.environ.get("TBF16_BACKEND", "cuda") == "cuda"
# TBF16_GEMM=fused|cublas forces the GEMM for M >= SMALL_M: the fused kernel (csrc/gemm_sm75.cu)
# or quantize + cuBLAS fp16 + epilogue. Unset, fused is used from FUSED_MIN_M rows up.
_GEMM = os.environ.get("TBF16_GEMM", "")
FUSED_MIN_M = 256
# Fused GEMM row scaling: max -> [2^14, 2^15) (fp32 accumulation, bf16 output, so no overflow bound
# beyond fp16's input range). (target exponent, scale by L1 norm instead of max)
_FUSED_SCALE = (15, False)


def _target_exp(K: int) -> int:
    """Rows are scaled to amax in [2^(t-1), 2^t). The fp16 GEMM output is then below K * 4^t,
    which must stay under fp16's 65504; t = 0 covers K <= 32768."""
    return min(0, (15 - math.ceil(math.log2(max(K, 1)))) // 2)


def _cuda_ok(*ts):
    return _USE_CUDA and available() and all(t.shape[-1] % 8 == 0 and t.stride(0) % 8 == 0 and t.data_ptr() % 16 == 0 for t in ts)


@triton.jit
def _pow2_scale(amax):
    # Exact power of two s with amax * s in [0.5, 1). Built from the exponent
    # bits so it is exact (exp2/log2 are approximate on the GPU).
    e = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
    e = tl.minimum(tl.maximum(e, 1), 252)  # zero/denormal rows and inf/nan stay finite
    return ((253 - e) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _quantize_rows_kernel(X, Y, S_INV, K, stride_x, stride_y, MUL, BLOCK: tl.constexpr, ONE_PASS: tl.constexpr,
                          L1: tl.constexpr):
    row = tl.program_id(0)
    x_row = X + row.to(tl.int64) * stride_x
    y_row = Y + row.to(tl.int64) * stride_y
    if ONE_PASS:  # the whole row fits in registers: read it once
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        s = _pow2_scale(tl.sum(tl.abs(x), 0) if L1 else tl.max(tl.abs(x), 0)) * MUL
        tl.store(y_row + offs, (x * s).to(tl.float16), mask=offs < K)
        tl.store(S_INV + row, 1.0 / s)
        return
    amax = tl.zeros((BLOCK,), tl.float32)
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        amax = amax + tl.abs(x) if L1 else tl.maximum(amax, tl.abs(x))
    s = _pow2_scale(tl.sum(amax, 0) if L1 else tl.max(amax, 0)) * MUL
    for k in range(0, K, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < K, other=0.0).to(tl.float32)
        tl.store(y_row + offs, (x * s).to(tl.float16), mask=offs < K)
    tl.store(S_INV + row, 1.0 / s)  # exact: s is a power of two


def quantize_rows(x: torch.Tensor, target_exp: int = 0, l1: bool = False):
    """bf16 (R, K) -> (fp16 (R, K), fp32 (R,) inverse scales). Lossless for normal bf16 values.
    Each row's max (or with l1=True its L1 norm) is scaled into [2^(t-1), 2^t)."""
    assert x.dim() == 2 and x.stride(1) == 1
    if _cuda_ok(x):
        return tuple(ext().quantize_rows(x, target_exp, l1))
    R, K = x.shape
    y = torch.empty((R, K), device=x.device, dtype=torch.float16)
    s_inv = torch.empty((R,), device=x.device, dtype=torch.float32)
    one_pass = K <= 16384
    block = triton.next_power_of_2(K) if one_pass else 4096
    warps = max(4, min(16, block // 512))
    _quantize_rows_kernel[(R,)](x, y, s_inv, K, x.stride(0), y.stride(0), 2.0 ** target_exp, BLOCK=block,
                                ONE_PASS=one_pass, L1=l1,
                                num_warps=warps)
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


class _RowCache:
    """Reuses quantize_rows results when the same activation feeds several linear layers
    (Q/K/V, gate/up). Keyed by tensor identity through a weakref plus the version counter,
    so freed or modified tensors never hit."""

    def __init__(self, size=8):
        self.size, self.entries = size, []

    def get(self, x, t, make):
        for ref, ver, tt, val in self.entries:
            if ref() is x and ver == x._version and tt == t:
                return val
        val = make()
        self.entries.append((weakref.ref(x), x._version, t, val))
        if len(self.entries) > self.size:
            self.entries.pop(0)
        return val


_row_cache = _RowCache()


# Inference rows: L1 norm below 2^15, so with |w16| <= 1 the fp16 GEMM output cannot overflow.
L1_TARGET = 15


def quantize_rows_shared(x: torch.Tensor, l1: bool = False):
    """quantize_rows of x viewed as (-1, K), shared across calls on the same tensor."""
    K = x.shape[-1]
    t = L1_TARGET if l1 else _target_exp(K)
    return _row_cache.get(x, (t, l1), lambda: quantize_rows(x.reshape(-1, K).contiguous(), t, l1))


def prepare_weight(w: torch.Tensor):
    """nn.Linear weight (N, K) bf16 -> (fp16 (N, K), fp32 (N,) inverse scales). Do this once."""
    return quantize_rows(w.contiguous(), _target_exp(w.shape[1]))


def _use_fused(x2, w16, bias):
    if not (_USE_CUDA and _GEMM != "cublas" and w16.shape[0] % 8 == 0 and _cuda_ok(x2, w16) and gemm_available()):
        return False
    if bias is not None and not (bias.dtype == torch.bfloat16 and bias.is_contiguous() and bias.data_ptr() % 4 == 0):
        return False
    return _GEMM == "fused" or x2.shape[0] >= FUSED_MIN_M


@torch.autocast("cuda", enabled=False)  # autocast would turn the fp16 GEMM back into bf16
def scaled_linear(x: torch.Tensor, w16: torch.Tensor, w_sinv: torch.Tensor, bias=None):
    """y = x @ W^T + b for bf16 x, with W given by prepare_weight. Returns bf16."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    if x2.shape[0] < SMALL_M:
        return small_m_linear(x2, w16, w_sinv, bias).reshape(*shape[:-1], w16.shape[0])
    if _use_fused(x2, w16, bias):  # one GEMM kernel: x converted to fp16 on the fly, bf16 out
        t, l1 = _FUSED_SCALE
        return gemm_ext().linear(x2, w16, w_sinv, bias, -1, t, l1).reshape(*shape[:-1], w16.shape[0])
    x16, x_sinv = quantize_rows_shared(x, l1=True)
    return linear_q(x16, x_sinv, w16, w_sinv, bias).reshape(*shape[:-1], w16.shape[0])


@torch.autocast("cuda", enabled=False)
def linear_q(x16, x_sinv, w16, w_sinv, bias=None):
    """GEMM + epilogue on already-quantized rows: (x16 * x_sinv) @ (w16 * w_sinv)^T + bias, bf16 (M, N)."""
    c = x16 @ w16.t()  # cuBLAS fp16 tensor cores, fp32 accumulate; |c| < K * 4^t <= 65504
    M, N = c.shape
    if _cuda_ok(c):
        return ext().epilogue(c, x_sinv, w_sinv, bias)
    out = torch.empty((M, N), device=c.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _epilogue_kernel[grid](c, x_sinv, w_sinv, bias if bias is not None else c, out, M, N,
                           HAS_BIAS=bias is not None, BM=64, BN=64, num_warps=4)
    return out


@triton.jit
def _small_m_kernel(X, W, WS, BIAS, OUT, M, N, K, stride_x, stride_w, stride_o, HAS_BIAS: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Few rows: memory bound, so plain fp32 FMAs on (bf16 x) * (fp16 w) are as fast as tensor
    # cores and need no activation scaling (fp32 has bf16's range).
    rm = tl.arange(0, BM)
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(0, K, BK):
        kk = k + rk
        x = tl.load(X + rm[:, None] * stride_x + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None].to(tl.int64) * stride_w + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K),
                    other=0.0)
        acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2)
    y = acc * tl.load(WS + rn, mask=rn < N, other=0.0)[None, :]
    if HAS_BIAS:
        y += tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None] * stride_o + rn[None, :], y.to(tl.bfloat16), mask=(rm[:, None] < M) & (rn[None, :] < N))


# Rows below this use the GEMV kernel. Native bf16 is slow from M=2 up (it leaves its GEMV
# path), while the fp16 tensor-core path is fast at any M >= 2.
SMALL_M = 2


def small_m_linear(x2: torch.Tensor, w16: torch.Tensor, w_sinv: torch.Tensor, bias=None):
    M, K = x2.shape
    if M == 1 and _cuda_ok(x2, w16):
        return ext().gemv(x2, w16, w_sinv, bias)
    N = w16.shape[0]
    out = torch.empty((M, N), device=x2.device, dtype=torch.bfloat16)
    BM = max(1, triton.next_power_of_2(M))
    BN, BK, warps = (4, 512, 1) if BM == 1 else (16, max(32, 512 // BM), 4)  # M=1 tuned on a 2080 Ti
    _small_m_kernel[(triton.cdiv(N, BN),)](x2, w16, w_sinv, bias if bias is not None else w_sinv, out, M, N, K,
                                           x2.stride(0), w16.stride(0), out.stride(0), HAS_BIAS=bias is not None,
                                           BM=BM, BN=BN, BK=BK, num_warps=warps)
    return out
