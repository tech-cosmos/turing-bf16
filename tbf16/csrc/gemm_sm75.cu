// Fused bf16 x fp16 -> bf16 GEMM for sm_75: y = x @ W^T * sb + bias, with W = (fp16 w16, scales sb).
// A cheap pre-pass finds each x row's power-of-two scale; the GEMM converts bf16 x tiles to fp16
// in registers while staging them into shared memory (exact, like quantize_rows), runs
// mma.m16n8k8 with fp32 accumulation and applies the scales and bias before storing bf16.
// No cp.async on sm_75, so global loads are register-staged with a double-buffered smem tile.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int BK = 32;  // k per smem stage: 64-byte rows of four 16-byte chunks

// Exact power of two s with amax * s in [0.5, 1), built from exponent bits.
__device__ __forceinline__ float pow2_scale(float amax) {
  int e = (__float_as_int(amax) >> 23) & 0xFF;
  e = min(max(e, 1), 252);
  return __int_as_float((253 - e) << 23);
}

// s_inv[r] = 1 / pow2_scale(max |x[r, :]|). One warp per row. K % 8 == 0, 16-byte aligned rows.
__global__ void row_scale_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ s_inv, int M, int K,
                                 int64_t ld) {
  const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (row >= M) return;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * ld);
  float amax = 0.f;
  for (int i = lane; i < K / 8; i += 32) {
    uint4 v = __ldg(xr + i);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(h[j]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
  if (lane == 0) s_inv[row] = 1.f / pow2_scale(amax);
}

// Byte offset of 16-byte chunk c of row r in a [rows][BK] fp16 tile. Two rows share a 128-byte
// bank line; xor-ing the chunk with (r / 2) % 4 puts the 8 rows of an ldmatrix phase and the
// 8 chunks of a quarter-warp store on distinct banks.
__device__ __forceinline__ int swz(int r, int c) { return r * 64 + ((c ^ ((r >> 1) & 3)) << 4); }

__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_1688(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b));
}

// 16-byte read-only load that also pulls the rest of its 128-byte line (the next k-tile) into L2.
__device__ __forceinline__ uint4 ldg_l2_128(const void* p) {
  uint4 v;
  asm("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// Two packed bf16 -> two fp16 scaled by a power of two: exact for results in fp16's normal range.
__device__ __forceinline__ uint32_t cvt2(uint32_t v, float s) {
  __half2 h = __floats2half2_rn(__uint_as_float(v << 16) * s, __uint_as_float(v & 0xffff0000u) * s);
  return *reinterpret_cast<uint32_t*>(&h);
}

// BM x BN output tile per block, WM x WN warps, each computing a (BM/WM) x (BN/WN) warp tile.
template <int BM, int BN, int WM, int WN>
__global__ void __launch_bounds__(WM * WN * 32)
gemm_kernel(const __nv_bfloat16* __restrict__ A, const __half* __restrict__ B, const float* __restrict__ sa,
            const float* __restrict__ sb, const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ C,
            int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc) {
  constexpr int NT = WM * WN * 32, TM = BM / WM, TN = BN / WN, MI = TM / 16, NI = TN / 8;
  constexpr int RP = NT / 4;                    // rows per pass of the block's 16-byte loads
  constexpr int AL = BM / RP, BL = BN / RP;     // 16-byte loads per thread per stage
  static_assert(MI % 2 == 0 && NI % 4 == 0 && AL * RP == BM && BL * RP == BN, "bad tile");
  __shared__ __align__(128) uint8_t smem[2][(BM + BN) * 64];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  // Grouped launch order: GM row tiles share W tiles while they are in L2.
  constexpr int GM = 8;
  const int tiles_m = (M + BM - 1) / BM, tiles_n = (N + BN - 1) / BN;
  const int group = blockIdx.x / (GM * tiles_n), first = group * GM, gsize = min(tiles_m - first, GM);
  const int m0 = (first + (int)blockIdx.x % gsize) * BM;
  const int n0 = ((int)blockIdx.x % (GM * tiles_n)) / gsize * BN;

  // Global -> register staging: thread loads chunk lc of rows lr + i * RP.
  const int lr = tid >> 2, lc = tid & 3;
  const __nv_bfloat16* ga = A + (int64_t)(m0 + lr) * lda + lc * 8;
  const __half* gb = B + (int64_t)(n0 + lr) * ldb + lc * 8;
  float as[AL];
  uint32_t aok = 0, bok = 0;
#pragma unroll
  for (int i = 0; i < AL; ++i) {
    const bool ok = m0 + lr + i * RP < M;
    aok |= ok << i;
    as[i] = ok ? __frcp_rn(sa[m0 + lr + i * RP]) : 0.f;
  }
#pragma unroll
  for (int i = 0; i < BL; ++i) bok |= (n0 + lr + i * RP < N) << i;

  uint4 ra[AL], rb[BL];
  int kleft = K - lc * 8;  // this thread's chunk of the next tile is in range while kleft > 0
  auto load = [&]() {
    const bool kok = kleft > 0;
#pragma unroll
    for (int i = 0; i < AL; ++i)
      ra[i] = kok && (aok >> i & 1) ? ldg_l2_128(ga + i * RP * lda) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < BL; ++i)
      rb[i] = kok && (bok >> i & 1) ? ldg_l2_128(gb + i * RP * ldb) : make_uint4(0, 0, 0, 0);
    ga += BK;
    gb += BK;
    kleft -= BK;
  };
  auto store = [&](int stg) {
    uint8_t* sA = smem[stg];
    uint8_t* sB = smem[stg] + BM * 64;
#pragma unroll
    for (int i = 0; i < AL; ++i) {
      uint4 h = make_uint4(cvt2(ra[i].x, as[i]), cvt2(ra[i].y, as[i]), cvt2(ra[i].z, as[i]), cvt2(ra[i].w, as[i]));
      *reinterpret_cast<uint4*>(sA + swz(lr + i * RP, lc)) = h;
    }
#pragma unroll
    for (int i = 0; i < BL; ++i) *reinterpret_cast<uint4*>(sB + swz(lr + i * RP, lc)) = rb[i];
  };

  // ldmatrix.x4: lane l addresses row base + l, so one call yields two m16 A tiles or four n8 B tiles.
  const uint32_t s0 = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t fa_off = (wm * TM + lane) * 64, fb_off = BM * 64 + (wn * TN + lane) * 64;
  uint32_t fa[2][MI][2], fb[2][NI];
  auto ldfrag = [&](int buf, int stg, int kg) {
    const uint32_t base = s0 + stg * (BM + BN) * 64 + ((kg ^ ((lane >> 1) & 3)) << 4);
#pragma unroll
    for (int p = 0; p < MI / 2; ++p)
      ldsm_x4(fa[buf][2 * p][0], fa[buf][2 * p][1], fa[buf][2 * p + 1][0], fa[buf][2 * p + 1][1], base + fa_off + p * 32 * 64);
#pragma unroll
    for (int q = 0; q < NI / 4; ++q)
      ldsm_x4(fb[buf][4 * q], fb[buf][4 * q + 1], fb[buf][4 * q + 2], fb[buf][4 * q + 3], base + fb_off + q * 32 * 64);
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  // Pipeline per k-tile (4 k8 steps on smem stage stg): at step 2 the registers holding tile kt + 1
  // go to stage stg ^ 1 and the loads of tile kt + 2 are issued (a full k-tile of mma hides them);
  // step 3 starts with the barrier, so its mma overlaps the first fragment loads of the next tile.
  // The body is branch-free (past the end the loads are predicated off and the smem writes are
  // unused), so ptxas sees each load and its use in one block and keeps them apart.
  constexpr int KG = BK / 8;
  const int KT = (K + BK - 1) / BK;
  load();
  store(0);
  load();
  __syncthreads();
  ldfrag(0, 0, 0);
  for (int kt = 0; kt < KT; ++kt) {
    const int stg = kt & 1;
#pragma unroll
    for (int kg = 0; kg < KG; ++kg) {
      if (kg == KG - 1) __syncthreads();
      ldfrag((kg + 1) & 1, kg < KG - 1 ? stg : stg ^ 1, (kg + 1) % KG);
      if (kg == KG - 2) {
        store(stg ^ 1);
        load();
      }
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) mma_1688(acc[i][j], fa[kg & 1][i][0], fa[kg & 1][i][1], fb[kg & 1][j]);
    }
  }

  // Epilogue: acc * sa[m] * sb[n] + bias[n] -> bf16. Each warp stages 32 rows of its tile in smem
  // so that global stores write whole rows; storing the fragments directly (16 bytes per row and
  // instruction, half an L2 sector) made the output write ~5x slower than cuBLAS's.
  constexpr int RS = TN * 2 + 16;  // staging row stride in bytes; the padding makes fragment writes conflict-free
  constexpr int CPR = TN / 8;      // 16-byte chunks per staged row
  static_assert(WM * WN * 32 * RS <= sizeof(smem), "staging does not fit");
  __syncthreads();  // all warps are done with the operand tiles
  uint8_t* st = &smem[0][0] + warp * 32 * RS;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  float2 cs[NI], cb[NI];
#pragma unroll
  for (int j = 0; j < NI; ++j) {
    const int n = n0 + wn * TN + j * 8 + c2;
    const bool ok = n < N;
    cs[j] = ok ? *reinterpret_cast<const float2*>(sb + n) : make_float2(0.f, 0.f);
    cb[j] = ok && bias ? __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + n)) : make_float2(0.f, 0.f);
  }
#pragma unroll
  for (int p = 0; p < MI; p += 2) {
#pragma unroll
    for (int i = p; i < p + 2; ++i)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rl = (i - p) * 16 + g + h * 8, r = m0 + wm * TM + i * 16 + g + h * 8;
        const float a = r < M ? sa[r] : 0.f;
#pragma unroll
        for (int j = 0; j < NI; ++j)
          *reinterpret_cast<__nv_bfloat162*>(st + rl * RS + (j * 8 + c2) * 2) = __floats2bfloat162_rn(
              acc[i][j][2 * h] * a * cs[j].x + cb[j].x, acc[i][j][2 * h + 1] * a * cs[j].y + cb[j].y);
      }
    __syncwarp();
#pragma unroll
    for (int e = lane; e < 32 * CPR; e += 32) {
      const int rl = e / CPR, cc = e % CPR, r = m0 + wm * TM + p * 16 + rl, n = n0 + wn * TN + cc * 8;
      if (r < M && n < N) *reinterpret_cast<uint4*>(C + r * ldc + n) = *reinterpret_cast<const uint4*>(st + rl * RS + cc * 16);
    }
    __syncwarp();
  }
}

bool aligned16(const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

template <int BM, int BN, int WM, int WN>
void launch(const torch::Tensor& x, const torch::Tensor& s, const torch::Tensor& w, const torch::Tensor& ws,
            const __nv_bfloat16* b, torch::Tensor& y) {
  const int M = x.size(0), K = x.size(1), N = w.size(0);
  const int grid = ((M + BM - 1) / BM) * ((N + BN - 1) / BN);
  gemm_kernel<BM, BN, WM, WN><<<grid, WM * WN * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __half*>(w.data_ptr()),
      s.data_ptr<float>(), ws.data_ptr<float>(), b, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N, K,
      x.stride(0), w.stride(0), y.stride(0));
}

}  // namespace

torch::Tensor row_scale(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1);
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 8 == 0 && x.stride(0) % 8 == 0 && aligned16(x), "row_scale: K and row stride must be multiples of 8");
  auto s_inv = torch::empty({M}, x.options().dtype(at::kFloat));
  if (M == 0) return s_inv;
  row_scale_kernel<<<(M + 7) / 8, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), s_inv.data_ptr<float>(), (int)M, (int)K, x.stride(0));
  return s_inv;
}

// y = x @ w16^T * ws + bias for bf16 x (M, K), fp16 w16 (N, K) with fp32 inverse scales ws (N,).
// cfg selects the tile shape (-1: by size).
torch::Tensor linear(torch::Tensor x, torch::Tensor w16, torch::Tensor ws, c10::optional<torch::Tensor> bias,
                     int64_t cfg) {
  TORCH_CHECK(w16.is_cuda() && w16.scalar_type() == at::kHalf && w16.dim() == 2 && w16.stride(1) == 1);
  TORCH_CHECK(ws.is_contiguous() && ws.scalar_type() == at::kFloat && ws.numel() == w16.size(0));
  const int64_t M = x.size(0), K = x.size(1), N = w16.size(0);
  TORCH_CHECK(w16.size(1) == K && N % 8 == 0 && w16.stride(0) % 8 == 0 && aligned16(w16), "linear: N and K must be multiples of 8");
  if (bias)
    TORCH_CHECK(bias->is_contiguous() && bias->scalar_type() == at::kBFloat16 && bias->numel() == N &&
                reinterpret_cast<uintptr_t>(bias->data_ptr()) % 4 == 0);
  auto s = row_scale(x);
  auto y = torch::empty({M, N}, x.options());
  if (M == 0 || N == 0) return y;
  const __nv_bfloat16* b = bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  if (cfg < 0) {  // fewest waves; a 128x128 tile takes ~half the time of a 128x256 one, at ~5% lower efficiency
    int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t t256 = ((M + 127) / 128) * ((N + 255) / 256), t128 = ((M + 127) / 128) * ((N + 127) / 128);
    cfg = (t128 + sms - 1) / sms * 0.525 < (t256 + sms - 1) / sms ? 2 : 1;
  }
  switch (cfg) {
    case 0: launch<256, 128, 4, 2>(x, s, w16, ws, b, y); break;
    case 1: launch<128, 256, 2, 4>(x, s, w16, ws, b, y); break;
    case 2: launch<128, 128, 2, 2>(x, s, w16, ws, b, y); break;
    case 3: launch<64, 128, 1, 2>(x, s, w16, ws, b, y); break;
    case 4: launch<128, 64, 2, 1>(x, s, w16, ws, b, y); break;
    default: TORCH_CHECK(false, "bad cfg");
  }
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("row_scale", &row_scale);
  m.def("linear", &linear);
}
