// Memory-bound helper kernels: row quantize, GEMM epilogue, M=1 GEMV.
// Plain CUDA so each call is one cheap launch (Triton's Python launcher costs ~45 us/call
// on slow host CPUs, which dominates LLM decoding).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

namespace {

// Exact power of two s with amax * s in [2^(t-1), 2^t), built from exponent bits.
__device__ __forceinline__ float pow2_scale(float amax, int t) {
  int e = (__float_as_int(amax) >> 23) & 0xFF;
  e = min(max(e, 1), 252);
  int se = 126 + t - (e - 127);  // biased exponent of the scale
  se = min(max(se, 1), 254);
  return __int_as_float(se << 23);
}

__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

// One block per row. K % 8 == 0 and 16-byte aligned rows (checked on the host).
__global__ void quantize_rows_kernel(const __nv_bfloat16* __restrict__ x, __half* __restrict__ y,
                                     float* __restrict__ s_inv, int K, int64_t sx, int64_t sy, int t) {
  const int64_t row = blockIdx.x;
  const uint4* xr = reinterpret_cast<const uint4*>(x + row * sx);
  uint4* yr = reinterpret_cast<uint4*>(y + row * sy);
  const int K8 = K / 8;
  float amax = 0.f;
  for (int i = threadIdx.x; i < K8; i += blockDim.x) {
    uint4 v = xr[i];
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(h[j]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  __shared__ float red[32];
  amax = warp_max(amax);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    v = warp_max(v);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float s = pow2_scale(red[0], t);
  for (int i = threadIdx.x; i < K8; i += blockDim.x) {
    uint4 v = xr[i], o;
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
    __half2* q = reinterpret_cast<__half2*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(h[j]);
      q[j] = __floats2half2_rn(f.x * s, f.y * s);
    }
    yr[i] = o;
  }
  if (threadIdx.x == 0) s_inv[row] = 1.f / s;
}

// out[m, n] = c[m, n] * sa[m] * sb[n] + bias[n], 8 columns per thread. N % 8 == 0.
__global__ void epilogue_kernel(const __half* __restrict__ c, const float* __restrict__ sa,
                                const float* __restrict__ sb, const __nv_bfloat16* __restrict__ bias,
                                __nv_bfloat16* __restrict__ out, int64_t M, int N) {
  const int N8 = N / 8;
  const int64_t total = M * N8;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t m = i / N8;
    const int n = (int)(i - m * N8) * 8;
    uint4 v = reinterpret_cast<const uint4*>(c)[i], o;
    const __half2* h = reinterpret_cast<const __half2*>(&v);
    __nv_bfloat162* b = reinterpret_cast<__nv_bfloat162*>(&o);
    const float a = sa[m];
    const float4 s0 = *reinterpret_cast<const float4*>(sb + n), s1 = *reinterpret_cast<const float4*>(sb + n + 4);
    const float s[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __half22float2(h[j]);
      float y0 = f.x * a * s[2 * j], y1 = f.y * a * s[2 * j + 1];
      if (bias) {
        y0 += __bfloat162float(bias[n + 2 * j]);
        y1 += __bfloat162float(bias[n + 2 * j + 1]);
      }
      b[j] = __floats2bfloat162_rn(y0, y1);
    }
    reinterpret_cast<uint4*>(out)[i] = o;
  }
}

// y[n] = (x . w16[n]) * ws[n] + bias[n] for a single bf16 row x. One warp per output.
// fp32 math directly on bf16 x, so no activation scaling is needed. K % 8 == 0.
__global__ void gemv_kernel(const __nv_bfloat16* __restrict__ x, const __half* __restrict__ w,
                            const float* __restrict__ ws, const __nv_bfloat16* __restrict__ bias,
                            __nv_bfloat16* __restrict__ y, int N, int K, int64_t sw) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
  if (n >= N) return;
  const uint4* xr = reinterpret_cast<const uint4*>(x);
  const uint4* wr = reinterpret_cast<const uint4*>(w + n * sw);
  float acc = 0.f;
  for (int i = lane; i < K / 8; i += 32) {
    uint4 xv = __ldg(xr + i), wv = __ldg(wr + i);
    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&xv);
    const __half2* wh = reinterpret_cast<const __half2*>(&wv);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 a = __bfloat1622float2(xb[j]), b = __half22float2(wh[j]);
      acc = fmaf(a.x, b.x, fmaf(a.y, b.y, acc));
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    float r = acc * ws[n];
    if (bias) r += __bfloat162float(bias[n]);
    y[n] = __float2bfloat16(r);
  }
}

// Column amax: each thread covers 8 adjacent columns over a chunk of rows, merged with
// atomicMax on the float bits (valid because the values are non-negative).
__global__ void col_amax_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ amax, int64_t R, int K,
                                int64_t sx, int64_t rows_per_chunk) {
  const int c8 = blockIdx.x * blockDim.x + threadIdx.x;
  if (c8 >= K / 8) return;
  const int64_t r0 = blockIdx.y * rows_per_chunk, r1 = min(R, r0 + rows_per_chunk);
  float m[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int64_t r = r0; r < r1; ++r) {
    uint4 v = __ldg(reinterpret_cast<const uint4*>(x + r * sx) + c8);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(h[j]);
      m[2 * j] = fmaxf(m[2 * j], fabsf(f.x));
      m[2 * j + 1] = fmaxf(m[2 * j + 1], fabsf(f.y));
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) atomicMax(reinterpret_cast<int*>(amax) + c8 * 8 + j, __float_as_int(m[j]));
}

// y = x * s[col] as fp16 with s from the column amax; row 0 also writes 1/s.
__global__ void col_convert_kernel(const __nv_bfloat16* __restrict__ x, const float* __restrict__ amax,
                                   __half* __restrict__ y, float* __restrict__ s_inv, int64_t R, int K, int64_t sx,
                                   int t) {
  const int K8 = K / 8;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < R * K8; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = i / K8;
    const int c = (int)(i - r * K8) * 8;
    uint4 v = __ldg(reinterpret_cast<const uint4*>(x + r * sx) + c / 8), o;
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
    __half2* q = reinterpret_cast<__half2*>(&o);
    float s[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) s[j] = pow2_scale(amax[c + j], t);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float2 f = __bfloat1622float2(h[j]);
      q[j] = __floats2half2_rn(f.x * s[2 * j], f.y * s[2 * j + 1]);
    }
    reinterpret_cast<uint4*>(y + r * K)[c / 8] = o;
    if (r == 0) {
#pragma unroll
      for (int j = 0; j < 8; ++j) s_inv[c + j] = 1.f / s[j];
    }
  }
}

// ---- Attention helpers: per-(batch, head) amax and power-of-two scale + cast ----
// x is (B, H, S, D) with any strides except a contiguous last dim; D % 8 == 0.

template <typename T> __device__ __forceinline__ void load8(const T* p, float* f);
template <> __device__ __forceinline__ void load8<__nv_bfloat16>(const __nv_bfloat16* p, float* f) {
  uint4 v = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) { float2 t = __bfloat1622float2(h[j]); f[2 * j] = t.x; f[2 * j + 1] = t.y; }
}
template <> __device__ __forceinline__ void load8<__half>(const __half* p, float* f) {
  uint4 v = *reinterpret_cast<const uint4*>(p);
  const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) { float2 t = __half22float2(h[j]); f[2 * j] = t.x; f[2 * j + 1] = t.y; }
}
template <typename T> __device__ __forceinline__ void store8(T* p, const float* f);
template <> __device__ __forceinline__ void store8<__nv_bfloat16>(__nv_bfloat16* p, const float* f) {
  uint4 v;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) h[j] = __floats2bfloat162_rn(f[2 * j], f[2 * j + 1]);
  *reinterpret_cast<uint4*>(p) = v;
}
template <> __device__ __forceinline__ void store8<__half>(__half* p, const float* f) {
  uint4 v;
  __half2* h = reinterpret_cast<__half2*>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(f[2 * j], f[2 * j + 1]);
  *reinterpret_cast<uint4*>(p) = v;
}

struct Strides4 { int64_t b, h, s; int H, S, D; };

// grid (B*H, chunks); amax over each (b, h) slice, merged with atomicMax on float bits.
template <typename T>
__global__ void head_amax_kernel(const T* __restrict__ x, float* __restrict__ out, Strides4 st, int rows_per_chunk) {
  const int bh = blockIdx.x, b = bh / st.H, h = bh % st.H;
  const T* base = x + b * st.b + h * st.h;
  const int D8 = st.D / 8, s0 = blockIdx.y * rows_per_chunk, s1 = min(st.S, s0 + rows_per_chunk);
  float m = 0.f;
  for (int i = threadIdx.x; i < (s1 - s0) * D8; i += blockDim.x) {
    const int s = s0 + i / D8, d = (i % D8) * 8;
    float f[8];
    load8<T>(base + s * st.s + d, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) m = fmaxf(m, fabsf(f[j]));
  }
  __shared__ float red[32];
  m = warp_max(m);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : 0.f;
    v = warp_max(v);
    if (threadIdx.x == 0) atomicMax(reinterpret_cast<int*>(out) + bh, __float_as_int(v));
  }
}

// y (contiguous B, H, S, D) = x * 2^e[b*H + h]; e is int32 per (b, h).
template <typename Tin, typename Tout>
__global__ void head_scale_cast_kernel(const Tin* __restrict__ x, const int* __restrict__ e, Tout* __restrict__ y,
                                       Strides4 st, int64_t total8) {
  const int D8 = st.D / 8;
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total8; i += (int64_t)gridDim.x * blockDim.x) {
    const int d = (int)(i % D8) * 8;
    const int64_t row = i / D8;
    const int s = (int)(row % st.S);
    const int64_t bh = row / st.S;
    const int b = (int)(bh / st.H), h = (int)(bh % st.H);
    const float sc = ldexpf(1.f, e[bh]);
    float f[8];
    load8<Tin>(x + b * st.b + h * st.h + s * st.s + d, f);
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] *= sc;
    store8<Tout>(y + i * 8, f);
  }
}

bool aligned16(const torch::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

Strides4 strides4(const torch::Tensor& x) {
  TORCH_CHECK(x.dim() == 4 && x.stride(3) == 1 && x.size(3) % 8 == 0 && aligned16(x) && x.stride(0) % 8 == 0 &&
              x.stride(1) % 8 == 0 && x.stride(2) % 8 == 0, "expected (B, H, S, D) with contiguous D % 8 == 0");
  return Strides4{x.stride(0), x.stride(1), x.stride(2), (int)x.size(1), (int)x.size(2), (int)x.size(3)};
}

}  // namespace

std::vector<torch::Tensor> quantize_rows(torch::Tensor x, int64_t target_exp) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1);
  const int64_t R = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 8 == 0 && x.stride(0) % 8 == 0 && aligned16(x), "quantize_rows: K and row stride must be multiples of 8");
  auto y = torch::empty({R, K}, x.options().dtype(at::kHalf));
  auto s_inv = torch::empty({R}, x.options().dtype(at::kFloat));
  if (R == 0) return {y, s_inv};
  const int threads = K >= 4096 ? 512 : K >= 1024 ? 256 : 128;
  quantize_rows_kernel<<<R, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<__half*>(y.data_ptr()),
      s_inv.data_ptr<float>(), (int)K, x.stride(0), y.stride(0), (int)target_exp);
  return {y, s_inv};
}

torch::Tensor epilogue(torch::Tensor c, torch::Tensor sa, torch::Tensor sb, c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(c.is_contiguous() && c.scalar_type() == at::kHalf && c.size(1) % 8 == 0);
  const int64_t M = c.size(0), N = c.size(1);
  auto out = torch::empty({M, N}, c.options().dtype(at::kBFloat16));
  const __nv_bfloat16* b = bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  const int64_t work = M * N / 8;
  if (work == 0) return out;
  const int threads = 256;
  const int blocks = (int)std::min<int64_t>((work + threads - 1) / threads, 68 * 32);
  epilogue_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(c.data_ptr()), sa.data_ptr<float>(), sb.data_ptr<float>(), b,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), M, (int)N);
  return out;
}

torch::Tensor gemv(torch::Tensor x, torch::Tensor w16, torch::Tensor ws, c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(x.numel() == x.size(-1) && x.is_contiguous() && x.scalar_type() == at::kBFloat16);
  const int64_t N = w16.size(0), K = w16.size(1);
  TORCH_CHECK(K % 8 == 0 && w16.stride(0) % 8 == 0 && aligned16(x) && aligned16(w16));
  auto y = torch::empty({1, N}, x.options());
  const __nv_bfloat16* b = bias ? reinterpret_cast<const __nv_bfloat16*>(bias->data_ptr()) : nullptr;
  const int warps = 8;
  gemv_kernel<<<(N + warps - 1) / warps, warps * 32, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __half*>(w16.data_ptr()),
      ws.data_ptr<float>(), b, reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), (int)N, (int)K, w16.stride(0));
  return y;
}

std::vector<torch::Tensor> quantize_cols(torch::Tensor x, int64_t target_exp) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1);
  const int64_t R = x.size(0), K = x.size(1);
  TORCH_CHECK(K % 8 == 0 && x.stride(0) % 8 == 0 && aligned16(x), "quantize_cols: K and row stride must be multiples of 8");
  auto y = torch::empty({R, K}, x.options().dtype(at::kHalf));
  auto s_inv = torch::empty({K}, x.options().dtype(at::kFloat));
  auto amax = torch::zeros({K}, x.options().dtype(at::kFloat));
  if (R == 0) return {y, s_inv.fill_(1.f)};
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int threads = 128;
  const int col_blocks = (int)((K / 8 + threads - 1) / threads);
  const int64_t chunks = std::max<int64_t>(1, std::min<int64_t>((R + 63) / 64, (68 * 16 + col_blocks - 1) / col_blocks));
  const int64_t rows_per_chunk = (R + chunks - 1) / chunks;
  col_amax_kernel<<<dim3(col_blocks, (unsigned)chunks), threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), amax.data_ptr<float>(), R, (int)K, x.stride(0), rows_per_chunk);
  const int64_t work = R * K / 8;
  const int blocks = (int)std::min<int64_t>((work + 255) / 256, 68 * 32);
  col_convert_kernel<<<blocks, 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), amax.data_ptr<float>(),
      reinterpret_cast<__half*>(y.data_ptr()), s_inv.data_ptr<float>(), R, (int)K, x.stride(0), (int)target_exp);
  return {y, s_inv};
}

torch::Tensor head_amax(torch::Tensor x) {
  auto st = strides4(x);
  const int BH = (int)(x.size(0) * x.size(1));
  auto out = torch::zeros({x.size(0), x.size(1)}, x.options().dtype(at::kFloat));
  if (x.numel() == 0) return out;
  const int chunks = std::max(1, std::min((68 * 8 + BH - 1) / BH, (st.S + 63) / 64));
  const int rows = (st.S + chunks - 1) / chunks;
  auto stream = c10::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kBFloat16)
    head_amax_kernel<__nv_bfloat16><<<dim3(BH, chunks), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), out.data_ptr<float>(), st, rows);
  else if (x.scalar_type() == at::kHalf)
    head_amax_kernel<__half><<<dim3(BH, chunks), 256, 0, stream>>>(
        reinterpret_cast<const __half*>(x.data_ptr()), out.data_ptr<float>(), st, rows);
  else
    TORCH_CHECK(false, "head_amax: bf16 or fp16");
  return out;
}

torch::Tensor head_scale_cast(torch::Tensor x, torch::Tensor e, at::ScalarType out_dtype) {
  auto st = strides4(x);
  TORCH_CHECK(e.scalar_type() == at::kInt && e.is_contiguous() && e.numel() == x.size(0) * x.size(1));
  auto y = torch::empty(x.sizes(), x.options().dtype(out_dtype));
  const int64_t total8 = x.numel() / 8;
  if (total8 == 0) return y;
  const int blocks = (int)std::min<int64_t>((total8 + 255) / 256, 68 * 32);
  auto stream = c10::cuda::getCurrentCUDAStream();
  const int* ep = e.data_ptr<int>();
#define LAUNCH(TI, TO) head_scale_cast_kernel<TI, TO><<<blocks, 256, 0, stream>>>( \
    reinterpret_cast<const TI*>(x.data_ptr()), ep, reinterpret_cast<TO*>(y.data_ptr()), st, total8)
  if (x.scalar_type() == at::kBFloat16 && out_dtype == at::kHalf) LAUNCH(__nv_bfloat16, __half);
  else if (x.scalar_type() == at::kHalf && out_dtype == at::kBFloat16) LAUNCH(__half, __nv_bfloat16);
  else if (x.scalar_type() == at::kBFloat16 && out_dtype == at::kBFloat16) LAUNCH(__nv_bfloat16, __nv_bfloat16);
  else if (x.scalar_type() == at::kHalf && out_dtype == at::kHalf) LAUNCH(__half, __half);
  else TORCH_CHECK(false, "head_scale_cast: bf16/fp16 only");
#undef LAUNCH
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("head_amax", &head_amax);
  m.def("head_scale_cast", &head_scale_cast);
  m.def("quantize_cols", &quantize_cols);
  m.def("quantize_rows", &quantize_rows);
  m.def("epilogue", &epilogue);
  m.def("gemv", &gemv);
}
