// UPDATED: BF16-safe backward kernels (host uses at::BFloat16; device uses __nv_bfloat16)

#include <iostream>
#include <algorithm>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// NEW: host/device BF16 types & helpers
#include <c10/util/BFloat16.h>   // at::BFloat16
#include <cuda_bf16.h>           // __nv_bfloat16 + intrinsics

namespace {

// -------- Device-side helpers to bridge at::BFloat16 <-> __nv_bfloat16 --------
__device__ __forceinline__ const __nv_bfloat16* as_nv(const at::BFloat16* p) {
  return reinterpret_cast<const __nv_bfloat16*>(p);
}
__device__ __forceinline__ __nv_bfloat16* as_nv(at::BFloat16* p) {
  return reinterpret_cast<__nv_bfloat16*>(p);
}
__device__ __forceinline__ __nv_bfloat16 BF16_LOAD(const at::BFloat16& x) {
  return *as_nv(&x);
}
__device__ __forceinline__ void BF16_STORE(at::BFloat16& dst, __nv_bfloat16 v) {
  *as_nv(&dst) = v;
}

// ======================= gradq =======================

__global__
void calc_gradq_unmasked0(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradq,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = 0; l < nk; ++l) {
      tv    = BF16_LOAD(k[ikv][l][m]);
      s[m]  = BF16_LOAD(v[ikv][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        tr[outer] += tv * s[oo*szr + outer];
      }
    }
    for (int l = 0; l < nq; ++l) {
      t      = __float2bfloat16(0.0f);
      s[d+m] = BF16_LOAD(grad_output[i][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        t += tr[outer] * s[d + oo*szr + outer];
      }
      atomicAdd(as_nv(&gradq[i][l][m]), t);
    }
  }
}

__global__
void calc_gradq_unmasked1(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradq,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = 0; l < nk; ++l) {
      tv = BF16_LOAD(k[ikv][l][m]);
      for (int outer = 0; outer < szr; ++outer) {
        tr[outer] += tv;
      }
    }
    for (int l = 0; l < nq; ++l) {
      t      = __float2bfloat16(0.0f);
      s[m]   = BF16_LOAD(o[i][l][m]);
      s[d+m] = BF16_LOAD(grad_output[i][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        int ooo = oo*szr + outer;
        t += tr[outer] * s[ooo] * s[d + ooo];
      }
      atomicAdd(as_nv(&gradq[i][l][m]), __float2bfloat16(-1.0f) * t);
    }
  }
}

__global__
void calc_gradq_masked0(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradq,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = 0; l < nk - nq; ++l) {
      tv   = BF16_LOAD(k[ikv][l][m]);
      s[m] = BF16_LOAD(v[ikv][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        tr[outer] += tv * s[oo*szr + outer];
      }
    }
    for (int l = 0; l < nq; ++l) {
      t      = __float2bfloat16(0.0f);
      tv     = BF16_LOAD(k[ikv][l][m]);
      s[m]   = BF16_LOAD(v[ikv][l][m]);
      s[d+m] = BF16_LOAD(grad_output[i][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        int ooo = oo*szr + outer;
        tr[outer] += tv * s[ooo];
        t         += tr[outer] * s[d + ooo];
      }
      atomicAdd(as_nv(&gradq[i][l][m]), t);
    }
  }
}

__global__
void calc_gradq_masked1(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradq,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;
  int ndiff = nk - nq;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = 0; l < nk - nq; ++l) {
      tv = BF16_LOAD(k[ikv][l][m]);
      for (int outer = 0; outer < szr; ++outer) {
        tr[outer] += tv;
      }
    }
    for (int l = 0; l < nq; ++l) {
      t      = __float2bfloat16(0.0f);
      tv     = BF16_LOAD(k[ikv][ndiff + l][m]);
      s[m]   = BF16_LOAD(o[i][l][m]);
      s[d+m] = BF16_LOAD(grad_output[i][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        int ooo = oo*szr + outer;
        tr[outer] += tv;
        t         += tr[outer] * s[ooo] * s[d + ooo];
      }
      atomicAdd(as_nv(&gradq[i][l][m]), __float2bfloat16(-1.0f) * t);
    }
  }
}

// ======================= gradk =======================

__global__
void calc_gradk_unmasked0(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradk,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = 0; l < nq; ++l) {
      tv   = BF16_LOAD(q[i][l][m]);
      s[m] = BF16_LOAD(grad_output[i][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        tr[outer] += tv * s[oo*szr + outer];
      }
    }
    for (int l = 0; l < nk; ++l) {
      t        = __float2bfloat16(0.0f);
      s[d+m]   = BF16_LOAD(v[ikv][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        t += tr[outer] * s[d + oo*szr + outer];
      }
      // non-atomic in original: emulate += via load/add/store
      __nv_bfloat16 cur = BF16_LOAD(gradk[ikv][l][m]);
      BF16_STORE(gradk[ikv][l][m], cur + t);
    }
  }
}

__global__
void calc_gradk_unmasked1(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradk,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = 0; l < nq; ++l) {
      t      = __float2bfloat16(0.0f);
      tv     = BF16_LOAD(q[i][l][m]);
      s[m]   = BF16_LOAD(grad_output[i][l][m]);
      s[d+m] = BF16_LOAD(o[i][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        int ooo = oo*szr + outer;
        tr[outer] += s[d + ooo] * tv * s[ooo];
      }
    }

    for (int l = 0; l < nk; ++l) {
      t = __float2bfloat16(0.0f);
      for (int outer = 0; outer < szr; ++outer) t += tr[outer];
      // emulate -= via load/add/store
      __nv_bfloat16 cur = BF16_LOAD(gradk[ikv][l][m]);
      BF16_STORE(gradk[ikv][l][m], cur + (__float2bfloat16(-1.0f) * t));
    }
  }
}

__global__
void calc_gradk_masked0(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradk,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = nq - 1; l >= 0; --l) {
      t      = __float2bfloat16(0.0f);
      tv     = BF16_LOAD(q[i][l][m]);
      s[m]   = BF16_LOAD(grad_output[i][l][m]);
      s[d+m] = BF16_LOAD(v[ikv][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        int ooo = oo*szr + outer;
        tr[outer] += tv * s[ooo];
        t         += tr[outer] * s[d + ooo];
      }
      // original direct +=
      __nv_bfloat16 cur = BF16_LOAD(gradk[ikv][l][m]);
      BF16_STORE(gradk[ikv][l][m], cur + t);
    }
    for (int l = nk - nq - 1; l >= 0; --l) {
      t        = __float2bfloat16(0.0f);
      s[d+m]   = BF16_LOAD(v[ikv][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        t += tr[outer] * s[d + oo*szr + outer];
      }
      atomicAdd(as_nv(&gradk[ikv][l][m]), t);
    }
  }
}

__global__
void calc_gradk_masked1(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradk,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int m  = threadIdx.x;
  const int oo = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (m < d && oo < szrb && i < bh) {
    ikv = i / bhratio;

    for (int outer = 0; outer < szr; ++outer) tr[outer] = __float2bfloat16(0.0f);
    for (int l = nq - 1; l >= 0; --l) {
      t      = __float2bfloat16(0.0f);
      tv     = BF16_LOAD(q[i][l][m]);
      s[m]   = BF16_LOAD(grad_output[i][l][m]);
      s[d+m] = BF16_LOAD(o[i][l][m]);
      __syncthreads();
      for (int outer = 0; outer < szr; ++outer) {
        int ooo = oo*szr + outer;
        tr[outer] += s[d + ooo] * tv * s[ooo];
        t         += tr[outer];
      }
      // original direct -=
      __nv_bfloat16 cur = BF16_LOAD(gradk[ikv][l][m]);
      BF16_STORE(gradk[ikv][l][m], cur + (__float2bfloat16(-1.0f) * t));
    }
    for (int l = nk - nq - 1; l >= 0; --l) {
      t = __float2bfloat16(0.0f);
      for (int outer = 0; outer < szr; ++outer) t += tr[outer];
      atomicAdd(as_nv(&gradk[ikv][l][m]), __float2bfloat16(-1.0f) * t);
    }
  }
}

// ======================= gradv =======================

__global__
void calc_gradv_unmasked0(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradv,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int i     = blockIdx.x;
  int ikv;
  __nv_bfloat16 t;
  int sz = std::min(32, d);

  if (i < bh && outer < d) {
    ikv = i / bhratio;

    t = __float2bfloat16(0.0f);
    for (int l = 0; l < nq; ++l) {
      t += BF16_LOAD(grad_output[i][l][outer]);
    }
    for (int l = 0; l < nk; ++l) {
      __nv_bfloat16 cur = BF16_LOAD(gradv[ikv][l][outer]);
      BF16_STORE(gradv[ikv][l][outer], cur + t);
    }
  }
}

__global__
void calc_gradv_unmasked1(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradv,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int mm    = blockIdx.x;
  const int i     = blockIdx.y;
  int ikv;
  __nv_bfloat16 t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (i < bh && mm < szrb && outer < d) {
    ikv = i / bhratio;

    for (int m = 0; m < szr; ++m) tr[m] = __float2bfloat16(0.0f);
    for (int l = 0; l < nq; ++l) {
      t        = __float2bfloat16(0.0f);
      s[outer] = BF16_LOAD(q[i][l][outer]);
      s[d+outer] = BF16_LOAD(k[ikv][l][outer]);
      __syncthreads();
      for (int m = 0; m < szr; ++m) {
        int mmm = mm*szr + m;
        tr[m] += s[mmm] * BF16_LOAD(grad_output[i][l][outer]);
      }
    }
    for (int l = 0; l < nk; ++l) {
      t          = __float2bfloat16(0.0f);
      s[d+outer] = BF16_LOAD(k[ikv][l][outer]);
      __syncthreads();
      for (int m = 0; m < szr; ++m) {
        t += tr[m] * s[d + mm*szr + m];
      }
      atomicAdd(as_nv(&gradv[ikv][l][outer]), t);
    }
  }
}

__global__
void calc_gradv_masked0(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradv,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int i     = blockIdx.x;
  int ikv;
  __nv_bfloat16 t;
  int sz = std::min(32, d);

  if (i < bh && outer < d) {
    ikv = i / bhratio;

    t = __float2bfloat16(0.0f);
    for (int l = nq - 1; l >= 0; --l) {
      t += BF16_LOAD(grad_output[i][l][outer]);
      __nv_bfloat16 cur = BF16_LOAD(gradv[ikv][l][outer]);
      BF16_STORE(gradv[ikv][l][outer], cur + t);
    }
    for (int l = nk - nq - 1; l >= 0; --l) {
      __nv_bfloat16 cur = BF16_LOAD(gradv[ikv][l][outer]);
      BF16_STORE(gradv[ikv][l][outer], cur + t);
    }
  }
}

__global__
void calc_gradv_masked1(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> gradv,
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int mm    = blockIdx.x;
  const int i     = blockIdx.y;
  int ikv;
  __nv_bfloat16 t;
  __nv_bfloat16 tr[32];
  int sz  = std::min(32, d);
  int szr = 8;
  int szrb = d / szr;

  if (i < bh && mm < szrb && outer < d) {
    ikv = i / bhratio;

    for (int m = 0; m < szr; ++m) tr[m] = __float2bfloat16(0.0f);
    for (int l = nq - 1; l >= 0; --l) {
      t          = __float2bfloat16(0.0f);
      s[outer]   = BF16_LOAD(q[i][l][outer]);
      s[d+outer] = BF16_LOAD(k[ikv][l][outer]);
      __syncthreads();
      for (int m = 0; m < szr; ++m) {
        int mmm = mm*szr + m;
        tr[m] += s[mmm] * BF16_LOAD(grad_output[i][l][outer]);
        t     += tr[m] * s[d + mmm];
      }
      atomicAdd(as_nv(&gradv[ikv][l][outer]), t);
    }
    for (int l = nk - nq - 1; l >= 0; --l) {
      t          = __float2bfloat16(0.0f);
      s[d+outer] = BF16_LOAD(k[ikv][l][outer]);
      __syncthreads();
      for (int m = 0; m < szr; ++m) {
        t += tr[m] * s[d + mm*szr + m];
      }
      atomicAdd(as_nv(&gradv[ikv][l][outer]), t);
    }
  }
}

// ======================= misc =======================

__global__
void div_grad_output(
  torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> grad_output,
  torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> denum,
  int bh, int nq, int d) {

  const int m  = threadIdx.x;
  const int mm = blockIdx.x;
  const int i  = blockIdx.y;
  if (m < d && mm < d && i < bh) {
    for (int l = mm; l < nq; l += d) {
      __nv_bfloat16 val = BF16_LOAD(grad_output[i][l][m]) / BF16_LOAD(denum[i][l]);
      BF16_STORE(grad_output[i][l][m], val);
    }
  }
}

} // namespace

// ======================= HOST WRAPPER =======================

std::vector<torch::Tensor> backward_cuda_bf16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor denum,
    torch::Tensor grad_output,
    bool mask) {

  // dtype/device/contiguity checks
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda() && denum.is_cuda() && grad_output.is_cuda(),
              "all tensors must be CUDA");
  TORCH_CHECK(q.dtype()==at::kBFloat16 && k.dtype()==at::kBFloat16 && v.dtype()==at::kBFloat16 &&
              o.dtype()==at::kBFloat16 && denum.dtype()==at::kBFloat16 && grad_output.dtype()==at::kBFloat16,
              "all tensors must be bfloat16");

  auto q_c   = q.contiguous();
  auto k_c   = k.contiguous();
  auto v_c   = v.contiguous();
  auto o_c   = o.contiguous();
  auto den_c = denum.contiguous();
  auto go_c  = grad_output.contiguous();

  const auto nq   = q_c.size(1);
  const auto nk   = k_c.size(1);
  const auto bh   = q_c.size(0);
  const auto bhkv = k_c.size(0);
  const auto d    = q_c.size(2);
  const int  bhratio = static_cast<int>(bh / bhkv);

  const int threads = d;
  const int blocks  = bh;
  int szr  = 8;
  int szrb = d / szr;

  auto opts  = q.options();
  auto gradq = torch::zeros({bh,   nq, d},  opts);
  auto gradk = torch::zeros({bhkv, nk, d},  opts);
  auto gradv = torch::zeros({bhkv, nk, d},  opts);

  // normalize grad_output by denum
  div_grad_output<<<dim3(d, blocks), threads>>>(
      go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
      den_c.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
      bh, nq, d);
  cudaDeviceSynchronize();

  if (mask) {
    calc_gradq_masked0<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradq.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradq_masked1<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradq.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradk_masked0<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradk.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradk_masked1<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradk.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradv_masked0<<<blocks, threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradv.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradv_masked1<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradv.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

  } else {
    calc_gradq_unmasked0<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradq.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradq_unmasked1<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradq.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradk_unmasked1<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradk.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradk_unmasked0<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradk.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradv_unmasked0<<<blocks, threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradv.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_gradv_unmasked1<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        gradv.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        go_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);
  }

  cudaDeviceSynchronize();
  return {gradq, gradk, gradv};
}
