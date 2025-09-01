// UPDATED: BF16-safe PyTorch extension with device-side __nv_bfloat16 math

#include <iostream>
#include <algorithm>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

// =========================== UNMASKED PART ===========================

__global__
void calc_unmasked_cons_and_denum(
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> denum,
    int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  __nv_bfloat16 t;

  if (outer < d && i < bh) {
    ikv = i / bhratio;

    // calc lin denum
    t = __float2bfloat16(0.0f);
    for (int l = 0; l < nk; ++l) {
      t += BF16_LOAD(k[ikv][l][outer]);
    }
    __nv_bfloat16 a0div = __float2bfloat16(static_cast<float>(nk) / static_cast<float>(d));
    for (int l = 0; l < nq; ++l) {
      __nv_bfloat16 qv = BF16_LOAD(q[i][l][outer]);
      __nv_bfloat16 val = qv * t + a0div;
      atomicAdd(as_nv(&denum[i][l]), val);
    }

    // calc cons
    t = __float2bfloat16(0.0f);
    for (int l = 0; l < nk; ++l) {
      t += BF16_LOAD(v[ikv][l][outer]);
    }
    for (int l = 0; l < nq; ++l) {
      BF16_STORE(o[i][l][outer], t);
    }
  }
}

__global__
void calc_unmasked_lin(
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> denum,
    int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i  = blockIdx.y;

  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  const int szr = 8; // number of reductions per thread
  const int szrb = d / szr;

  if (outer < d && mm < szrb && i < bh) {
    ikv = i / bhratio;

    #pragma unroll
    for (int m = 0; m < szr; ++m) tr[m] = __float2bfloat16(0.0f);

    for (int l = 0; l < nk; ++l) {
      tv = BF16_LOAD(v[ikv][l][outer]);
      s[outer + d] = BF16_LOAD(k[ikv][l][outer]);
      __syncthreads();
      #pragma unroll
      for (int m = 0; m < szr; ++m) {
        tr[m] += s[d + mm*szr + m] * tv;
      }
    }

    for (int l = 0; l < nq; ++l) {
      s[outer] = BF16_LOAD(q[i][l][outer]);
      __syncthreads();
      t = __float2bfloat16(0.0f);
      #pragma unroll
      for (int m = 0; m < szr; ++m) {
        t += tr[m] * s[mm*szr + m];
      }
      atomicAdd(as_nv(&o[i][l][outer]), t);
    }
  }
}

// =========================== MASKED PART ============================

__global__
void calc_masked_cons_and_denum(
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> denum,
    int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  __nv_bfloat16 t;
  int ndiff = nk - nq;

  if (outer < d && i < bh) {
    ikv = i / bhratio;

    // calc lin denum
    t = __float2bfloat16(0.0f);
    for (int l = 0; l < nk - nq; ++l) {
      t += BF16_LOAD(k[ikv][l][outer]);
    }
    __nv_bfloat16 a0div = __float2bfloat16(1.0f / static_cast<float>(d));
    int ndiff1 = ndiff + 1;
    for (int l = 0; l < nq; ++l) {
      t += BF16_LOAD(k[ikv][ndiff + l][outer]);
      __nv_bfloat16 qv = BF16_LOAD(q[i][l][outer]);
      __nv_bfloat16 extra = a0div * __int2bfloat16_ru(ndiff1 + l);
      atomicAdd(as_nv(&denum[i][l]), qv * t + extra);
    }

    // calc cons
    t = __float2bfloat16(0.0f);
    for (int l = 0; l < nk - nq; ++l) {
      t += BF16_LOAD(v[ikv][l][outer]);
    }
    for (int l = 0; l < nq; ++l) {
      t += BF16_LOAD(v[ikv][ndiff + l][outer]);
      BF16_STORE(o[i][l][outer], t);
    }
  }
}

__global__
void calc_masked_lin(
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> q,
    const torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> k,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> v,
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> denum,
    int bh, int nq, int nk, int d, int bhratio) {

  extern __shared__ __nv_bfloat16 s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i  = blockIdx.y;
  int ikv;
  __nv_bfloat16 tv, t;
  __nv_bfloat16 tr[32];
  const int szr = 8;
  const int ndiff = nk - nq;

  // original code had boundary checks disabled; keep behavior
  if (true) {
    ikv = i / bhratio;

    #pragma unroll
    for (int m = 0; m < szr; ++m) tr[m] = __float2bfloat16(0.0f);

    for (int l = 0; l < nk - nq; ++l) {
      tv = BF16_LOAD(v[ikv][l][outer]);
      s[outer + d] = BF16_LOAD(k[ikv][l][outer]);
      __syncthreads();
      #pragma unroll
      for (int m = 0; m < szr; ++m) {
        tr[m] += s[d + mm*szr + m] * tv;
      }
    }

    for (int l = 0; l < nq; ++l) {
      tv = BF16_LOAD(v[ikv][ndiff + l][outer]);
      s[outer + d] = BF16_LOAD(k[ikv][ndiff + l][outer]);
      s[outer]     = BF16_LOAD(q[i][l][outer]);
      __syncthreads();
      t = __float2bfloat16(0.0f);
      #pragma unroll
      for (int m = 0; m < szr; ++m) {
        int mmm = mm*szr + m;
        tr[m] += s[d + mmm] * tv;
        t     += tr[m] * s[mmm];
      }
      atomicAdd(as_nv(&o[i][l][outer]), t);
    }
  }
}

__global__
void calc_div(
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> o,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> denum,
    int bh, int nq, int d) {

  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i  = blockIdx.y;

  const int szr  = 8;
  const int szrb = d / szr;

  if (true) {
    for (int l = mm; l < nq; l += szrb) {
      __nv_bfloat16 val = BF16_LOAD(o[i][l][outer]) / BF16_LOAD(denum[i][l]);
      BF16_STORE(o[i][l][outer], val);
    }
  }
}

__global__
void calc_norms(
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> norms,
    int bh, int n, int d, int th) {
  const int ii = threadIdx.x;
  const int j  = blockIdx.x;
  const int l  = blockIdx.y;
  __nv_bfloat16 t;
  int i;

  if (l < n && ii < th && j < ((bh - 1) / th + 1)) {
    i = j * th + ii;
    if (i < bh) {
      t = __float2bfloat16(0.0f);
      for (int m = 0; m < d; m++) {
        __nv_bfloat16 av = BF16_LOAD(a[i][l][m]);
        t += av * av;
      }
      BF16_STORE(norms[i][l], t);
    }
  }
}

__global__
void find_max(
    torch::PackedTensorAccessor32<at::BFloat16,2,torch::RestrictPtrTraits> norms,
    torch::PackedTensorAccessor32<at::BFloat16,1,torch::RestrictPtrTraits> maxes,
    int bh, int n, int th) {

  const int ii = threadIdx.x;
  const int j  = blockIdx.x;
  __nv_bfloat16 t = __float2bfloat16(0.0f);
  int i;

  if (ii < th && j < ((bh - 1) / th + 1)) {
    i = j * th + ii;
    if (i < bh) {
      for (int l = 0; l < n; ++l) {
        __nv_bfloat16 nv = BF16_LOAD(norms[i][l]);
        if (t < nv) t = nv;
      }
      BF16_STORE(maxes[i], t);
    }
  }
}

__global__
void apply_norm(
    torch::PackedTensorAccessor32<at::BFloat16,3,torch::RestrictPtrTraits> a,
    torch::PackedTensorAccessor32<at::BFloat16,1,torch::RestrictPtrTraits> maxes,
    int bh, int n, int d, int n_seg) {

  const int m = threadIdx.x;
  const int i = blockIdx.x;
  const int j = blockIdx.y;
  const int seg_len = int(n / n_seg);

  if (m < d && i < bh) {
    __nv_bfloat16 mx = BF16_LOAD(maxes[i]);
    if (mx < __float2bfloat16(0.1f)) mx = __float2bfloat16(0.1f);
    int l_begin = j * seg_len;
    int l_end   = min(n, (j + 1) * seg_len);
    for (int l = l_begin; l < l_end; ++l) {
      __nv_bfloat16 val = BF16_LOAD(a[i][l][m]) / mx;
      BF16_STORE(a[i][l][m], val);
    }
  }
}

} // namespace

// ============================== HOST WRAPPER ==============================

std::vector<torch::Tensor> forward_cuda_bf16(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool mask) {

  // dtype/device/contiguity checks
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
  TORCH_CHECK(q.dtype() == at::kBFloat16 && k.dtype() == at::kBFloat16 && v.dtype() == at::kBFloat16,
              "q/k/v must be bfloat16");
  auto q_c = q.contiguous();
  auto k_c = k.contiguous();
  auto v_c = v.contiguous();

  // Dimensions (q: [bh, nq, d], k/v: [bhkv, nk, d])
  const auto nq    = q_c.size(1);
  const auto nk    = k_c.size(1);
  const auto bh    = q_c.size(0);
  const auto bhkv  = k_c.size(0);
  const auto d     = q_c.size(2);
  const int  bhratio = static_cast<int>(bh / bhkv);

  const int threads = d;        // assume d <= 1024
  const int blocks  = bh;

  const int n_seg = 128;        // parallelization of context length
  const int szr   = 8;
  const int szrb  = d / szr;

  auto opts = q.options();
  auto denum  = torch::zeros({bh,  nq}, opts);
  auto o      = torch::zeros({bh,  nq, d}, opts);
  auto qnorms = torch::zeros({bh,  nq}, opts);
  auto knorms = torch::zeros({bhkv, nk}, opts);
  auto qmaxes = torch::zeros({bh}, opts);
  auto kmaxes = torch::zeros({bhkv}, opts);

  // Normalize q/k by per-(batch*head) max L2 across sequence
  {
    const long th_lim = 1024;
    int th = std::min<long>(th_lim, bh);

    calc_norms<<<dim3((bh - 1) / th + 1, nq), th>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        qnorms.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        bh, nq, d, th);

    // FIXED: pass nq (not nk)
    find_max<<<(bh - 1) / th + 1, th>>>(
        qnorms.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        qmaxes.packed_accessor32<at::BFloat16,1,torch::RestrictPtrTraits>(),
        bh, nq, th);

    for (int rep = 0; rep < int(nq / n_seg); ++rep) {
      apply_norm<<<dim3(blocks, n_seg), threads>>>(
          q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
          qmaxes.packed_accessor32<at::BFloat16,1,torch::RestrictPtrTraits>(),
          bh, nq, d, n_seg);
    }

    th = std::min<long>(th_lim, bhkv);
    calc_norms<<<dim3((bhkv - 1) / th + 1, nk), th>>>(
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        knorms.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        bhkv, nk, d, th);

    // FIXED: pass nk (not nq)
    find_max<<<(bhkv - 1) / th + 1, th>>>(
        knorms.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        kmaxes.packed_accessor32<at::BFloat16,1,torch::RestrictPtrTraits>(),
        bhkv, nk, th);

    for (int rep = 0; rep < int(nk / n_seg); ++rep) {
      apply_norm<<<dim3(bhkv, n_seg), threads>>>(
          k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
          kmaxes.packed_accessor32<at::BFloat16,1,torch::RestrictPtrTraits>(),
          bhkv, nk, d, n_seg);
    }
  }

  // Main computation
  if (mask) {
    calc_masked_cons_and_denum<<<blocks, threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        denum.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_masked_lin<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        denum.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);
  } else {
    calc_unmasked_cons_and_denum<<<blocks, threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        denum.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);

    calc_unmasked_lin<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
        q_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        k_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        v_c.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        o.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
        denum.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
        bh, nq, nk, d, bhratio);
  }

  // FIXED: pass nq here
  calc_div<<<dim3(szrb, blocks), threads, 2 * d * sizeof(__nv_bfloat16)>>>(
      o.packed_accessor32<at::BFloat16,3,torch::RestrictPtrTraits>(),
      denum.packed_accessor32<at::BFloat16,2,torch::RestrictPtrTraits>(),
      bh, nq, d);

  cudaDeviceSynchronize();

  return {o, denum};
}
