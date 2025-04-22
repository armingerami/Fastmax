#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>



namespace {

// UNMASKED PART ////////////////////////////
__global__
void calc_unmasked_cons_and_denum(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  // int szr = 16; //number of reductions happeing in each thread; should be ~sqrt(d)
  // int szrb = 4; //number of reductions happeing in each thread; should be d/szr
  if(outer < d && i < bh){
    ikv = i/bhratio;
    // calc lin denum
    t = 0;
    for(int l = 0; l < nk; ++l){
      t += k[ikv][l][outer];
    }
    float a0div = nk/d;
    for(int l = 0; l < nq; ++l){
      atomicAdd(&o[i][l][d], q[i][l][outer]*t + a0div);
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk;  ++l){
      t += v[ikv][l][outer];
    }
    for(int l = 0; l < nq;  ++l){
      o[i][l][outer] = t;
    }

  }
}


__global__
void calc_unmasked_lin(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  int szr = 8; //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions happeing in each thread; should be d/szr
  if(outer < d && mm < szrb && i < bh){
    ikv = i/bhratio;
    // calc lin
    for(int m = 0; m < szr; ++m) tr[m] = 0;
    for(int l = 0; l < nk;  ++l){
      tv = v[ikv][l][outer];
      s[outer+d] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        tr[m] += s[d+mm*szr+m]*tv;
      }
    }
    for(int l = 0; l < nq;  ++l){
      s[outer] = q[i][l][outer];
      __syncthreads();
      t = 0;
      for(int m = 0; m < szr; ++m){
        t += tr[m]*s[mm*szr+m];
      }
      atomicAdd(&o[i][l][outer],t);
    }

    // for(int l = mm; l < nq; l += szrb) o[i][l][outer] /= o[i][l][d];
  }
}

// UNMASKED PART ////////////////////////////
__global__
void calc_masked_cons_and_denum(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  int ndiff = nk-nq;
  // int szr = 16; //number of reductions happeing in each thread; should be ~sqrt(d)
  // int szrb = 4; //number of reductions happeing in each thread; should be d/szr
  //     s[d+outer] = 0;
//     __syncthreads();
//     for(int l = 0; l < nk-nq; ++l){
//       s[d+outer] += k[ikv][l][outer];
//     }
//     __syncthreads();
//     for(int l = 0; l < nq; ++l){
//       s[d+outer] += k[ikv][nk-nq+l][outer];
//       s[outer] = q[i][l][outer];
//       __syncthreads();
//       if(outer == 0){
//         t = 0;
//         for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
//         o[i][l][d] = t + (nk-nq+l+1);
//       }
//     }
  if(outer < d && i < bh){
    ikv = i/bhratio;
    // calc lin denum
    t = 0;
    for(int l = 0; l < nk-nq; ++l){
      t += k[ikv][l][outer];
    }
    float a0div = 1/d;
    int ndiff1 = ndiff+1;
    for(int l = 0; l < nq; ++l){
      t += k[ikv][ndiff+l][outer];
      atomicAdd(&o[i][l][d], q[i][l][outer]*t + a0div*(ndiff1+l));
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk-nq; ++l){
      t += v[ikv][l][outer];
    }
    for(int l = 0; l < nq; ++l){
      t += v[ikv][ndiff+l][outer];
      o[i][l][outer] = t;
    }

  }
}


__global__
void calc_masked_lin(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  int loc1, loc2;
  float tr[32];
  int sz = min(32,d);
  int szr = 8; //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions happeing in each thread; should be d/szr
  int ndiff = nk-nq;
  if(outer < d && mm < szrb && i < bh){
    ikv = i/bhratio;
    // calc lin
    for(int m = 0; m < szr; ++m) tr[m] = 0;
    for(int l = 0; l < nk-nq;  ++l){
      tv = v[ikv][l][outer];
      s[outer+d] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        tr[m] += s[d+mm*szr+m]*tv;
      }
    }
    for(int l = 0; l < nq;  ++l){
      tv = v[ikv][ndiff+l][outer];
      s[outer+d] = k[ikv][ndiff+l][outer];
      s[outer] = q[i][l][outer];
      __syncthreads();
      t = 0;
      for(int m = 0; m < szr; ++m){
        int mmm = mm*szr+m;
        tr[m] += s[d+mmm]*tv;
        t += tr[m]*s[mmm];
      }
      atomicAdd(&o[i][l][outer],t);
    }

  }
}

__global__
void calc_div(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int d){
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int szr = 8; //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions happeing in each thread; should be d/szr
  if(outer < d && mm < szrb && i < bh){
    
    for(int l = mm; l < nq; l += szrb) o[i][l][outer] /= o[i][l][d];
  }
}


__global__
void calc_norms(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, int bh, int n, int d, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  const int l = blockIdx.y;
  float t;
  int i;
  if(l < n && ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    t = 0;
    for(int m = 0; m < d; m++){
      t += a[i][l][m]*a[i][l][m];
    }
    norms[i][l] = t;
  }
}

__global__
void find_max(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  float t = 0;
  int i;
  if(ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    for(int l = 0; l < n; ++l){
      t = max(t,norms[i][l]);
    }
    maxes[i] = t;
  }
}

__global__
void apply_norm(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int d, int n_seg){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  const int j = blockIdx.y;
  const int np = int(n/n_seg);
  float mx;
  if(m < d && i < bh){
    mx = maxes[i];
    if(mx < 0.1) mx = 0.1;
    for(int l = j*np; l < min(n,(j+1)*np); ++l){
      a[i][l][m] /= mx;
    }
  }
}


__global__
void apply_permute(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a_p, int b, int h, int n, int d, int dir){
  const int m = threadIdx.x;
  const int j = blockIdx.x;
  const int i = blockIdx.y;
  if(m < d && i < b && j < h){
    for(int l = 0; l < n; ++l){
      if(dir == 0) a_p[l][i*h+j][m] = a[i][l][j][m];
      else a[i][l][j][m] = a_p[l][i*h+j][m];
    }
  }
}

} // namespace

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool mask){
    // q: (nq,bh,d)
    // k: (nk,bh,d)
    // v: (nk,bh,d)

  // const auto nq = q_old.size(1);
  // const auto nk = k_old.size(1);
  // const auto b = q_old.size(0);
  // const auto h = q_old.size(2);
  // const auto d = q_old.size(3);
  // const auto bh = b*h;

  // const auto nq = q.size(0);
  // const auto nk = k.size(0);
  // const auto bh = q.size(1);
  // const auto d = q.size(2);

  const auto nq = q.size(1);
  const auto nk = k.size(1);
  const auto bh = q.size(0);
  const auto bhkv = k.size(0);
  const auto d = q.size(2);
  const int bhratio = bh/bhkv;

  const int threads = d; // threads = 256
  const int blocks = bh;

  const int n_seg = 128; // breaks context length into segments of n_seg, which are parallelized; i.e., paralleizes the code n_seg times
  // int szr = int(sqrt(d)); //number of reductions happeing in each thread; should be ~sqrt(d)
  // szr = 16;
  // int szrb = int(d/szr); //number of blocks performing reduction; should be ~sqrt(d)
  int szr = 8;
  int szrb = d/szr; //number of blocks performing reduction; should be ~sqrt(d)

  auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
  // auto q = torch::zeros({nq,bh,d},opts);
  // auto k = torch::zeros({nk,bh,d},opts);
  // auto v = torch::zeros({nk,bh,d},opts);
  // auto drop_noise = torch::zeros({nq,bh,d},opts);
  // auto o = torch::zeros({nq,bh,d+1},opts);
  auto o = torch::zeros({bh,nq,d+1},opts);
  // auto out = torch::zeros({b,nq,h,d+1},opts);
  auto qnorms = torch::zeros({bh,nq},opts);
  auto knorms = torch::zeros({bh,nk},opts);
  auto qmaxes = torch::zeros({bh},opts);
  auto kmaxes = torch::zeros({bh},opts);


  // apply_permute<<<dim3(h,b),threads>>>(q_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(k_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(v_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(drop_noise_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);


  if(false){
    const long th_lim = 1024;
    int th = min(th_lim, bh);
    calc_norms<<<dim3((bh-1)/th + 1, nq),th>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nq,d,th);
    find_max<<<(bh-1)/th + 1,th>>>(qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk,th);
    for(int np = 0; np < int(nq/n_seg); ++np){
      apply_norm<<<dim3(blocks,n_seg),threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq,d,n_seg);
    }
    th = min(th_lim, bhkv);
    calc_norms<<<dim3((bhkv-1)/th + 1, nk),th>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bhkv,nk,d,th);
    find_max<<<(bhkv-1)/th + 1,th>>>(knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bhkv,nq,th);
    for(int np = 0; np < int(nk/n_seg); ++np){
      apply_norm<<<dim3(bhkv,n_seg),threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bhkv,nk,d,n_seg);
    }
  }

  if(mask){
    calc_masked_cons_and_denum<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
    calc_masked_lin<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
  }
  else{
    calc_unmasked_cons_and_denum<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
    calc_unmasked_lin<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,bhratio);
  }
  calc_div<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nk,d);
  

  cudaDeviceSynchronize();

  // apply_permute<<<dim3(h,b),threads+1>>>(out.packed_accessor32<float,4,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d+1,1);

  // delete q;
  // delete k;
  // delete v;
  // delete drop_noise;
  // delete o;

  return o;
  // return out;
}