#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


namespace {

__global__
void calc_gradq_unmasked0(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
    // calc q 0 
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = 0; l < nk; ++l){
      tv = k[ikv][l][m];
      s[m] = v[ikv][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        tr[outer] += tv*s[oo*szr+outer];
      }
    }
    for(int l = 0; l < nq; ++l){
      t = 0;
      s[d+m] = grad_output[i][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        t += tr[outer]*s[d+oo*szr+outer];
      }
      atomicAdd(&gradq[i][l][m], t);
    }
  }
}


__global__
void calc_gradq_unmasked1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d, int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
    // calc q 0 
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = 0; l < nk; ++l){
      tv = k[ikv][l][m];
      for(int outer = 0; outer < szr; ++outer){
        tr[outer] += tv;
      }
    }
    for(int l = 0; l < nq; ++l){
      t = 0;
      s[m] = o[i][l][m];
      s[d+m] = grad_output[i][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        int ooo = oo*szr+outer;
        t += tr[outer]*s[ooo]*s[d+ooo];
      }
      atomicAdd(&gradq[i][l][m], -t);
    }

  }
}


__global__
void calc_gradq_masked0(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc q 0 
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = 0; l < nk-nq; ++l){
      tv = k[ikv][l][m];
      s[m] = v[ikv][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        tr[outer] += tv*s[oo*szr+outer];
      }
    }
    for(int l = 0; l < nq; ++l){
      t = 0;
      tv = k[ikv][l][m];
      s[m] = v[ikv][l][m];
      s[d+m] = grad_output[i][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        int ooo = oo*szr+outer;
        tr[outer] += tv*s[ooo];
        t += tr[outer]*s[d+ooo];
      }
      atomicAdd(&gradq[i][l][m], t);
    }

  }
}


__global__
void calc_gradq_masked1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradq, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  int ndiff = nk-nq;
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc q 0 
    
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = 0; l < nk-nq; ++l){
      tv = k[ikv][l][m];
      for(int outer = 0; outer < szr; ++outer){
        tr[outer] += tv;
      }
    }
    for(int l = 0; l < nq; ++l){
      t = 0;
      tv = k[ikv][ndiff+l][m];
      s[m] = o[i][l][m];
      s[d+m] = grad_output[i][l][m];
      __syncthreads();        
      for(int outer = 0; outer < szr; ++outer){
        int ooo = oo*szr+outer;
        tr[outer] += tv;
        t += tr[outer]*s[ooo]*s[d+ooo];
      }
      atomicAdd(&gradq[i][l][m], -t);
    }
  }
}


__global__
void calc_gradk_unmasked0(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){

  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = 0; l < nq; ++l){
      tv = q[i][l][m];
      s[m] = grad_output[i][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        tr[outer] += tv*s[oo*szr+outer];
      }
    }
    for(int l = 0; l < nk; ++l){
      t = 0;
      s[d+m] = v[ikv][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        t += tr[outer]*s[d+oo*szr+outer];
      }
      atomicAdd(&gradk[ikv][l][m], t);
    }
  }
}


__global__
void calc_gradk_unmasked1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){

  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // UNMASKED PART ////////////////////////////
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = 0; l < nq; ++l){
      t = 0;
      tv = q[i][l][m];
      s[m] = grad_output[i][l][m];
      s[d+m] = o[i][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        int ooo = oo*szr + outer;
        tr[outer] += s[d+ooo]*tv*s[ooo];
      }
    }

    for(int l = 0; l < nk; ++l){
      t = 0;
      for(int outer = 0; outer < szr; ++outer){
        t += tr[outer];
      }
      atomicAdd(&gradk[ikv][l][m], -t);
    }
  }
}


__global__
void calc_gradk_masked0(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc k 0 
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = nq-1; l >= 0; --l){
      t = 0;
      tv = q[i][l][m];
      s[m] = grad_output[i][l][m];
      s[d+m] = v[ikv][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        int ooo = oo*szr + outer;
        tr[outer] += tv*s[ooo];
        t += tr[outer]*s[d+ooo];
      }
      gradk[ikv][l][m] += t;
    }
    for(int l = nk-nq-1; l >= 0; --l){
      t = 0;
      s[d+m] = v[ikv][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        t += tr[outer]*s[d+oo*szr+outer];
      }
      atomicAdd(&gradk[ikv][l][m], t);
    }
  }
}


__global__
void calc_gradk_masked1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradk, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){
  extern __shared__ float s[];
  const int m = threadIdx.x;
  const int oo = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(m < d && oo < szrb && i < bh){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    for(int outer = 0; outer < szr; ++outer) tr[outer] = 0;
    for(int l = nq-1; l >= 0; --l){
      t = 0;
      tv = q[i][l][m];
      s[m] = grad_output[i][l][m];
      s[d+m] = o[i][l][m];
      __syncthreads();
      for(int outer = 0; outer < szr; ++outer){
        int ooo = oo*szr + outer;
        tr[outer] += s[d+ooo]*tv*s[ooo];
        t += tr[outer];
      }
      gradk[ikv][l][m] -= t;
    }
    for(int l = nk-nq-1; l >= 0; --l){
      t = 0;
      for(int outer = 0; outer < szr; ++outer){
        t += tr[outer];
      }
      atomicAdd(&gradk[ikv][l][m], -t);
    }
  }
}


__global__
void calc_gradv_unmasked0(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(i < bh && outer < d){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc v 0    
    t = 0;
    for(int l = 0; l < nq; ++l){
      t += grad_output[i][l][outer];
    }
    for(int l = 0; l < nk; ++l){
      gradv[ikv][l][outer] += t;
    }

  }
}


__global__
void calc_gradv_unmasked1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(i < bh && mm < szrb && outer < d){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc v 1    
    for(int m = 0; m < szr; ++m) tr[m] = 0;
    for(int l = 0; l < nq; ++l){
      t = 0;
      s[outer] = q[i][l][outer];
      s[d+outer] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        int mmm = mm*szr + m;
        tr[m] += s[mmm] * grad_output[i][l][outer];
      }
    }
    for(int l = 0; l < nk; ++l){
      t = 0;
      s[d+outer] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        t += tr[m]*s[d+mm*szr+m];
      }
      atomicAdd(&gradv[ikv][l][outer], t);
    }
  }
}


__global__
void calc_gradv_masked0(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(i < bh && outer < d){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc v 0    
    t = 0;
    for(int l = nq-1; l >= 0; --l){
      t += grad_output[i][l][outer];
      gradv[ikv][l][outer] += t;
    }
    for(int l = nk-nq-1; l >= 0; --l){
      gradv[ikv][l][outer] += t;
    }

  }
}


__global__
void calc_gradv_masked1(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gradv, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, int bh, int nq, int nk, int d,int bhratio){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  int ikv;
  float tv, t;
  float tr[32];
  int sz = min(32,d);
  int szr = 8;  //number of reductions happeing in each thread; should be ~sqrt(d)
  int szrb = d/szr; //number of reductions blocks; should be d/szr
  if(i < bh && mm < szrb && outer < d){
    ikv = i/bhratio;
    // MASKED PART ////////////////////////////
    // calc v 1    
    for(int m = 0; m < szr; ++m) tr[m] = 0;
    for(int l = nq-1; l >= 0; --l){
      t = 0;
      s[outer] = q[i][l][outer];
      s[d+outer] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        int mmm = mm*szr + m;
        tr[m] += s[mmm] * grad_output[i][l][outer];
        t += tr[m]*s[d+mmm];
      }
      atomicAdd(&gradv[ikv][l][outer], t);
    }
    for(int l = nk-nq-1; l >= 0; --l){
      t = 0;
      s[d+outer] = k[ikv][l][outer];
      __syncthreads();
      for(int m = 0; m < szr; ++m){
        t += tr[m]*s[d+mm*szr+m];
      }
      atomicAdd(&gradv[ikv][l][outer], t);
    }
   
  }
}


__global__
void div_grad_output(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> grad_output, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int d){
  const int m = threadIdx.x;
  const int mm = blockIdx.x;
  const int i = blockIdx.y;
  if(m < d && mm < d && i < bh){
    for(int l = mm; l < nq; l += d) grad_output[i][l][m] /= o[i][l][d];
  }
}


} // namespace

std::vector<torch::Tensor> backward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor o,
    torch::Tensor grad_output,  
    bool mask){


    const auto nq = q.size(1);
    const auto nk = k.size(1);
    const auto bh = q.size(0);
    const auto bhkv = k.size(0);
    const auto d = q.size(2);
    const int bhratio = bh/bhkv;

    const int threads = d;
    const int blocks = bh;
    // int szr = int(sqrt(d)); //number of reductions happeing in each thread; should be ~sqrt(d)
    // szr = 16;
    // int szrb = int(d/szr); //number of blocks performing reduction; should be ~sqrt(d)
    int szr = 8;
    int szrb = d/szr; //number of reduction blocks; should be ~sqrt(d)
    
    auto opts =  torch::TensorOptions().dtype(q.dtype()).layout(torch::kStrided).device(q.device());

    auto gradq = torch::zeros({bh,nq,d},opts);
    auto gradk = torch::zeros({bhkv,nk,d},opts);
    auto gradv = torch::zeros({bhkv,nk,d},opts);
  
    div_grad_output<<<dim3(d,blocks),threads>>>(grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,d);
    cudaDeviceSynchronize();

    if(mask){
      calc_gradq_masked0<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradq_masked1<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradk_masked0<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradk_masked1<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradv_masked0<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradv_masked1<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
    }
    else{
      calc_gradq_unmasked0<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradq_unmasked1<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradq.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradk_unmasked1<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradk_unmasked0<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradk.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradv_unmasked0<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
      calc_gradv_unmasked1<<<dim3(szrb,blocks),threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(), k.packed_accessor32<float,3,torch::RestrictPtrTraits>(), v.packed_accessor32<float,3,torch::RestrictPtrTraits>(), o.packed_accessor32<float,3,torch::RestrictPtrTraits>(), gradv.packed_accessor32<float,3,torch::RestrictPtrTraits>(), grad_output.packed_accessor32<float,3,torch::RestrictPtrTraits>(), bh,nq,nk,d,bhratio);
    }

    cudaDeviceSynchronize();

    // auto out = {gradq,gradk,gradv};
    return {gradq,gradk,gradv};
}
