#include <iostream>
#include <vector>
#include <math.h>
#include <torch/extension.h>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
using namespace std;


torch::Tensor forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool mask);

vector<torch::Tensor> backward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, bool mask);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor forwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool mask){

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  return forward_cuda(q, k, v, mask);
}

vector<torch::Tensor> backwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, bool mask){
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
  return backward_cuda(q, k, v, o, grad_output, mask);
}

PYBIND11_MODULE(fastmax_cuda, m) {
  m.def("forwardpass", &forwardpass, "forwardpass");
  m.def("backwardpass", &backwardpass, "backwardpass");
}
