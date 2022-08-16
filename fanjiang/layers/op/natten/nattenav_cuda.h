#pragma once
#include <torch/types.h>
#include <vector>


#if defined(WITH_CUDA) || defined(WITH_HIP)
// CUDA forward declaration
torch::Tensor nattenav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value);

// CUDA backward declaration
std::vector<torch::Tensor> nattenav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value);
#endif

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline torch::Tensor nattenav_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value) {
  CHECK_INPUT(attn);
  CHECK_INPUT(value);

#if defined(WITH_CUDA) || defined(WITH_HIP)
  return nattenav_cuda_forward(attn, value);
#endif

}

inline std::vector<torch::Tensor> nattenav_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value) {
  CHECK_INPUT(d_out);
  CHECK_INPUT(attn);
  CHECK_INPUT(value);
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return nattenav_cuda_backward(d_out, attn, value);
#endif

}


