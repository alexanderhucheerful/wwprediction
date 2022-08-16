#pragma once
#include <torch/types.h>
#include <vector>


#if defined(WITH_CUDA) || defined(WITH_HIP)
// CUDA forward declarations
torch::Tensor nattenqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb);

// CUDA backward declarations
std::vector<torch::Tensor> nattenqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key);
#endif


// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline torch::Tensor nattenqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb) {
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_INPUT(rpb);
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return nattenqkrpb_cuda_forward(query, key, rpb);
#endif

}

inline std::vector<torch::Tensor> nattenqkrpb_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key) {
  CHECK_INPUT(d_attn);
  CHECK_INPUT(query);
  CHECK_INPUT(key);
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return nattenqkrpb_cuda_backward(d_attn, query, key);
#endif

}
