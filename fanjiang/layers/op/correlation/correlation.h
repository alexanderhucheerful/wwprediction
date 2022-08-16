#pragma once
#include <torch/types.h>
#include <vector>

#if defined(WITH_CUDA) || defined(WITH_HIP)
std::vector<at::Tensor> corr_cuda_forward(
    at::Tensor fmap1,
    at::Tensor fmap2,
    at::Tensor coords,
    int radius);

std::vector<at::Tensor> corr_cuda_backward(
  at::Tensor fmap1,
  at::Tensor fmap2,
  at::Tensor coords,
  at::Tensor corr_grad,
  int radius);
#endif


inline std::vector<at::Tensor> corr_forward(
    at::Tensor fmap1,
    at::Tensor fmap2,
    at::Tensor coords,
    int radius) {

#if defined(WITH_CUDA) || defined(WITH_HIP)
  return corr_cuda_forward(fmap1, fmap2, coords, radius);
#endif


}


inline std::vector<at::Tensor> corr_backward(
    at::Tensor fmap1,
    at::Tensor fmap2,
    at::Tensor coords,
    at::Tensor corr_grad,
    int radius) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  return corr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
#endif

}


