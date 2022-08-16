#pragma once
#include <torch/types.h>


#if defined(WITH_CUDA) || defined(WITH_HIP)
void resample2d_kernel_forward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& output,
    int kernel_size,
    bool bilinear);

void resample2d_kernel_backward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1, 
    at::Tensor& gradInput2, 
    int kernel_size,
    bool bilinear);
#endif


 inline int resample2d_forward(
    at::Tensor& input1,
    at::Tensor& input2, 
    at::Tensor& output,
    int kernel_size, bool bilinear) {

#if defined(WITH_CUDA) || defined(WITH_HIP)
      resample2d_kernel_forward(input1, input2, output, kernel_size, bilinear);
#endif

    return 1;
}

inline int resample2d_backward(
    at::Tensor& input1, 
    at::Tensor& input2,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1, 
    at::Tensor& gradInput2, 
    int kernel_size, bool bilinear) {

#if defined(WITH_CUDA) || defined(WITH_HIP)
        resample2d_kernel_backward(input1, input2, gradOutput, gradInput1, gradInput2, kernel_size, bilinear);
#endif

    return 1;
}
