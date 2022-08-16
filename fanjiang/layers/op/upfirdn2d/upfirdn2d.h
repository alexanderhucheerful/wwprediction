#pragma once
#include <torch/types.h>


#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor upfirdn2d_op(const at::Tensor& input, const at::Tensor& kernel,
                            int up_x, int up_y, int down_x, int down_y,
                            int pad_x0, int pad_x1, int pad_y0, int pad_y1);
#endif


inline at::Tensor upfirdn2d(const at::Tensor& input, const at::Tensor& kernel,
                        int up_x, int up_y, int down_x, int down_y,
                        int pad_x0, int pad_x1, int pad_y0, int pad_y1) {

#if defined(WITH_CUDA) || defined(WITH_HIP)
    return upfirdn2d_op(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1);
#endif

}
