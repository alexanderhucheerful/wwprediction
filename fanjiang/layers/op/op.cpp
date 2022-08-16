#include <torch/extension.h>
#include "correlation/correlation.h"
#include "resample2d/resample2d.h"
#include "upfirdn2d/upfirdn2d.h"
#include "natten/nattenav_cuda.h"
#include "natten/nattenqkrpb_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("resample2d_forward", &resample2d_forward, "resample2d_forward");
  m.def("resample2d_backward", &resample2d_backward, "resample2d_backward");
  m.def("corr_forward", &corr_forward, "corr_forward");
  m.def("corr_backward", &corr_backward, "corr_backward");  
  m.def("upfirdn2d", &upfirdn2d, "upfirdn2d");  
  m.def("nattenqkrpb_forward", &nattenqkrpb_forward, "NATTENQK+RPB forward (CUDA)");
  m.def("nattenqkrpb_backward", &nattenqkrpb_backward, "NATTENQK+RPB backward (CUDA)");
  m.def("nattenav_forward", &nattenav_forward, "NATTENAV forward (CUDA)");
  m.def("nattenav_backward", &nattenav_backward, "NATTENAV backward (CUDA)");

}
