// baracuda-kernels Phase 9 Category T — non-max suppression.

#include "../include/baracuda_nms.cuh"

BARACUDA_KERNELS_NMS_INSTANTIATE(nms_f32, float)
BARACUDA_KERNELS_NMS_INSTANTIATE(nms_f64, double)
