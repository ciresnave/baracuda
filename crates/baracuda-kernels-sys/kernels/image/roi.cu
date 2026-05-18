// baracuda-kernels Phase 9 Category T — roi_align + roi_pool FW + BW.

#include "../include/baracuda_roi.cuh"

BARACUDA_KERNELS_ROI_ALIGN_INSTANTIATE(roi_align_f32, float)
BARACUDA_KERNELS_ROI_ALIGN_INSTANTIATE(roi_align_f64, double)

BARACUDA_KERNELS_ROI_ALIGN_BACKWARD_INSTANTIATE(roi_align_backward_f32, float)
BARACUDA_KERNELS_ROI_ALIGN_BACKWARD_INSTANTIATE(roi_align_backward_f64, double)

BARACUDA_KERNELS_ROI_POOL_INSTANTIATE(roi_pool_f32, float)
BARACUDA_KERNELS_ROI_POOL_INSTANTIATE(roi_pool_f64, double)

BARACUDA_KERNELS_ROI_POOL_BACKWARD_INSTANTIATE(roi_pool_backward_f32, float)
BARACUDA_KERNELS_ROI_POOL_BACKWARD_INSTANTIATE(roi_pool_backward_f64, double)
