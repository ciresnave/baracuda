// baracuda-kernels Phase 4.5: dropout BW kernels.
//
// `dx = dy * mask * scale`. Mask is the packed-Bool tensor the matching
// forward kernel emitted; `scale = 1 / (1 - p)`. f32 and f64 only.

#include "../include/baracuda_random.cuh"

BARACUDA_KERNELS_DROPOUT_BACKWARD_INSTANTIATE(f32, float, float)
BARACUDA_KERNELS_DROPOUT_BACKWARD_INSTANTIATE(f64, double, double)
