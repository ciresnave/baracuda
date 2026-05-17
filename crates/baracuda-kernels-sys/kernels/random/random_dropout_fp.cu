// baracuda-kernels Phase 4.5: dropout FW kernels.
//
// `y = mask * x * scale`, `mask = (rand < (1 - p))` Bool, where
// `scale = 1 / (1 - p)`. f32 and f64 only — f16 / bf16 random deferred
// (cuRAND has limited half-precision support; the f32-detour would slow
// dropout down without buying meaningful accuracy).

#include "../include/baracuda_random.cuh"

BARACUDA_KERNELS_DROPOUT_INSTANTIATE(f32, float, float)
BARACUDA_KERNELS_DROPOUT_INSTANTIATE(f64, double, double)
