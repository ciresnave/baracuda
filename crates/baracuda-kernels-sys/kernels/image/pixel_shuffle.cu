// baracuda-kernels Phase 9 Category T — pixel_shuffle / pixel_unshuffle.

#include "../include/baracuda_pixel_shuffle.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

BARACUDA_KERNELS_PIXEL_SHUFFLE_INSTANTIATE(pixel_shuffle_f32, float)
BARACUDA_KERNELS_PIXEL_SHUFFLE_INSTANTIATE(pixel_shuffle_f64, double)
BARACUDA_KERNELS_PIXEL_SHUFFLE_INSTANTIATE(pixel_shuffle_f16, __half)
BARACUDA_KERNELS_PIXEL_SHUFFLE_INSTANTIATE(pixel_shuffle_bf16, __nv_bfloat16)

BARACUDA_KERNELS_PIXEL_UNSHUFFLE_INSTANTIATE(pixel_unshuffle_f32, float)
BARACUDA_KERNELS_PIXEL_UNSHUFFLE_INSTANTIATE(pixel_unshuffle_f64, double)
BARACUDA_KERNELS_PIXEL_UNSHUFFLE_INSTANTIATE(pixel_unshuffle_f16, __half)
BARACUDA_KERNELS_PIXEL_UNSHUFFLE_INSTANTIATE(pixel_unshuffle_bf16, __nv_bfloat16)
