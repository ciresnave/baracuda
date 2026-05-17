// baracuda-kernels Milestone 6.4: in-place scale helpers for cuFFT
// inverse-transform 1/N normalization.
//
// cuFFT's inverse transforms are unnormalized (`cufftExecC2C` /
// `cufftExecZ2Z` / `cufftExecC2R` / `cufftExecZ2D` return N · IFFT(x)).
// PyTorch's `norm="backward"` convention wants IFFT(x). The safe-plan
// layer multiplies the output by 1/N after the inverse exec via these
// helpers. Two flavors per output kind:
//
//   * c32 / c64 — complex output, used after IFFT (C2C).
//   * real_f32 / real_f64 — real output, used after IRFFT (C2R / Z2D).

#include "../include/baracuda_fft.cuh"

BARACUDA_KERNELS_SCALE_INPLACE_C32_INSTANTIATE()
BARACUDA_KERNELS_SCALE_INPLACE_C64_INSTANTIATE()
BARACUDA_KERNELS_SCALE_INPLACE_REAL_INSTANTIATE(f32, float)
BARACUDA_KERNELS_SCALE_INPLACE_REAL_INSTANTIATE(f64, double)
