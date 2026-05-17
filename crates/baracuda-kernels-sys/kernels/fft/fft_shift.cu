// baracuda-kernels Milestone 6.4: fftshift / ifftshift.
//
// Element-width specialized — same templated kernel re-instantiated at
// 4 / 8 / 16-byte cell widths so it covers `f32`, `f64` / `Complex32`,
// and `Complex64` without templating on the concrete float / struct
// types. The safe-plan layer dispatches on `std::mem::size_of::<T>()`.

#include "../include/baracuda_fft.cuh"

BARACUDA_KERNELS_FFTSHIFT_INSTANTIATE(4,  uint32_t)
BARACUDA_KERNELS_FFTSHIFT_INSTANTIATE(8,  uint2)
BARACUDA_KERNELS_FFTSHIFT_INSTANTIATE(16, uint4)

BARACUDA_KERNELS_IFFTSHIFT_INSTANTIATE(4,  uint32_t)
BARACUDA_KERNELS_IFFTSHIFT_INSTANTIATE(8,  uint2)
BARACUDA_KERNELS_IFFTSHIFT_INSTANTIATE(16, uint4)
