// baracuda-kernels Milestone 6.10: N-D fftshift / ifftshift.
//
// Single-pass general-permutation kernel — one thread per output cell,
// each thread decomposes its flat output index into per-axis coords and
// rotates the subset of "shifted" axes by their per-axis shift amount.
// Beats chained 1-D shifts at any rank > 1 (one read + one write per
// element vs N reads + N writes for the chained variant).
//
// Element-width specialized (4 / 8 / 16-byte cells) matching the 1-D
// variant — fftshift is a pure index permutation so the element type is
// irrelevant beyond its byte width. The safe-plan layer dispatches on
// `std::mem::size_of::<T>()`.

#include "../include/baracuda_fft.cuh"

BARACUDA_KERNELS_FFTSHIFT_ND_INSTANTIATE(4,  uint32_t)
BARACUDA_KERNELS_FFTSHIFT_ND_INSTANTIATE(8,  uint2)
BARACUDA_KERNELS_FFTSHIFT_ND_INSTANTIATE(16, uint4)
