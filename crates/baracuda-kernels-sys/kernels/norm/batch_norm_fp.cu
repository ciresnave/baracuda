// baracuda-kernels Phase 5.1 Category G: BatchNorm FW for FP types.
//
// Training mode only: computes per-channel mean / inv_std from the batch
// + spatial dimensions, writes them to saved buffers for BW reuse.
// Inference mode (using running statistics) is deferred — when wired
// it'll be a thin wrapper that becomes a per-channel affine multiply.
//
// Caller pre-collapses the input to logical [N, C, S] (S = product of
// spatial dims). group_kind=0 selects the BN dispatch in the shared
// BN/GN kernel.

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_BN_GN_INSTANTIATE(batch_norm_f32, float)
BARACUDA_KERNELS_BN_GN_INSTANTIATE(batch_norm_f16, __half)
BARACUDA_KERNELS_BN_GN_INSTANTIATE(batch_norm_bf16, __nv_bfloat16)
BARACUDA_KERNELS_BN_GN_INSTANTIATE(batch_norm_f64, double)
