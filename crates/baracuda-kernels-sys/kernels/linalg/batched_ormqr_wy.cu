// baracuda-kernels Phase 6 Category Linalg — WY-blocked batched-`ormqr`
// (Milestone 6.17).
//
// Companion to `batched_ormqr.cu` (Milestone 6.14). Two bespoke kernels:
//
//  - `batched_ormqr_wy_build_t` — builds the block-reflector matrix T per
//    LAPACK DLARFT.
//  - `batched_ormqr_wy_extract_v` — materializes the dense V column
//    panel for one block (implicit-1 → explicit) so cuBLAS GEMM can
//    consume it.
//
// The three GEMMs that apply each block reflector (V^T·C, T·W, C-V·W)
// are issued at the safe-plan layer through `cublas{S,D}gemmStridedBatched`
// rather than in this header — keeps the .cu free of cuBLAS headers
// and keeps the dispatch in the Rust plan.
//
// Scope (trailblazer): Side = Left, op ∈ {N, T}, dtype ∈ {f32, f64}.

#include "../include/baracuda_batched_ormqr_wy.cuh"

BARACUDA_KERNELS_BATCHED_ORMQR_WY_BUILD_T_INSTANTIATE(batched_ormqr_wy_build_t_f32, float, float)
BARACUDA_KERNELS_BATCHED_ORMQR_WY_BUILD_T_INSTANTIATE(batched_ormqr_wy_build_t_f64, double, double)
BARACUDA_KERNELS_BATCHED_ORMQR_WY_EXTRACT_V_INSTANTIATE(batched_ormqr_wy_extract_v_f32, float)
BARACUDA_KERNELS_BATCHED_ORMQR_WY_EXTRACT_V_INSTANTIATE(batched_ormqr_wy_extract_v_f64, double)
