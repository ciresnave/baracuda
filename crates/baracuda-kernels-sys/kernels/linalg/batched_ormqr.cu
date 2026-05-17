// baracuda-kernels Phase 6 Category Linalg — Bespoke batched-`ormqr`
// (Milestone 6.14).
//
// One launch applies the implicit Householder-encoded Q (or Q^T) from a
// `BatchedQrPlan` packed output to every batch slot. cuSOLVER's `ormqr`
// is non-batched, so for the small-matrix regime where batched-QR is
// most useful per-slot launch latency dominates. This bespoke kernel
// amortizes one launch over the whole batch.
//
// Scope (Milestone 6.18): Side ∈ {Left, Right}, op ∈ {N, T, C},
// dtype ∈ {f32, f64, Complex32, Complex64}. WY blocking is a future
// optimization — the kernel is correctness-first GEMV-rates.

#include <cuComplex.h>
#include "../include/baracuda_batched_ormqr.cuh"

// Apply-Q / Apply-Q^T / Apply-Q^H batched.
// Accumulator type matches the storage type (no precision boost).
BARACUDA_KERNELS_BATCHED_ORMQR_INSTANTIATE(batched_ormqr_f32,       float,           float)
BARACUDA_KERNELS_BATCHED_ORMQR_INSTANTIATE(batched_ormqr_f64,       double,          double)
BARACUDA_KERNELS_BATCHED_ORMQR_INSTANTIATE(batched_ormqr_complex32, cuFloatComplex,  cuFloatComplex)
BARACUDA_KERNELS_BATCHED_ORMQR_INSTANTIATE(batched_ormqr_complex64, cuDoubleComplex, cuDoubleComplex)
