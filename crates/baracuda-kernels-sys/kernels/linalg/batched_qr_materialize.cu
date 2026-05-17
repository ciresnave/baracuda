// baracuda-kernels Phase 6 Category Linalg — Batched-QR dense Q/R
// materialization helpers (Milestone 6.14, Piece 2).
//
// Two tiny bespoke kernels that together (with `BatchedOrmqrPlan`) turn
// the cuBLAS `geqrfBatched` packed output (A packed with R in upper +
// Householder reflectors in lower; tau vector) into dense `Q [B, M, M]`
// and `R [B, K, N]` tensors.
//
//   1. Upper-triangle copy → R: one block per (batch, column), threads
//      stride over output rows. R cell (i, j) = A_packed[i, j] if i ≤ j
//      else 0.
//
//   2. Identity stage → Q: one block per (batch, column), threads
//      stride over rows. Q cell (i, j) = 1 if i == j else 0. Caller
//      follows with `BatchedOrmqrPlan` (Left, op=N) to overwrite Q
//      in place with the dense Q matrix from the Householder
//      reflectors.
//
// Dtype scope mirrors `BatchedOrmqrPlan`: f32 + f64 only.

#include "../include/baracuda_batched_ormqr.cuh"

BARACUDA_KERNELS_BATCHED_QR_MATERIALIZE_R_INSTANTIATE(batched_qr_materialize_r_f32, float)
BARACUDA_KERNELS_BATCHED_QR_MATERIALIZE_R_INSTANTIATE(batched_qr_materialize_r_f64, double)

BARACUDA_KERNELS_BATCHED_QR_MATERIALIZE_IDENTITY_INSTANTIATE(batched_qr_materialize_identity_f32, float)
BARACUDA_KERNELS_BATCHED_QR_MATERIALIZE_IDENTITY_INSTANTIATE(batched_qr_materialize_identity_f64, double)
