// baracuda-kernels Phase 7 Milestone 7.3 — `gather` FW + BW kernels.
//
// Trailblazer dtype coverage: f32, f64, i32 for FW; f32, f64 for BW
// (BW uses atomicAdd — restrict to FP for native support).
//
// Phase 11.5 (Fuel team feedback #7): adds i64 index instantiations
// alongside the original i32 ones. PyTorch defaults to int64 for
// indices, so the i64 variants spare callers a cast pass. Legacy
// `_f32 / _f64 / _i32` names keep the i32-index ABI; `_i64idx_*`
// suffix names are the new i64 entry points.

#include "../include/baracuda_indexing.cuh"

// i32 index — legacy / default surface (kept under the original names).
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_f32, float,   int32_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_f64, double,  int32_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i32, int32_t, int32_t)

// i64 index — Phase 11.5 additions.
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i64idx_f32, float,   int64_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i64idx_f64, double,  int64_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_i64idx_i32, int32_t, int64_t)

// Backward — atomicAdd into dsrc, FP-only.
BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_f32, float,  int32_t)
BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_f64, double, int32_t)

BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_GATHER_BACKWARD_INSTANTIATE(gather_backward_i64idx_f64, double, int64_t)

// Phase 39 (Fuel 6c.4 Gap 5) — u8 idx fanout for gather FW.
// The kernel templates compile fine against `uint8_t` (the kernel's
// `(int64_t)index[off]` zero-extends), and the bounds check
// `idx_val < 0 || idx_val >= src_dim_size` correctly accepts the
// 0..=255 unsigned range. No IndexElement Rust trait extension yet
// (sealed trait change deferred to a follow-up); these symbols are
// callable directly by FFI consumers.
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_u8idx_f32, float,  uint8_t)
BARACUDA_KERNELS_GATHER_INSTANTIATE(gather_u8idx_f64, double, uint8_t)
