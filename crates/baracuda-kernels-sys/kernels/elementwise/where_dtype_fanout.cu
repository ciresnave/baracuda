// Phase 38 (Fuel 6c.4 Gap 3) — `where_cond` ternary dtype-matrix fanout.
//
// Today's coverage (alpha.52) was U8-cond × {f32, f16, bf16, f64}
// (the trailblazer set in `where_fp.cu`). This file ships the full
// matrix Fuel needs:
//
//   - cond dtypes : U8 (already in `where_fp.cu` for the 4 fp values),
//                   U32, I64                                              (3 total)
//   - value dtypes: U8, I8, U32, I16, I32, I64, F16, Bf16, F32, F64,
//                   Fp8E4M3                                              (11 total)
//
// Naming convention follows the brief: the explicit `<cond>cond_` prefix
// is REQUIRED for U32 / I64 cond; for backward source-compat, U8-cond
// keeps the no-prefix `where_<value>_` symbols (`where_f32_run`, etc.)
// already shipped in `where_fp.cu`. New symbols here:
//
//   * U32-cond / I64-cond × 4 fp values × {contig, strided}            = 16
//   * U8 / U32 / I64-cond × 6 int values × {contig, strided}           = 36
//   * U8 / U32 / I64-cond × Fp8E4M3 × {contig, strided}                =  6
//
// Total Phase 38 = 58 new symbol pairs, each pair = `<sym>_run` + (for
// the contig form) `<sym>_can_implement`. The strided form has no
// `_can_implement` companion (matching the existing where family).
//
// Cond-type semantics: any non-zero value selects `a`, zero selects `b`.
// The template body uses `cond != Cond(0)`, which compiles to the
// natural `setp.ne` PTX instruction for any integer width — no perf
// penalty vs the original u8-cond instantiation.
//
// Fp8E4M3 transport: 1-byte storage type, treated as raw `uint8_t` at
// the kernel level (FP8 E4M3 encoding is bit-identical to `uint8_t`).
// The kernel body never compares Fp8E4M3 values for ordering — it only
// performs pure element selection — so the host-side dtype tag is
// irrelevant once the values land on the device.

#include "../include/baracuda_elementwise.cuh"

// ============================================================================
// (a) U32 / I64 cond × {f32, f16, bf16, f64} — 16 new symbols
// ============================================================================

BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_f32, float, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_f32, float, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_f64, double, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_f64, double, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_f16, __half, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_f16, __half, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_bf16, __nv_bfloat16, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_bf16, __nv_bfloat16, uint32_t)

BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_f32, float, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_f32, float, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_f64, double, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_f64, double, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_f16, __half, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_f16, __half, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_bf16, __nv_bfloat16, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_bf16, __nv_bfloat16, int64_t)

// ============================================================================
// (b) U8 / U32 / I64 cond × {u8, i8, u32, i16, i32, i64} — 36 new symbols
// ============================================================================

// --- U8-cond × int values
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u8cond_u8, uint8_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u8cond_u8, uint8_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u8cond_i8, int8_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u8cond_i8, int8_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u8cond_u32, uint32_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u8cond_u32, uint32_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u8cond_i16, int16_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u8cond_i16, int16_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u8cond_i32, int32_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u8cond_i32, int32_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u8cond_i64, int64_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u8cond_i64, int64_t, uint8_t)

// --- U32-cond × int values
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_u8, uint8_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_u8, uint8_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_i8, int8_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_i8, int8_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_u32, uint32_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_u32, uint32_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_i16, int16_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_i16, int16_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_i32, int32_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_i32, int32_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_i64, int64_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_i64, int64_t, uint32_t)

// --- I64-cond × int values
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_u8, uint8_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_u8, uint8_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_i8, int8_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_i8, int8_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_u32, uint32_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_u32, uint32_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_i16, int16_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_i16, int16_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_i32, int32_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_i32, int32_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_i64, int64_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_i64, int64_t, int64_t)

// ============================================================================
// (c) U8 / U32 / I64 cond × Fp8E4M3 — 6 new symbols (treated as uint8_t)
// ============================================================================
//
// FP8 E4M3 is 1-byte storage, bit-identical to `uint8_t`. Pure
// element selection has no FP semantics, so the kernel template
// instantiated on `uint8_t` produces bit-exact output regardless of how
// the host-side caller interprets the bytes. Symbol name uses the
// `fp8e4m3` value-dtype tag to make the FFI surface explicit.

BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u8cond_fp8e4m3, uint8_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u8cond_fp8e4m3, uint8_t, uint8_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_u32cond_fp8e4m3, uint8_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_u32cond_fp8e4m3, uint8_t, uint32_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE(where_i64cond_fp8e4m3, uint8_t, int64_t)
BARACUDA_KERNELS_WHERE_COND_INSTANTIATE_STRIDED(where_i64cond_fp8e4m3, uint8_t, int64_t)
