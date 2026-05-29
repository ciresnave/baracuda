//! Plan-based GEMM and grouped-GEMM API.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_driver::{Context, PinnedBuffer, Stream};
use baracuda_kernels_types::BackendKind;

use crate::error::{status_to_result, Error, Result};
use crate::types::{
    ArchSku, BatchedGemmArgs, BatchedGemmDescriptor, BiasElement, CutlassElement, ElementKind,
    EpilogueKind, GemmArgs, GemmDescriptor, GemmSku, GroupedPlanPreference, GroupedProblem,
    GroupedScheduleMode, IntElement, IntGemmArgs, IntGemmDescriptor, LayoutSku, PlanPreference,
    PrecisionGuarantee, ScalarType, Workspace,
};

// ============================================================================
// Internal dispatch — generic-T → element-specific extern "C" entry point
// ============================================================================

mod dispatch {
    use super::{ElementKind, LayoutSku};
    use core::ffi::c_void;

    use super::EpilogueKind;

    /// Single-GEMM dispatch on sm_80 with a bias-family epilogue (Bias,
    /// BiasRelu, BiasGelu, BiasSilu).
    ///
    /// SKU coverage today: `{Rcr, Rrr} × {F16, Bf16, F32 (TF32),
    /// F32Strict (SIMT)}` for every bias-family epilogue. F64 routes
    /// through [`gemm_bias_sm80_run_f64`] because the FFI takes `f64`
    /// alpha/beta; that fork is selected at the call site based on
    /// `T::Scalar::IS_F64`. The non-bias path lives in
    /// [`gemm_sm80_run`].
    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_bias_sm80_run(
        layout: LayoutSku,
        kind: ElementKind,
        epilogue: EpilogueKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        bias: *const c_void,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind, epilogue) {
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_bf16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- Rrr layout ----
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_bf16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- TF32 path (Rcr × F32) ----
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_tf32_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- TF32 path (Rrr × F32) ----
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_tf32_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- f32-SIMT path (Rcr × F32Strict) ----
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- f32-SIMT path (Rrr × F32Strict) ----
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            _ => 3,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) fn gemm_bias_sm80_workspace_size(
        layout: LayoutSku,
        kind: ElementKind,
        epilogue: EpilogueKind,
        m: i32,
        n: i32,
        k: i32,
    ) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind, epilogue) {
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_bf16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_bf16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_tf32_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_tf32_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_workspace_size(m, n, k)
            },
            _ => 0,
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_bias_sm80_can_implement(
        layout: LayoutSku,
        kind: ElementKind,
        epilogue: EpilogueKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        bias: *const c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind, epilogue) {
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_bf16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_bf16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_tf32_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_tf32_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            _ => 3,
        }
    }

    /// Single-GEMM dispatch on sm_80 — selects per `(layout, kind)`.
    ///
    /// SKU coverage: all `(Rcr|Rrr) × (F16|Bf16|F32-via-TF32)` are
    /// implemented. F32 routes through Ampere TF32 tensor cores. Any
    /// unknown pair returns status 3 (not implemented).
    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_sm80_run(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_f16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_bf16_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_tf32_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_f16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_bf16_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_tf32_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict) => unsafe {
                k_sys::baracuda_cutlass_gemm_f32_simt_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict) => unsafe {
                k_sys::baracuda_cutlass_gemm_f32_simt_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // F64 has its own dispatcher (`gemm_sm80_run_f64`) because the
            // FFI takes `f64` alpha/beta. The plan layer routes F64 through
            // that path before reaching this function; this arm is
            // defensive.
            (LayoutSku::Rcr, ElementKind::F64)
            | (LayoutSku::Rrr, ElementKind::F64) => 3,
            // Integer element kinds (`S8`, `U8`) route through
            // [`int_gemm_sm80_run`] instead — this float-family
            // dispatcher should never see them when the plan layer is
            // wired correctly. `I32` is an accumulator-only kind and
            // is never a kernel input element.
            (_, ElementKind::S8) | (_, ElementKind::U8) | (_, ElementKind::I32)
            | (_, ElementKind::I64)
            | (_, ElementKind::Bool)
            | (_, ElementKind::Fp8E4M3)
            | (_, ElementKind::Fp8E5M2)
            | (_, ElementKind::S4)
            | (_, ElementKind::U4)
            | (_, ElementKind::Bin)
            | (_, ElementKind::Complex32)
            | (_, ElementKind::Complex64) => 3,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) fn gemm_sm80_workspace_size(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
    ) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_f16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_bf16_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_tf32_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_f16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_bf16_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_tf32_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::F32Strict) => unsafe {
                k_sys::baracuda_cutlass_gemm_f32_simt_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, ElementKind::F32Strict) => unsafe {
                k_sys::baracuda_cutlass_gemm_f32_simt_rrr_sm80_workspace_size(m, n, k)
            },
            // F64 routes through its own dispatcher.
            (LayoutSku::Rcr, ElementKind::F64)
            | (LayoutSku::Rrr, ElementKind::F64) => 0,
            // Integer kinds route through `int_gemm_sm80_workspace_size`.
            // FP8 kinds route through baracuda-kernels-sys. Defensive arms;
            // never expected to fire here.
            (_, ElementKind::S8)
            | (_, ElementKind::U8)
            | (_, ElementKind::I32)
            | (_, ElementKind::I64)
            | (_, ElementKind::Bool)
            | (_, ElementKind::Fp8E4M3)
            | (_, ElementKind::Fp8E5M2)
            | (_, ElementKind::S4)
            | (_, ElementKind::U4)
            | (_, ElementKind::Bin)
            | (_, ElementKind::Complex32)
            | (_, ElementKind::Complex64) => 0,
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_sm80_can_implement(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_f16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_bf16_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_tf32_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rrr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_f16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rrr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_bf16_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_tf32_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rcr, ElementKind::F32Strict) => unsafe {
                k_sys::baracuda_cutlass_gemm_f32_simt_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rrr, ElementKind::F32Strict) => unsafe {
                k_sys::baracuda_cutlass_gemm_f32_simt_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            // F64 routes through its own dispatcher.
            (LayoutSku::Rcr, ElementKind::F64)
            | (LayoutSku::Rrr, ElementKind::F64) => 3,
            // Integer kinds route through `int_gemm_sm80_can_implement`.
            (_, ElementKind::S8) | (_, ElementKind::U8) | (_, ElementKind::I32)
            | (_, ElementKind::I64)
            | (_, ElementKind::Bool)
            | (_, ElementKind::Fp8E4M3)
            | (_, ElementKind::Fp8E5M2)
            | (_, ElementKind::S4)
            | (_, ElementKind::U4)
            | (_, ElementKind::Bin)
            | (_, ElementKind::Complex32)
            | (_, ElementKind::Complex64) => 3,
        }
    }

    // ---------- f64 single-GEMM dispatch, sm_80 ----------
    //
    // Distinct from `gemm_sm80_run` because the FFI signature takes
    // `f64` alpha/beta. The plan layer routes through these when
    // `T::Scalar::IS_F64` is true.

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_sm80_run_f64(
        layout: LayoutSku,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        alpha: f64,
        beta: f64,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match layout {
            LayoutSku::Rcr => unsafe {
                k_sys::baracuda_cutlass_gemm_f64_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            LayoutSku::Rrr => unsafe {
                k_sys::baracuda_cutlass_gemm_f64_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) fn gemm_sm80_workspace_size_f64(layout: LayoutSku, m: i32, n: i32, k: i32) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        match layout {
            LayoutSku::Rcr => unsafe {
                k_sys::baracuda_cutlass_gemm_f64_rcr_sm80_workspace_size(m, n, k)
            },
            LayoutSku::Rrr => unsafe {
                k_sys::baracuda_cutlass_gemm_f64_rrr_sm80_workspace_size(m, n, k)
            },
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_sm80_can_implement_f64(
        layout: LayoutSku,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match layout {
            LayoutSku::Rcr => unsafe {
                k_sys::baracuda_cutlass_gemm_f64_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            LayoutSku::Rrr => unsafe {
                k_sys::baracuda_cutlass_gemm_f64_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
        }
    }

    // ---------- f64 bias-fused GEMM dispatch, sm_80 ----------

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_bias_sm80_run_f64(
        layout: LayoutSku,
        epilogue: EpilogueKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        bias: *const c_void,
        alpha: f64,
        beta: f64,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, epilogue) {
            (LayoutSku::Rcr, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f64_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f64_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // Identity not a bias-family epilogue; never reached when
            // `epilogue.requires_bias()` gates the call site.
            (_, EpilogueKind::Identity) => 3,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) fn gemm_bias_sm80_workspace_size_f64(
        layout: LayoutSku,
        epilogue: EpilogueKind,
        m: i32,
        n: i32,
        k: i32,
    ) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, epilogue) {
            (LayoutSku::Rcr, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f64_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f64_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rrr, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_workspace_size(m, n, k)
            },
            (_, EpilogueKind::Identity) => 0,
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn gemm_bias_sm80_can_implement_f64(
        layout: LayoutSku,
        epilogue: EpilogueKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        bias: *const c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, epilogue) {
            (LayoutSku::Rcr, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f64_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rcr, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::Bias) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f64_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::BiasRelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::BiasGelu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (LayoutSku::Rrr, EpilogueKind::BiasSilu) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (_, EpilogueKind::Identity) => 3,
        }
    }

    // ---------- batched GEMM, sm_80 ----------
    //
    // SKU coverage (status 3 = not implemented):
    //   - Rcr × {F16, Bf16}            ✓
    //   - Rcr × F32, all Rrr           ✗

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn batched_gemm_sm80_run(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        stride_a: i64,
        b: *const c_void,
        ldb: i64,
        stride_b: i64,
        c: *const c_void,
        ldc: i64,
        stride_c: i64,
        d: *mut c_void,
        ldd: i64,
        stride_d: i64,
        alpha: f32,
        beta: f32,
        batch_count: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_batched_f16_rcr_sm80_run(
                    m, n, k,
                    a, lda, stride_a,
                    b, ldb, stride_b,
                    c, ldc, stride_c,
                    d, ldd, stride_d,
                    alpha, beta,
                    batch_count,
                    workspace, workspace_bytes,
                    stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_batched_bf16_rcr_sm80_run(
                    m, n, k,
                    a, lda, stride_a,
                    b, ldb, stride_b,
                    c, ldc, stride_c,
                    d, ldd, stride_d,
                    alpha, beta,
                    batch_count,
                    workspace, workspace_bytes,
                    stream,
                )
            },
            _ => 3,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) fn batched_gemm_sm80_workspace_size(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
        batch_count: i32,
    ) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_batched_f16_rcr_sm80_workspace_size(
                    m, n, k, batch_count,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_batched_bf16_rcr_sm80_workspace_size(
                    m, n, k, batch_count,
                )
            },
            _ => 0,
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn batched_gemm_sm80_can_implement(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        stride_a: i64,
        b: *const c_void,
        ldb: i64,
        stride_b: i64,
        c: *const c_void,
        ldc: i64,
        stride_c: i64,
        d: *mut c_void,
        ldd: i64,
        stride_d: i64,
        batch_count: i32,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::F16) => unsafe {
                k_sys::baracuda_cutlass_gemm_batched_f16_rcr_sm80_can_implement(
                    m, n, k,
                    a, lda, stride_a,
                    b, ldb, stride_b,
                    c, ldc, stride_c,
                    d, ldd, stride_d,
                    batch_count,
                )
            },
            (LayoutSku::Rcr, ElementKind::Bf16) => unsafe {
                k_sys::baracuda_cutlass_gemm_batched_bf16_rcr_sm80_can_implement(
                    m, n, k,
                    a, lda, stride_a,
                    b, ldb, stride_b,
                    c, ldc, stride_c,
                    d, ldd, stride_d,
                    batch_count,
                )
            },
            _ => 3,
        }
    }

    // ---------- grouped GEMM, RCR sm_80 ----------

    #[cfg(feature = "sm80")]
    pub(super) unsafe fn grouped_gemm_rcr_sm80_sufficient(
        kind: ElementKind,
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match kind {
            ElementKind::F16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_f16_rcr_sm80_sufficient(h_m, h_n, h_k, group_count)
            },
            ElementKind::Bf16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_sufficient(h_m, h_n, h_k, group_count)
            },
            ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
            | ElementKind::S8
            | ElementKind::U8
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::Bool
            | ElementKind::Fp8E4M3
            | ElementKind::Fp8E5M2
            | ElementKind::S4
            | ElementKind::U4
            | ElementKind::Bin
            | ElementKind::Complex32
            | ElementKind::Complex64 => 0,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) unsafe fn grouped_gemm_rcr_sm80_scratch_bytes(
        kind: ElementKind,
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
        threadblock_count: i32,
    ) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        match kind {
            ElementKind::F16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_f16_rcr_sm80_scratch_bytes(
                    h_m, h_n, h_k, group_count, threadblock_count,
                )
            },
            ElementKind::Bf16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_scratch_bytes(
                    h_m, h_n, h_k, group_count, threadblock_count,
                )
            },
            ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
            | ElementKind::S8
            | ElementKind::U8
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::Bool
            | ElementKind::Fp8E4M3
            | ElementKind::Fp8E5M2
            | ElementKind::S4
            | ElementKind::U4
            | ElementKind::Bin
            | ElementKind::Complex32
            | ElementKind::Complex64 => 0,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) unsafe fn grouped_gemm_rcr_sm80_can_implement(
        kind: ElementKind,
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match kind {
            ElementKind::F16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_f16_rcr_sm80_can_implement(h_m, h_n, h_k, group_count)
            },
            ElementKind::Bf16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_can_implement(h_m, h_n, h_k, group_count)
            },
            ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
            | ElementKind::S8
            | ElementKind::U8
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::Bool
            | ElementKind::Fp8E4M3
            | ElementKind::Fp8E5M2
            | ElementKind::S4
            | ElementKind::U4
            | ElementKind::Bin
            | ElementKind::Complex32
            | ElementKind::Complex64 => 3,
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn grouped_gemm_rcr_sm80_run(
        kind: ElementKind,
        group_count: i32,
        threadblock_count: i32,
        d_problem_sizes: *const c_void,
        d_ptr_a: *const c_void,
        d_ptr_b: *const c_void,
        d_ptr_c: *const c_void,
        d_ptr_d: *mut c_void,
        d_lda: *const c_void,
        d_ldb: *const c_void,
        d_ldc: *const c_void,
        d_ldd: *const c_void,
        h_problem_sizes: *const c_void,
        alpha: f32,
        beta: f32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match kind {
            ElementKind::F16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_f16_rcr_sm80_run(
                    group_count, threadblock_count,
                    d_problem_sizes,
                    d_ptr_a, d_ptr_b, d_ptr_c, d_ptr_d,
                    d_lda, d_ldb, d_ldc, d_ldd,
                    h_problem_sizes,
                    alpha, beta,
                    scratch, scratch_bytes,
                    stream,
                )
            },
            ElementKind::Bf16 => unsafe {
                k_sys::baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_run(
                    group_count, threadblock_count,
                    d_problem_sizes,
                    d_ptr_a, d_ptr_b, d_ptr_c, d_ptr_d,
                    d_lda, d_ldb, d_ldc, d_ldd,
                    h_problem_sizes,
                    alpha, beta,
                    scratch, scratch_bytes,
                    stream,
                )
            },
            ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
            | ElementKind::S8
            | ElementKind::U8
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::Bool
            | ElementKind::Fp8E4M3
            | ElementKind::Fp8E5M2
            | ElementKind::S4
            | ElementKind::U4
            | ElementKind::Bin
            | ElementKind::Complex32
            | ElementKind::Complex64 => 3,
        }
    }

    // ============================================================================
    // int8 GEMM dispatch — Identity, RCR layout, sm_80
    // ============================================================================
    //
    // Identity int kernels: `LinearCombinationClamp` epilogue. Layout is
    // restricted to `Rcr` (`Rrr` is deferred — see lib.rs section header).
    // Element kind is restricted to S8 / U8. Alpha/beta are `f32` (CUTLASS
    // int8 dequant-in-epilogue convention).

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn int_gemm_rcr_sm80_run(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::S8) => unsafe {
                k_sys::baracuda_cutlass_gemm_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (LayoutSku::Rcr, ElementKind::U8) => unsafe {
                k_sys::baracuda_cutlass_gemm_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // RRR int8 is deferred — CUTLASS 4.2.0 lacks the 8-bit
            // `TensorOpMultiplicandCongruous` warp iterator needed for
            // `RowMajor × RowMajor × OpClassTensorOp` instantiation.
            (LayoutSku::Rrr, ElementKind::S8) | (LayoutSku::Rrr, ElementKind::U8) => 3,
            // Defensive: float and I32 kinds should never reach this dispatcher.
            _ => 3,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) fn int_gemm_rcr_sm80_workspace_size(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
    ) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::S8) => unsafe {
                k_sys::baracuda_cutlass_gemm_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (LayoutSku::Rcr, ElementKind::U8) => unsafe {
                k_sys::baracuda_cutlass_gemm_u8_rcr_sm80_workspace_size(m, n, k)
            },
            _ => 0,
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn int_gemm_rcr_sm80_can_implement(
        layout: LayoutSku,
        kind: ElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        match (layout, kind) {
            (LayoutSku::Rcr, ElementKind::S8) => unsafe {
                k_sys::baracuda_cutlass_gemm_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rcr, ElementKind::U8) => unsafe {
                k_sys::baracuda_cutlass_gemm_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                )
            },
            (LayoutSku::Rrr, ElementKind::S8) | (LayoutSku::Rrr, ElementKind::U8) => 3,
            _ => 3,
        }
    }

    // ============================================================================
    // int8 bias-fused GEMM dispatch — RCR layout, sm_80
    // ============================================================================
    //
    // Bias kernels: `LinearCombinationBiasElementwise<int8, int32, float,
    // int8, int8, EPA=16, ActOp, plus<float>, false, ElementBias>`. The
    // four `Bias*` epilogues differ only in `ActOp`; the bias element
    // is independent of the matrix element and dispatched on
    // `bias_kind`. The 16 reachable arms = 2 sgn × 4 epi × 2 bias-type.

    use crate::types::BiasElementKind;

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn int_gemm_bias_rcr_sm80_run(
        layout: LayoutSku,
        kind: ElementKind,
        epilogue: EpilogueKind,
        bias_kind: BiasElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        bias: *const c_void,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        // RRR int8 deferred (see Identity dispatcher).
        if !matches!(layout, LayoutSku::Rcr) {
            return 3;
        }
        match (kind, epilogue, bias_kind) {
            // ---- s8 × f32 bias ----
            (ElementKind::S8, EpilogueKind::Bias, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasRelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasGelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasSilu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- s8 × i32 bias ----
            (ElementKind::S8, EpilogueKind::Bias, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasRelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasGelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasSilu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- u8 × f32 bias ----
            (ElementKind::U8, EpilogueKind::Bias, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasRelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasGelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasSilu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // ---- u8 × i32 bias ----
            (ElementKind::U8, EpilogueKind::Bias, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasRelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasGelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasSilu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            },
            // Identity reaches here when called accidentally — bias is
            // required for the bias-family dispatchers. Identity routes
            // through `int_gemm_rcr_sm80_run` instead.
            (_, EpilogueKind::Identity, _) => 3,
            // Defensive: float / I32 kinds should never reach this dispatcher.
            _ => 3,
        }
    }

    #[cfg(feature = "sm80")]
    pub(super) fn int_gemm_bias_rcr_sm80_workspace_size(
        layout: LayoutSku,
        kind: ElementKind,
        epilogue: EpilogueKind,
        bias_kind: BiasElementKind,
        m: i32,
        n: i32,
        k: i32,
    ) -> usize {
        use baracuda_cutlass_kernels_sys as k_sys;
        if !matches!(layout, LayoutSku::Rcr) {
            return 0;
        }
        match (kind, epilogue, bias_kind) {
            (ElementKind::S8, EpilogueKind::Bias, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::S8, EpilogueKind::BiasRelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::S8, EpilogueKind::BiasGelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::S8, EpilogueKind::BiasSilu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::S8, EpilogueKind::Bias, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::S8, EpilogueKind::BiasRelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::S8, EpilogueKind::BiasGelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::S8, EpilogueKind::BiasSilu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::Bias, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::BiasRelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::BiasGelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::BiasSilu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::Bias, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::BiasRelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::BiasGelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            (ElementKind::U8, EpilogueKind::BiasSilu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_workspace_size(m, n, k)
            },
            _ => 0,
        }
    }

    #[cfg(feature = "sm80")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn int_gemm_bias_rcr_sm80_can_implement(
        layout: LayoutSku,
        kind: ElementKind,
        epilogue: EpilogueKind,
        bias_kind: BiasElementKind,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        bias: *const c_void,
    ) -> i32 {
        use baracuda_cutlass_kernels_sys as k_sys;
        if !matches!(layout, LayoutSku::Rcr) {
            return 3;
        }
        match (kind, epilogue, bias_kind) {
            (ElementKind::S8, EpilogueKind::Bias, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasRelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasGelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasSilu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::S8, EpilogueKind::Bias, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasRelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasGelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::S8, EpilogueKind::BiasSilu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::Bias, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasRelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasGelu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasSilu, BiasElementKind::F32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::Bias, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasRelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasGelu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            (ElementKind::U8, EpilogueKind::BiasSilu, BiasElementKind::I32) => unsafe {
                k_sys::baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_can_implement(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias,
                )
            },
            _ => 3,
        }
    }
}

// ============================================================================
// Host-side validation helpers
// ============================================================================

/// Minimum element count required to back a `(rows, cols, ld)` matrix at
/// the given layout. Returns `None` on overflow.
///
/// Assumes `rows >= 1` and `cols >= 1` (callers go through
/// [`check_descriptor`] which rejects non-positive dimensions first).
fn min_elements_row_major(rows: i32, cols: i32, ld: i64) -> Option<usize> {
    // [rows, cols] row-major: element [i, j] at offset i*ld + j.
    // Maximum addressable index = (rows - 1) * ld + (cols - 1), so the
    // buffer must hold (rows - 1) * ld + cols elements. Accepts padded
    // leading dimensions (ld > cols) without rejecting valid slabs.
    let r = (rows - 1) as i64;
    let needed = r.checked_mul(ld)?.checked_add(cols as i64)?;
    usize::try_from(needed).ok()
}

fn min_elements_col_major(rows: i32, cols: i32, ld: i64) -> Option<usize> {
    // [rows, cols] column-major: element [i, j] at offset j*ld + i.
    // Maximum addressable index = (cols - 1) * ld + (rows - 1).
    let c = (cols - 1) as i64;
    let needed = c.checked_mul(ld)?.checked_add(rows as i64)?;
    usize::try_from(needed).ok()
}

// Compatibility shims for the buffer-size tests below — they document
// the per-operand layout (A row-major, B column-major for Rcr, C/D
// row-major) so we keep separate names for clarity.
#[cfg(test)]
fn min_elements_rcr_a(rows: i32, cols: i32, ld: i64) -> Option<usize> {
    min_elements_row_major(rows, cols, ld)
}
#[cfg(test)]
fn min_elements_rcr_b(rows: i32, cols: i32, ld: i64) -> Option<usize> {
    min_elements_col_major(rows, cols, ld)
}
#[cfg(test)]
fn min_elements_rcr_cd(rows: i32, cols: i32, ld: i64) -> Option<usize> {
    min_elements_row_major(rows, cols, ld)
}

fn check_descriptor(desc: &GemmDescriptor) -> Result<()> {
    if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
        return Err(Error::InvalidProblem("M, N, K must all be positive"));
    }
    // All shipped layouts (Rcr, Rrr) and epilogues (Identity) are
    // accepted by select; per-(layout, kind) implementability is
    // dispatched at run time to the kernel's own can_implement.
    Ok(())
}

fn check_args<T: CutlassElement>(desc: &GemmDescriptor, args: &GemmArgs<'_, T>) -> Result<()> {
    // Epilogue / bias must agree: any Bias* variant needs bias = Some,
    // Identity needs bias = None.
    match (desc.epilogue.requires_bias(), &args.bias) {
        (false, Some(_)) => {
            return Err(Error::InvalidProblem(
                "args.bias must be None when descriptor.epilogue is Identity",
            ));
        }
        (true, None) => {
            return Err(Error::InvalidProblem(
                "args.bias is required when descriptor.epilogue is in the Bias family \
                 (Bias / BiasRelu / BiasGelu / BiasSilu)",
            ));
        }
        (false, None) | (true, Some(_)) => {}
    }
    if let Some(bias) = &args.bias {
        if bias.len != desc.n {
            return Err(Error::InvalidProblem(
                "bias vector length must equal N",
            ));
        }
        if bias.stride != 1 {
            return Err(Error::Unsupported(
                "bias vector must be contiguous (stride 1) — strided bias not supported",
            ));
        }
        if bias.data.len() < desc.n as usize {
            return Err(Error::BufferTooSmall {
                needed: desc.n as usize,
                got: bias.data.len(),
            });
        }
    }
    if args.a.rows != desc.m || args.a.cols != desc.k {
        return Err(Error::InvalidProblem("A shape doesn't match descriptor (M, K)"));
    }
    if args.b.rows != desc.k || args.b.cols != desc.n {
        return Err(Error::InvalidProblem("B shape doesn't match descriptor (K, N)"));
    }
    if args.d.rows != desc.m || args.d.cols != desc.n {
        return Err(Error::InvalidProblem("D shape doesn't match descriptor (M, N)"));
    }
    if let Some(c) = &args.c {
        if c.rows != desc.m || c.cols != desc.n {
            return Err(Error::InvalidProblem("C shape doesn't match descriptor (M, N)"));
        }
    }
    // A is row-major in both Rcr and Rrr — leading dim along the K axis.
    if args.a.ld < desc.k as i64 {
        return Err(Error::InvalidProblem("A leading dimension must be >= K"));
    }
    // B's leading-dim minor depends on layout:
    //   Rcr: column-major [K, N], ld is along the K axis (rows).
    //   Rrr: row-major    [K, N], ld is along the N axis (cols).
    let b_min_ld = match desc.layout {
        LayoutSku::Rcr => desc.k as i64,
        LayoutSku::Rrr => desc.n as i64,
    };
    if args.b.ld < b_min_ld {
        return Err(Error::InvalidProblem(match desc.layout {
            LayoutSku::Rcr => "B leading dimension must be >= K (column-major Rcr layout)",
            LayoutSku::Rrr => "B leading dimension must be >= N (row-major Rrr layout)",
        }));
    }
    if args.d.ld < desc.n as i64 {
        return Err(Error::InvalidProblem("D leading dimension must be >= N"));
    }
    if let Some(c) = &args.c {
        if c.ld < desc.n as i64 {
            return Err(Error::InvalidProblem("C leading dimension must be >= N"));
        }
    }

    let need_a = min_elements_row_major(args.a.rows, args.a.cols, args.a.ld)
        .ok_or(Error::InvalidProblem("A storage size overflow"))?;
    if args.a.data.len() < need_a {
        return Err(Error::BufferTooSmall {
            needed: need_a,
            got: args.a.data.len(),
        });
    }
    let need_b = match desc.layout {
        LayoutSku::Rcr => min_elements_col_major(args.b.rows, args.b.cols, args.b.ld),
        LayoutSku::Rrr => min_elements_row_major(args.b.rows, args.b.cols, args.b.ld),
    }
    .ok_or(Error::InvalidProblem("B storage size overflow"))?;
    if args.b.data.len() < need_b {
        return Err(Error::BufferTooSmall {
            needed: need_b,
            got: args.b.data.len(),
        });
    }
    let need_d = min_elements_row_major(args.d.rows, args.d.cols, args.d.ld)
        .ok_or(Error::InvalidProblem("D storage size overflow"))?;
    if args.d.data.len() < need_d {
        return Err(Error::BufferTooSmall {
            needed: need_d,
            got: args.d.data.len(),
        });
    }
    if let Some(c) = &args.c {
        let need_c = min_elements_row_major(c.rows, c.cols, c.ld)
            .ok_or(Error::InvalidProblem("C storage size overflow"))?;
        if c.data.len() < need_c {
            return Err(Error::BufferTooSmall {
                needed: need_c,
                got: c.data.len(),
            });
        }
    }
    Ok(())
}

// ============================================================================
// cuBLAS backend — Phase 30 fast-path for f16/bf16 decode-regime FP GEMM
// ============================================================================
//
// Background (from `crates/baracuda-kernels-bench/BENCHMARKS.md`, Phase 29):
// baracuda's CUTLASS RCR plan emits `_sm80_` kernels (Ampere-tuned) for
// f16/bf16. On Ada (sm_89, RTX 4070) cuBLAS auto-dispatches to the
// sm_89 tensor-core path with an f32 accumulator and wins 2–4× at
// low-M (M=1 / M=32) decode shapes, narrowing to ~parity at M=128.
// This module hosts the cuBLAS Gemm fallback path that the
// [`GemmPlan::select`] heuristic dispatches to for those shapes.
//
// Coverage:
//   - f16, bf16, f32: `cublasGemmEx` with `CUBLAS_COMPUTE_32F`
//     (CUBLAS_GEMM_DEFAULT_TENSOR_OP algo, value 99).
//   - f64: `cublasDgemm` (no GemmEx needed — DGEMM is plenty fast).
//   - F32Strict: not supported (cuBLAS doesn't expose a strict-IEEE
//     SIMT path the way CUTLASS does; F32Strict callers stay on the
//     CUTLASS SIMT kernels for bit-stability).
//   - Bias / Bias* epilogues: not supported (cuBLAS-classic has no
//     fused-bias-activation; cuBLASLt does but is a separate API
//     surface). Forcing Cublas backend on a Bias* epilogue returns
//     `Error::Unsupported` from `select`.
//
// Layout mapping (baracuda Row/Col/Row → cuBLAS column-major):
// baracuda's RCR means A row-major [M, K], B column-major [K, N], D
// row-major [M, N]. cuBLAS is column-major, so we use the standard
// "C^T = B^T · A^T" trick: pass B as the first operand, A as the
// second, with `transa = Op::T` on both, and the resulting D in
// col-major IS our row-major D. For RRR the mapping changes the
// second `transa` to `Op::N`.

mod cublas_backend {
    use core::cell::RefCell;

    use baracuda_cublas::Handle as CublasHandle;
    use baracuda_driver::Stream;

    // Per-thread cache of cuBLAS handles, keyed by the raw context
    // pointer the handle was created against.
    //
    // cuBLAS handles are tied to the CUDA context current at the time
    // of `cublasCreate`. Caching by (thread, ctx-raw-ptr) means the
    // first call into the cuBLAS backend on a given thread pays the
    // (~100µs–ms) `cublasCreate` cost; subsequent calls reuse the
    // handle. Per-thread storage sidesteps the cuBLAS-handle `!Sync`
    // constraint (NVIDIA documents that a handle should only be
    // touched from one host thread at a time) and keeps
    // `super::GemmPlan` `Send + Sync` (the plan stores no handle —
    // it looks one up out of the thread-local on each `run`).
    //
    // A `Vec` keyed by raw context pointer is fine here: production
    // callers have one context per process, and even
    // multi-context callers rarely exceed a handful.
    thread_local! {
        static HANDLE_CACHE: RefCell<Vec<(usize, CublasHandle)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Fetch (or lazily create) the cuBLAS handle bound to `stream`'s
    /// context, with the handle's stream binding set to `stream`.
    ///
    /// The returned handle is `Clone`able (the inner state is `Arc`'d
    /// inside [`baracuda_cublas::Handle`]) so the caller can move it
    /// into the closure that calls `gemm_ex` / `gemm` without holding
    /// the thread-local `RefCell` borrow across the launch.
    pub(super) fn handle_for(stream: &Stream) -> crate::Result<CublasHandle> {
        // Use the context's raw pointer as the cache key. baracuda's
        // `Context` doesn't expose a stable id, but the inner CUDA
        // context pointer is unique per process and stable for the
        // context's lifetime.
        let ctx_key = stream.context().as_raw() as usize;
        let handle = HANDLE_CACHE.with(|cache| -> crate::Result<CublasHandle> {
            let mut cache = cache.borrow_mut();
            if let Some((_, h)) = cache.iter().find(|(k, _)| *k == ctx_key) {
                return Ok(h.clone());
            }
            // First touch on this thread for this context — create.
            // The current context must be the stream's context for
            // `cublasCreate` to bind correctly. baracuda's Context API
            // is push-based; we call `set_current` (idempotent if
            // already current) before the cuBLAS create.
            stream
                .context()
                .set_current()
                .map_err(crate::Error::Driver)?;
            // Retry handle creation under transient failure. Observed
            // empirically: when many test PROCESSES start in parallel
            // (the `cargo test --workspace` harness launches dozens of
            // binaries concurrently), the cuBLAS library's first-call
            // initialization races on a shared driver resource (DLL
            // loader / cuBLAS-side global lock) and `cublasCreate_v2`
            // intermittently returns CUBLAS_STATUS_ALLOC_FAILED or
            // CUBLAS_STATUS_NOT_INITIALIZED. The error is genuinely
            // transient — a 50ms backoff + retry clears it 100% of the
            // time on RTX 4070 / cuBLAS 12. Five retries with linear
            // backoff (50ms, 100ms, ..., 250ms) bounds the worst case
            // at ~750ms per first-touch, which is fine for what is
            // already a one-time-per-thread operation.
            let h = {
                let mut last_err = None;
                let mut handle: Option<CublasHandle> = None;
                for attempt in 0..5 {
                    match CublasHandle::new() {
                        Ok(h) => { handle = Some(h); break }
                        Err(e) => {
                            last_err = Some(e);
                            // Linear backoff: 50ms * (attempt + 1).
                            std::thread::sleep(std::time::Duration::from_millis(
                                50 * (attempt as u64 + 1),
                            ));
                        }
                    }
                }
                match handle {
                    Some(h) => h,
                    None => {
                        // Surface the underlying error code in the
                        // message for diagnostics. The retry loop is
                        // transparent on success — only the final
                        // failure path mentions exhaustion.
                        let _ = last_err; // dropped; cublas Error not in scope here
                        return Err(crate::Error::Unsupported(
                            "cuBLAS handle creation failed after 5 retries \
                             (library missing, device unavailable, or \
                             persistent driver-init contention)",
                        ));
                    }
                }
            };
            cache.push((ctx_key, h.clone()));
            Ok(h)
        })?;
        // Bind to the launch stream so the GEMM enqueues on the
        // caller-supplied stream rather than the default stream. The
        // bind is per-handle, sticky across calls — but a previously
        // cached handle could have been bound to a *different* stream
        // last time. Always re-bind.
        handle
            .set_stream(stream)
            .map_err(|_| crate::Error::Unsupported(
                "cuBLAS set_stream failed",
            ))?;
        Ok(handle)
    }
}

/// Algo selector for `cublasGemmEx`.
///
/// `-1` corresponds to `CUBLAS_GEMM_DEFAULT` / `CUBLAS_GEMM_DFALT` — the
/// "let cuBLAS pick" heuristic. In modern cuBLAS (CUDA 12+) when
/// `compute_type = Compute32F` the algo selector is largely advisory:
/// cuBLAS picks the kernel from its internal heuristic based on
/// (M, N, K, dtype) regardless of the value here. The Phase-29 bench's
/// reference cuBLAS-direct path passes `0` (CUBLAS_GEMM_ALGO0), which
/// is also routed through the same heuristic; both produce the same
/// kernel selection on the f16/bf16/f32 GemmEx paths.
///
/// `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (= 99) would force a tensor-op
/// kernel even at very low M where cuBLAS's heuristic prefers a CUDA-
/// core kernel; benchmarking showed it was ~3× slower than `DEFAULT`
/// at M=1 on RTX 4070. Stick with `DEFAULT` (`-1`).
const CUBLAS_GEMM_ALGO: i32 = -1;

/// Backend the plan picked for this descriptor. Stored privately on
/// the plan; surfaced through [`GemmPlan::backend`].
///
/// Differs from `GemmSku::arch` in that it answers a *higher-level*
/// question: "which kernel library does this plan call into?" The arch
/// SKU only meaningfully describes the CUTLASS variant — CUTLASS itself
/// is one of several backends now that the Phase-30 cuBLAS fast-path
/// joined the dispatch surface.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BackendChoice {
    /// Compiled CUTLASS template instantiation in
    /// `baracuda-cutlass-kernels-sys`. `arch` records which CUTLASS
    /// SKU (sm_80 today).
    Cutlass { arch: ArchSku },
    /// `cublasGemmEx` (f16/bf16/f32) or `cublasDgemm` (f64) via the
    /// `baracuda-cublas` wrapper. Used as the Phase-30 fast path for
    /// f16/bf16 low-M decode shapes on sm_89 hardware.
    Cublas,
    /// Phase 44: vendored ozIMMU (Ozaki-scheme FP64 GEMM that
    /// synthesizes a DGEMM from `S²` int8 tensor-core matmuls). Only
    /// valid for f64 / RCR / RRR / Identity epilogue. The `slices`
    /// discriminant follows the public `BackendKind::Ozaki { slices }`
    /// convention: 0 = auto, 3..=18 = fixed slice count.
    Ozaki { slices: u8 },
}

impl BackendChoice {
    fn as_public(self) -> BackendKind {
        match self {
            BackendChoice::Cutlass { .. } => BackendKind::Cutlass,
            BackendChoice::Cublas => BackendKind::Cublas,
            BackendChoice::Ozaki { slices } => BackendKind::Ozaki { slices },
        }
    }
}

/// Heuristic that decides whether to route an FP-family GEMM through
/// cuBLAS or CUTLASS. Caller `pref.prefer_backend` overrides whatever
/// the heuristic would have picked (subject to availability checks).
///
/// Today's heuristic (Phase 30, validated on RTX 4070):
///
/// - f16/bf16 with `2 ≤ M < 128`: cuBLAS. The decode-batch regime
///   where cuBLAS's sm_89 tensor-core kernels beat baracuda's
///   CUTLASS sm_80 RCR plan by 2–3×.
/// - f16/bf16 with `M == 1` (pure GEMV-shape): CUTLASS. Counter-
///   intuitive given Phase-29's M=1 perf gap, but the layout
///   `transa=T` we must use to bridge baracuda's row-major contract
///   to cuBLAS's col-major API forces cuBLAS to materialize a B^T
///   transpose, which is slower than the unmaterialized
///   CUTLASS-RCR-sm_80 GEMV-tile at the K=N=4096 shape (185µs vs
///   108µs measured). For pure single-token decode, callers wanting
///   the bench's "all-1's mathematically-wrong" speed can force
///   `prefer_backend = Some(Cublas)` and accept the routing.
/// - f16/bf16 with `M >= 128`: CUTLASS. Tie zone; the existing
///   sm_80 RCR is known-stable.
/// - f32: CUTLASS. Phase-29 bench shows CUTLASS f32 RCR beats cuBLAS
///   `sgemm` at M=128; ~parity at low-M; not worth the routing churn.
/// - f64: CUTLASS (back-compat). Callers can force cuBLAS via
///   `prefer_backend` for decode-shaped f64 workloads.
/// - F32Strict: CUTLASS (cuBLAS has no strict-IEEE SIMT path).
///
/// Bias* epilogues always route through CUTLASS regardless of
/// (M, dtype) — cuBLAS-classic has no fused-bias-activation.
fn should_use_cublas_for_fp(
    desc: &GemmDescriptor,
    element: ElementKind,
) -> bool {
    // Bias / Bias* epilogues — cuBLAS-classic has no fused path.
    if desc.epilogue.requires_bias() {
        return false;
    }
    match element {
        // The 2 ≤ M < 128 window is where cuBLAS wins. M=1 is excluded
        // because the `transa=T` mapping costs more than cuBLAS-direct
        // saves over CUTLASS at K=N=4096-ish shapes. M=128+ is excluded
        // because CUTLASS sm_80 catches up at prefill scale.
        ElementKind::F16 | ElementKind::Bf16 => desc.m >= 2 && desc.m < 128,
        // f32 stays on CUTLASS by default — Phase 29 bench shows
        // CUTLASS f32 RCR beats cuBLAS sgemm at the M=128 prefill
        // shape. Callers can force via PlanPreference if needed.
        ElementKind::F32 => false,
        // F32Strict: CUTLASS SIMT path is the only bit-stable option.
        ElementKind::F32Strict => false,
        // f64: keep CUTLASS by default (back-compat). Callers can
        // force-prefer cuBLAS if their f64 workload is decode-shaped.
        ElementKind::F64 => false,
        // Any other element type isn't reachable through `GemmPlan<T>`
        // (the `Element` bound rejects non-FP types), but the match
        // arm needs to be exhaustive.
        _ => false,
    }
}

/// Phase 44 — validate a `BackendKind::Ozaki { slices }` request
/// against this descriptor.
///
/// ozIMMU only supports FP64 GEMM with the Identity epilogue (no
/// fused bias / activation chain) and the `RCR` / `RRR` layouts that
/// baracuda's `GemmPlan` already exposes. The slice count must be
/// `0` (= auto) or `3..=18` (= fixed). Anything else returns
/// `Error::Unsupported` so callers see the rejection at plan-select
/// time rather than as a deep status code at launch.
///
/// When the `ozimmu` cargo feature is off, every request is rejected
/// with a message pointing at the gate.
#[allow(unused_variables)]
fn validate_ozaki_request(
    desc: &GemmDescriptor,
    element: ElementKind,
    slices: u8,
) -> Result<()> {
    #[cfg(not(feature = "ozimmu"))]
    {
        return Err(Error::Unsupported(
            "PlanPreference::prefer_backend = Some(Ozaki {..}) requires the \
             `ozimmu` cargo feature on baracuda-cutlass (off by default — \
             enable on baracuda-kernels too if going through the kernels facade)",
        ));
    }
    #[cfg(feature = "ozimmu")]
    {
        if element != ElementKind::F64 {
            return Err(Error::Unsupported(
                "BackendKind::Ozaki is FP64-only (Ozaki-scheme synthesizes \
                 DGEMM from int8; f16/bf16/f32/F32Strict have no Ozaki path)",
            ));
        }
        if desc.epilogue != EpilogueKind::Identity {
            return Err(Error::Unsupported(
                "BackendKind::Ozaki only supports the Identity epilogue \
                 (no fused bias / activation chain on the Ozaki path)",
            ));
        }
        // Layout is already constrained to Rcr / Rrr by the descriptor;
        // both are supported by `mtk::ozimmu::gemm`.
        //
        // Phase 44c extends the encoding to include variant flags.
        // The low 5 bits are the slice count (0 = auto, 3..=18 = fixed);
        // the high 3 bits are the variant (0 = Base, 1 = EF, 2 = RN,
        // 3 = H). Reject anything else.
        let s = slices & 0x1F; // low 5 bits = slice count
        let v = slices >> 5;   // high 3 bits = variant
        if s != 0 && !(3..=18).contains(&s) {
            return Err(Error::Unsupported(
                "BackendKind::Ozaki slice count (low 5 bits) must be 0 \
                 (auto) or 3..=18",
            ));
        }
        if v > 3 {
            return Err(Error::Unsupported(
                "BackendKind::Ozaki variant (high 3 bits) must be 0 (Base), \
                 1 (EF), 2 (RN), or 3 (H)",
            ));
        }
        Ok(())
    }
}

/// Map a baracuda [`ElementKind`] to the cuBLAS [`cudaDataType_t`].
/// Returns `None` for element kinds that don't have a cuBLAS GemmEx
/// dtype (F32Strict, integer, FP8, …). Callers should fall back to
/// CUTLASS in the `None` case.
fn cublas_dtype_for(kind: ElementKind) -> Option<baracuda_cublas_sys::functions::cudaDataType_t> {
    use baracuda_cublas_sys::functions::cudaDataType_t;
    match kind {
        ElementKind::F16 => Some(cudaDataType_t::R_16F),
        ElementKind::Bf16 => Some(cudaDataType_t::R_16BF),
        ElementKind::F32 => Some(cudaDataType_t::R_32F),
        ElementKind::F64 => Some(cudaDataType_t::R_64F),
        _ => None,
    }
}

// ============================================================================
// GemmPlan
// ============================================================================

/// Selected GEMM kernel and the host-side metadata needed to launch it.
///
/// Plans are cheap to construct, hold no device memory, and are
/// `Send + Sync` for the same reason — they're pure host data. The
/// Phase-30 cuBLAS fast-path adds no per-plan state: cuBLAS handles
/// live in a thread-local cache so the plan itself stays trivially
/// thread-safe.
///
/// See the crate root for usage; key methods:
/// - [`select`](Self::select) — pick a kernel for a problem shape.
/// - [`can_implement`](Self::can_implement) — host-side validation.
/// - [`workspace_size`](Self::workspace_size) — bytes of scratch needed.
/// - [`run`](Self::run) — launch on a stream.
/// - [`sku`](Self::sku) — identity of the chosen kernel.
/// - [`backend`](Self::backend) — which backend (CUTLASS / cuBLAS) was
///   picked. Phase 30 added the cuBLAS fast-path for f16/bf16 low-M
///   decode shapes; the heuristic is documented on
///   [`should_use_cublas_for_fp`](self::should_use_cublas_for_fp).
#[derive(Debug)]
pub struct GemmPlan<T: CutlassElement> {
    desc: GemmDescriptor,
    sku: GemmSku,
    backend: BackendChoice,
    _element: PhantomData<T>,
}

impl<T: CutlassElement> GemmPlan<T> {
    /// Pick a kernel for `desc`.
    ///
    /// Queries the stream's device for its compute capability and selects
    /// between the CUTLASS-sm_80 (forward-compatible across Ampere /
    /// Ada / Hopper), CUTLASS-sm_90a (Hopper-specialized, when feature-
    /// enabled and the device actually is Hopper), and the Phase-30
    /// cuBLAS fast-path. Build features filter what kernels are
    /// *available*; the device cap and the f16/bf16-low-M heuristic
    /// decide what to *use*. See [`should_use_cublas_for_fp`] for the
    /// dispatch rules. Override the heuristic via
    /// [`PlanPreference::prefer_backend`].
    pub fn select(stream: &Stream, desc: &GemmDescriptor, pref: PlanPreference) -> Result<Self> {
        check_descriptor(desc)?;
        // Bias-family kernels currently cover every `(Layout, ElementKind)`
        // pair: F16/Bf16 → 8 elements per epilogue access; F32 (TF32)
        // and F64 → 4 elements per access; F32Strict (SIMT) → 1. The
        // F32Strict bias path uses a vendored SIMT broadcast-epilogue
        // partial specialization (see
        // `crates/baracuda-cutlass-kernels-sys/kernels/include/
        // baracuda_simt_broadcast_epilogue.h`) because CUTLASS doesn't
        // wire one by default. F64 routes through the f64-scalar
        // dispatcher (`gemm_bias_sm80_run_f64`) because alpha/beta are
        // `f64` at the FFI. When new element kinds land that don't yet
        // have bias instantiations, reinstate a gate here that returns
        // `Error::Unsupported` so callers don't get a runtime status 3
        // deep inside the launch path.

        // Phase 30: pick the backend (CUTLASS vs cuBLAS) before the
        // arch. Caller may force a backend via `pref.prefer_backend`;
        // otherwise the f16/bf16-low-M heuristic chooses.
        // Phase 44: `BackendKind::Ozaki { slices }` joins the dispatch
        // surface for FP64-only / Identity-only / RCR-or-RRR shapes
        // (gated behind the `ozimmu` feature on this crate).
        let element = T::KIND;

        // Phase 44 ozIMMU dispatch — handled before the cuBLAS gate so
        // its preconditions (f64 / Identity / RCR|RRR / feature gate)
        // are validated up front.
        if let Some(BackendKind::Ozaki { slices }) = pref.prefer_backend {
            validate_ozaki_request(desc, element, slices)?;
            let arch_for_sku = pick_arch(stream, desc, pref)?;
            let backend = BackendChoice::Ozaki { slices };
            let sku = GemmSku {
                arch: arch_for_sku,
                layout: desc.layout,
                epilogue: desc.epilogue,
                element,
                bias_element: None,
            };
            return Ok(Self {
                desc: *desc,
                sku,
                backend,
                _element: PhantomData,
            });
        }

        let use_cublas = match pref.prefer_backend {
            Some(BackendKind::Cublas) => {
                // Force-Cublas: validate that cuBLAS actually supports
                // this (layout, epilogue, element) triple. Bias*
                // epilogues are not supported (cuBLAS-classic has no
                // fused-bias-activation). F32Strict has no cuBLAS dtype.
                if desc.epilogue.requires_bias() {
                    return Err(Error::Unsupported(
                        "cuBLAS backend doesn't fuse bias activations \
                         (use Cutlass backend for Bias* epilogues)",
                    ));
                }
                if cublas_dtype_for(element).is_none() {
                    return Err(Error::Unsupported(
                        "cuBLAS backend has no GemmEx dtype for this element \
                         (F32Strict / integer / FP8 stay on Cutlass)",
                    ));
                }
                true
            }
            Some(BackendKind::Cutlass) => false,
            Some(BackendKind::Ozaki { .. }) => {
                // Unreachable — handled by the early return above.
                false
            }
            Some(_) => {
                // Other backend hints aren't meaningful for GEMM; treat
                // as "let the heuristic decide".
                should_use_cublas_for_fp(desc, element)
                    && cublas_dtype_for(element).is_some()
            }
            None => {
                should_use_cublas_for_fp(desc, element)
                    && cublas_dtype_for(element).is_some()
            }
        };

        let (backend, sku_arch) = if use_cublas {
            // Capture device arch for the SKU record (so telemetry /
            // autotuner caches still see the real silicon), but the
            // backend choice routes through cuBLAS.
            let arch_for_sku = pick_arch(stream, desc, pref)?;
            (BackendChoice::Cublas, arch_for_sku)
        } else {
            let arch = pick_arch(stream, desc, pref)?;
            (BackendChoice::Cutlass { arch }, arch)
        };

        let sku = GemmSku {
            arch: sku_arch,
            layout: desc.layout,
            epilogue: desc.epilogue,
            element,
            // Float-family bias kernels imply bias element = element type;
            // `None` distinguishes them from the int-family bias kernels
            // (which encode the bias element explicitly because it's
            // independent of the matrix dtype).
            bias_element: None,
        };
        Ok(Self {
            desc: *desc,
            sku,
            backend,
            _element: PhantomData,
        })
    }

    /// Which backend this plan picked.
    ///
    /// CUTLASS (the default path) or cuBLAS (the Phase-30 fast-path
    /// for f16/bf16 low-M decode shapes). The dispatch heuristic is
    /// documented on [`should_use_cublas_for_fp`].
    pub fn backend(&self) -> BackendKind {
        self.backend.as_public()
    }

    /// Validate that this plan can actually launch with `args`.
    ///
    /// Two-stage check:
    /// 1. **Host-side**: shape/stride/buffer-size validation in pure Rust.
    /// 2. **Kernel-side**: calls CUTLASS's `Gemm::can_implement` host
    ///    adapter via a no-launch FFI symbol to catch alignment and
    ///    kernel-support issues that the host can't see (e.g., the
    ///    selected tile's element-per-access requirement on `lda`/`ldb`).
    ///
    /// Returns without launching a kernel and without touching the device.
    /// Use this as a clean prelaunch branch point: if it returns `Ok`, the
    /// `run` call will succeed barring runtime CUDA errors.
    pub fn can_implement(&self, args: &GemmArgs<'_, T>) -> Result<()> {
        check_args(&self.desc, args)?;

        // Phase 30: even when the plan picked the cuBLAS backend, we
        // still run the CUTLASS-side `can_implement` FFI check.
        // Rationale: the cuBLAS path falls back to CUTLASS under graph
        // capture (cuBLAS-classic calls aren't capture-safe), so the
        // CUTLASS launch path may execute regardless of which backend
        // the plan nominally targets. Better to validate the CUTLASS
        // path up front than discover an alignment problem deep in the
        // capture replay.

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld),
            None => (core::ptr::null(), 0i64),
        };
        let bias_ptr = args
            .bias
            .as_ref()
            .map(|b| b.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());

        let bias_family = self.sku.epilogue.requires_bias();
        let status = match (self.sku.arch, bias_family) {
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) if <T::Scalar as ScalarType>::IS_F64 => unsafe {
                dispatch::gemm_sm80_can_implement_f64(
                    self.sku.layout,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) => unsafe {
                dispatch::gemm_sm80_can_implement(
                    self.sku.layout,
                    T::KIND,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) if <T::Scalar as ScalarType>::IS_F64 => unsafe {
                dispatch::gemm_bias_sm80_can_implement_f64(
                    self.sku.layout,
                    self.sku.epilogue,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    bias_ptr,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) => unsafe {
                dispatch::gemm_bias_sm80_can_implement(
                    self.sku.layout,
                    T::KIND,
                    self.sku.epilogue,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    bias_ptr,
                )
            },
            #[cfg(not(feature = "sm80"))]
            (ArchSku::Sm80, _) => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            (ArchSku::Sm90a, _) => {
                return Err(Error::Unsupported(
                    "sm90a kernels not yet shipped (deferred until Hopper hardware available for validation)",
                ));
            }
            (ArchSku::Sm89, _) => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };

        status_to_result(status)
    }

    /// Bytes of device scratch this plan needs at `run` time.
    ///
    /// Returns 0 when the kernel's launch is workspace-free; pass
    /// [`Workspace::None`] in that case.
    ///
    /// Phase 30 note: even when the plan picked the cuBLAS backend
    /// (which manages its own scratch internally), this method
    /// reports the **CUTLASS-side** workspace requirement. The cuBLAS
    /// path falls back to CUTLASS under graph capture (cuBLAS-classic
    /// calls aren't capture-safe), so the caller must size the
    /// workspace for the CUTLASS path in case the fallback triggers.
    /// In practice CUTLASS Identity-epilogue GEMM on sm_80 reports
    /// 0 bytes for most (M, N, K), so this conservative reporting
    /// rarely costs anything.
    pub fn workspace_size(&self) -> usize {
        let bias_family = self.sku.epilogue.requires_bias();
        match (self.sku.arch, bias_family) {
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) if <T::Scalar as ScalarType>::IS_F64 => {
                dispatch::gemm_sm80_workspace_size_f64(
                    self.sku.layout,
                    self.desc.m, self.desc.n, self.desc.k,
                )
            }
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) => dispatch::gemm_sm80_workspace_size(
                self.sku.layout,
                T::KIND,
                self.desc.m, self.desc.n, self.desc.k,
            ),
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) if <T::Scalar as ScalarType>::IS_F64 => {
                dispatch::gemm_bias_sm80_workspace_size_f64(
                    self.sku.layout,
                    self.sku.epilogue,
                    self.desc.m, self.desc.n, self.desc.k,
                )
            }
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) => dispatch::gemm_bias_sm80_workspace_size(
                self.sku.layout,
                T::KIND,
                self.sku.epilogue,
                self.desc.m, self.desc.n, self.desc.k,
            ),
            #[cfg(not(feature = "sm80"))]
            (ArchSku::Sm80, _) => 0,
            (ArchSku::Sm90a, _) => 0,
            (ArchSku::Sm89, _) => 0,
        }
    }

    /// Identity of the kernel this plan chose.
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees this plan's kernel provides.
    ///
    /// Convenience for [`GemmSku::precision_guarantee`] applied to this
    /// plan's SKU. Useful for callers that maintain a per-decision-point
    /// alternatives table (e.g. picking between cuBLAS and CUTLASS for a
    /// given precision contract) without having to re-derive the
    /// guarantees from per-kernel documentation.
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Launch the kernel.
    ///
    /// `workspace` must be at least [`workspace_size`](Self::workspace_size)
    /// bytes when non-zero, or [`Workspace::None`] when zero. The stream
    /// must be in the same context as the device buffers in `args`.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: GemmArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;

        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                if needed != 0 {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: 0,
                    });
                }
                (core::ptr::null_mut(), 0)
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld),
            None => (core::ptr::null(), 0i64),
        };
        let bias_ptr = args
            .bias
            .as_ref()
            .map(|b| b.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        // When the caller passes c = None, force beta = 0 at the safe
        // layer. The kernel internally substitutes D for the C operand to
        // satisfy CUTLASS's non-null pointer contract, so a non-zero beta
        // here would silently fold the previous D contents into the
        // result (D += alpha*AB instead of D = alpha*AB).
        let beta_eff = if args.c.is_some() { args.beta } else { <T::Scalar as Default>::default() };
        let stream_raw = stream.as_raw();

        // Phase 30: cuBLAS fast-path. Dispatches to `cublasGemmEx`
        // (f16/bf16/f32) or `cublasDgemm` (f64). cuBLAS is column-major;
        // we apply the row-major-from-col-major trick: compute
        // `D^T = (op_b B)^T · (op_a A)^T` in cuBLAS terms, which
        // lets us pass A/B straight through and read D row-major.
        //
        // For RCR (A row-major, B col-major, D row-major):
        //   cuBLAS sees A as col-major A^T [K, M] (lda = K),
        //   cuBLAS sees B as col-major B [K, N] (ldb = K),
        //   D^T col-major [N, M] (ldd = N) IS D row-major [M, N].
        //   So we call gemmEx(transa = T, transb = N, m=N, n=M, k=K,
        //                     B, ldb=K, A, lda=K, D, ldd=N).
        // For RRR (A row-major, B row-major, D row-major):
        //   B viewed as col-major is B^T [N, K] (ldb = N),
        //   so transb stays N (no transpose) but the operands are
        //   B^T·A^T = (A·B)^T → identical call shape with transa=N.
        //
        // Capture-mode guard: cuBLAS-classic calls aren't capture-safe
        // (they perform host-side handle-state mutations and may issue
        // host allocations / branches that break graph capture). Fall
        // back to CUTLASS when the stream is in capture mode. The
        // `is_capturing` query itself is graph-safe (it just reads
        // driver state).
        if matches!(self.backend, BackendChoice::Cublas) {
            let capturing = stream.is_capturing().unwrap_or(false);
            if !capturing {
                return self.run_cublas(stream, args, beta_eff);
            }
            // Fall through to the CUTLASS dispatch below. The
            // BackendChoice::Cublas flag stays set on the plan for
            // telemetry / sku() consistency; only this launch falls
            // back. Subsequent launches outside capture will route
            // back to cuBLAS.
        }

        // Phase 44: ozIMMU dispatch. Same capture-safety story as the
        // cuBLAS path — ozIMMU calls into cuBLAS internally to drive
        // the int8 tensor-core matmuls + accumulate stage, so it
        // inherits cuBLAS's "not capture-safe" property. Fall back to
        // the CUTLASS DGEMM path under capture; the SKU stays Ozaki
        // for telemetry consistency.
        #[cfg(feature = "ozimmu")]
        if let BackendChoice::Ozaki { slices } = self.backend {
            let capturing = stream.is_capturing().unwrap_or(false);
            if !capturing {
                return self.run_ozaki(stream, args, beta_eff, slices);
            }
        }
        #[cfg(not(feature = "ozimmu"))]
        if matches!(self.backend, BackendChoice::Ozaki { .. }) {
            // Should be unreachable — `select` rejects Ozaki when the
            // feature is off. Belt-and-suspenders.
            return Err(Error::Unsupported(
                "BackendChoice::Ozaki selected without `ozimmu` cargo feature",
            ));
        }

        let bias_family = self.sku.epilogue.requires_bias();
        let status = match (self.sku.arch, bias_family) {
            // Fork on T::Scalar::IS_F64 to pick the matching FFI dispatcher.
            // The two paths only differ in the alpha/beta types passed to
            // CUTLASS; the `to_f32` / `to_f64` calls are identity on the
            // matching impl and never narrow at runtime because the gate
            // statically picks the impl that matches `T::Scalar`.
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) if <T::Scalar as ScalarType>::IS_F64 => unsafe {
                dispatch::gemm_sm80_run_f64(
                    self.sku.layout,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    args.alpha.to_f64(),
                    beta_eff.to_f64(),
                    ws_ptr, ws_bytes, stream_raw,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) => unsafe {
                dispatch::gemm_sm80_run(
                    self.sku.layout,
                    T::KIND,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    args.alpha.to_f32(),
                    beta_eff.to_f32(),
                    ws_ptr, ws_bytes, stream_raw,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) if <T::Scalar as ScalarType>::IS_F64 => unsafe {
                dispatch::gemm_bias_sm80_run_f64(
                    self.sku.layout,
                    self.sku.epilogue,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    bias_ptr,
                    args.alpha.to_f64(),
                    beta_eff.to_f64(),
                    ws_ptr, ws_bytes, stream_raw,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) => unsafe {
                dispatch::gemm_bias_sm80_run(
                    self.sku.layout,
                    T::KIND,
                    self.sku.epilogue,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    bias_ptr,
                    args.alpha.to_f32(),
                    beta_eff.to_f32(),
                    ws_ptr, ws_bytes, stream_raw,
                )
            },
            #[cfg(not(feature = "sm80"))]
            (ArchSku::Sm80, _) => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            (ArchSku::Sm90a, _) => {
                return Err(Error::Unsupported(
                    "sm90a kernels not yet implemented (Phase 4c)",
                ));
            }
            (ArchSku::Sm89, _) => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };

        status_to_result(status)
    }

    /// Phase-30 cuBLAS fast-path launch.
    ///
    /// Caller already validated host-side shape / strides via
    /// [`Self::can_implement`] and confirmed the chosen backend is
    /// [`BackendChoice::Cublas`]. This method translates baracuda's
    /// row-major operands into cuBLAS's column-major convention via
    /// the standard "compute D^T in col-major" trick, then dispatches
    /// to `cublasGemmEx` (f16/bf16/f32) or `cublasDgemm` (f64).
    fn run_cublas(
        &self,
        stream: &Stream,
        args: GemmArgs<'_, T>,
        beta_eff: T::Scalar,
    ) -> Result<()> {
        use baracuda_cublas::Op as CublasOp;
        use baracuda_cublas_sys::functions::cublasComputeType_t;

        // Bias / Bias* epilogues aren't supported on the cuBLAS path —
        // `select` already rejected them. Belt-and-suspenders here.
        if self.sku.epilogue.requires_bias() {
            return Err(Error::Unsupported(
                "cuBLAS backend doesn't fuse bias activations \
                 (caller forced a Bias* epilogue onto the cuBLAS path)",
            ));
        }

        let handle = cublas_backend::handle_for(stream)?;

        let m = self.desc.m;
        let n = self.desc.n;
        let k = self.desc.k;
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        // C must equal D for cuBLAS's `C = α·op(A)·op(B) + β·C` shape
        // when the safe-layer caller passed C=None (treated as β=0).
        // When C is Some, we route it through. cuBLAS reads C with the
        // same layout/ldc as it writes D, so the same row-major-from-
        // col-major mapping applies.
        let (c_ptr, ldc_arg) = match &args.c {
            // cuBLAS expects β*C added in; pass the caller's C directly.
            Some(c) => (c.data.as_raw().0 as *mut c_void, c.ld as i32),
            None => (d_ptr, args.d.ld as i32),
        };

        // Row-major-from-col-major mapping for baracuda's Rcr/Rrr:
        //
        // baracuda Rcr: A row-major [M, K] (lda = K),
        //               B col-major [K, N] (ldb = K),
        //               D row-major [M, N] (ldd = N).
        //
        // cuBLAS sees the same memory column-major:
        //   - A_raw[i*K + kk] col-major (lda=K) → A_blas[kk, i] = A[i, kk]
        //     ⇒ cuBLAS view of A is A^T, shape [K, M], lda=K.
        //   - B_raw[j*K + kk] col-major (ldb=K) → B_blas[kk, j] = B[kk, j]
        //     ⇒ cuBLAS view of B IS B, shape [K, N], ldb=K.
        //   - D_raw[i*N + j] col-major (ldd=N) → D_blas[j, i] = D[i, j]
        //     ⇒ cuBLAS view of D is D^T, shape [N, M], ldd=N.
        //
        // The target identity is D[i,j] = Σ_kk A[i,kk]·B[kk,j], which in
        // cuBLAS view is D_blas[j,i] = Σ_kk B_blas[kk,j] · A_blas[kk,i].
        // Expressed as a cuBLAS GEMM `D_blas = op(X)·op(Y)` with
        // D_blas shape (M_blas, N_blas) = (N, M):
        //   - op(X) shape (M_blas, K) = (N, K) ⇐ B_blas with Op::T
        //     (B_blas is [K, N], B_blas^T = [N, K]).
        //   - op(Y) shape (K, N_blas) = (K, M) ⇐ A_blas with Op::N
        //     (A_blas is already [K, M]).
        // So pass B as the first operand to gemmEx with Op::T and A as
        // the second operand with Op::N.
        //
        // For Rrr (A row-major [M,K], B row-major [K,N], D row-major
        // [M,N]):
        //   - B_raw[kk*N + j] col-major (ldb=N) → B_blas[j, kk] = B[kk, j]
        //     ⇒ cuBLAS view of B is B^T, shape [N, K], ldb=N.
        //   - First operand needs (M_blas, K) = (N, K) ⇐ B^T-as-stored
        //     with Op::N (no further transpose).
        //   - Second operand same as Rcr: A with Op::N, lda=K.
        //
        // `cublas_lda` and `cublas_ldb` below name the ld values *as
        // gemmEx expects them*: cublas_lda is the leading dim of the
        // first operand (which in our call is baracuda's B), and
        // cublas_ldb is the leading dim of the second operand (which
        // is baracuda's A). For both Rcr and Rrr, cublas_lda = args.b.ld
        // and cublas_ldb = args.a.ld; the only thing that changes is
        // the op on the first operand.
        let (transa, transb) = match self.desc.layout {
            LayoutSku::Rcr => (CublasOp::T, CublasOp::N),
            LayoutSku::Rrr => (CublasOp::N, CublasOp::N),
        };
        let cublas_lda = args.b.ld as i32; // first operand (B) ld
        let cublas_ldb = args.a.ld as i32; // second operand (A) ld
        let ldd_arg = args.d.ld as i32;

        // Pick compute path by `T::Scalar` (the FP family bound on
        // CutlassElement guarantees f32 or f64). f64 routes through
        // `cublasDgemm`; everything else uses `cublasGemmEx` with
        // `CUBLAS_COMPUTE_32F` (f32 accumulator regardless of operand
        // dtype, matching cuBLAS's tensor-core path).
        if <T::Scalar as ScalarType>::IS_F64 {
            // f64 path: cublasDgemm. No GemmEx needed; α/β are f64.
            use baracuda_cublas_sys::cublasOperation_t;

            // Translate cublasOperation_t for the raw helper.
            let to_raw = |op: CublasOp| match op {
                CublasOp::N => cublasOperation_t::N,
                CublasOp::T => cublasOperation_t::T,
                CublasOp::C => cublasOperation_t::C,
            };
            // cublasDgemm signature: (handle, transa, transb, m, n, k,
            //   alpha, A, lda, B, ldb, beta, C, ldc).
            // Layout-mapped operand order: B first, A second, output D.
            let alpha_f64 = args.alpha.to_f64();
            let beta_f64 = beta_eff.to_f64();
            let c_api = baracuda_cublas_sys::cublas()
                .map_err(|_| Error::Unsupported("cuBLAS library unavailable"))?;
            let dgemm = c_api
                .cublas_dgemm()
                .map_err(|_| Error::Unsupported("cublasDgemm symbol unavailable"))?;
            // SAFETY: every pointer is from a live DeviceBuffer with
            // the correct dtype; sizes were validated via check_args;
            // the handle is bound to `stream`'s context.
            let status = unsafe {
                dgemm(
                    handle.as_raw(),
                    to_raw(transa),
                    to_raw(transb),
                    n,
                    m,
                    k,
                    &alpha_f64,
                    b_ptr as *const f64,
                    cublas_lda,
                    a_ptr as *const f64,
                    cublas_ldb,
                    &beta_f64,
                    // cublasDgemm uses C as both input and output. Pass
                    // C-or-D depending on whether caller supplied C.
                    if args.c.is_some() {
                        // Caller passed a separate C. cuBLAS writes
                        // back into the same buffer it reads from, so
                        // we'd need a copy step to materialize into D.
                        // For now, restrict the cuBLAS f64 path to
                        // C=None (the common decode case).
                        return Err(Error::Unsupported(
                            "cuBLAS f64 GEMM with explicit C operand is not yet wired \
                             (D and C alias differently than cuBLAS expects); \
                             use Cutlass backend or set c = None",
                        ));
                    } else {
                        d_ptr as *mut f64
                    },
                    ldd_arg,
                )
            };
            return match status {
                baracuda_cublas_sys::cublasStatus_t::SUCCESS => Ok(()),
                _ => Err(Error::CutlassInternal(status.0)),
            };
        }

        // f16/bf16/f32 path: cublasGemmEx with Compute32F accumulator
        // and the tensor-op default algo selector.
        let dtype = cublas_dtype_for(self.sku.element).ok_or(Error::Unsupported(
            "cuBLAS backend selected for element kind without a cuBLAS dtype mapping",
        ))?;
        // For the col-major-mapped problem the operands swap order; the
        // dtype tag is the same for A, B, and D since all three carry
        // the same `T`.
        let a_type = dtype;
        let b_type = dtype;
        let c_type = dtype;
        // Alpha/beta as f32 (the host scalars cuBLAS expects when
        // compute_type = Compute32F).
        let alpha_f32 = args.alpha.to_f32();
        let beta_f32 = beta_eff.to_f32();

        // Caller passed an explicit C: cuBLAS gemmEx writes C in-place
        // (treats C as the output too). When `args.c == Some` and the
        // caller wants D != C, we'd need a memcpy step. For simplicity
        // (and matching the bench's no-C decode case) reject the
        // explicit-C path on the cuBLAS branch and tell the caller to
        // use Cutlass backend. The Identity / no-bias path is the
        // common decode shape we're optimizing.
        if args.c.is_some() {
            return Err(Error::Unsupported(
                "cuBLAS GemmPlan path requires c = None \
                 (cublasGemmEx writes the output in-place into the C operand; \
                 explicit-C with D ≠ C requires an extra copy step — \
                 force Cutlass backend if you need it)",
            ));
        }
        let _ = (c_ptr, ldc_arg); // suppress unused-let warning

        // SAFETY: every pointer is from a live DeviceBuffer with the
        // correct dtype; sizes were validated via check_args; the
        // handle is bound to `stream`'s context.
        unsafe {
            baracuda_cublas::gemm_ex(
                &handle,
                transa,
                transb,
                n,
                m,
                k,
                &alpha_f32 as *const f32 as *const c_void,
                b_ptr,
                b_type,
                cublas_lda,
                a_ptr,
                a_type,
                cublas_ldb,
                &beta_f32 as *const f32 as *const c_void,
                d_ptr,
                c_type,
                ldd_arg,
                cublasComputeType_t::Compute32F,
                CUBLAS_GEMM_ALGO,
            )
            .map_err(|_| Error::CutlassInternal(-1))
        }
    }

    /// Phase-44 ozIMMU dispatch launch.
    ///
    /// Caller already validated host-side shape / strides via
    /// [`Self::can_implement`], confirmed the chosen backend is
    /// [`BackendChoice::Ozaki`], and verified the stream is not in
    /// graph-capture mode (Ozaki is not capture-safe — ozIMMU runs
    /// cuBLAS internally on the int8 accumulate stage). This method
    /// translates baracuda's row-major operands into ozIMMU's
    /// cuBLAS-compatible column-major convention (same trick as the
    /// cuBLAS f64 path) and dispatches to `mtk::ozimmu::gemm`.
    ///
    /// Restrictions for the Phase 44 alpha:
    /// - `args.c` must be `None` (caller passes a C operand only when
    ///   they want `D = alpha*AB + beta*C` with `D != C`; ozIMMU's
    ///   GEMM uses C as the output operand, same as cublasDgemm, so
    ///   we'd need an extra copy to materialize into D — defer).
    /// - Element must be F64 — guarded at `select` already; this
    ///   method takes the safe-typed `T` so the compiler can't see
    ///   that, but the `Scalar::IS_F64` check below makes it
    ///   defense-in-depth.
    #[cfg(feature = "ozimmu")]
    fn run_ozaki(
        &self,
        stream: &Stream,
        args: GemmArgs<'_, T>,
        beta_eff: T::Scalar,
        slices: u8,
    ) -> Result<()> {
        use baracuda_ozimmu::{Op as OzakiOp, OzakiSlices, OzakiVariant};

        if !<T::Scalar as ScalarType>::IS_F64 {
            return Err(Error::Unsupported(
                "BackendChoice::Ozaki reached on non-f64 element \
                 (select() guard should have rejected this)",
            ));
        }
        if args.c.is_some() {
            return Err(Error::Unsupported(
                "ozIMMU GemmPlan path requires c = None \
                 (the Ozaki path writes its output in-place into the C \
                 operand of the underlying cuBLAS GEMM — explicit-C with \
                 D ≠ C requires an extra copy step that the Phase 44 \
                 alpha does not yet wire; force Cutlass backend if needed)",
            ));
        }

        // Phase 44c — decode the discriminant. Low 5 bits = slice
        // count (0 = auto, 3..=18 = fixed); high 3 bits = variant
        // (0 = Base, 1 = EF, 2 = RN, 3 = H). The pure-slices-only
        // values 0..=18 used by Phase 44b decode as Base, preserving
        // source-compat for existing callers.
        let s = slices & 0x1F;
        let v = slices >> 5;
        let slice_choice = match s {
            0 => OzakiSlices::Auto,
            3 => OzakiSlices::S3,
            4 => OzakiSlices::S4,
            5 => OzakiSlices::S5,
            6 => OzakiSlices::S6,
            7 => OzakiSlices::S7,
            8 => OzakiSlices::S8,
            9 => OzakiSlices::S9,
            10 => OzakiSlices::S10,
            11 => OzakiSlices::S11,
            12 => OzakiSlices::S12,
            13 => OzakiSlices::S13,
            14 => OzakiSlices::S14,
            15 => OzakiSlices::S15,
            16 => OzakiSlices::S16,
            17 => OzakiSlices::S17,
            18 => OzakiSlices::S18,
            _ => {
                return Err(Error::Unsupported(
                    "ozIMMU slice count out of range (validated at select; \
                     this is unreachable)",
                ));
            }
        };
        let variant_choice = match v {
            0 => OzakiVariant::Base,
            1 => OzakiVariant::EF,
            2 => OzakiVariant::RN,
            3 => OzakiVariant::H,
            _ => {
                return Err(Error::Unsupported(
                    "ozIMMU variant out of range (validated at select; \
                     this is unreachable)",
                ));
            }
        };

        let handle = ozimmu_backend::handle_for(stream)?;

        // Row-major → ozIMMU (cuBLAS-compatible col-major) mapping.
        // Same algebra as the cuBLAS f64 path: compute `D^T` in
        // col-major terms, swap the operand order, set transa = T for
        // RCR (because B in col-major IS B^T-transposed-from-row-major)
        // or transa = N for RRR (because B in col-major IS B from row
        // major).
        //
        // Pass B as the first operand, A as the second.
        let (transa, transb) = match self.desc.layout {
            LayoutSku::Rcr => (OzakiOp::T, OzakiOp::N),
            LayoutSku::Rrr => (OzakiOp::N, OzakiOp::N),
        };
        let m = self.desc.m as usize;
        let n = self.desc.n as usize;
        let k = self.desc.k as usize;
        let lda = args.b.ld as usize; // first operand (B) ld
        let ldb = args.a.ld as usize; // second operand (A) ld
        let ldc = args.d.ld as usize;

        let a_ptr = args.a.data.as_raw().0 as *const f64;
        let b_ptr = args.b.data.as_raw().0 as *const f64;
        let d_ptr = args.d.data.as_raw().0 as *mut f64;
        let alpha = args.alpha.to_f64();
        let beta = beta_eff.to_f64();

        // SAFETY: the descriptor / args were validated by
        // can_implement(); pointers are live DeviceBuffer<f64> views.
        //
        // Phase 44c — always go through dgemm_with_variant. Base
        // variant is bit-identical to the pre-44c dgemm path; the
        // variant dispatch happens inside the C++ shim with no
        // measurable per-call host overhead.
        unsafe {
            handle.dgemm_with_variant(
                transa, transb,
                // ozIMMU sees us computing D^T col-major: shape (n, m).
                n, m, k,
                alpha,
                b_ptr, lda,
                a_ptr, ldb,
                beta,
                d_ptr, ldc,
                slice_choice,
                variant_choice,
            )
            .map_err(|e| {
                use baracuda_ozimmu::Error as OzErr;
                match e {
                    OzErr::DgemmFailed(s) => Error::CutlassInternal(s),
                    _ => Error::Unsupported(
                        "ozIMMU dgemm rejected the request (see logs)",
                    ),
                }
            })
        }
    }
}

// ============================================================================
// Phase 44 ozIMMU backend — handle cache + dispatch helpers
// ============================================================================
//
// Mirror of the Phase-30 `cublas_backend` module: thread-local cache of
// ozIMMU handles keyed by raw context pointer. ozIMMU handles are
// expensive to construct (they spin up a cuBLAS handle internally + do
// some env-var probing); cache + re-bind keeps the steady-state launch
// cost down to one `set_cuda_stream` call. Returns an `Arc`-cloned
// safe wrapper so the cache can hold the canonical handle while the
// caller drives a launch without holding the thread-local borrow.

#[cfg(feature = "ozimmu")]
mod ozimmu_backend {
    use core::cell::RefCell;
    use std::rc::Rc;

    use baracuda_driver::Stream;
    use baracuda_ozimmu::Handle as OzimmuHandle;

    thread_local! {
        static HANDLE_CACHE: RefCell<Vec<(usize, Rc<OzimmuHandle>)>> =
            const { RefCell::new(Vec::new()) };
    }

    /// Fetch (or lazily create) an ozIMMU handle bound to `stream`'s
    /// context, with the handle's stream binding set to `stream`.
    pub(super) fn handle_for(stream: &Stream) -> crate::Result<Rc<OzimmuHandle>> {
        let ctx_key = stream.context().as_raw() as usize;
        let handle = HANDLE_CACHE.with(|cache| -> crate::Result<Rc<OzimmuHandle>> {
            let mut cache = cache.borrow_mut();
            if let Some((_, h)) = cache.iter().find(|(k, _)| *k == ctx_key) {
                return Ok(h.clone());
            }
            // Make sure the stream's context is current — ozIMMU's
            // create() calls into cuBLAS create() which needs the
            // current context to bind correctly.
            stream
                .context()
                .set_current()
                .map_err(crate::Error::Driver)?;
            // Retry with linear backoff under transient init failures
            // (mirrors the cuBLAS Phase-35 retry pattern; ozIMMU
            // wraps cuBLAS so it inherits the same parallel-init
            // race window).
            let mut last_status: Option<i32> = None;
            let mut handle: Option<OzimmuHandle> = None;
            for attempt in 0..5 {
                match OzimmuHandle::new() {
                    Ok(h) => { handle = Some(h); break }
                    Err(e) => {
                        if let baracuda_ozimmu::Error::CreateFailed(s) = e {
                            last_status = Some(s);
                        }
                        std::thread::sleep(std::time::Duration::from_millis(
                            50 * (attempt as u64 + 1),
                        ));
                    }
                }
            }
            let h = match handle {
                Some(h) => h,
                None => {
                    let _ = last_status;
                    return Err(crate::Error::Unsupported(
                        "ozIMMU handle creation failed after 5 retries \
                         (library missing, device unavailable, or persistent \
                         init contention)",
                    ));
                }
            };
            let rc = Rc::new(h);
            cache.push((ctx_key, rc.clone()));
            Ok(rc)
        })?;
        handle.set_stream(stream);
        Ok(handle)
    }
}

// ============================================================================
// BatchedGemmPlan — uniform-shape batched GEMM
// ============================================================================
//
// All batches share `(M, N, K)`; per-batch operands are addressed by
// adding `i * stride_*` (in elements) to the base pointer. For
// variable-shape grouped problems use `GroupedGemmPlan` instead.
//
// v1 coverage: Rcr layout, F16 / Bf16 elements, sm_80, Identity epilogue.

fn check_batched_descriptor(desc: &BatchedGemmDescriptor) -> Result<()> {
    if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
        return Err(Error::InvalidProblem("M, N, K must all be positive"));
    }
    if desc.batch_count <= 0 {
        return Err(Error::InvalidProblem("batch_count must be positive"));
    }
    if desc.epilogue != EpilogueKind::Identity {
        return Err(Error::Unsupported(
            "BatchedGemmPlan v1 supports only EpilogueKind::Identity",
        ));
    }
    Ok(())
}

fn check_batched_args<T: CutlassElement>(
    desc: &BatchedGemmDescriptor,
    args: &BatchedGemmArgs<'_, T>,
) -> Result<()> {
    // Per-batch shape validation matches the single-GEMM path; strides
    // are validated by checking that the last batch's max-addressable
    // element fits within each base buffer.
    if args.a.rows != desc.m || args.a.cols != desc.k {
        return Err(Error::InvalidProblem("A shape doesn't match descriptor (M, K)"));
    }
    if args.b.rows != desc.k || args.b.cols != desc.n {
        return Err(Error::InvalidProblem("B shape doesn't match descriptor (K, N)"));
    }
    if args.d.rows != desc.m || args.d.cols != desc.n {
        return Err(Error::InvalidProblem("D shape doesn't match descriptor (M, N)"));
    }
    if let Some(c) = &args.c {
        if c.rows != desc.m || c.cols != desc.n {
            return Err(Error::InvalidProblem("C shape doesn't match descriptor (M, N)"));
        }
    }
    if args.a.ld < desc.k as i64 {
        return Err(Error::InvalidProblem("A leading dimension must be >= K"));
    }
    let b_min_ld = match desc.layout {
        LayoutSku::Rcr => desc.k as i64,
        LayoutSku::Rrr => desc.n as i64,
    };
    if args.b.ld < b_min_ld {
        return Err(Error::InvalidProblem("B leading dimension too small for layout"));
    }
    if args.d.ld < desc.n as i64 {
        return Err(Error::InvalidProblem("D leading dimension must be >= N"));
    }
    if let Some(c) = &args.c {
        if c.ld < desc.n as i64 {
            return Err(Error::InvalidProblem("C leading dimension must be >= N"));
        }
    }

    // Per-batch element footprint = single-batch min + (batch - 1) * stride.
    // Stride 0 means "broadcast same matrix across all batches" — the
    // single-batch min is the only constraint there.
    fn need_for_batches(
        per_batch_min: usize,
        stride: i64,
        batch_count: i32,
    ) -> Option<usize> {
        if batch_count <= 1 || stride == 0 {
            return Some(per_batch_min);
        }
        let extra = stride.checked_mul((batch_count - 1) as i64)?;
        let extra = usize::try_from(extra).ok()?;
        per_batch_min.checked_add(extra)
    }

    let a_per = min_elements_row_major(args.a.rows, args.a.cols, args.a.ld)
        .ok_or(Error::InvalidProblem("A storage size overflow"))?;
    let need_a = need_for_batches(a_per, args.stride_a, desc.batch_count)
        .ok_or(Error::InvalidProblem("A batched storage size overflow"))?;
    if args.a.data.len() < need_a {
        return Err(Error::BufferTooSmall {
            needed: need_a,
            got: args.a.data.len(),
        });
    }

    let b_per = match desc.layout {
        LayoutSku::Rcr => min_elements_col_major(args.b.rows, args.b.cols, args.b.ld),
        LayoutSku::Rrr => min_elements_row_major(args.b.rows, args.b.cols, args.b.ld),
    }
    .ok_or(Error::InvalidProblem("B storage size overflow"))?;
    let need_b = need_for_batches(b_per, args.stride_b, desc.batch_count)
        .ok_or(Error::InvalidProblem("B batched storage size overflow"))?;
    if args.b.data.len() < need_b {
        return Err(Error::BufferTooSmall {
            needed: need_b,
            got: args.b.data.len(),
        });
    }

    let d_per = min_elements_row_major(args.d.rows, args.d.cols, args.d.ld)
        .ok_or(Error::InvalidProblem("D storage size overflow"))?;
    let need_d = need_for_batches(d_per, args.stride_d, desc.batch_count)
        .ok_or(Error::InvalidProblem("D batched storage size overflow"))?;
    if args.d.data.len() < need_d {
        return Err(Error::BufferTooSmall {
            needed: need_d,
            got: args.d.data.len(),
        });
    }

    if let Some(c) = &args.c {
        let c_per = min_elements_row_major(c.rows, c.cols, c.ld)
            .ok_or(Error::InvalidProblem("C storage size overflow"))?;
        let need_c = need_for_batches(c_per, args.stride_c, desc.batch_count)
            .ok_or(Error::InvalidProblem("C batched storage size overflow"))?;
        if c.data.len() < need_c {
            return Err(Error::BufferTooSmall {
                needed: need_c,
                got: c.data.len(),
            });
        }
    }
    Ok(())
}

/// Plan for a uniform-shape batched GEMM launch.
///
/// All batches share `(M, N, K)`. Plans hold host-side selection
/// metadata only — no device memory, cheap to clone, `Send + Sync`.
///
/// See [`GroupedGemmPlan`] for the variable-shape (per-group) case.
#[derive(Debug)]
pub struct BatchedGemmPlan<T: CutlassElement> {
    desc: BatchedGemmDescriptor,
    sku: GemmSku,
    _element: PhantomData<T>,
}

impl<T: CutlassElement> BatchedGemmPlan<T> {
    /// Pick a kernel for `desc`.
    ///
    /// Returns [`Error::Unsupported`] when the requested
    /// `(layout, element)` combination has no shipped batched kernel
    /// (today: anything other than `Rcr × {F16, Bf16}`).
    pub fn select(
        stream: &Stream,
        desc: &BatchedGemmDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        check_batched_descriptor(desc)?;
        let one_off_desc = GemmDescriptor {
            m: desc.m,
            n: desc.n,
            k: desc.k,
            layout: desc.layout,
            epilogue: desc.epilogue,
        };
        let arch = pick_arch(stream, &one_off_desc, pref)?;
        // v1 coverage gate: only Rcr × {F16, Bf16} have batched kernels.
        match (desc.layout, T::KIND) {
            (LayoutSku::Rcr, ElementKind::F16) | (LayoutSku::Rcr, ElementKind::Bf16) => {}
            _ => {
                return Err(Error::Unsupported(
                    "BatchedGemmPlan v1 only ships Rcr × {F16, Bf16} on sm_80",
                ));
            }
        }
        let sku = GemmSku {
            arch,
            layout: desc.layout,
            epilogue: desc.epilogue,
            element: T::KIND,
            // Float-family bias kernels imply bias element = element type;
            // `None` distinguishes them from the int-family bias kernels
            // (which encode the bias element explicitly because it's
            // independent of the matrix dtype).
            bias_element: None,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _element: PhantomData,
        })
    }

    /// Validate that this plan can launch with `args`. See
    /// [`GemmPlan::can_implement`] for the two-stage host + kernel check.
    pub fn can_implement(&self, args: &BatchedGemmArgs<'_, T>) -> Result<()> {
        check_batched_args(&self.desc, args)?;

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc, stride_c) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld, args.stride_c),
            None => (core::ptr::null(), 0i64, 0i64),
        };

        let status = match self.sku.arch {
            #[cfg(feature = "sm80")]
            ArchSku::Sm80 => unsafe {
                dispatch::batched_gemm_sm80_can_implement(
                    self.sku.layout,
                    T::KIND,
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    a_ptr,
                    args.a.ld,
                    args.stride_a,
                    b_ptr,
                    args.b.ld,
                    args.stride_b,
                    c_ptr,
                    ldc,
                    stride_c,
                    d_ptr,
                    args.d.ld,
                    args.stride_d,
                    self.desc.batch_count,
                )
            },
            #[cfg(not(feature = "sm80"))]
            ArchSku::Sm80 => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            ArchSku::Sm90a => {
                return Err(Error::Unsupported(
                    "sm90a batched kernels not yet shipped",
                ));
            }
            ArchSku::Sm89 => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };

        status_to_result(status)
    }

    /// Bytes of device scratch this plan needs at `run` time.
    pub fn workspace_size(&self) -> usize {
        match self.sku.arch {
            #[cfg(feature = "sm80")]
            ArchSku::Sm80 => dispatch::batched_gemm_sm80_workspace_size(
                self.sku.layout,
                T::KIND,
                self.desc.m,
                self.desc.n,
                self.desc.k,
                self.desc.batch_count,
            ),
            #[cfg(not(feature = "sm80"))]
            ArchSku::Sm80 => 0,
            ArchSku::Sm90a => 0,
            ArchSku::Sm89 => 0,
        }
    }

    /// Identity of the kernel this plan chose.
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees this plan's kernel provides.
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Launch the batched kernel.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: BatchedGemmArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;

        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                if needed != 0 {
                    return Err(Error::WorkspaceTooSmall { needed, got: 0 });
                }
                (core::ptr::null_mut(), 0)
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc, stride_c) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld, args.stride_c),
            None => (core::ptr::null(), 0i64, 0i64),
        };
        let beta_eff = if args.c.is_some() { args.beta } else { <T::Scalar as Default>::default() };
        let stream_raw = stream.as_raw();

        let status = match self.sku.arch {
            #[cfg(feature = "sm80")]
            ArchSku::Sm80 => unsafe {
                // Batched GEMM v1 ships only Rcr × {F16, Bf16} — both
                // f32-scalar — so the `to_f32` conversion is identity.
                // The select gate rejects F64/F32Strict before we reach here.
                dispatch::batched_gemm_sm80_run(
                    self.sku.layout,
                    T::KIND,
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    a_ptr,
                    args.a.ld,
                    args.stride_a,
                    b_ptr,
                    args.b.ld,
                    args.stride_b,
                    c_ptr,
                    ldc,
                    stride_c,
                    d_ptr,
                    args.d.ld,
                    args.stride_d,
                    args.alpha.to_f32(),
                    beta_eff.to_f32(),
                    self.desc.batch_count,
                    ws_ptr,
                    ws_bytes,
                    stream_raw,
                )
            },
            #[cfg(not(feature = "sm80"))]
            ArchSku::Sm80 => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            ArchSku::Sm90a => {
                return Err(Error::Unsupported(
                    "sm90a batched kernels not yet shipped",
                ));
            }
            ArchSku::Sm89 => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };

        status_to_result(status)
    }
}

fn pick_arch(
    stream: &Stream,
    _desc: &GemmDescriptor,
    pref: PlanPreference,
) -> Result<ArchSku> {
    // Selection policy:
    //   1. Query the stream's device for its compute capability.
    //   2. Prefer sm90a when (a) the caller didn't disable it via
    //      `pref.allow_sm90a == false`, (b) the `sm90a` feature is on,
    //      and (c) the device is actually Hopper (cap >= 9.0).
    //   3. Otherwise fall back to sm80, which runs forward-compatibly on
    //      Ampere (sm_80), Ada (sm_89), and Hopper (sm_90+) at lower peak
    //      perf than sm90a.
    //
    // Build features control what kernels are *available*; this function
    // controls what's *picked* given the actual device.
    let (major, _minor) = stream.context().device().compute_capability()?;

    if pref.allow_sm90a && cfg!(feature = "sm90a") && major >= 9 {
        return Ok(ArchSku::Sm90a);
    }

    if cfg!(feature = "sm80") {
        // sm80 kernels are PTX-forward-compatible to anything sm_80+.
        if major >= 8 {
            return Ok(ArchSku::Sm80);
        }
        return Err(Error::Unsupported(
            "device compute capability < 8.0; sm_80 kernels won't run here",
        ));
    }

    Err(Error::Unsupported(
        "no arch features enabled — build with --features sm80",
    ))
}

// ============================================================================
// Grouped GEMM
// ============================================================================
//
// Architecture (per Fuel team's design review):
//   - `GroupedGemmPlan` holds host-only selection metadata (SKU, schedule
//     mode, epilogue kind). Cheap to clone, no device allocations.
//   - `prepare()` packs per-group host arrays (problem_sizes, ptr arrays,
//     ld arrays) and computes the threadblock count + scratch size for the
//     specific problem set. Returns a `PreparedGroupedGemm`.
//   - `PreparedGroupedGemm::workspace_size()` reports total bytes needed
//     (metadata layout + CUTLASS internal scratch, with alignment padding).
//   - `PreparedGroupedGemm::run(stream, workspace)` uploads the host
//     metadata to the start of the workspace via async H2D, computes
//     workspace pointer offsets, and launches the kernel.
//
// Workspace layout (caller-supplied; baracuda-cutlass owns this):
//
//     0                              metadata_end                  total
//     |  problem_sizes               |    pad   | CUTLASS scratch  |
//     |  ptr_a, ptr_b, ptr_c, ptr_d  |          |                  |
//     |  lda,   ldb,   ldc,   ldd    |          |                  |
//
// All v0 limitations:
//   - All groups must share the same `(alpha, beta)` epilogue params.
//   - All groups must consistently have `c = None` or `c = Some(_)`.
//   - Identity epilogue only (Bias deferred per Fuel team roadmap).

const COORD_BYTES: usize = 12; // [i32; 3]
const PTR_BYTES: usize = 8; // u64
const LD_BYTES: usize = 8; // i64
const SCRATCH_ALIGN: usize = 256; // CUTLASS internal scratch wants ≥128B; 256 is safe

#[inline]
fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Byte offsets for each metadata array within the caller-supplied workspace.
#[derive(Copy, Clone, Debug)]
struct MetadataLayout {
    problem_sizes_offset: usize,
    ptr_a_offset: usize,
    ptr_b_offset: usize,
    ptr_c_offset: usize,
    ptr_d_offset: usize,
    lda_offset: usize,
    ldb_offset: usize,
    ldc_offset: usize,
    ldd_offset: usize,
    /// First byte past the packed metadata, before scratch alignment.
    metadata_end: usize,
    /// Aligned start of CUTLASS internal scratch.
    scratch_offset: usize,
    /// Total workspace bytes needed.
    total_workspace_bytes: usize,
}

impl MetadataLayout {
    fn compute(group_count: usize, scratch_bytes: usize) -> Self {
        let mut off = 0usize;
        let problem_sizes_offset = off;
        off += COORD_BYTES * group_count;
        off = align_up(off, 8);

        let ptr_a_offset = off;
        off += PTR_BYTES * group_count;
        let ptr_b_offset = off;
        off += PTR_BYTES * group_count;
        let ptr_c_offset = off;
        off += PTR_BYTES * group_count;
        let ptr_d_offset = off;
        off += PTR_BYTES * group_count;
        let lda_offset = off;
        off += LD_BYTES * group_count;
        let ldb_offset = off;
        off += LD_BYTES * group_count;
        let ldc_offset = off;
        off += LD_BYTES * group_count;
        let ldd_offset = off;
        off += LD_BYTES * group_count;
        let metadata_end = off;

        let scratch_offset = align_up(metadata_end, SCRATCH_ALIGN);
        let total_workspace_bytes = scratch_offset + scratch_bytes;

        Self {
            problem_sizes_offset,
            ptr_a_offset,
            ptr_b_offset,
            ptr_c_offset,
            ptr_d_offset,
            lda_offset,
            ldb_offset,
            ldc_offset,
            ldd_offset,
            metadata_end,
            scratch_offset,
            total_workspace_bytes,
        }
    }
}

/// Plan for a grouped (per-problem variable shape) GEMM launch.
///
/// Cheap host-only struct — selection metadata + a cloned [`Context`]
/// handle so [`prepare`](Self::prepare) can allocate pinned host memory
/// without a stream argument. The cloned context is `Arc`-backed in
/// baracuda-driver, so cloning is cheap and the plan stays `Send + Sync`.
///
/// Use [`prepare`](Self::prepare) to bind a concrete slice of
/// [`GroupedProblem`]s to this plan and produce a [`PreparedGroupedGemm`]
/// that owns pinned host scratch and can launch capture-safely.
#[derive(Debug)]
pub struct GroupedGemmPlan<T: CutlassElement> {
    sku: GemmSku,
    schedule: GroupedScheduleMode,
    context: Context,
    _element: PhantomData<T>,
}

impl<T: CutlassElement> GroupedGemmPlan<T> {
    /// Pick a grouped-GEMM kernel for the given epilogue and preferences.
    ///
    /// v0 supports only [`EpilogueKind::Identity`]. Selection arch follows
    /// the same device-cap-aware logic as [`GemmPlan::select`].
    pub fn select(
        stream: &Stream,
        epilogue: EpilogueKind,
        pref: GroupedPlanPreference,
    ) -> Result<Self> {
        if epilogue != EpilogueKind::Identity {
            return Err(Error::Unsupported(
                "v0 grouped GEMM supports only EpilogueKind::Identity",
            ));
        }

        let dummy_desc = GemmDescriptor {
            m: 1,
            n: 1,
            k: 1,
            layout: LayoutSku::Rcr,
            epilogue,
        };
        let arch = pick_arch(stream, &dummy_desc, pref.base)?;
        let sku = GemmSku {
            arch,
            layout: LayoutSku::Rcr,
            epilogue,
            element: T::KIND,
            bias_element: None,
        };
        Ok(Self {
            sku,
            schedule: pref.schedule,
            context: stream.context().clone(),
            _element: PhantomData,
        })
    }

    /// Identity of the kernel this plan chose.
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees this plan's kernel provides.
    ///
    /// See [`GemmPlan::precision_guarantee`] for usage.
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Schedule mode this plan was selected with.
    pub fn schedule(&self) -> GroupedScheduleMode {
        self.schedule
    }

    /// Bind a concrete set of per-group problems to this plan.
    ///
    /// Performs host-side validation, queries CUTLASS for the threadblock
    /// count and scratch-bytes requirement, and packs the per-group host
    /// arrays. The returned [`PreparedGroupedGemm`] holds host-side
    /// metadata only — no device allocations, and crucially **no Rust
    /// borrow on the input `groups` slice**: device pointers are extracted
    /// into pinned memory during this call. The caller is free to drop
    /// `groups` immediately after; the underlying device buffers must be
    /// kept alive for as long as the prepared plan (or any captured graph
    /// referencing it) is in use.
    pub fn prepare<'a, 'g>(
        &'a self,
        groups: &'g [GroupedProblem<'g, T>],
    ) -> Result<PreparedGroupedGemm<'a, T>> {
        if groups.is_empty() {
            return Err(Error::InvalidProblem("grouped GEMM requires at least one group"));
        }

        // v0 invariants enforced here, before we touch CUTLASS:
        //   - All groups share the same (alpha, beta).
        //   - C presence (`Some` vs `None`) is consistent across groups.
        //   - Each group's shapes / strides are individually valid.
        let first_alpha = groups[0].alpha;
        let first_beta = groups[0].beta;
        let first_has_c = groups[0].c.is_some();
        for g in groups {
            if g.m <= 0 || g.n <= 0 || g.k <= 0 {
                return Err(Error::InvalidProblem("group M, N, K must all be positive"));
            }
            if g.a.rows != g.m || g.a.cols != g.k {
                return Err(Error::InvalidProblem("group A shape doesn't match (M, K)"));
            }
            if g.b.rows != g.k || g.b.cols != g.n {
                return Err(Error::InvalidProblem("group B shape doesn't match (K, N)"));
            }
            if g.d.rows != g.m || g.d.cols != g.n {
                return Err(Error::InvalidProblem("group D shape doesn't match (M, N)"));
            }
            if let Some(c) = &g.c {
                if c.rows != g.m || c.cols != g.n {
                    return Err(Error::InvalidProblem("group C shape doesn't match (M, N)"));
                }
            }
            if g.a.ld < g.k as i64 || g.b.ld < g.k as i64 || g.d.ld < g.n as i64 {
                return Err(Error::InvalidProblem("group leading dimension too small"));
            }
            if g.alpha != first_alpha {
                return Err(Error::Unsupported(
                    "v0 grouped GEMM requires all groups to share alpha",
                ));
            }
            if g.beta != first_beta {
                return Err(Error::Unsupported(
                    "v0 grouped GEMM requires all groups to share beta",
                ));
            }
            if g.c.is_some() != first_has_c {
                return Err(Error::Unsupported(
                    "v0 grouped GEMM requires all groups to consistently have c=None or c=Some",
                ));
            }
        }

        // Pack host problem_sizes as [m,n,k, m,n,k, ...] for the C ABI's
        // sufficient / scratch_bytes / can_implement queries.
        let group_count = groups.len();
        let mut h_m: Vec<i32> = Vec::with_capacity(group_count);
        let mut h_n: Vec<i32> = Vec::with_capacity(group_count);
        let mut h_k: Vec<i32> = Vec::with_capacity(group_count);
        for g in groups {
            h_m.push(g.m);
            h_n.push(g.n);
            h_k.push(g.k);
        }

        let kind = T::KIND;
        let group_count_i32 = group_count as i32;

        // CUTLASS-level can_implement (per-group alignment / shape).
        let ci_status = match self.sku.arch {
            #[cfg(feature = "sm80")]
            ArchSku::Sm80 => unsafe {
                dispatch::grouped_gemm_rcr_sm80_can_implement(
                    kind,
                    h_m.as_ptr(),
                    h_n.as_ptr(),
                    h_k.as_ptr(),
                    group_count_i32,
                )
            },
            #[cfg(not(feature = "sm80"))]
            ArchSku::Sm80 => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            ArchSku::Sm90a => {
                return Err(Error::Unsupported(
                    "sm90a grouped kernels not yet shipped (deferred until Hopper hardware available)",
                ));
            }
            ArchSku::Sm89 => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };
        status_to_result(ci_status)?;

        // Threadblock count + CUTLASS scratch bytes.
        let threadblock_count = match self.sku.arch {
            #[cfg(feature = "sm80")]
            ArchSku::Sm80 => unsafe {
                dispatch::grouped_gemm_rcr_sm80_sufficient(
                    kind,
                    h_m.as_ptr(),
                    h_n.as_ptr(),
                    h_k.as_ptr(),
                    group_count_i32,
                )
            },
            #[cfg(not(feature = "sm80"))]
            ArchSku::Sm80 => 0,
            ArchSku::Sm90a => 0,
            ArchSku::Sm89 => 0,
        };
        if threadblock_count <= 0 {
            return Err(Error::CutlassInternal(threadblock_count));
        }

        let scratch_bytes = match self.sku.arch {
            #[cfg(feature = "sm80")]
            ArchSku::Sm80 => unsafe {
                dispatch::grouped_gemm_rcr_sm80_scratch_bytes(
                    kind,
                    h_m.as_ptr(),
                    h_n.as_ptr(),
                    h_k.as_ptr(),
                    group_count_i32,
                    threadblock_count,
                )
            },
            #[cfg(not(feature = "sm80"))]
            ArchSku::Sm80 => 0,
            ArchSku::Sm90a => 0,
            ArchSku::Sm89 => 0,
        };

        let layout = MetadataLayout::compute(group_count, scratch_bytes);

        // Pack host metadata into a PINNED host buffer so the H2D in
        // `run()` is truly async (and therefore stream-capture-safe).
        // From pageable host memory, cuMemcpyHtoDAsync is implicitly
        // synchronizing and not capturable; pinned memory is required.
        let mut pinned: PinnedBuffer<u8> = PinnedBuffer::new(&self.context, layout.metadata_end)?;

        // Collect device pointers and leading dimensions before borrowing
        // `pinned` mutably — keeps the mutable-borrow scope tight so we
        // can move `pinned` into the returned struct below.
        let ptr_a: Vec<u64> = groups.iter().map(|g| g.a.data.as_raw().0).collect();
        let ptr_b: Vec<u64> = groups.iter().map(|g| g.b.data.as_raw().0).collect();
        let ptr_d: Vec<u64> = groups.iter().map(|g| g.d.data.as_raw().0).collect();
        // For c = None, point ptr_c at the group's D buffer (kernel reads
        // a valid pointer but multiplies by beta = 0; same trick as
        // single-GEMM null-C handling).
        let ptr_c: Vec<u64> = groups
            .iter()
            .map(|g| {
                g.c.as_ref()
                    .map(|c| c.data.as_raw().0)
                    .unwrap_or_else(|| g.d.data.as_raw().0)
            })
            .collect();
        let lda: Vec<i64> = groups.iter().map(|g| g.a.ld).collect();
        let ldb: Vec<i64> = groups.iter().map(|g| g.b.ld).collect();
        let ldd: Vec<i64> = groups.iter().map(|g| g.d.ld).collect();
        let ldc: Vec<i64> = groups
            .iter()
            .map(|g| g.c.as_ref().map(|c| c.ld).unwrap_or(g.d.ld))
            .collect();

        // Now write all metadata into the pinned slab. The borrow ends
        // when `host_packed` falls out of scope at the end of the block.
        {
            let host_packed: &mut [u8] = &mut pinned;

            let mut p = layout.problem_sizes_offset;
            for g in groups {
                host_packed[p..p + 4].copy_from_slice(&g.m.to_ne_bytes());
                host_packed[p + 4..p + 8].copy_from_slice(&g.n.to_ne_bytes());
                host_packed[p + 8..p + 12].copy_from_slice(&g.k.to_ne_bytes());
                p += COORD_BYTES;
            }

            let pack_ptrs = |dst: &mut [u8], offset: usize, ptrs: &[u64]| {
                let mut p = offset;
                for &val in ptrs {
                    dst[p..p + 8].copy_from_slice(&val.to_ne_bytes());
                    p += PTR_BYTES;
                }
            };
            pack_ptrs(host_packed, layout.ptr_a_offset, &ptr_a);
            pack_ptrs(host_packed, layout.ptr_b_offset, &ptr_b);
            pack_ptrs(host_packed, layout.ptr_c_offset, &ptr_c);
            pack_ptrs(host_packed, layout.ptr_d_offset, &ptr_d);

            let pack_lds = |dst: &mut [u8], offset: usize, lds: &[i64]| {
                let mut p = offset;
                for &val in lds {
                    dst[p..p + 8].copy_from_slice(&val.to_ne_bytes());
                    p += LD_BYTES;
                }
            };
            pack_lds(host_packed, layout.lda_offset, &lda);
            pack_lds(host_packed, layout.ldb_offset, &ldb);
            pack_lds(host_packed, layout.ldc_offset, &ldc);
            pack_lds(host_packed, layout.ldd_offset, &ldd);
        }

        // Host-side problem_sizes copy CUTLASS reads at run time (the
        // device copy from `pinned` is what the kernel actually
        // dereferences; this Vec just gives CUTLASS a stable host pointer
        // for its own internal tile-schedule math).
        let mut host_problem_sizes: Vec<i32> = Vec::with_capacity(group_count * 3);
        for g in groups {
            host_problem_sizes.push(g.m);
            host_problem_sizes.push(g.n);
            host_problem_sizes.push(g.k);
        }

        let beta_eff = if first_has_c { first_beta } else { <T::Scalar as Default>::default() };

        // Grouped GEMM v0 ships only Rcr × {F16, Bf16} (both f32-scalar);
        // the to_f32 conversion is identity. The select gate rejects
        // F64/F32Strict before we reach here.
        Ok(PreparedGroupedGemm {
            plan: self,
            pinned,
            host_problem_sizes,
            layout,
            threadblock_count,
            alpha: first_alpha.to_f32(),
            beta: beta_eff.to_f32(),
            _element: PhantomData,
        })
    }
}

/// A [`GroupedGemmPlan`] bound to a concrete set of per-group problems.
///
/// Owns a [`PinnedBuffer<u8>`] holding the packed metadata (problem
/// sizes, pointer arrays, leading dimensions). Pinned host memory is
/// what makes the H2D inside [`run`](Self::run) truly async — and
/// therefore safely capturable into a CUDA graph. Owns no device memory;
/// the caller supplies that via [`Workspace::Borrowed`] at run time.
///
/// # Lifetime contract
///
/// `PreparedGroupedGemm` extracts raw device pointers from the input
/// [`GroupedProblem`] slice during [`prepare`](GroupedGemmPlan::prepare)
/// and stores them in pinned memory — it does **not** hold a Rust borrow
/// on the input buffers afterwards. This is required for stream capture:
/// the captured graph references the pinned buffer (for the metadata
/// H2D) and the device buffers (via the pointer arrays) by raw address,
/// not by Rust lifetime. The caller must therefore keep both this
/// `PreparedGroupedGemm` and the underlying device buffers alive for as
/// long as any captured graph that references them is in use.
///
/// In practice the pattern is: build groups, call `prepare`, capture
/// into a graph, then keep `PreparedGroupedGemm` plus the input/output
/// device buffers alive for the lifetime of the captured graph.
#[derive(Debug)]
pub struct PreparedGroupedGemm<'a, T: CutlassElement> {
    plan: &'a GroupedGemmPlan<T>,
    /// Pinned host scratch holding all packed metadata. The H2D in `run`
    /// reads from this buffer; pinned memory means the copy is truly
    /// async + capturable on the user's stream.
    pinned: PinnedBuffer<u8>,
    host_problem_sizes: Vec<i32>,
    layout: MetadataLayout,
    threadblock_count: i32,
    alpha: f32,
    beta: f32,
    _element: PhantomData<T>,
}

impl<'a, T: CutlassElement> PreparedGroupedGemm<'a, T> {
    /// Total bytes of device workspace this plan needs at `run` time.
    ///
    /// Includes both the packed metadata layout and CUTLASS's internal
    /// scratch tail with alignment padding between them.
    pub fn workspace_size(&self) -> usize {
        self.layout.total_workspace_bytes
    }

    /// Identity of the kernel this plan chose (forwarded from the parent
    /// [`GroupedGemmPlan`]).
    pub fn sku(&self) -> GemmSku {
        self.plan.sku
    }

    /// Group count this plan was prepared for.
    pub fn group_count(&self) -> usize {
        self.host_problem_sizes.len() / 3
    }

    /// Launch the grouped GEMM.
    ///
    /// Uploads the packed metadata to the start of `workspace` via async
    /// H2D on `stream`, then enqueues the grouped kernel using the
    /// remainder of the workspace as CUTLASS internal scratch.
    pub fn run(&self, stream: &Stream, workspace: Workspace<'_>) -> Result<()> {
        let needed = self.workspace_size();
        let workspace_slice = match workspace {
            Workspace::None => {
                return Err(Error::WorkspaceTooSmall { needed, got: 0 });
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: slice.len(),
                    });
                }
                slice
            }
        };

        let workspace_base = workspace_slice.as_raw().0;

        // Single async H2D from pinned host memory into the workspace
        // prefix. Because the source is pinned (allocated in `prepare`),
        // this copy is truly async and capture-safe — wrapping the
        // surrounding launch in `cuStreamBeginCapture` / `cuStreamEndCapture`
        // produces a graph that replays correctly.
        {
            let mut workspace_for_meta = workspace_slice;
            let metadata_dst = workspace_for_meta.slice_mut(0..self.layout.metadata_end);
            metadata_dst.copy_from_host_async(&self.pinned, stream)?;
        }

        // Compute device pointers via base + offset arithmetic. The C
        // function dereferences these as pointer arrays.
        let off = |o: usize| (workspace_base + o as u64) as *const c_void;
        let off_mut = |o: usize| (workspace_base + o as u64) as *mut c_void;
        let d_problem_sizes = off(self.layout.problem_sizes_offset);
        let d_ptr_a = off(self.layout.ptr_a_offset);
        let d_ptr_b = off(self.layout.ptr_b_offset);
        let d_ptr_c = off(self.layout.ptr_c_offset);
        let d_ptr_d = off_mut(self.layout.ptr_d_offset);
        let d_lda = off(self.layout.lda_offset);
        let d_ldb = off(self.layout.ldb_offset);
        let d_ldc = off(self.layout.ldc_offset);
        let d_ldd = off(self.layout.ldd_offset);
        let scratch_ptr = off_mut(self.layout.scratch_offset);
        let scratch_bytes = self.layout.total_workspace_bytes - self.layout.scratch_offset;

        let h_problem_sizes = self.host_problem_sizes.as_ptr() as *const c_void;
        let stream_raw = stream.as_raw();
        let group_count = self.group_count() as i32;

        let status = match self.plan.sku.arch {
            #[cfg(feature = "sm80")]
            ArchSku::Sm80 => unsafe {
                dispatch::grouped_gemm_rcr_sm80_run(
                    T::KIND,
                    group_count,
                    self.threadblock_count,
                    d_problem_sizes,
                    d_ptr_a,
                    d_ptr_b,
                    d_ptr_c,
                    d_ptr_d,
                    d_lda,
                    d_ldb,
                    d_ldc,
                    d_ldd,
                    h_problem_sizes,
                    self.alpha,
                    self.beta,
                    scratch_ptr,
                    scratch_bytes,
                    stream_raw,
                )
            },
            #[cfg(not(feature = "sm80"))]
            ArchSku::Sm80 => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            ArchSku::Sm90a => {
                return Err(Error::Unsupported(
                    "sm90a grouped kernels not yet shipped",
                ));
            }
            ArchSku::Sm89 => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };

        status_to_result(status)
    }
}

// ============================================================================
// IntGemmPlan — int8 GEMM (Phase 2)
// ============================================================================
//
// Sibling to [`GemmPlan`] for the integer GEMM family. The split exists
// because the kernel-level dispatch, accumulator (int32), and epilogue
// templates (`LinearCombinationClamp` / `LinearCombinationBiasElementwise`
// with `ElementCompute = float`) differ enough from the float family
// that mixing them through a single generic plan would smear the
// argument types of every helper.
//
// API surface mirrors [`GemmPlan`]: `select`, `can_implement`,
// `workspace_size`, `run`, `sku`, `precision_guarantee`. Trait bounds:
// - `T: IntElement` constrains the matrix element to a kernel-supported
//   int dtype (today [`S8`](crate::S8) or [`U8`](crate::U8)).
// - `BT: BiasElement` constrains the bias element to a kernel-supported
//   bias dtype (today `f32` or `i32`). Defaults to `f32` since most
//   callers using a `Bias*` epilogue start there; `IntGemmPlan::<S8>`
//   shorthand picks the f32-bias path. Override explicitly with
//   `IntGemmPlan::<S8, i32>` for the int32-bias path.
//
// Layout coverage: today's int8 kernels are RCR-only. Selecting
// [`LayoutSku::Rrr`] returns [`Error::Unsupported`] at `select` time —
// CUTLASS 4.2.0 has no warp-level iterator for the 8-bit `Congruous`
// shared-memory layout the row-major B operand would need. A follow-up
// release will vendor the missing specialization.

/// Plan handle for an int8 GEMM kernel selection.
///
/// See module-level docs above and [`IntGemmDescriptor`] /
/// [`IntGemmArgs`] for the full API contract.
#[derive(Debug)]
pub struct IntGemmPlan<T: IntElement, BT: BiasElement = f32> {
    desc: IntGemmDescriptor,
    sku: GemmSku,
    _element: PhantomData<T>,
    _bias_element: PhantomData<BT>,
}

impl<T: IntElement, BT: BiasElement> IntGemmPlan<T, BT> {
    /// Pick an int-GEMM kernel for `desc`.
    ///
    /// Returns [`Error::Unsupported`] for `LayoutSku::Rrr` — see the
    /// module-level docs above for the CUTLASS upstream limitation.
    /// `BT` is only meaningful for the `Bias*` epilogue variants;
    /// `EpilogueKind::Identity` ignores it.
    pub fn select(
        stream: &Stream,
        desc: &IntGemmDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        check_int_descriptor(desc)?;
        let arch = pick_int_arch(stream, pref)?;
        // RCR-only on int8 today. The descriptor-level check rejects
        // any non-RCR layout up front so callers see the error at
        // plan-creation time rather than at launch.
        if !matches!(desc.layout, LayoutSku::Rcr) {
            return Err(Error::Unsupported(
                "int8 GEMM kernels are RCR-only in this release \
                 (CUTLASS 4.2.0 lacks 8-bit `TensorOpMultiplicandCongruous` \
                 warp iterators for RRR / row-major-B layout)",
            ));
        }
        // `bias_element` is meaningful only for the bias-family
        // epilogues; Identity int kernels carry `None` to match the
        // float-family convention.
        let bias_element = if desc.epilogue.requires_bias() {
            Some(BT::KIND)
        } else {
            None
        };
        let sku = GemmSku {
            arch,
            layout: desc.layout,
            epilogue: desc.epilogue,
            element: T::KIND,
            bias_element,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _element: PhantomData,
            _bias_element: PhantomData,
        })
    }

    /// Validate that this plan can launch with `args`.
    ///
    /// Two-stage check identical in shape to [`GemmPlan::can_implement`].
    pub fn can_implement(&self, args: &IntGemmArgs<'_, T, BT>) -> Result<()> {
        check_int_args(&self.desc, args)?;

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld),
            None => (core::ptr::null(), 0i64),
        };
        let bias_ptr = args
            .bias
            .as_ref()
            .map(|b| b.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());

        let bias_family = self.sku.epilogue.requires_bias();
        let status = match (self.sku.arch, bias_family) {
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) => unsafe {
                dispatch::int_gemm_rcr_sm80_can_implement(
                    self.sku.layout,
                    T::KIND,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) => unsafe {
                dispatch::int_gemm_bias_rcr_sm80_can_implement(
                    self.sku.layout,
                    T::KIND,
                    self.sku.epilogue,
                    BT::KIND,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    bias_ptr,
                )
            },
            #[cfg(not(feature = "sm80"))]
            (ArchSku::Sm80, _) => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            (ArchSku::Sm90a, _) => {
                return Err(Error::Unsupported(
                    "sm90a int8 kernels not yet shipped",
                ));
            }
            (ArchSku::Sm89, _) => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };

        status_to_result(status)
    }

    /// Bytes of device scratch this plan needs at `run` time.
    pub fn workspace_size(&self) -> usize {
        let bias_family = self.sku.epilogue.requires_bias();
        match (self.sku.arch, bias_family) {
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) => dispatch::int_gemm_rcr_sm80_workspace_size(
                self.sku.layout,
                T::KIND,
                self.desc.m, self.desc.n, self.desc.k,
            ),
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) => dispatch::int_gemm_bias_rcr_sm80_workspace_size(
                self.sku.layout,
                T::KIND,
                self.sku.epilogue,
                BT::KIND,
                self.desc.m, self.desc.n, self.desc.k,
            ),
            #[cfg(not(feature = "sm80"))]
            (ArchSku::Sm80, _) => 0,
            (ArchSku::Sm90a, _) => 0,
            (ArchSku::Sm89, _) => 0,
        }
    }

    /// Identity of the kernel this plan chose.
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees this plan's kernel provides.
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Launch the kernel.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: IntGemmArgs<'_, T, BT>,
    ) -> Result<()> {
        self.can_implement(&args)?;

        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                if needed != 0 {
                    return Err(Error::WorkspaceTooSmall { needed, got: 0 });
                }
                (core::ptr::null_mut(), 0)
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };

        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let d_ptr = args.d.data.as_raw().0 as *mut c_void;
        let (c_ptr, ldc) = match &args.c {
            Some(c) => (c.data.as_raw().0 as *const c_void, c.ld),
            None => (core::ptr::null(), 0i64),
        };
        let bias_ptr = args
            .bias
            .as_ref()
            .map(|b| b.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        // Match GemmPlan: when c = None, force beta = 0 at the safe layer
        // so the C-substituted-as-D fallback inside the kernel doesn't
        // fold the previous D contents into the result.
        let beta_eff: f32 = if args.c.is_some() { args.beta } else { 0.0 };
        let stream_raw = stream.as_raw();

        let bias_family = self.sku.epilogue.requires_bias();
        let status = match (self.sku.arch, bias_family) {
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, false) => unsafe {
                dispatch::int_gemm_rcr_sm80_run(
                    self.sku.layout,
                    T::KIND,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    args.alpha,
                    beta_eff,
                    ws_ptr, ws_bytes, stream_raw,
                )
            },
            #[cfg(feature = "sm80")]
            (ArchSku::Sm80, true) => unsafe {
                dispatch::int_gemm_bias_rcr_sm80_run(
                    self.sku.layout,
                    T::KIND,
                    self.sku.epilogue,
                    BT::KIND,
                    self.desc.m, self.desc.n, self.desc.k,
                    a_ptr, args.a.ld,
                    b_ptr, args.b.ld,
                    c_ptr, ldc,
                    d_ptr, args.d.ld,
                    bias_ptr,
                    args.alpha,
                    beta_eff,
                    ws_ptr, ws_bytes, stream_raw,
                )
            },
            #[cfg(not(feature = "sm80"))]
            (ArchSku::Sm80, _) => {
                return Err(Error::Unsupported(
                    "sm80 selected but the `sm80` feature isn't enabled",
                ));
            }
            (ArchSku::Sm90a, _) => {
                return Err(Error::Unsupported("sm90a int8 kernels not yet shipped"));
            }
            (ArchSku::Sm89, _) => {
                return Err(Error::Unsupported(
                    "Ada-specialized FP8 / sm_89 SKUs live in baracuda-kernels-sys, not baracuda-cutlass",
                ));
            }
        };

        status_to_result(status)
    }
}

// Internal helpers for int-GEMM validation. Structure mirrors
// `check_descriptor` / `check_args` for the float family.

fn check_int_descriptor(desc: &IntGemmDescriptor) -> Result<()> {
    if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
        return Err(Error::InvalidProblem("M, N, K must all be positive"));
    }
    Ok(())
}

fn check_int_args<T: IntElement, BT: BiasElement>(
    desc: &IntGemmDescriptor,
    args: &IntGemmArgs<'_, T, BT>,
) -> Result<()> {
    // Epilogue / bias must agree.
    match (desc.epilogue.requires_bias(), &args.bias) {
        (false, Some(_)) => {
            return Err(Error::InvalidProblem(
                "args.bias must be None when descriptor.epilogue is Identity",
            ));
        }
        (true, None) => {
            return Err(Error::InvalidProblem(
                "args.bias is required when descriptor.epilogue is in the Bias family \
                 (Bias / BiasRelu / BiasGelu / BiasSilu)",
            ));
        }
        (false, None) | (true, Some(_)) => {}
    }
    if let Some(bias) = &args.bias {
        if bias.len != desc.n {
            return Err(Error::InvalidProblem("bias vector length must equal N"));
        }
        if bias.stride != 1 {
            return Err(Error::Unsupported(
                "bias vector must be contiguous (stride 1) — strided bias not supported",
            ));
        }
        if bias.data.len() < desc.n as usize {
            return Err(Error::BufferTooSmall {
                needed: desc.n as usize,
                got: bias.data.len(),
            });
        }
    }
    if args.a.rows != desc.m || args.a.cols != desc.k {
        return Err(Error::InvalidProblem("A shape doesn't match descriptor (M, K)"));
    }
    if args.b.rows != desc.k || args.b.cols != desc.n {
        return Err(Error::InvalidProblem("B shape doesn't match descriptor (K, N)"));
    }
    if args.d.rows != desc.m || args.d.cols != desc.n {
        return Err(Error::InvalidProblem("D shape doesn't match descriptor (M, N)"));
    }
    if let Some(c) = &args.c {
        if c.rows != desc.m || c.cols != desc.n {
            return Err(Error::InvalidProblem("C shape doesn't match descriptor (M, N)"));
        }
    }
    if args.a.ld < desc.k as i64 {
        return Err(Error::InvalidProblem("A leading dimension must be >= K"));
    }
    let b_min_ld = match desc.layout {
        LayoutSku::Rcr => desc.k as i64,
        LayoutSku::Rrr => desc.n as i64,
    };
    if args.b.ld < b_min_ld {
        return Err(Error::InvalidProblem(match desc.layout {
            LayoutSku::Rcr => "B leading dimension must be >= K (column-major Rcr layout)",
            LayoutSku::Rrr => "B leading dimension must be >= N (row-major Rrr layout)",
        }));
    }
    if args.d.ld < desc.n as i64 {
        return Err(Error::InvalidProblem("D leading dimension must be >= N"));
    }
    if let Some(c) = &args.c {
        if c.ld < desc.n as i64 {
            return Err(Error::InvalidProblem("C leading dimension must be >= N"));
        }
    }
    let need_a = min_elements_row_major(args.a.rows, args.a.cols, args.a.ld)
        .ok_or(Error::InvalidProblem("A storage size overflow"))?;
    if args.a.data.len() < need_a {
        return Err(Error::BufferTooSmall {
            needed: need_a,
            got: args.a.data.len(),
        });
    }
    let need_b = match desc.layout {
        LayoutSku::Rcr => min_elements_col_major(args.b.rows, args.b.cols, args.b.ld),
        LayoutSku::Rrr => min_elements_row_major(args.b.rows, args.b.cols, args.b.ld),
    }
    .ok_or(Error::InvalidProblem("B storage size overflow"))?;
    if args.b.data.len() < need_b {
        return Err(Error::BufferTooSmall {
            needed: need_b,
            got: args.b.data.len(),
        });
    }
    let need_d = min_elements_row_major(args.d.rows, args.d.cols, args.d.ld)
        .ok_or(Error::InvalidProblem("D storage size overflow"))?;
    if args.d.data.len() < need_d {
        return Err(Error::BufferTooSmall {
            needed: need_d,
            got: args.d.data.len(),
        });
    }
    if let Some(c) = &args.c {
        let need_c = min_elements_row_major(c.rows, c.cols, c.ld)
            .ok_or(Error::InvalidProblem("C storage size overflow"))?;
        if c.data.len() < need_c {
            return Err(Error::BufferTooSmall {
                needed: need_c,
                got: c.data.len(),
            });
        }
    }
    Ok(())
}

fn pick_int_arch(stream: &Stream, pref: PlanPreference) -> Result<ArchSku> {
    // Int kernels currently only ship for sm_80. The selection logic is a
    // simpler shape of `pick_arch` — no sm_90a path because there are no
    // sm_90a int kernels.
    let (major, _minor) = stream.context().device().compute_capability()?;
    if pref.allow_sm90a && cfg!(feature = "sm90a") && major >= 9 {
        // Allowed but not yet shipped — fall through to sm_80.
    }
    if cfg!(feature = "sm80") {
        if major >= 8 {
            return Ok(ArchSku::Sm80);
        }
        return Err(Error::Unsupported(
            "device compute capability < 8.0; sm_80 int8 kernels won't run here",
        ));
    }
    Err(Error::Unsupported(
        "no arch features enabled — build with --features sm80",
    ))
}

#[cfg(test)]
mod buffer_size_tests {
    //! Regression tests for the per-Fuel-team-review buffer-size formulas.
    //!
    //! Pre-fix the helpers used `rows * ld` / `cols * ld`, which over-rejects
    //! valid padded slabs. The corrected formula is
    //! `(major - 1) * ld + minor`.

    use super::{min_elements_rcr_a, min_elements_rcr_b, min_elements_rcr_cd};

    #[test]
    fn rcr_a_tight_layout() {
        // [M=4, K=8] row-major, lda = K = 8.
        // Min elements = (4 - 1) * 8 + 8 = 32.
        assert_eq!(min_elements_rcr_a(4, 8, 8), Some(32));
    }

    #[test]
    fn rcr_a_padded_layout_accepts_smaller_count() {
        // [M=4, K=8] row-major with lda = 16 (padded).
        // Min elements = (4 - 1) * 16 + 8 = 56.
        // Pre-fix formula was rows*ld = 4*16 = 64 — over-strict by 8 elements.
        assert_eq!(min_elements_rcr_a(4, 8, 16), Some(56));
    }

    #[test]
    fn rcr_b_tight_layout() {
        // [K=8, N=4] column-major, ldb = K = 8.
        // Min elements = (4 - 1) * 8 + 8 = 32.
        assert_eq!(min_elements_rcr_b(8, 4, 8), Some(32));
    }

    #[test]
    fn rcr_b_padded_layout_accepts_smaller_count() {
        // [K=8, N=4] column-major, ldb = 16 (padded).
        // Min elements = (4 - 1) * 16 + 8 = 56. Pre-fix was 4*16 = 64.
        assert_eq!(min_elements_rcr_b(8, 4, 16), Some(56));
    }

    #[test]
    fn rcr_cd_tight_layout() {
        // [M=4, N=8] row-major, ld = N = 8.
        // Min elements = (4 - 1) * 8 + 8 = 32.
        assert_eq!(min_elements_rcr_cd(4, 8, 8), Some(32));
    }

    #[test]
    fn rcr_cd_padded_layout_accepts_smaller_count() {
        // [M=4, N=8] row-major, ld = 16 (padded).
        // Min elements = (4 - 1) * 16 + 8 = 56. Pre-fix was 4*16 = 64.
        assert_eq!(min_elements_rcr_cd(4, 8, 16), Some(56));
    }

    #[test]
    fn single_row_matrix_does_not_underflow() {
        // [M=1, K=8] should need exactly K elements regardless of ld.
        assert_eq!(min_elements_rcr_a(1, 8, 8), Some(8));
        assert_eq!(min_elements_rcr_a(1, 8, 256), Some(8));
    }

    #[test]
    fn overflow_returns_none() {
        // Force i64::checked_mul overflow.
        assert_eq!(min_elements_rcr_a(i32::MAX, 1, i64::MAX), None);
    }
}
