//! Dynamic loader for cuBLAS.

#![allow(non_camel_case_types)]

use std::sync::OnceLock;

use baracuda_core::{platform, Library, LoaderError};

use super::functions::*;

/// OS-specific cuBLAS library candidates, in preference order.
fn cublas_candidates() -> Vec<String> {
    platform::versioned_library_candidates("cublas", &["13", "12", "11"])
}

macro_rules! cublas_fns {
    ($($(#[$attr:meta])* fn $name:ident as $sym:literal : $pfn:ty;)*) => {
        pub struct Cublas {
            lib: Library,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Cublas {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cublas").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl Cublas {
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }

            $(
                $(#[$attr])*
                #[doc = concat!("Resolve `", $sym, "`.")]
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

cublas_fns! {
    fn cublas_create as "cublasCreate_v2": PFN_cublasCreate;
    fn cublas_destroy as "cublasDestroy_v2": PFN_cublasDestroy;
    fn cublas_get_version as "cublasGetVersion_v2": PFN_cublasGetVersion;
    fn cublas_set_stream as "cublasSetStream_v2": PFN_cublasSetStream;
    fn cublas_get_stream as "cublasGetStream_v2": PFN_cublasGetStream;
    fn cublas_set_pointer_mode as "cublasSetPointerMode_v2": PFN_cublasSetPointerMode;
    fn cublas_set_math_mode as "cublasSetMathMode": PFN_cublasSetMathMode;
    fn cublas_sgemm as "cublasSgemm_v2": PFN_cublasSgemm;
    fn cublas_dgemm as "cublasDgemm_v2": PFN_cublasDgemm;
    fn cublas_sgemm_strided_batched as "cublasSgemmStridedBatched": PFN_cublasSgemmStridedBatched;
    fn cublas_dgemm_strided_batched as "cublasDgemmStridedBatched": PFN_cublasDgemmStridedBatched;
    fn cublas_saxpy as "cublasSaxpy_v2": PFN_cublasSaxpy;
    fn cublas_daxpy as "cublasDaxpy_v2": PFN_cublasDaxpy;
    fn cublas_sdot as "cublasSdot_v2": PFN_cublasSdot;
    fn cublas_ddot as "cublasDdot_v2": PFN_cublasDdot;

    // L1 scalar ops
    fn cublas_sscal as "cublasSscal_v2": PFN_cublasSscal;
    fn cublas_dscal as "cublasDscal_v2": PFN_cublasDscal;
    fn cublas_snrm2 as "cublasSnrm2_v2": PFN_cublasSnrm2;
    fn cublas_dnrm2 as "cublasDnrm2_v2": PFN_cublasDnrm2;
    fn cublas_sasum as "cublasSasum_v2": PFN_cublasSasum;
    fn cublas_dasum as "cublasDasum_v2": PFN_cublasDasum;
    fn cublas_isamax as "cublasIsamax_v2": PFN_cublasIsamax;
    fn cublas_idamax as "cublasIdamax_v2": PFN_cublasIdamax;
    fn cublas_isamin as "cublasIsamin_v2": PFN_cublasIsamin;
    fn cublas_idamin as "cublasIdamin_v2": PFN_cublasIdamin;
    fn cublas_scopy as "cublasScopy_v2": PFN_cublasScopy;
    fn cublas_dcopy as "cublasDcopy_v2": PFN_cublasDcopy;

    // L2 GEMV
    fn cublas_sgemv as "cublasSgemv_v2": PFN_cublasSgemv;
    fn cublas_dgemv as "cublasDgemv_v2": PFN_cublasDgemv;
    fn cublas_cgemv as "cublasCgemv_v2": PFN_cublasCgemv;
    fn cublas_zgemv as "cublasZgemv_v2": PFN_cublasZgemv;

    // L1 complex
    fn cublas_caxpy as "cublasCaxpy_v2": PFN_cublasCaxpy;
    fn cublas_zaxpy as "cublasZaxpy_v2": PFN_cublasZaxpy;
    fn cublas_cscal as "cublasCscal_v2": PFN_cublasCscal;
    fn cublas_zscal as "cublasZscal_v2": PFN_cublasZscal;
    fn cublas_csscal as "cublasCsscal_v2": PFN_cublasCsscal;
    fn cublas_zdscal as "cublasZdscal_v2": PFN_cublasZdscal;
    fn cublas_scnrm2 as "cublasScnrm2_v2": PFN_cublasScnrm2;
    fn cublas_dznrm2 as "cublasDznrm2_v2": PFN_cublasDznrm2;
    fn cublas_scasum as "cublasScasum_v2": PFN_cublasScasum;
    fn cublas_dzasum as "cublasDzasum_v2": PFN_cublasDzasum;
    fn cublas_cdotu as "cublasCdotu_v2": PFN_cublasCdotu;
    fn cublas_cdotc as "cublasCdotc_v2": PFN_cublasCdotc;
    fn cublas_zdotu as "cublasZdotu_v2": PFN_cublasZdotu;
    fn cublas_zdotc as "cublasZdotc_v2": PFN_cublasZdotc;
    fn cublas_ccopy as "cublasCcopy_v2": PFN_cublasCcopy;
    fn cublas_zcopy as "cublasZcopy_v2": PFN_cublasZcopy;
    fn cublas_sswap as "cublasSswap_v2": PFN_cublasSswap;
    fn cublas_dswap as "cublasDswap_v2": PFN_cublasDswap;
    fn cublas_cswap as "cublasCswap_v2": PFN_cublasCswap;
    fn cublas_zswap as "cublasZswap_v2": PFN_cublasZswap;
    fn cublas_icamax as "cublasIcamax_v2": PFN_cublasIcamax;
    fn cublas_izamax as "cublasIzamax_v2": PFN_cublasIzamax;
    fn cublas_icamin as "cublasIcamin_v2": PFN_cublasIcamin;
    fn cublas_izamin as "cublasIzamin_v2": PFN_cublasIzamin;
    fn cublas_srot as "cublasSrot_v2": PFN_cublasSrot;
    fn cublas_drot as "cublasDrot_v2": PFN_cublasDrot;

    // L2 symmetric / triangular / rank-1
    fn cublas_ssymv as "cublasSsymv_v2": PFN_cublasSsymv;
    fn cublas_dsymv as "cublasDsymv_v2": PFN_cublasDsymv;
    fn cublas_strmv as "cublasStrmv_v2": PFN_cublasStrmv;
    fn cublas_dtrmv as "cublasDtrmv_v2": PFN_cublasDtrmv;
    fn cublas_strsv as "cublasStrsv_v2": PFN_cublasStrsv;
    fn cublas_dtrsv as "cublasDtrsv_v2": PFN_cublasDtrsv;
    fn cublas_sger as "cublasSger_v2": PFN_cublasSger;
    fn cublas_dger as "cublasDger_v2": PFN_cublasDger;
    fn cublas_ssyr as "cublasSsyr_v2": PFN_cublasSsyr;
    fn cublas_dsyr as "cublasDsyr_v2": PFN_cublasDsyr;

    // L3 — complex GEMM + Ex
    fn cublas_cgemm as "cublasCgemm_v2": PFN_cublasCgemm;
    fn cublas_zgemm as "cublasZgemm_v2": PFN_cublasZgemm;
    fn cublas_gemm_ex as "cublasGemmEx": PFN_cublasGemmEx;
    fn cublas_gemm_strided_batched_ex as "cublasGemmStridedBatchedEx":
        PFN_cublasGemmStridedBatchedEx;

    // L3 SYMM/HEMM/SYRK/HERK/TRSM
    fn cublas_ssymm as "cublasSsymm_v2": PFN_cublasSsymm;
    fn cublas_dsymm as "cublasDsymm_v2": PFN_cublasDsymm;
    fn cublas_csymm as "cublasCsymm_v2": PFN_cublasCsymm;
    fn cublas_zsymm as "cublasZsymm_v2": PFN_cublasZsymm;
    fn cublas_chemm as "cublasChemm_v2": PFN_cublasChemm;
    fn cublas_zhemm as "cublasZhemm_v2": PFN_cublasZhemm;
    fn cublas_ssyrk as "cublasSsyrk_v2": PFN_cublasSsyrk;
    fn cublas_dsyrk as "cublasDsyrk_v2": PFN_cublasDsyrk;
    fn cublas_csyrk as "cublasCsyrk_v2": PFN_cublasCsyrk;
    fn cublas_zsyrk as "cublasZsyrk_v2": PFN_cublasZsyrk;
    fn cublas_cherk as "cublasCherk_v2": PFN_cublasCherk;
    fn cublas_zherk as "cublasZherk_v2": PFN_cublasZherk;
    fn cublas_strsm as "cublasStrsm_v2": PFN_cublasStrsm;
    fn cublas_dtrsm as "cublasDtrsm_v2": PFN_cublasDtrsm;
    fn cublas_ctrsm as "cublasCtrsm_v2": PFN_cublasCtrsm;
    fn cublas_ztrsm as "cublasZtrsm_v2": PFN_cublasZtrsm;
    fn cublas_strmm as "cublasStrmm_v2": PFN_cublasStrmm;
    fn cublas_dtrmm as "cublasDtrmm_v2": PFN_cublasDtrmm;
    fn cublas_ctrmm as "cublasCtrmm_v2": PFN_cublasCtrmm;
    fn cublas_ztrmm as "cublasZtrmm_v2": PFN_cublasZtrmm;

    // Batched
    fn cublas_sgemm_batched as "cublasSgemmBatched": PFN_cublasSgemmBatched;
    fn cublas_dgemm_batched as "cublasDgemmBatched": PFN_cublasDgemmBatched;
    fn cublas_cgemm_batched as "cublasCgemmBatched": PFN_cublasCgemmBatched;
    fn cublas_zgemm_batched as "cublasZgemmBatched": PFN_cublasZgemmBatched;
    fn cublas_cgemm_strided_batched as "cublasCgemmStridedBatched":
        PFN_cublasCgemmStridedBatched;
    fn cublas_zgemm_strided_batched as "cublasZgemmStridedBatched":
        PFN_cublasZgemmStridedBatched;

    // Mixed-precision Ex variants for BLAS-1
    fn cublas_axpy_ex as "cublasAxpyEx": PFN_cublasAxpyEx;
    fn cublas_dot_ex as "cublasDotEx": PFN_cublasDotEx;
    fn cublas_dotc_ex as "cublasDotcEx": PFN_cublasDotcEx;
    fn cublas_nrm2_ex as "cublasNrm2Ex": PFN_cublasNrm2Ex;
    fn cublas_scal_ex as "cublasScalEx": PFN_cublasScalEx;
    fn cublas_rot_ex as "cublasRotEx": PFN_cublasRotEx;

    // Batched direct solvers
    fn cublas_sgetrf_batched as "cublasSgetrfBatched": PFN_cublasSgetrfBatched;
    fn cublas_dgetrf_batched as "cublasDgetrfBatched": PFN_cublasDgetrfBatched;
    fn cublas_cgetrf_batched as "cublasCgetrfBatched": PFN_cublasCgetrfBatched;
    fn cublas_zgetrf_batched as "cublasZgetrfBatched": PFN_cublasZgetrfBatched;
    fn cublas_sgetri_batched as "cublasSgetriBatched": PFN_cublasSgetriBatched;
    fn cublas_dgetri_batched as "cublasDgetriBatched": PFN_cublasDgetriBatched;
    fn cublas_cgetri_batched as "cublasCgetriBatched": PFN_cublasCgetriBatched;
    fn cublas_zgetri_batched as "cublasZgetriBatched": PFN_cublasZgetriBatched;
    fn cublas_sgetrs_batched as "cublasSgetrsBatched": PFN_cublasSgetrsBatched;
    fn cublas_dgetrs_batched as "cublasDgetrsBatched": PFN_cublasDgetrsBatched;
    fn cublas_cgetrs_batched as "cublasCgetrsBatched": PFN_cublasCgetrsBatched;
    fn cublas_zgetrs_batched as "cublasZgetrsBatched": PFN_cublasZgetrsBatched;
    fn cublas_smatinv_batched as "cublasSmatinvBatched": PFN_cublasSmatinvBatched;
    fn cublas_dmatinv_batched as "cublasDmatinvBatched": PFN_cublasDmatinvBatched;
    fn cublas_cmatinv_batched as "cublasCmatinvBatched": PFN_cublasCmatinvBatched;
    fn cublas_zmatinv_batched as "cublasZmatinvBatched": PFN_cublasZmatinvBatched;

    // cuBLASXt (same shared library as cuBLAS)
    fn cublas_xt_create as "cublasXtCreate": PFN_cublasXtCreate;
    fn cublas_xt_destroy as "cublasXtDestroy": PFN_cublasXtDestroy;
    fn cublas_xt_device_select as "cublasXtDeviceSelect": PFN_cublasXtDeviceSelect;
    fn cublas_xt_set_block_dim as "cublasXtSetBlockDim": PFN_cublasXtSetBlockDim;
    fn cublas_xt_get_block_dim as "cublasXtGetBlockDim": PFN_cublasXtGetBlockDim;
    fn cublas_xt_sgemm as "cublasXtSgemm": PFN_cublasXtSgemm;
    fn cublas_xt_dgemm as "cublasXtDgemm": PFN_cublasXtDgemm;
    fn cublas_xt_cgemm as "cublasXtCgemm": PFN_cublasXtCgemm;
    fn cublas_xt_zgemm as "cublasXtZgemm": PFN_cublasXtZgemm;
}

/// Lazily-initialized process-wide cuBLAS singleton.
pub fn cublas() -> Result<&'static Cublas, LoaderError> {
    static CUBLAS: OnceLock<Cublas> = OnceLock::new();
    if let Some(c) = CUBLAS.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = cublas_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("cublas", candidates_leaked)?;
    let c = Cublas::empty(lib);
    let _ = CUBLAS.set(c);
    Ok(CUBLAS.get().expect("OnceLock set or lost race"))
}

// ===================================================================
// cuBLASLt (separate shared library)
// ===================================================================

fn cublaslt_candidates() -> Vec<String> {
    platform::versioned_library_candidates("cublasLt", &["13", "12", "11"])
}

macro_rules! cublaslt_fns {
    ($($(#[$attr:meta])* fn $name:ident as $sym:literal : $pfn:ty;)*) => {
        pub struct CublasLt {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for CublasLt {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("CublasLt").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl CublasLt {
            fn empty(lib: Library) -> Self { Self { lib, $($name: OnceLock::new(),)* } }
            $(
                $(#[$attr])*
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

cublaslt_fns! {
    fn cublas_lt_create as "cublasLtCreate": PFN_cublasLtCreate;
    fn cublas_lt_destroy as "cublasLtDestroy": PFN_cublasLtDestroy;
    fn cublas_lt_matmul_desc_create as "cublasLtMatmulDescCreate":
        PFN_cublasLtMatmulDescCreate;
    fn cublas_lt_matmul_desc_destroy as "cublasLtMatmulDescDestroy":
        PFN_cublasLtMatmulDescDestroy;
    fn cublas_lt_matmul_desc_set_attribute as "cublasLtMatmulDescSetAttribute":
        PFN_cublasLtMatmulDescSetAttribute;
    fn cublas_lt_matmul_desc_get_attribute as "cublasLtMatmulDescGetAttribute":
        PFN_cublasLtMatmulDescGetAttribute;
    fn cublas_lt_matrix_layout_create as "cublasLtMatrixLayoutCreate":
        PFN_cublasLtMatrixLayoutCreate;
    fn cublas_lt_matrix_layout_destroy as "cublasLtMatrixLayoutDestroy":
        PFN_cublasLtMatrixLayoutDestroy;
    fn cublas_lt_matrix_layout_set_attribute as "cublasLtMatrixLayoutSetAttribute":
        PFN_cublasLtMatrixLayoutSetAttribute;
    fn cublas_lt_matmul_preference_create as "cublasLtMatmulPreferenceCreate":
        PFN_cublasLtMatmulPreferenceCreate;
    fn cublas_lt_matmul_preference_destroy as "cublasLtMatmulPreferenceDestroy":
        PFN_cublasLtMatmulPreferenceDestroy;
    fn cublas_lt_matmul_preference_set_attribute as "cublasLtMatmulPreferenceSetAttribute":
        PFN_cublasLtMatmulPreferenceSetAttribute;
    fn cublas_lt_matmul_algo_get_heuristic as "cublasLtMatmulAlgoGetHeuristic":
        PFN_cublasLtMatmulAlgoGetHeuristic;
    fn cublas_lt_matmul as "cublasLtMatmul": PFN_cublasLtMatmul;
    fn cublas_lt_get_version as "cublasLtGetVersion": PFN_cublasLtGetVersion;
    fn cublas_lt_get_cudart_version as "cublasLtGetCudartVersion":
        PFN_cublasLtGetCudartVersion;
}

pub fn cublas_lt() -> Result<&'static CublasLt, LoaderError> {
    static LT: OnceLock<CublasLt> = OnceLock::new();
    if let Some(c) = LT.get() {
        return Ok(c);
    }
    let candidates: Vec<&'static str> = cublaslt_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());
    let lib = Library::open("cublasLt", leaked)?;
    let _ = LT.set(CublasLt::empty(lib));
    Ok(LT.get().expect("OnceLock set or lost race"))
}
