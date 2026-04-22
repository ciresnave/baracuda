//! cuBLASLt — the descriptor-based, heuristic-tuned GEMM API.
//!
//! Typical flow:
//!
//! ```no_run
//! use baracuda_cublas::lt::{LtHandle, MatmulDesc, MatrixLayout, MatmulPreference};
//! use baracuda_cublas_sys::functions::{
//!     cublasComputeType_t, cudaDataType_t, cublasOperation_t,
//! };
//!
//! # fn demo() -> baracuda_cublas::Result<()> {
//! let lt = LtHandle::new()?;
//! let desc = MatmulDesc::new(cublasComputeType_t::Compute32F, cudaDataType_t::R_32F)?;
//! desc.set_transa(cublasOperation_t::N)?;
//! desc.set_transb(cublasOperation_t::N)?;
//!
//! let a_layout = MatrixLayout::new(cudaDataType_t::R_32F, 128, 64, 128)?;
//! let b_layout = MatrixLayout::new(cudaDataType_t::R_32F, 64, 256, 64)?;
//! let c_layout = MatrixLayout::new(cudaDataType_t::R_32F, 128, 256, 128)?;
//!
//! let pref = MatmulPreference::new()?;
//! # let _ = (lt, desc, a_layout, b_layout, c_layout, pref);
//! # Ok(()) }
//! ```
//!
//! The handle type is distinct from the regular [`crate::Handle`] because
//! cuBLASLt lives in a separate shared library (`libcublasLt.so` /
//! `cublasLt64_*.dll`) with its own loader.

use core::ffi::c_void;

use baracuda_cublas_sys::functions::{
    cublasComputeType_t, cublasLtHandle_t, cublasLtMatmulAlgo_t, cublasLtMatmulDesc_t,
    cublasLtMatmulDescAttributes_t, cublasLtMatmulHeuristicResult_t, cublasLtMatmulPreference_t,
    cublasLtMatmulPreferenceAttributes_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayoutAttribute_t, cudaDataType_t,
};
use baracuda_cublas_sys::{cublasOperation_t, cublasStatus_t};
use baracuda_cublas_sys::loader::cublas_lt;
use baracuda_driver::Stream;

use crate::error::{check, Result};

/// Owned cuBLASLt handle.
#[derive(Debug)]
pub struct LtHandle {
    raw: cublasLtHandle_t,
}

impl LtHandle {
    pub fn new() -> Result<Self> {
        let lt = cublas_lt()?;
        let create = lt.cublas_lt_create()?;
        let mut h: cublasLtHandle_t = core::ptr::null_mut();
        check(unsafe { create(&mut h) })?;
        Ok(Self { raw: h })
    }

    pub fn as_raw(&self) -> cublasLtHandle_t {
        self.raw
    }
}

impl Drop for LtHandle {
    fn drop(&mut self) {
        if let Ok(lt) = cublas_lt() {
            if let Ok(d) = lt.cublas_lt_destroy() {
                let _ = unsafe { d(self.raw) };
            }
        }
    }
}

// Handles are thread-safe enough to move between threads but cuBLASLt
// documents per-handle usage from a single thread at a time.
unsafe impl Send for LtHandle {}

/// Matmul descriptor — owns the compute type + scalar type + trans flags.
#[derive(Debug)]
pub struct MatmulDesc {
    raw: cublasLtMatmulDesc_t,
}

impl MatmulDesc {
    pub fn new(compute: cublasComputeType_t, scale: cudaDataType_t) -> Result<Self> {
        let lt = cublas_lt()?;
        let create = lt.cublas_lt_matmul_desc_create()?;
        let mut d: cublasLtMatmulDesc_t = core::ptr::null_mut();
        check(unsafe { create(&mut d, compute, scale) })?;
        Ok(Self { raw: d })
    }

    fn set_attr<T>(&self, attr: cublasLtMatmulDescAttributes_t, value: &T) -> Result<()> {
        let lt = cublas_lt()?;
        let set = lt.cublas_lt_matmul_desc_set_attribute()?;
        check(unsafe {
            set(
                self.raw,
                attr,
                value as *const T as *const c_void,
                core::mem::size_of::<T>(),
            )
        })
    }

    pub fn set_transa(&self, op: cublasOperation_t) -> Result<()> {
        self.set_attr(cublasLtMatmulDescAttributes_t::Transa, &(op as i32))
    }

    pub fn set_transb(&self, op: cublasOperation_t) -> Result<()> {
        self.set_attr(cublasLtMatmulDescAttributes_t::Transb, &(op as i32))
    }

    pub fn set_epilogue(&self, epilogue: i32) -> Result<()> {
        self.set_attr(cublasLtMatmulDescAttributes_t::Epilogue, &epilogue)
    }

    pub fn set_bias_pointer(&self, ptr: *const c_void) -> Result<()> {
        self.set_attr(cublasLtMatmulDescAttributes_t::BiasPointer, &ptr)
    }

    pub fn as_raw(&self) -> cublasLtMatmulDesc_t {
        self.raw
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        if let Ok(lt) = cublas_lt() {
            if let Ok(d) = lt.cublas_lt_matmul_desc_destroy() {
                let _ = unsafe { d(self.raw) };
            }
        }
    }
}

/// Matrix layout — element type, shape, and leading-dim.
#[derive(Debug)]
pub struct MatrixLayout {
    raw: cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    pub fn new(element_type: cudaDataType_t, rows: u64, cols: u64, ld: i64) -> Result<Self> {
        let lt = cublas_lt()?;
        let create = lt.cublas_lt_matrix_layout_create()?;
        let mut l: cublasLtMatrixLayout_t = core::ptr::null_mut();
        check(unsafe { create(&mut l, element_type, rows, cols, ld) })?;
        Ok(Self { raw: l })
    }

    fn set_attr<T>(&self, attr: cublasLtMatrixLayoutAttribute_t, value: &T) -> Result<()> {
        let lt = cublas_lt()?;
        let set = lt.cublas_lt_matrix_layout_set_attribute()?;
        check(unsafe {
            set(
                self.raw,
                attr,
                value as *const T as *const c_void,
                core::mem::size_of::<T>(),
            )
        })
    }

    pub fn set_batch_count(&self, n: i32) -> Result<()> {
        self.set_attr(cublasLtMatrixLayoutAttribute_t::BatchCount, &n)
    }

    pub fn set_strided_batch_offset(&self, stride: i64) -> Result<()> {
        self.set_attr(cublasLtMatrixLayoutAttribute_t::StridedBatchOffset, &stride)
    }

    pub fn as_raw(&self) -> cublasLtMatrixLayout_t {
        self.raw
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        if let Ok(lt) = cublas_lt() {
            if let Ok(d) = lt.cublas_lt_matrix_layout_destroy() {
                let _ = unsafe { d(self.raw) };
            }
        }
    }
}

/// Algorithm-preference container.
#[derive(Debug)]
pub struct MatmulPreference {
    raw: cublasLtMatmulPreference_t,
}

impl MatmulPreference {
    pub fn new() -> Result<Self> {
        let lt = cublas_lt()?;
        let create = lt.cublas_lt_matmul_preference_create()?;
        let mut p: cublasLtMatmulPreference_t = core::ptr::null_mut();
        check(unsafe { create(&mut p) })?;
        Ok(Self { raw: p })
    }

    pub fn set_max_workspace_bytes(&self, bytes: usize) -> Result<()> {
        let lt = cublas_lt()?;
        let set = lt.cublas_lt_matmul_preference_set_attribute()?;
        let v = bytes as u64;
        check(unsafe {
            set(
                self.raw,
                cublasLtMatmulPreferenceAttributes_t::MaxWorkspaceBytes,
                &v as *const _ as *const c_void,
                core::mem::size_of::<u64>(),
            )
        })
    }

    pub fn as_raw(&self) -> cublasLtMatmulPreference_t {
        self.raw
    }
}

impl Drop for MatmulPreference {
    fn drop(&mut self) {
        if let Ok(lt) = cublas_lt() {
            if let Ok(d) = lt.cublas_lt_matmul_preference_destroy() {
                let _ = unsafe { d(self.raw) };
            }
        }
    }
}

/// A single algorithm candidate returned by the heuristic search.
#[derive(Copy, Clone, Debug)]
pub struct MatmulHeuristic {
    pub algo: cublasLtMatmulAlgo_t,
    pub workspace_size: usize,
    pub waves_count: f32,
}

/// Ask cuBLASLt for up to `capacity` matching algorithms. Returns only the
/// algorithms whose status is Success.
#[allow(clippy::too_many_arguments)]
pub fn heuristics_search(
    handle: &LtHandle,
    desc: &MatmulDesc,
    a: &MatrixLayout,
    b: &MatrixLayout,
    c: &MatrixLayout,
    d: &MatrixLayout,
    pref: &MatmulPreference,
    capacity: i32,
) -> Result<Vec<MatmulHeuristic>> {
    let lt = cublas_lt()?;
    let heur = lt.cublas_lt_matmul_algo_get_heuristic()?;
    let mut out: Vec<cublasLtMatmulHeuristicResult_t> = vec![
        cublasLtMatmulHeuristicResult_t {
            algo: cublasLtMatmulAlgo_t {
                data: [0u64; 8],
            },
            workspace_size: 0,
            state: cublasStatus_t::SUCCESS,
            waves_count: 0.0,
            reserved: [0; 4],
        };
        capacity as usize
    ];
    let mut n_returned: i32 = 0;
    check(unsafe {
        heur(
            handle.as_raw(),
            desc.as_raw(),
            a.as_raw(),
            b.as_raw(),
            c.as_raw(),
            d.as_raw(),
            pref.as_raw(),
            capacity,
            out.as_mut_ptr(),
            &mut n_returned,
        )
    })?;
    let mut results = Vec::with_capacity(n_returned as usize);
    for r in out.iter().take(n_returned as usize) {
        if r.state.is_success() {
            results.push(MatmulHeuristic {
                algo: r.algo,
                workspace_size: r.workspace_size,
                waves_count: r.waves_count,
            });
        }
    }
    Ok(results)
}

/// Execute the matmul. Pointers are raw to stay type-agnostic over all
/// datatypes cuBLASLt supports (fp8, fp4, int4, etc.). A workspace may be
/// required depending on the chosen algorithm.
///
/// # Safety
/// All pointers must be device pointers of the correct size/type; the
/// descriptor/layouts must match the actual buffers. The stream must be
/// valid; using the cuBLASLt default (null) stream is allowed.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul(
    handle: &LtHandle,
    desc: &MatmulDesc,
    alpha: *const c_void,
    a: *const c_void,
    a_layout: &MatrixLayout,
    b: *const c_void,
    b_layout: &MatrixLayout,
    beta: *const c_void,
    c: *const c_void,
    c_layout: &MatrixLayout,
    d: *mut c_void,
    d_layout: &MatrixLayout,
    algo: Option<&cublasLtMatmulAlgo_t>,
    workspace: *mut c_void,
    workspace_size: usize,
    stream: Option<&Stream>,
) -> Result<()> {
    let lt = cublas_lt()?;
    let f = lt.cublas_lt_matmul()?;
    let stream_raw = stream.map(|s| s.as_raw() as _).unwrap_or(core::ptr::null_mut());
    let algo_ptr = algo
        .map(|a| a as *const _)
        .unwrap_or(core::ptr::null());
    check(f(
        handle.as_raw(),
        desc.as_raw(),
        alpha,
        a,
        a_layout.as_raw(),
        b,
        b_layout.as_raw(),
        beta,
        c,
        c_layout.as_raw(),
        d,
        d_layout.as_raw(),
        algo_ptr,
        workspace,
        workspace_size,
        stream_raw,
    ))
}
