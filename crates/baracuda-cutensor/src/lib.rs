//! Safe Rust wrappers for NVIDIA cuTENSOR (v2 API).
//!
//! cuTENSOR is NVIDIA's high-performance tensor-primitive library —
//! einsum-style contractions, element-wise ops, reductions, and
//! permutations. This crate wraps the full v2 host API surface.
//!
//! # Concepts
//!
//! - [`Handle`] — per-process library handle; owns the plan cache.
//! - [`TensorDescriptor`] — shape + strides + dtype for one tensor.
//! - [`OperationDescriptor`] — an *un*-compiled op (contraction,
//!   reduction, elementwise binary/trinary, permutation). Created via
//!   [`Contraction::new`], [`Reduction::new`], [`ElementwiseBinary::new`],
//!   [`ElementwiseTrinary::new`], or [`Permutation::new`].
//! - [`PlanPreference`] — algorithm selection + JIT mode.
//! - [`Plan`] — compiled op, bound to a workspace size.
//! - [`Plan::contract`] / [`Plan::reduce`] / etc. — execute the plan.
//!
//! # Example — `D = α * A ⊗ B + β * C` (matmul via contraction)
//!
//! ```no_run
//! use baracuda_cutensor::*;
//! let handle = Handle::new()?;
//! let m = 64i64; let n = 64i64; let k = 32i64;
//! let a = TensorDescriptor::new(&handle, &[m, k], None, DataType::F32, 128)?;
//! let b = TensorDescriptor::new(&handle, &[k, n], None, DataType::F32, 128)?;
//! let c = TensorDescriptor::new(&handle, &[m, n], None, DataType::F32, 128)?;
//! let modes_a = &[0i32, 2]; // [m, k]
//! let modes_b = &[2, 1];     // [k, n]
//! let modes_c = &[0, 1];     // [m, n]
//! let op = unsafe {
//!     Contraction::new(&handle, &a, modes_a, &b, modes_b, &c, modes_c, &c, modes_c,
//!         core::ptr::null())
//! }?;
//! let pref = PlanPreference::default_for(&handle)?;
//! let ws = op.estimate_workspace(&pref, WorkspaceKind::Default)?;
//! let plan = Plan::new(&op, &pref, ws)?;
//! # Result::<(), Error>::Ok(())
//! ```

#![warn(missing_debug_implementations)]

use core::ffi::c_void;
use std::ffi::CString;

use baracuda_cutensor_sys::{
    cutensor, cutensorAlgo, cutensorDataType, cutensorHandle_t, cutensorJitMode,
    cutensorOperationDescriptor_t, cutensorOperator, cutensorPlanPreference_t, cutensorPlan_t,
    cutensorStatus_t, cutensorTensorDescriptor_t, cutensorWorksizePreference,
};

/// Error type for cuTENSOR operations.
pub type Error = baracuda_core::Error<cutensorStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: cutensorStatus_t) -> Result<()> {
    Error::check(status)
}

/// Verify cuTENSOR is loadable on this host.
pub fn probe() -> Result<()> {
    cutensor()?;
    Ok(())
}

/// Encoded integer version from `cutensorGetVersion`. Decode as
/// `major = v / 10000, minor = (v / 100) % 100, patch = v % 100`.
pub fn version() -> Result<usize> {
    let c = cutensor()?;
    let cu = c.cutensor_get_version()?;
    Ok(unsafe { cu() })
}

/// cuTENSOR's view of the CUDART version it was built against.
pub fn cudart_version() -> Result<usize> {
    let c = cutensor()?;
    let cu = c.cutensor_get_cudart_version()?;
    Ok(unsafe { cu() })
}

/// Set the cuTENSOR logger verbosity (0 = off, 1 = error, 2 = trace).
pub fn set_log_level(level: i32) -> Result<()> {
    let c = cutensor()?;
    let cu = c.cutensor_logger_set_level()?;
    check(unsafe { cu(level) })
}

/// Bitmask of log categories (API calls, hints, traces, …). Full value
/// list in cuTENSOR headers.
pub fn set_log_mask(mask: i32) -> Result<()> {
    let c = cutensor()?;
    let cu = c.cutensor_logger_set_mask()?;
    check(unsafe { cu(mask) })
}

/// Open a log file path for cuTENSOR output.
pub fn open_log_file(path: &str) -> Result<()> {
    let cpath = std::ffi::CString::new(path).map_err(|_| Error::Status {
        status: cutensorStatus_t::INVALID_VALUE,
    })?;
    let c = cutensor()?;
    let cu = c.cutensor_logger_open_file()?;
    check(unsafe { cu(cpath.as_ptr()) })
}

/// Force-disable all cuTENSOR logging (tightest possible quiet).
pub fn force_disable_logging() -> Result<()> {
    let c = cutensor()?;
    let cu = c.cutensor_logger_force_disable()?;
    check(unsafe { cu() })
}

/// Element dtype for tensor descriptors.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DataType {
    F16,
    BF16,
    F32,
    F64,
    ComplexF32,
    ComplexF64,
    I8,
    U8,
    I32,
    U32,
}

impl DataType {
    #[inline]
    fn raw(self) -> i32 {
        match self {
            DataType::F16 => cutensorDataType::R_16F,
            DataType::BF16 => cutensorDataType::R_16BF,
            DataType::F32 => cutensorDataType::R_32F,
            DataType::F64 => cutensorDataType::R_64F,
            DataType::ComplexF32 => cutensorDataType::C_32F,
            DataType::ComplexF64 => cutensorDataType::C_64F,
            DataType::I8 => cutensorDataType::R_8I,
            DataType::U8 => cutensorDataType::R_8U,
            DataType::I32 => cutensorDataType::R_32I,
            DataType::U32 => cutensorDataType::R_32U,
        }
    }
}

/// Per-operand unary operator (applied to A/B/C before the main op).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum UnaryOp {
    Identity,
    Sqrt,
    Relu,
    Conj,
    Rcp,
    Sigmoid,
    Tanh,
}

impl UnaryOp {
    #[inline]
    fn raw(self) -> i32 {
        match self {
            UnaryOp::Identity => cutensorOperator::IDENTITY,
            UnaryOp::Sqrt => cutensorOperator::SQRT,
            UnaryOp::Relu => cutensorOperator::RELU,
            UnaryOp::Conj => cutensorOperator::CONJ,
            UnaryOp::Rcp => cutensorOperator::RCP,
            UnaryOp::Sigmoid => cutensorOperator::SIGMOID,
            UnaryOp::Tanh => cutensorOperator::TANH,
        }
    }
}

/// Binary combining operator (used between operands in elementwise /
/// reduction ops).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Mul,
    Max,
    Min,
}

impl BinaryOp {
    #[inline]
    fn raw(self) -> i32 {
        match self {
            BinaryOp::Add => cutensorOperator::ADD,
            BinaryOp::Mul => cutensorOperator::MUL,
            BinaryOp::Max => cutensorOperator::MAX,
            BinaryOp::Min => cutensorOperator::MIN,
        }
    }
}

/// cuTENSOR library handle.
#[derive(Debug)]
pub struct Handle {
    handle: cutensorHandle_t,
}

unsafe impl Send for Handle {}

impl Handle {
    pub fn new() -> Result<Self> {
        let c = cutensor()?;
        let cu = c.cutensor_create()?;
        let mut h: cutensorHandle_t = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self { handle: h })
    }

    #[inline]
    pub fn as_raw(&self) -> cutensorHandle_t {
        self.handle
    }

    /// Resize the internal plan cache — larger = more cached plans,
    /// faster re-invocations. Default is 64.
    pub fn resize_plan_cache(&self, num_entries: u32) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_handle_resize_plan_cache()?;
        check(unsafe { cu(self.handle, num_entries) })
    }

    /// Persist the plan cache to disk.
    pub fn write_plan_cache_to_file(&self, path: &str) -> Result<()> {
        let cpath = CString::new(path).map_err(|_| Error::Status {
            status: cutensorStatus_t::INVALID_VALUE,
        })?;
        let c = cutensor()?;
        let cu = c.cutensor_handle_write_plan_cache_to_file()?;
        check(unsafe { cu(self.handle, cpath.as_ptr()) })
    }

    /// Read a previously-written plan cache from disk.
    pub fn read_plan_cache_from_file(&self, path: &str) -> Result<()> {
        let cpath = CString::new(path).map_err(|_| Error::Status {
            status: cutensorStatus_t::INVALID_VALUE,
        })?;
        let c = cutensor()?;
        let cu = c.cutensor_handle_read_plan_cache_from_file()?;
        check(unsafe { cu(self.handle, cpath.as_ptr()) })
    }

    /// Persist the **kernel cache** (compiled binary kernels) to disk.
    /// Separate from plan cache — kernel cache survives across planner
    /// changes.
    pub fn write_kernel_cache_to_file(&self, path: &str) -> Result<()> {
        let cpath = CString::new(path).map_err(|_| Error::Status {
            status: cutensorStatus_t::INVALID_VALUE,
        })?;
        let c = cutensor()?;
        let cu = c.cutensor_write_kernel_cache_to_file()?;
        check(unsafe { cu(self.handle, cpath.as_ptr()) })
    }

    pub fn read_kernel_cache_from_file(&self, path: &str) -> Result<()> {
        let cpath = CString::new(path).map_err(|_| Error::Status {
            status: cutensorStatus_t::INVALID_VALUE,
        })?;
        let c = cutensor()?;
        let cu = c.cutensor_read_kernel_cache_from_file()?;
        check(unsafe { cu(self.handle, cpath.as_ptr()) })
    }

    /// Fetch cuTENSOR's pre-defined `CUTENSOR_COMPUTE_DESC_32F` descriptor.
    /// Pass this (or one of the sibling accessors) as `compute_desc` to
    /// any op constructor.
    pub fn compute_desc_32f(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_32f()?)
    }
    pub fn compute_desc_64f(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_64f()?)
    }
    pub fn compute_desc_16f(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_16f()?)
    }
    pub fn compute_desc_16bf(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_16bf()?)
    }
    pub fn compute_desc_tf32(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_tf32()?)
    }
    pub fn compute_desc_3xtf32(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_3xtf32()?)
    }
    pub fn compute_desc_4x16f(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_4x16f()?)
    }
    pub fn compute_desc_8xint8(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_8xint8()?)
    }
    pub fn compute_desc_9x16bf(&self) -> Result<*const c_void> {
        Ok(cutensor()?.compute_desc_9x16bf()?)
    }
}

/// A custom [compute descriptor]. Prefer the pre-defined ones
/// ([`Handle::compute_desc_32f`], …) unless you need attribute
/// customization.
#[derive(Debug)]
pub struct ComputeDescriptor<'h> {
    desc: baracuda_cutensor_sys::cutensorComputeDescriptor_t,
    _handle: &'h Handle,
}

impl<'h> ComputeDescriptor<'h> {
    pub fn new(handle: &'h Handle) -> Result<Self> {
        let c = cutensor()?;
        let cu = c.cutensor_create_compute_descriptor()?;
        let mut desc: baracuda_cutensor_sys::cutensorComputeDescriptor_t = core::ptr::null();
        check(unsafe { cu(handle.as_raw(), &mut desc as *mut _ as *mut _) })?;
        Ok(Self {
            desc,
            _handle: handle,
        })
    }

    #[inline]
    pub fn as_raw(&self) -> baracuda_cutensor_sys::cutensorComputeDescriptor_t {
        self.desc
    }

    /// # Safety
    ///
    /// `value` points at a buffer of `size_bytes` matching `attr`.
    pub unsafe fn set_attribute(
        &self,
        attr: i32,
        value: *const c_void,
        size_bytes: usize,
    ) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_compute_descriptor_set_attribute()?;
        check(cu(
            self._handle.as_raw(),
            self.desc,
            attr,
            value,
            size_bytes,
        ))
    }

    /// # Safety
    ///
    /// `value` points at a writable buffer of `size_bytes`.
    pub unsafe fn get_attribute(
        &self,
        attr: i32,
        value: *mut c_void,
        size_bytes: usize,
    ) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_compute_descriptor_get_attribute()?;
        check(cu(
            self._handle.as_raw(),
            self.desc,
            attr,
            value,
            size_bytes,
        ))
    }
}

impl Drop for ComputeDescriptor<'_> {
    fn drop(&mut self) {
        if let Ok(c) = cutensor() {
            if let Ok(cu) = c.cutensor_destroy_compute_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// A block-sparse tensor descriptor (cuTENSOR 2.x). Used on the A
/// operand of a [`BlockSparseContraction`].
#[derive(Debug)]
pub struct BlockSparseTensorDescriptor<'h> {
    desc: baracuda_cutensor_sys::cutensorBlockSparseTensorDescriptor_t,
    _handle: &'h Handle,
}

impl<'h> BlockSparseTensorDescriptor<'h> {
    /// Build a block-sparse tensor:
    ///
    /// - `extents` — full dense shape
    /// - `block_size` — size per dim of each non-zero block (same length as extents)
    /// - `strides` — optional custom strides; `None` = packed
    /// - `block_indices` — array of `num_modes × block_count` ints identifying
    ///   the non-zero block locations (index per mode per block)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        handle: &'h Handle,
        extents: &[i64],
        block_size: &[i64],
        strides: Option<&[i64]>,
        block_indices: &[i32],
        dtype: DataType,
        alignment_bytes: u32,
    ) -> Result<Self> {
        assert_eq!(block_size.len(), extents.len());
        if let Some(s) = strides {
            assert_eq!(s.len(), extents.len());
        }
        let num_modes = extents.len() as u32;
        let block_count = (block_indices.len() / extents.len()) as i64;
        let c = cutensor()?;
        let cu = c.cutensor_create_block_sparse_tensor_descriptor()?;
        let mut desc: baracuda_cutensor_sys::cutensorBlockSparseTensorDescriptor_t =
            core::ptr::null_mut();
        check(unsafe {
            cu(
                handle.as_raw(),
                &mut desc,
                num_modes,
                extents.as_ptr(),
                block_size.as_ptr(),
                strides.map_or(core::ptr::null(), |s| s.as_ptr()),
                block_count,
                block_indices.as_ptr(),
                dtype.raw(),
                alignment_bytes,
            )
        })?;
        Ok(Self {
            desc,
            _handle: handle,
        })
    }

    #[inline]
    pub fn as_raw(&self) -> baracuda_cutensor_sys::cutensorBlockSparseTensorDescriptor_t {
        self.desc
    }
}

impl Drop for BlockSparseTensorDescriptor<'_> {
    fn drop(&mut self) {
        if let Ok(c) = cutensor() {
            if let Ok(cu) = c.cutensor_destroy_block_sparse_tensor_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Block-sparse contraction: the A operand is block-sparse, B/C/D dense.
#[derive(Debug)]
pub struct BlockSparseContraction;

impl BlockSparseContraction {
    /// # Safety
    ///
    /// `compute_desc` must be null or a live `cutensorComputeDescriptor_t`.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub unsafe fn new<'h>(
        handle: &'h Handle,
        a: &BlockSparseTensorDescriptor<'h>,
        modes_a: &[i32],
        b: &TensorDescriptor<'h>,
        modes_b: &[i32],
        c: &TensorDescriptor<'h>,
        modes_c: &[i32],
        d: &TensorDescriptor<'h>,
        modes_d: &[i32],
        compute_desc: *const c_void,
    ) -> Result<OperationDescriptor<'h>> {
        let lib = cutensor()?;
        let cu = lib.cutensor_create_block_sparse_contraction()?;
        let mut desc: cutensorOperationDescriptor_t = core::ptr::null_mut();
        check(cu(
            handle.as_raw(),
            &mut desc,
            a.as_raw(),
            modes_a.as_ptr(),
            cutensorOperator::IDENTITY,
            b.as_raw(),
            modes_b.as_ptr(),
            cutensorOperator::IDENTITY,
            c.as_raw(),
            modes_c.as_ptr(),
            cutensorOperator::IDENTITY,
            d.as_raw(),
            modes_d.as_ptr(),
            compute_desc,
        ))?;
        Ok(OperationDescriptor {
            desc,
            handle,
            kind: OpKind::BlockSparseContraction,
        })
    }
}

/// A ternary contraction op: `E[mE] = α·op_a(A)·op_b(B)·op_c(C) + β·op_d(D)`.
#[derive(Debug)]
pub struct TrinaryContraction;

impl TrinaryContraction {
    /// # Safety
    ///
    /// `compute_desc` must be null or a live `cutensorComputeDescriptor_t`.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub unsafe fn new<'h>(
        handle: &'h Handle,
        a: &TensorDescriptor<'h>,
        modes_a: &[i32],
        b: &TensorDescriptor<'h>,
        modes_b: &[i32],
        c: &TensorDescriptor<'h>,
        modes_c: &[i32],
        d: &TensorDescriptor<'h>,
        modes_d: &[i32],
        e: &TensorDescriptor<'h>,
        modes_e: &[i32],
        compute_desc: *const c_void,
    ) -> Result<OperationDescriptor<'h>> {
        let lib = cutensor()?;
        let cu = lib.cutensor_create_contraction_trinary()?;
        let mut desc: cutensorOperationDescriptor_t = core::ptr::null_mut();
        check(cu(
            handle.as_raw(),
            &mut desc,
            a.as_raw(),
            modes_a.as_ptr(),
            cutensorOperator::IDENTITY,
            b.as_raw(),
            modes_b.as_ptr(),
            cutensorOperator::IDENTITY,
            c.as_raw(),
            modes_c.as_ptr(),
            cutensorOperator::IDENTITY,
            d.as_raw(),
            modes_d.as_ptr(),
            cutensorOperator::IDENTITY,
            e.as_raw(),
            modes_e.as_ptr(),
            compute_desc,
        ))?;
        Ok(OperationDescriptor {
            desc,
            handle,
            kind: OpKind::TrinaryContraction,
        })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Ok(c) = cutensor() {
            if let Ok(cu) = c.cutensor_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A tensor descriptor: modes + extents + dtype + stride layout.
#[derive(Debug)]
pub struct TensorDescriptor<'h> {
    desc: cutensorTensorDescriptor_t,
    _handle: &'h Handle,
}

impl<'h> TensorDescriptor<'h> {
    /// `extents[i]` is the size along mode `i`. `strides = None` gets a
    /// row-major packed layout.
    pub fn new(
        handle: &'h Handle,
        extents: &[i64],
        strides: Option<&[i64]>,
        dtype: DataType,
        alignment_bytes: u32,
    ) -> Result<Self> {
        let c = cutensor()?;
        let cu = c.cutensor_create_tensor_descriptor()?;
        let num_modes = extents.len() as u32;
        if let Some(s) = strides {
            assert_eq!(s.len(), extents.len(), "strides length mismatch");
        }
        let mut desc: cutensorTensorDescriptor_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                handle.as_raw(),
                &mut desc,
                num_modes,
                extents.as_ptr(),
                strides.map_or(core::ptr::null(), |s| s.as_ptr()),
                dtype.raw(),
                alignment_bytes,
            )
        })?;
        Ok(Self {
            desc,
            _handle: handle,
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cutensorTensorDescriptor_t {
        self.desc
    }

    /// Low-level tensor-descriptor attribute setter.
    ///
    /// # Safety
    ///
    /// `buf` must point at `size_bytes` matching `attr`.
    pub unsafe fn set_attribute(
        &self,
        attr: i32,
        buf: *const c_void,
        size_bytes: usize,
    ) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_tensor_descriptor_set_attribute()?;
        check(cu(self._handle.as_raw(), self.desc, attr, buf, size_bytes))
    }
}

impl Drop for TensorDescriptor<'_> {
    fn drop(&mut self) {
        if let Ok(c) = cutensor() {
            if let Ok(cu) = c.cutensor_destroy_tensor_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Internal: what kind of op a descriptor wraps — needed to dispatch
/// the right `execute` path on the compiled [`Plan`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum OpKind {
    Contraction,
    TrinaryContraction,
    BlockSparseContraction,
    Reduction,
    ElementwiseBinary,
    ElementwiseTrinary,
    Permutation,
}

/// An un-compiled operation descriptor. Users typically create these
/// through constructors on [`Contraction`], [`Reduction`],
/// [`ElementwiseBinary`], [`ElementwiseTrinary`], or [`Permutation`].
#[derive(Debug)]
pub struct OperationDescriptor<'h> {
    desc: cutensorOperationDescriptor_t,
    handle: &'h Handle,
    kind: OpKind,
}

impl<'h> OperationDescriptor<'h> {
    #[inline]
    pub fn as_raw(&self) -> cutensorOperationDescriptor_t {
        self.desc
    }

    /// Estimate the scratch workspace required by a plan built from
    /// this descriptor + `pref`.
    pub fn estimate_workspace(
        &self,
        pref: &PlanPreference<'h>,
        kind: WorkspaceKind,
    ) -> Result<u64> {
        let c = cutensor()?;
        let cu = c.cutensor_estimate_workspace_size()?;
        let mut size: u64 = 0;
        check(unsafe {
            cu(
                self.handle.as_raw(),
                self.desc,
                pref.as_raw(),
                kind.raw(),
                &mut size,
            )
        })?;
        Ok(size)
    }

    /// Estimated runtime in milliseconds for this op at the given
    /// algorithm (`cutensorAlgo::DEFAULT` for auto).
    pub fn estimate_runtime(&self, pref: &PlanPreference<'h>, algo: i32) -> Result<f32> {
        let c = cutensor()?;
        let cu = c.cutensor_operation_estimate_runtime()?;
        let mut ms: f32 = 0.0;
        check(unsafe {
            cu(
                self.handle.as_raw(),
                self.desc,
                pref.as_raw(),
                algo,
                &mut ms,
            )
        })?;
        Ok(ms)
    }

    /// Number of algorithms cuTENSOR has for this op shape.
    pub fn num_algos(&self) -> Result<i32> {
        let c = cutensor()?;
        let cu = c.cutensor_operation_num_algos()?;
        let mut n: i32 = 0;
        check(unsafe { cu(self.desc, &mut n) })?;
        Ok(n)
    }

    /// Low-level attribute getter (for attributes not exposed as typed fns).
    ///
    /// # Safety
    ///
    /// `buf` must be writable for `size_bytes` matching `attr`.
    pub unsafe fn get_attribute(
        &self,
        attr: i32,
        buf: *mut c_void,
        size_bytes: usize,
    ) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_operation_descriptor_get_attribute()?;
        check(cu(self.handle.as_raw(), self.desc, attr, buf, size_bytes))
    }

    /// Low-level attribute setter.
    ///
    /// # Safety
    ///
    /// `buf` must point at a buffer of `size_bytes` matching `attr`.
    pub unsafe fn set_attribute(
        &self,
        attr: i32,
        buf: *const c_void,
        size_bytes: usize,
    ) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_operation_descriptor_set_attribute()?;
        check(cu(self.handle.as_raw(), self.desc, attr, buf, size_bytes))
    }
}

impl Drop for OperationDescriptor<'_> {
    fn drop(&mut self) {
        if let Ok(c) = cutensor() {
            if let Ok(cu) = c.cutensor_destroy_operation_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// A contraction op: `D[mD] = α * op_a(A[mA]) * op_b(B[mB]) + β * op_c(C[mC])`.
#[derive(Debug)]
pub struct Contraction;

impl Contraction {
    /// Build a contraction descriptor.
    ///
    /// `compute_desc` is an opaque pointer — pass `core::ptr::null()`
    /// for the library default (compute-type matches C's dtype).
    ///
    /// # Safety
    ///
    /// `compute_desc` must be null or a valid `cutensorComputeDescriptor_t`.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub unsafe fn new<'h>(
        handle: &'h Handle,
        a: &TensorDescriptor<'h>,
        modes_a: &[i32],
        b: &TensorDescriptor<'h>,
        modes_b: &[i32],
        c: &TensorDescriptor<'h>,
        modes_c: &[i32],
        d: &TensorDescriptor<'h>,
        modes_d: &[i32],
        compute_desc: *const c_void,
    ) -> Result<OperationDescriptor<'h>> {
        let cu_lib = cutensor()?;
        let cu = cu_lib.cutensor_create_contraction()?;
        let mut desc: cutensorOperationDescriptor_t = core::ptr::null_mut();
        check(cu(
            handle.as_raw(),
            &mut desc,
            a.as_raw(),
            modes_a.as_ptr(),
            cutensorOperator::IDENTITY,
            b.as_raw(),
            modes_b.as_ptr(),
            cutensorOperator::IDENTITY,
            c.as_raw(),
            modes_c.as_ptr(),
            cutensorOperator::IDENTITY,
            d.as_raw(),
            modes_d.as_ptr(),
            compute_desc,
        ))?;
        Ok(OperationDescriptor {
            desc,
            handle,
            kind: OpKind::Contraction,
        })
    }
}

/// A reduction op: `D[mD] = reduce(A[mA])` with user-chosen reduce op.
#[derive(Debug)]
pub struct Reduction;

impl Reduction {
    /// Build a reduction. `modes_d` is a subset of `modes_a` — all
    /// modes in `a` that do NOT appear in `d` are reduced. `op_reduce`
    /// is ADD for sum, MUL for product, MAX/MIN for min-max.
    ///
    /// # Safety
    ///
    /// `compute_desc` must be null or valid.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub unsafe fn new<'h>(
        handle: &'h Handle,
        a: &TensorDescriptor<'h>,
        modes_a: &[i32],
        c: &TensorDescriptor<'h>,
        modes_c: &[i32],
        d: &TensorDescriptor<'h>,
        modes_d: &[i32],
        op_reduce: BinaryOp,
        compute_desc: *const c_void,
    ) -> Result<OperationDescriptor<'h>> {
        let lib = cutensor()?;
        let cu = lib.cutensor_create_reduction()?;
        let mut desc: cutensorOperationDescriptor_t = core::ptr::null_mut();
        check(cu(
            handle.as_raw(),
            &mut desc,
            a.as_raw(),
            modes_a.as_ptr(),
            cutensorOperator::IDENTITY,
            c.as_raw(),
            modes_c.as_ptr(),
            cutensorOperator::IDENTITY,
            d.as_raw(),
            modes_d.as_ptr(),
            op_reduce.raw(),
            compute_desc,
        ))?;
        Ok(OperationDescriptor {
            desc,
            handle,
            kind: OpKind::Reduction,
        })
    }
}

/// Elementwise binary op: `D[mD] = (α * op_a(A[mA])) op_ac (γ * op_c(C[mC]))`.
#[derive(Debug)]
pub struct ElementwiseBinary;

impl ElementwiseBinary {
    /// # Safety
    ///
    /// `compute_desc` must be null or valid.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub unsafe fn new<'h>(
        handle: &'h Handle,
        a: &TensorDescriptor<'h>,
        modes_a: &[i32],
        op_a: UnaryOp,
        c: &TensorDescriptor<'h>,
        modes_c: &[i32],
        op_c: UnaryOp,
        d: &TensorDescriptor<'h>,
        modes_d: &[i32],
        op_ac: BinaryOp,
        compute_desc: *const c_void,
    ) -> Result<OperationDescriptor<'h>> {
        let lib = cutensor()?;
        let cu = lib.cutensor_create_elementwise_binary()?;
        let mut desc: cutensorOperationDescriptor_t = core::ptr::null_mut();
        check(cu(
            handle.as_raw(),
            &mut desc,
            a.as_raw(),
            modes_a.as_ptr(),
            op_a.raw(),
            c.as_raw(),
            modes_c.as_ptr(),
            op_c.raw(),
            d.as_raw(),
            modes_d.as_ptr(),
            op_ac.raw(),
            compute_desc,
        ))?;
        Ok(OperationDescriptor {
            desc,
            handle,
            kind: OpKind::ElementwiseBinary,
        })
    }
}

/// Elementwise trinary op:
/// `D[mD] = ((α * op_a(A) op_ab β * op_b(B)) op_abc γ * op_c(C))`.
#[derive(Debug)]
pub struct ElementwiseTrinary;

impl ElementwiseTrinary {
    /// # Safety
    ///
    /// `compute_desc` must be null or valid.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub unsafe fn new<'h>(
        handle: &'h Handle,
        a: &TensorDescriptor<'h>,
        modes_a: &[i32],
        op_a: UnaryOp,
        b: &TensorDescriptor<'h>,
        modes_b: &[i32],
        op_b: UnaryOp,
        c: &TensorDescriptor<'h>,
        modes_c: &[i32],
        op_c: UnaryOp,
        d: &TensorDescriptor<'h>,
        modes_d: &[i32],
        op_ab: BinaryOp,
        op_abc: BinaryOp,
        compute_desc: *const c_void,
    ) -> Result<OperationDescriptor<'h>> {
        let lib = cutensor()?;
        let cu = lib.cutensor_create_elementwise_trinary()?;
        let mut desc: cutensorOperationDescriptor_t = core::ptr::null_mut();
        check(cu(
            handle.as_raw(),
            &mut desc,
            a.as_raw(),
            modes_a.as_ptr(),
            op_a.raw(),
            b.as_raw(),
            modes_b.as_ptr(),
            op_b.raw(),
            c.as_raw(),
            modes_c.as_ptr(),
            op_c.raw(),
            d.as_raw(),
            modes_d.as_ptr(),
            op_ab.raw(),
            op_abc.raw(),
            compute_desc,
        ))?;
        Ok(OperationDescriptor {
            desc,
            handle,
            kind: OpKind::ElementwiseTrinary,
        })
    }
}

/// Tensor permutation (axis shuffle + optional unary op):
/// `B[mB] = α * op_a(A[mA])`.
#[derive(Debug)]
pub struct Permutation;

impl Permutation {
    /// # Safety
    ///
    /// `compute_desc` must be null or valid.
    #[allow(clippy::too_many_arguments, clippy::new_ret_no_self)]
    pub unsafe fn new<'h>(
        handle: &'h Handle,
        a: &TensorDescriptor<'h>,
        modes_a: &[i32],
        op_a: UnaryOp,
        b: &TensorDescriptor<'h>,
        modes_b: &[i32],
        compute_desc: *const c_void,
    ) -> Result<OperationDescriptor<'h>> {
        let lib = cutensor()?;
        let cu = lib.cutensor_create_permutation()?;
        let mut desc: cutensorOperationDescriptor_t = core::ptr::null_mut();
        check(cu(
            handle.as_raw(),
            &mut desc,
            a.as_raw(),
            modes_a.as_ptr(),
            op_a.raw(),
            b.as_raw(),
            modes_b.as_ptr(),
            compute_desc,
        ))?;
        Ok(OperationDescriptor {
            desc,
            handle,
            kind: OpKind::Permutation,
        })
    }
}

/// Plan preferences — algorithm selection + JIT mode.
#[derive(Debug)]
pub struct PlanPreference<'h> {
    pref: cutensorPlanPreference_t,
    _handle: &'h Handle,
}

impl<'h> PlanPreference<'h> {
    pub fn new(handle: &'h Handle, algo: i32, jit_mode: i32) -> Result<Self> {
        let c = cutensor()?;
        let cu = c.cutensor_create_plan_preference()?;
        let mut p: cutensorPlanPreference_t = core::ptr::null_mut();
        check(unsafe { cu(handle.as_raw(), &mut p, algo, jit_mode) })?;
        Ok(Self {
            pref: p,
            _handle: handle,
        })
    }

    /// Default preferences — library's best guess at algorithm, JIT off.
    pub fn default_for(handle: &'h Handle) -> Result<Self> {
        Self::new(handle, cutensorAlgo::DEFAULT, cutensorJitMode::NONE)
    }

    #[inline]
    pub fn as_raw(&self) -> cutensorPlanPreference_t {
        self.pref
    }

    /// Set a plan-preference attribute (see cuTENSOR's
    /// `cutensorPlanPreferenceAttribute_t`).
    ///
    /// # Safety
    ///
    /// `value` must point at a buffer of at least `size_bytes` for the
    /// attribute kind being set.
    pub unsafe fn set_attribute(
        &self,
        attr: i32,
        value: *const c_void,
        size_bytes: usize,
    ) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_plan_preference_set_attribute()?;
        check(cu(
            self._handle.as_raw(),
            self.pref,
            attr,
            value,
            size_bytes,
        ))
    }

    /// Read a plan-preference attribute.
    ///
    /// # Safety
    ///
    /// `value` must be writable for `size_bytes` matching `attr`.
    pub unsafe fn get_attribute(
        &self,
        attr: i32,
        value: *mut c_void,
        size_bytes: usize,
    ) -> Result<()> {
        let c = cutensor()?;
        let cu = c.cutensor_plan_preference_get_attribute()?;
        check(cu(
            self._handle.as_raw(),
            self.pref,
            attr,
            value,
            size_bytes,
        ))
    }
}

impl Drop for PlanPreference<'_> {
    fn drop(&mut self) {
        if let Ok(c) = cutensor() {
            if let Ok(cu) = c.cutensor_destroy_plan_preference() {
                let _ = unsafe { cu(self.pref) };
            }
        }
    }
}

/// Workspace-size preference tier.
#[derive(Copy, Clone, Debug)]
pub enum WorkspaceKind {
    Min,
    Default,
    Max,
}

impl WorkspaceKind {
    #[inline]
    fn raw(self) -> i32 {
        match self {
            WorkspaceKind::Min => cutensorWorksizePreference::MIN,
            WorkspaceKind::Default => cutensorWorksizePreference::DEFAULT,
            WorkspaceKind::Max => cutensorWorksizePreference::MAX,
        }
    }
}

/// A compiled operation plan. Dispatch to the matching `execute` method
/// based on the op kind that built it.
#[derive(Debug)]
pub struct Plan<'h> {
    plan: cutensorPlan_t,
    handle: &'h Handle,
    kind: OpKind,
}

impl<'h> Plan<'h> {
    /// Compile an operation descriptor into a plan.
    /// `workspace_size_limit` bytes — pass the estimate.
    pub fn new(
        op: &OperationDescriptor<'h>,
        pref: &PlanPreference<'h>,
        workspace_size_limit: u64,
    ) -> Result<Self> {
        let c = cutensor()?;
        let cu = c.cutensor_create_plan()?;
        let mut p: cutensorPlan_t = core::ptr::null_mut();
        check(unsafe {
            cu(
                op.handle.as_raw(),
                &mut p,
                op.as_raw(),
                pref.as_raw(),
                workspace_size_limit,
            )
        })?;
        Ok(Self {
            plan: p,
            handle: op.handle,
            kind: op.kind,
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cutensorPlan_t {
        self.plan
    }

    /// Execute a contraction plan. Aborts if `self` wasn't built from a
    /// [`Contraction`] descriptor.
    ///
    /// # Safety
    ///
    /// All device pointers must be live, tensor-descriptor-conforming,
    /// and aligned. `workspace` must be at least the estimated size.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn contract(
        &self,
        alpha: *const c_void,
        a: *const c_void,
        b: *const c_void,
        beta: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        assert_eq!(self.kind, OpKind::Contraction, "plan is not a contraction");
        let lib = cutensor()?;
        let cu = lib.cutensor_contract()?;
        check(cu(
            self.handle.as_raw(),
            self.plan,
            alpha,
            a,
            b,
            beta,
            c,
            d,
            workspace,
            workspace_bytes,
            stream,
        ))
    }

    /// Execute a reduction plan.
    ///
    /// # Safety
    ///
    /// Same as [`Self::contract`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn reduce(
        &self,
        alpha: *const c_void,
        a: *const c_void,
        beta: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        assert_eq!(self.kind, OpKind::Reduction, "plan is not a reduction");
        let lib = cutensor()?;
        let cu = lib.cutensor_reduce()?;
        check(cu(
            self.handle.as_raw(),
            self.plan,
            alpha,
            a,
            beta,
            c,
            d,
            workspace,
            workspace_bytes,
            stream,
        ))
    }

    /// Execute an elementwise-binary plan.
    ///
    /// # Safety
    ///
    /// Same as [`Self::contract`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn elementwise_binary(
        &self,
        alpha: *const c_void,
        a: *const c_void,
        gamma: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        stream: *mut c_void,
    ) -> Result<()> {
        assert_eq!(
            self.kind,
            OpKind::ElementwiseBinary,
            "plan is not an elementwise-binary"
        );
        let lib = cutensor()?;
        let cu = lib.cutensor_elementwise_binary_execute()?;
        check(cu(
            self.handle.as_raw(),
            self.plan,
            alpha,
            a,
            gamma,
            c,
            d,
            stream,
        ))
    }

    /// Execute an elementwise-trinary plan.
    ///
    /// # Safety
    ///
    /// Same as [`Self::contract`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn elementwise_trinary(
        &self,
        alpha: *const c_void,
        a: *const c_void,
        beta: *const c_void,
        b: *const c_void,
        gamma: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        stream: *mut c_void,
    ) -> Result<()> {
        assert_eq!(
            self.kind,
            OpKind::ElementwiseTrinary,
            "plan is not an elementwise-trinary"
        );
        let lib = cutensor()?;
        let cu = lib.cutensor_elementwise_trinary_execute()?;
        check(cu(
            self.handle.as_raw(),
            self.plan,
            alpha,
            a,
            beta,
            b,
            gamma,
            c,
            d,
            stream,
        ))
    }

    /// Execute a permutation plan.
    ///
    /// # Safety
    ///
    /// Same as [`Self::contract`].
    pub unsafe fn permute(
        &self,
        alpha: *const c_void,
        a: *const c_void,
        b: *mut c_void,
        stream: *mut c_void,
    ) -> Result<()> {
        assert_eq!(self.kind, OpKind::Permutation, "plan is not a permutation");
        let lib = cutensor()?;
        let cu = lib.cutensor_permute()?;
        check(cu(self.handle.as_raw(), self.plan, alpha, a, b, stream))
    }

    /// Execute a block-sparse contraction plan.
    ///
    /// # Safety
    ///
    /// Same as [`Self::contract`]; `a` must be a block-sparse device
    /// buffer matching the `BlockSparseTensorDescriptor` passed to
    /// [`BlockSparseContraction::new`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn contract_block_sparse(
        &self,
        alpha: *const c_void,
        a: *const c_void,
        b: *const c_void,
        beta: *const c_void,
        c: *const c_void,
        d: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        assert_eq!(
            self.kind,
            OpKind::BlockSparseContraction,
            "plan is not a block-sparse contraction"
        );
        let lib = cutensor()?;
        let cu = lib.cutensor_block_sparse_contract()?;
        check(cu(
            self.handle.as_raw(),
            self.plan,
            alpha,
            a,
            b,
            beta,
            c,
            d,
            workspace,
            workspace_bytes,
            stream,
        ))
    }

    /// Execute a trinary-contraction plan:
    /// `E = α·op_a(A)·op_b(B)·op_c(C) + β·op_d(D)`.
    ///
    /// # Safety
    ///
    /// Same as [`Self::contract`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn contract_trinary(
        &self,
        alpha: *const c_void,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        beta: *const c_void,
        d: *const c_void,
        e: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        assert_eq!(
            self.kind,
            OpKind::TrinaryContraction,
            "plan is not a trinary-contraction"
        );
        let lib = cutensor()?;
        let cu = lib.cutensor_contract_trinary()?;
        check(cu(
            self.handle.as_raw(),
            self.plan,
            alpha,
            a,
            b,
            c,
            beta,
            d,
            e,
            workspace,
            workspace_bytes,
            stream,
        ))
    }
}

impl Drop for Plan<'_> {
    fn drop(&mut self) {
        if let Ok(c) = cutensor() {
            if let Ok(cu) = c.cutensor_destroy_plan() {
                let _ = unsafe { cu(self.plan) };
            }
        }
    }
}
