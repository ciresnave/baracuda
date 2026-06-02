//! GGUF MMVQ plan — fused dequant + matrix-vector multiply.
//!
//! `out[r] = Σ_c W_q[r, c] · y[c]`, where `W_q` is GGUF-packed
//! (one block-row of `packed_cols_bytes` per matrix row) and `y` /
//! `out` are dense FP. This is the inference-time "decode-step"
//! matmul used by llama.cpp on GGUF weights.
//!
//! All 11 GGUF block formats are supported. Q8_K MMVQ (Phase 11.4) is
//! a bespoke baracuda addition — upstream llama.cpp / Fuel ship only
//! the Q8_K dequant kernel and treat the format as a CPU-side
//! intermediate. baracuda adds a fused MMVQ to avoid the 2× memory
//! traffic of dequantize-then-GEMV on the inference decode step.
//!
//! **Activation dtype**: Phase 18.1 added `f16` and `bf16` activation
//! paths alongside the original `f32`. Output dtype always matches the
//! activation dtype (PyTorch convention). The accumulator stays f32 in
//! every variant — the dtype cast happens at the activation load and
//! the destination store sites only.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, GgufBlockFormat, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace,
    U8,
};
use half::{bf16, f16};

use crate::quantize::map_status;

/// Sealed marker for activation / destination element types accepted by
/// [`GgufMmvqPlan`].
///
/// Three members: `f32` (the original, retained as the default for
/// source-compat), `f16`, and `bf16` (Phase 18.1). Each implementation
/// pins the FFI dispatch to the matching `*_run` / `*_actstrided_run`
/// symbols in `baracuda-kernels-sys`.
pub trait GgufMmvqActivation: Element + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

impl GgufMmvqActivation for f32 {}
impl GgufMmvqActivation for f16 {}
impl GgufMmvqActivation for bf16 {}

/// Descriptor for a GGUF MMVQ op.
#[derive(Copy, Clone, Debug)]
pub struct GgufMmvqDescriptor {
    /// Number of output rows (= rows of the packed weight matrix).
    pub nrows: i32,
    /// Number of unpacked columns (= length of the activation vector).
    /// Must be a multiple of `block_format.block_size()`.
    pub ncols: i32,
    /// GGUF block format of the packed weight matrix.
    pub block_format: GgufBlockFormat,
    /// Byte offset into the `weight` allocation at which this matrix
    /// starts. `0` is the default and falls through the contig fast
    /// path; non-zero engages the actstrided FFI sibling and lets a
    /// single device allocation host multiple GGUF matrices.
    ///
    /// Phase 14.5 — defaults to `0` for source-compat.
    pub w_start_byte_offset: i64,
}

impl Default for GgufMmvqDescriptor {
    fn default() -> Self {
        Self {
            nrows: 0,
            ncols: 0,
            block_format: GgufBlockFormat::Q8_0,
            w_start_byte_offset: 0,
        }
    }
}

/// Args bundle for a GGUF MMVQ launch.
///
/// Generic on the activation / destination dtype `T`. Default
/// `T = f32` keeps the original (pre-Phase-18.1) source surface.
pub struct GgufMmvqArgs<'a, T: GgufMmvqActivation = f32> {
    /// Packed GGUF weight bytes. Length must equal
    /// `nrows * (ncols / block_size) * type_size`.
    pub weight: TensorRef<'a, U8, 1>,
    /// Activation vector `[ncols]`.
    pub activation: TensorRef<'a, T, 1>,
    /// Output vector `[nrows]`, same dtype as the activation (PyTorch
    /// convention — `f16 activation → f16 output`, etc.).
    pub output: TensorMut<'a, T, 1>,
}

/// `gguf_mmvq` plan.
///
/// `out[r] = Σ_c W_q[r, c] · y[c]` — fused dequant + matrix-vector
/// multiply over a GGUF-packed weight matrix.
///
/// **When to use**: inference-time "decode-step" matmul in
/// llama.cpp-style LLM serving — single activation vector, full FP
/// output, no intermediate dequant materialization. For ahead-of-time
/// unpack use [`GgufDequantizePlan`](crate::GgufDequantizePlan).
///
/// **Dtypes**: weight is GGUF-packed `u8` bytes; activation and
/// output share dtype `T` ∈ {`f32`, `f16`, `bf16`}. f32 is the
/// default for source-compat; f16 / bf16 are Phase 18.1 additions.
///
/// **Block formats**: all 11 GGUF block formats (type-0/1 + k-quants
/// including `Q8_K`). The `Q8_K` kernel is a bespoke baracuda addition
/// (Phase 11.4) — upstream llama.cpp ships only the dequant kernel.
///
/// **Shape limits**: `ncols` must be a multiple of the block size;
/// weight byte length must equal `nrows * (ncols / block_size) * type_size`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable on identical
/// hardware. Single-pass warp reduction; no atomics. f32 accumulator
/// in every variant (f16 / bf16 activations are cast to f32 at load).
pub struct GgufMmvqPlan<T: GgufMmvqActivation = f32> {
    desc: GgufMmvqDescriptor,
    sku: KernelSku,
    _phantom: PhantomData<T>,
}

impl<T: GgufMmvqActivation> GgufMmvqPlan<T> {
    /// Pick a kernel for `desc`. Errors on non-positive dims or `ncols`
    /// that doesn't tile to the block size. All 11 GGUF block formats
    /// are supported as of Phase 11.4 (Q8_K MMVQ is a bespoke baracuda
    /// addition; upstream llama.cpp ships only the dequant kernel).
    pub fn select(
        _stream: &Stream,
        desc: &GgufMmvqDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.nrows < 0 || desc.ncols < 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: nrows / ncols must be non-negative",
            ));
        }
        if !desc.block_format.has_mmvq() {
            return Err(Error::Unsupported(
                "GgufMmvqPlan: block format reports no MMVQ kernel",
            ));
        }
        let bs = desc.block_format.block_size() as i32;
        if desc.ncols % bs != 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: ncols must be a multiple of the block size",
            ));
        }
        // Note: type-0/1 MMVQ has an implicit `ncols >= 64` minimum
        // because threads `tid=16..31` always read columns `32..62`.
        // For single-matrix callers (this plan) the OOB reads land in
        // unallocated zero memory and produce the right answer.
        // GgufMmvqBatchedPlan adds a debug-build assertion against this
        // because contiguous-batched activations make the OOB reads
        // hit adjacent tokens' rows — see Phase 22 / Phase 20.1.
        // Phase 15.1 — debug-build alignment guard for the W-offset.
        // GGUF block structs have natural alignment ≥ 2 (`half`) or
        // ≥ 4 (`half2` / `float`). A misaligned `w_start_byte_offset`
        // turns the host-side pointer arithmetic `(const u8*)x + off`
        // into a misaligned `const block_qX_*` reload inside the
        // kernel — silently producing garbage results. Reject up-front
        // in debug builds; release builds skip the check (zero cost).
        #[cfg(debug_assertions)]
        {
            if desc.w_start_byte_offset < 0 {
                return Err(Error::InvalidProblem(
                    "GgufMmvqPlan: w_start_byte_offset must be non-negative",
                ));
            }
            let alignment = required_alignment(desc.block_format);
            if desc.w_start_byte_offset % alignment != 0 {
                return Err(Error::InvalidProblem(
                    "GgufMmvqPlan: w_start_byte_offset must be aligned to the block format's natural alignment (Q4_1/Q5_1/Q2K/Q4K/Q5K/Q8K = 4, others = 2)",
                ));
            }
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(desc.block_format, T::KIND),
            _phantom: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &GgufMmvqArgs<'_, T>) -> Result<()> {
        if args.activation.shape != [self.desc.ncols] {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: activation shape != [ncols]",
            ));
        }
        if args.output.shape != [self.desc.nrows] {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: output shape != [nrows]",
            ));
        }
        let blocks_per_row = self.desc.ncols / self.desc.block_format.block_size() as i32;
        let expected_bytes =
            (self.desc.nrows as i64) * (blocks_per_row as i64) * (self.desc.block_format.type_size() as i64);
        // When the host shares one allocation across multiple GGUF
        // matrices (`w_start_byte_offset > 0`) the weight TensorRef's
        // shape covers the *whole* allocation, not just this matrix.
        // Require the slice to start at the matrix's first byte (i.e.
        // shape[0] == matrix_bytes when offset == 0) and otherwise to
        // simply contain enough bytes after the offset.
        let weight_len_bytes = args.weight.shape[0] as i64;
        if self.desc.w_start_byte_offset < 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: w_start_byte_offset must be non-negative",
            ));
        }
        let need_bytes = self.desc.w_start_byte_offset + expected_bytes;
        if weight_len_bytes < need_bytes {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: weight byte length < w_start_byte_offset + nrows * blocks_per_row * type_size",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the selected kernel.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    ///
    /// Dispatch policy:
    ///   * `T::KIND` selects the dtype-suffixed FFI family
    ///     (f32 → un-suffixed; f16 → `_f16_run`; bf16 → `_bf16_run`).
    ///   * within each dtype family:
    ///     - `w_start_byte_offset == 0 && activation.stride[0] == 1`
    ///       → contig FFI (no overhead).
    ///     - otherwise → activation-strided + W-offset sibling FFI.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: GgufMmvqArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.nrows == 0 || self.desc.ncols == 0 {
            return Ok(());
        }
        let w_ptr = args.weight.data.as_raw().0 as *const c_void;
        let y_ptr = args.activation.data.as_raw().0 as *const c_void;
        let dst_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let ncols = self.desc.ncols;
        let nrows = self.desc.nrows;
        let w_off = self.desc.w_start_byte_offset;
        let stride_y = args.activation.stride[0];

        let use_strided = w_off != 0 || stride_y != 1;

        let status =
            unsafe { dispatch_ffi::<T>(self.desc.block_format, use_strided, ncols, nrows, w_ptr,
                w_off, stride_y, y_ptr, dst_ptr, stream_ptr) };
        map_status(status)
    }
}

/// Dispatch to the right FFI symbol given (block format, dtype, contig/strided).
///
/// One match per (dtype, strided) quadrant. Factored out of `run()` to
/// keep the launch path readable.
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_ffi<T: GgufMmvqActivation>(
    fmt: GgufBlockFormat,
    use_strided: bool,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    w_off: i64,
    stride_y: i64,
    y_ptr: *const c_void,
    dst_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) -> i32 { unsafe {
    match T::KIND {
        ElementKind::F32 => {
            if !use_strided {
                dispatch_f32_contig(fmt, ncols, nrows, w_ptr, y_ptr, dst_ptr, stream_ptr)
            } else {
                dispatch_f32_strided(fmt, ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, stream_ptr)
            }
        }
        ElementKind::F16 => {
            if !use_strided {
                dispatch_f16_contig(fmt, ncols, nrows, w_ptr, y_ptr, dst_ptr, stream_ptr)
            } else {
                dispatch_f16_strided(fmt, ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, stream_ptr)
            }
        }
        ElementKind::Bf16 => {
            if !use_strided {
                dispatch_bf16_contig(fmt, ncols, nrows, w_ptr, y_ptr, dst_ptr, stream_ptr)
            } else {
                dispatch_bf16_strided(fmt, ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, stream_ptr)
            }
        }
        _ => unreachable!("GgufMmvqActivation is sealed to f32 / f16 / bf16"),
    }
}}

// ---- f32 dispatch (legacy contig + strided) ------------------------------

unsafe fn dispatch_f32_contig(
    fmt: GgufBlockFormat,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    y_ptr: *const c_void,
    dst_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) -> i32 { unsafe {
    let ws = core::ptr::null_mut();
    match fmt {
        GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_0_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_1_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_0_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_1_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_mmvq_q3_K_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_K_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_K_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_mmvq_q6_K_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8K => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_K_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`.
        // Status 3 maps to `Error::Unsupported` via the shared
        // `map_status` translator in `quantize::mod`.
        _ => 3,
    }
}}

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_f32_strided(
    fmt: GgufBlockFormat,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    w_off: i64,
    stride_y: i64,
    y_ptr: *const c_void,
    dst_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) -> i32 { unsafe {
    let ws = core::ptr::null_mut();
    match fmt {
        GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_0_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_1_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_0_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_1_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_mmvq_q3_K_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_K_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_K_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_mmvq_q6_K_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8K => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_K_actstrided_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`.
        _ => 3,
    }
}}

// ---- f16 dispatch (Phase 18.1) -------------------------------------------

unsafe fn dispatch_f16_contig(
    fmt: GgufBlockFormat,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    y_ptr: *const c_void,
    dst_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) -> i32 { unsafe {
    let ws = core::ptr::null_mut();
    match fmt {
        GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_0_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_1_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_0_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_1_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_mmvq_q3_K_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_K_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_K_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_mmvq_q6_K_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8K => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_K_f16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`.
        _ => 3,
    }
}}

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_f16_strided(
    fmt: GgufBlockFormat,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    w_off: i64,
    stride_y: i64,
    y_ptr: *const c_void,
    dst_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) -> i32 { unsafe {
    let ws = core::ptr::null_mut();
    match fmt {
        GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_0_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_1_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_0_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_1_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_mmvq_q3_K_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_K_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_K_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_mmvq_q6_K_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8K => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_K_actstrided_f16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`.
        _ => 3,
    }
}}

// ---- bf16 dispatch (Phase 18.1) ------------------------------------------

unsafe fn dispatch_bf16_contig(
    fmt: GgufBlockFormat,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    y_ptr: *const c_void,
    dst_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) -> i32 { unsafe {
    let ws = core::ptr::null_mut();
    match fmt {
        GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_0_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_1_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_0_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_1_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_mmvq_q3_K_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_K_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_K_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_mmvq_q6_K_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8K => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_K_bf16_run(
            ncols, nrows, w_ptr, y_ptr, dst_ptr, ws, 0, stream_ptr),
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`.
        _ => 3,
    }
}}

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_bf16_strided(
    fmt: GgufBlockFormat,
    ncols: i32,
    nrows: i32,
    w_ptr: *const c_void,
    w_off: i64,
    stride_y: i64,
    y_ptr: *const c_void,
    dst_ptr: *mut c_void,
    stream_ptr: *mut c_void,
) -> i32 { unsafe {
    let ws = core::ptr::null_mut();
    match fmt {
        GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_0_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_1_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_0_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_1_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_mmvq_q3_K_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_K_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_K_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_mmvq_q6_K_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        GgufBlockFormat::Q8K => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_K_actstrided_bf16_run(
            ncols, nrows, w_ptr, w_off, stride_y, y_ptr, dst_ptr, ws, 0, stream_ptr),
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`.
        _ => 3,
    }
}}

/// Natural byte alignment of a packed GGUF block struct, derived from
/// the layouts in `baracuda_gguf.cuh`:
///
/// * `block_q4_0` / `block_q5_0` / `block_q8_0` — `half d` first
///   → alignment 2.
/// * `block_q3_K` / `block_q6_K` — `half d` (after `uint8_t[]` arrays
///   that are 1-aligned) → alignment 2.
/// * `block_q4_1` / `block_q5_1` / `block_q2_K` / `block_q4_K` /
///   `block_q5_K` — contain `half2 dm` → alignment 4.
/// * `block_q8_K` — `float d` first + `int16_t bsums[]` → alignment 4.
///
/// Used only by the Phase 15.1 debug-build alignment guard for
/// `w_start_byte_offset`. Release builds elide the call entirely.
#[cfg(debug_assertions)]
#[inline]
fn required_alignment(format: GgufBlockFormat) -> i64 {
    match format {
        // `half`-first or `half`-only fp scale → 2-byte aligned.
        GgufBlockFormat::Q4_0
        | GgufBlockFormat::Q5_0
        | GgufBlockFormat::Q8_0
        | GgufBlockFormat::Q3K
        | GgufBlockFormat::Q6K => 2,
        // `half2 dm` (4-byte aligned) or `float d` → 4-byte aligned.
        GgufBlockFormat::Q4_1
        | GgufBlockFormat::Q5_1
        | GgufBlockFormat::Q2K
        | GgufBlockFormat::Q4K
        | GgufBlockFormat::Q5K
        | GgufBlockFormat::Q8K => 4,
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`. The
        // most conservative (largest) alignment is used here so a
        // newly-added format won't silently underestimate the
        // requirement; the asserting caller will then fail loudly if
        // the pointer isn't actually aligned.
        _ => 4,
    }
}

fn build_sku(_block_format: GgufBlockFormat, act_kind: ElementKind) -> KernelSku {
    KernelSku {
        category: OpCategory::Quantization,
        op: QuantizeKind::GgufMmvq as u16,
        // Element on the SKU records the activation / output dtype.
        // f32 (legacy), f16 / bf16 (Phase 18.1).
        element: act_kind,
        aux_element: Some(ElementKind::U8),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee: PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // MMVQ uses warp-shuffle reduction (no atomicAdd), so it's
            // bit-stable on identical hardware. Determinism follows from
            // the fixed thread-block geometry per (nrows, ncols).
            bit_stable_on_same_hardware: true,
            deterministic: true,
        },
    }
}
