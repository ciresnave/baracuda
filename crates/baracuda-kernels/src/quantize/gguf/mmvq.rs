//! GGUF MMVQ plan — fused dequant + matrix-vector multiply.
//!
//! `out[r] = Σ_c W_q[r, c] · y[c]`, where `W_q` is GGUF-packed
//! (one block-row of `packed_cols_bytes` per matrix row) and `y` /
//! `out` are dense FP32. This is the inference-time "decode-step"
//! matmul used by llama.cpp on GGUF weights.
//!
//! `GgufBlockFormat::Q8K` is NOT supported — llama.cpp / Fuel reserve
//! Q8_K as a CPU-side intermediate and ship no MMVQ kernel for it.
//! Plan `select()` returns `Error::Unsupported` for that case.

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, GgufBlockFormat, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace, U8,
};

use crate::quantize::map_status;

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
}

/// Args bundle for a GGUF MMVQ launch.
pub struct GgufMmvqArgs<'a> {
    /// Packed GGUF weight bytes. Length must equal
    /// `nrows * (ncols / block_size) * type_size`.
    pub weight: TensorRef<'a, U8, 1>,
    /// f32 activation vector `[ncols]`.
    pub activation: TensorRef<'a, f32, 1>,
    /// f32 output vector `[nrows]`.
    pub output: TensorMut<'a, f32, 1>,
}

/// `gguf_mmvq` plan.
pub struct GgufMmvqPlan {
    desc: GgufMmvqDescriptor,
    sku: KernelSku,
}

impl GgufMmvqPlan {
    /// Pick a kernel for `desc`. Errors on `Q8_K` (no upstream MMVQ),
    /// non-positive dims, or `ncols` that doesn't tile to the block size.
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
                "GgufMmvqPlan: Q8_K MMVQ is not supported (upstream llama.cpp ships no Q8_K MMVQ kernel)",
            ));
        }
        let bs = desc.block_format.block_size() as i32;
        if desc.ncols % bs != 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: ncols must be a multiple of the block size",
            ));
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(desc.block_format),
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &GgufMmvqArgs<'_>) -> Result<()> {
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
        if args.weight.shape != [expected_bytes as i32] {
            return Err(Error::InvalidProblem(
                "GgufMmvqPlan: weight byte length != nrows * blocks_per_row * type_size",
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
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: GgufMmvqArgs<'_>,
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

        let status = unsafe {
            match self.desc.block_format {
                GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_0_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_1_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_0_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_1_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_mmvq_q3_K_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_mmvq_q4_K_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_mmvq_q5_K_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_mmvq_q6_K_run(
                    ncols, nrows, w_ptr, y_ptr, dst_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q8K => {
                    return Err(Error::Unsupported(
                        "GgufMmvqPlan: Q8_K MMVQ is not supported (select should have caught)",
                    ));
                }
            }
        };
        map_status(status)
    }
}

fn build_sku(_block_format: GgufBlockFormat) -> KernelSku {
    KernelSku {
        category: OpCategory::Quantization,
        op: QuantizeKind::GgufMmvq as u16,
        element: ElementKind::F32,
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
