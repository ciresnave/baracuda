//! GGUF dequantize plan — unpacks a GGUF-packed weight buffer into a
//! dense f32 tensor.
//!
//! Block format is selected at descriptor time via [`GgufBlockFormat`];
//! all eleven block formats are supported. Output dtype today is f32
//! only (f16 output deferred to a follow-up milestone).

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, GgufBlockFormat, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace, U8,
};

use crate::quantize::map_status;

/// Descriptor for a GGUF dequantize op.
#[derive(Copy, Clone, Debug)]
pub struct GgufDequantizeDescriptor {
    /// Number of FP elements in the output tensor. Must be a multiple
    /// of `block_format.block_size()` (32 for type-0/1, 256 for k-quants).
    pub numel: i64,
    /// GGUF block format of the packed input.
    pub block_format: GgufBlockFormat,
}

/// Args bundle for a GGUF dequantize launch.
///
/// The input weight buffer is carried as a `TensorRef<u8, 1>` over the
/// raw packed bytes — its `shape[0]` must equal
/// `(numel / block_size) * block_format.type_size()`.
pub struct GgufDequantizeArgs<'a> {
    /// Packed GGUF weight bytes.
    pub input: TensorRef<'a, U8, 1>,
    /// Output f32 tensor, length `numel`.
    pub output: TensorMut<'a, f32, 1>,
}

/// `gguf_dequantize` plan.
pub struct GgufDequantizePlan {
    desc: GgufDequantizeDescriptor,
    sku: KernelSku,
}

impl GgufDequantizePlan {
    /// Pick a kernel for `desc`. Errors if `numel` is not a multiple of
    /// the block size.
    pub fn select(
        _stream: &Stream,
        desc: &GgufDequantizeDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "GgufDequantizePlan: numel must be non-negative",
            ));
        }
        let bs = desc.block_format.block_size() as i64;
        if desc.numel % bs != 0 {
            return Err(Error::InvalidProblem(
                "GgufDequantizePlan: numel must be a multiple of the block size",
            ));
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(desc.block_format, QuantizeKind::GgufDequantize),
        })
    }

    /// Validate args at run time.
    pub fn can_implement(&self, args: &GgufDequantizeArgs<'_>) -> Result<()> {
        if args.output.shape != [self.desc.numel as i32] {
            return Err(Error::InvalidProblem(
                "GgufDequantizePlan: output shape != [numel]",
            ));
        }
        let blocks = self.desc.numel / self.desc.block_format.block_size() as i64;
        let expected_bytes = blocks * self.desc.block_format.type_size() as i64;
        if args.input.shape != [expected_bytes as i32] {
            return Err(Error::InvalidProblem(
                "GgufDequantizePlan: input byte length != blocks * type_size",
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
        args: GgufDequantizeArgs<'_>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let y_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let numel = self.desc.numel;

        let status = unsafe {
            match self.desc.block_format {
                GgufBlockFormat::Q4_0 => baracuda_kernels_sys::baracuda_kernels_dequantize_q4_0_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q4_1 => baracuda_kernels_sys::baracuda_kernels_dequantize_q4_1_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q5_0 => baracuda_kernels_sys::baracuda_kernels_dequantize_q5_0_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q5_1 => baracuda_kernels_sys::baracuda_kernels_dequantize_q5_1_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q8_0 => baracuda_kernels_sys::baracuda_kernels_dequantize_q8_0_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q2K => baracuda_kernels_sys::baracuda_kernels_dequantize_q2_K_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q3K => baracuda_kernels_sys::baracuda_kernels_dequantize_q3_K_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q4K => baracuda_kernels_sys::baracuda_kernels_dequantize_q4_K_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q5K => baracuda_kernels_sys::baracuda_kernels_dequantize_q5_K_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q6K => baracuda_kernels_sys::baracuda_kernels_dequantize_q6_K_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                GgufBlockFormat::Q8K => baracuda_kernels_sys::baracuda_kernels_dequantize_q8_K_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
            }
        };
        map_status(status)
    }
}

pub(crate) fn build_sku(_block_format: GgufBlockFormat, op: QuantizeKind) -> KernelSku {
    // `_block_format` is kept on the signature for future-proofing —
    // a follow-up milestone will key the SKU off the block format when
    // f16 output is added (different math-precision / element pairs).
    KernelSku {
        category: OpCategory::Quantization,
        op: op as u16,
        // Element on the SKU records the OUTPUT FP dtype (f32 today).
        element: ElementKind::F32,
        // Aux records "byte-packed quants" via U8.
        aux_element: Some(ElementKind::U8),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee: PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Bit-stable on same hardware: dequant is pure arithmetic
            // on values loaded from device memory; no atomicAdd.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        },
    }
}
