//! `pixel_shuffle` plan — Category T.
//!
//! `[N, C·r², H, W] → [N, C, H·r, W·r]`. Pure index permutation —
//! no arithmetic, output is bit-exact across every wired dtype.
//!
//! Its backward is [`crate::image::PixelUnshufflePlan`] (the two are
//! each other's inverse).
//!
//! Dtype coverage: `f32, f64, f16, bf16` (memory-bound).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for `pixel_shuffle`.
#[derive(Copy, Clone, Debug)]
pub struct PixelShuffleDescriptor {
    /// Batch.
    pub n: i32,
    /// Output channel count (input has `c * r * r`).
    pub c: i32,
    /// Input height (output is `h * r`).
    pub h: i32,
    /// Input width (output is `w * r`).
    pub w: i32,
    /// Upscale factor.
    pub upscale_factor: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for `pixel_shuffle`.
pub struct PixelShuffleArgs<'a, T: Element> {
    /// Input `[N, C * r * r, H, W]`.
    pub input: TensorRef<'a, T, 4>,
    /// Output `[N, C, H * r, W * r]`.
    pub output: TensorMut<'a, T, 4>,
}

/// `pixel_shuffle` plan.
///
/// `[N, C·r², H, W] → [N, C, H·r, W·r]` — sub-pixel-conv rearrange
/// (PyTorch `F.pixel_shuffle`). Pure index permutation.
///
/// **When to use**: super-resolution / efficient upsample. Its BW
/// is [`PixelUnshufflePlan`](crate::PixelUnshufflePlan) — the two
/// are each other's exact inverse.
///
/// **Dtypes**: `{f32, f64, f16, bf16}` (memory-bound; arithmetic-
/// free so all FP dtypes work uniformly).
///
/// **Shape limits**: rank-4 NCHW; input channel count must equal
/// `c * r * r`; `r ≥ 1`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact at
/// every dtype. No arithmetic.
pub struct PixelShufflePlan<T: Element> {
    desc: PixelShuffleDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> PixelShufflePlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &PixelShuffleDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PixelShufflePlan: descriptor element != T",
            ));
        }
        if desc.n < 0 || desc.c < 0 || desc.h < 0 || desc.w < 0 || desc.upscale_factor <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PixelShufflePlan: extents must be non-negative; r > 0",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::PixelShufflePlan: only `f32`, `f64`, `f16`, `bf16` wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            // Pure copy — bit-stable, deterministic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Image,
            op: ImageKind::PixelShuffle as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &PixelShuffleArgs<'_, T>) -> Result<()> {
        let r = self.desc.upscale_factor;
        let cin = self.desc.c * r * r;
        if args.input.shape != [self.desc.n, cin, self.desc.h, self.desc.w] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PixelShufflePlan: input shape must be [N, C*r*r, H, W]",
            ));
        }
        if args.output.shape != [self.desc.n, self.desc.c, self.desc.h * r, self.desc.w * r] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PixelShufflePlan: output shape must be [N, C, H*r, W*r]",
            ));
        }
        Ok(())
    }

    /// Workspace (zero).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity.
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
        args: PixelShuffleArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.output.numel() == 0 {
            return Ok(());
        }
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let (n, c, h, w, r) = (
            self.desc.n,
            self.desc.c,
            self.desc.h,
            self.desc.w,
            self.desc.upscale_factor,
        );
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_shuffle_f32_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_shuffle_f64_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_shuffle_f16_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_shuffle_bf16_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PixelShufflePlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
