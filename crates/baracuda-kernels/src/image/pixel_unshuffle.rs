//! `pixel_unshuffle` plan — Category T.
//!
//! `[N, C, H·r, W·r] → [N, C·r², H, W]`. Inverse permutation of
//! [`crate::image::PixelShufflePlan`]; the two are each other's
//! backward.
//!
//! Dtype coverage: `f32, f64, f16, bf16`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for `pixel_unshuffle`.
#[derive(Copy, Clone, Debug)]
pub struct PixelUnshuffleDescriptor {
    /// Batch.
    pub n: i32,
    /// Input channel count (output has `c * r * r`).
    pub c: i32,
    /// Output height (input is `h * r`).
    pub h: i32,
    /// Output width (input is `w * r`).
    pub w: i32,
    /// Downscale factor.
    pub downscale_factor: i32,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for `pixel_unshuffle`.
pub struct PixelUnshuffleArgs<'a, T: Element> {
    /// Input `[N, C, H * r, W * r]`.
    pub input: TensorRef<'a, T, 4>,
    /// Output `[N, C * r * r, H, W]`.
    pub output: TensorMut<'a, T, 4>,
}

/// `pixel_unshuffle` plan.
///
/// `[N, C, H·r, W·r] → [N, C·r², H, W]` — inverse of
/// [`PixelShufflePlan`](crate::PixelShufflePlan).
///
/// **When to use**: forward pixel-unshuffle. Also serves as the BW
/// of `pixel_shuffle` (and vice versa).
///
/// **Dtypes**: `{f32, f64, f16, bf16}` (memory-bound).
///
/// **Shape limits**: rank-4 NCHW; H and W must be multiples of `r`;
/// `r ≥ 1`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact.
pub struct PixelUnshufflePlan<T: Element> {
    desc: PixelUnshuffleDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> PixelUnshufflePlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &PixelUnshuffleDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PixelUnshufflePlan: descriptor element != T",
            ));
        }
        if desc.n < 0 || desc.c < 0 || desc.h < 0 || desc.w < 0 || desc.downscale_factor <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PixelUnshufflePlan: extents must be non-negative; r > 0",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::PixelUnshufflePlan: only `f32`, `f64`, `f16`, `bf16` wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: T::KIND,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Image,
            op: ImageKind::PixelUnshuffle as u16,
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
    pub fn can_implement(&self, args: &PixelUnshuffleArgs<'_, T>) -> Result<()> {
        let r = self.desc.downscale_factor;
        if args.input.shape != [self.desc.n, self.desc.c, self.desc.h * r, self.desc.w * r] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PixelUnshufflePlan: input shape must be [N, C, H*r, W*r]",
            ));
        }
        let cout = self.desc.c * r * r;
        if args.output.shape != [self.desc.n, cout, self.desc.h, self.desc.w] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PixelUnshufflePlan: output shape must be [N, C*r*r, H, W]",
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
        args: PixelUnshuffleArgs<'_, T>,
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
            self.desc.downscale_factor,
        );
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_unshuffle_f32_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_unshuffle_f64_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_unshuffle_f16_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pixel_unshuffle_bf16_run(
                    n, c, h, w, r, input_ptr, out_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PixelUnshufflePlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
