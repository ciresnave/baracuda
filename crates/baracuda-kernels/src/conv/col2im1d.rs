//! Col2Im1d — inverse of [`super::Im2Col1dPlan`] (Phase 19.3).
//!
//! Scatters a column-shaped matrix back into its NCL source. Used by
//! Fuel for the conv-1d backward filter-gradient path and for any
//! `unfold → … → fold` round-trip.
//!
//! Input shape: `[N, C·kl, l_out]`.
//! Output shape: `[N, C, L_in]`.
//!
//! Parameters mirror [`super::Im2Col1dDescriptor`] plus the original
//! `L_in` (which the col-shape doesn't carry).
//!
//! **Caller must pre-zero the output buffer** before calling `run` —
//! the kernel uses `atomicAdd` scatter to accumulate overlapping
//! window contributions (when `stride_l < kl`). This matches the
//! Phase 16 pool-BW family's convention.
//!
//! **Dtypes**: `f16, bf16, f32, f64`. half / bf16 atomicAdd routes
//! through `baracuda::atomic::add<T>` (the 32-bit-CAS path from
//! Phase 11.3) for universal availability across arch + CUDA-version
//! combinations.
//!
//! **Precision guarantee**: deterministic in the per-thread arithmetic
//! sense, but **not** bit-stable across runs — `atomicAdd` ordering is
//! non-deterministic across launches.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ConvKind, Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::im2col::{build_im2col_sku, map_im2col_status};
use super::im2col1d::{compute_im2col_1d_l_out, validate_im2col_1d, Im2Col1dDescriptor};

/// Descriptor for `Col2Im1d`. The `kl` / `stride_l` / `pad_l` /
/// `dilation_l` / `l_in` fields must match the original
/// [`super::Im2Col1dDescriptor`] used to produce the col-shaped input;
/// `l_out` is derived from those.
#[derive(Copy, Clone, Debug)]
pub struct Col2Im1dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Target output length `L_in` — the input length of the original
    /// `Im2Col1d` pass.
    pub l_in: i32,
    /// Kernel length `kl`.
    pub kl: i32,
    /// Stride along the length axis.
    pub stride_l: i32,
    /// Zero-padding on each side of the length axis.
    pub pad_l: i32,
    /// Dilation along the length axis.
    pub dilation_l: i32,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a `Col2Im1d` launch.
pub struct Col2Im1dArgs<'a, T: Element> {
    /// Input column matrix `[N, C·kl, l_out]` contiguous.
    pub input: TensorRef<'a, T, 3>,
    /// Output NCL tensor `[N, C, L_in]` — **must be pre-zeroed by
    /// the caller** (atomicAdd scatter).
    pub output: TensorMut<'a, T, 3>,
}

/// 1-D col2im plan (bespoke) — inverse of [`super::Im2Col1dPlan`].
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`. half/bf16 atomicAdd via
/// `baracuda::atomic::add<T>`.
///
/// **Pre-zero contract**: the caller is responsible for zeroing the
/// output buffer before invoking [`Self::run`]. The kernel uses
/// `atomicAdd` to accumulate overlapping window cells when `stride <
/// kl`, so a non-zero starting buffer would corrupt the result.
///
/// **Precision guarantee**: deterministic per-thread arithmetic but
/// **not** bit-stable across runs (`atomicAdd` ordering).
///
/// **Workspace**: none.
pub struct Col2Im1dPlan<T: Element> {
    desc: Col2Im1dDescriptor,
    l_out: i32,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> Col2Im1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Col2Im1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        // Reuse the Im2Col1d validation — same field set + same
        // constraints (positive batch/channels/l_in/kl/stride/dilation,
        // non-negative pad).
        let im2col_desc = Im2Col1dDescriptor {
            batch: desc.batch,
            channels: desc.channels,
            l_in: desc.l_in,
            kl: desc.kl,
            stride_l: desc.stride_l,
            pad_l: desc.pad_l,
            dilation_l: desc.dilation_l,
            element: desc.element,
        };
        validate_im2col_1d::<T>(&im2col_desc).map_err(|e| match e {
            Error::Unsupported(_) => Error::Unsupported(
                "baracuda-kernels::Col2Im1dPlan: dtype/descriptor unsupported",
            ),
            Error::InvalidProblem(_) => Error::InvalidProblem(
                "baracuda-kernels::Col2Im1dPlan: invalid problem dimensions",
            ),
            other => other,
        })?;
        let l_out = compute_im2col_1d_l_out(&im2col_desc).map_err(|_| {
            Error::InvalidProblem("baracuda-kernels::Col2Im1dPlan: computed l_out <= 0")
        })?;
        let sku = build_im2col_sku::<T>(ConvKind::Col2Im1d);
        Ok(Self {
            desc: *desc,
            l_out,
            sku,
            _marker: PhantomData,
        })
    }

    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Workspace size in bytes. Always `0`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// `l_out` of the col-shaped input.
    #[inline]
    pub fn input_l_out(&self) -> i32 {
        self.l_out
    }

    /// Run col2im. **Caller must zero `args.output` first.**
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Col2Im1dArgs<'_, T>,
    ) -> Result<()> {
        self.check_args(&args)?;
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let output_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let d = &self.desc;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_col2im_1d_f32_run(
                    d.batch, d.channels, d.l_in, self.l_out,
                    d.kl, d.stride_l, d.pad_l, d.dilation_l,
                    input_ptr, output_ptr, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_col2im_1d_f64_run(
                    d.batch, d.channels, d.l_in, self.l_out,
                    d.kl, d.stride_l, d.pad_l, d.dilation_l,
                    input_ptr, output_ptr, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_col2im_1d_f16_run(
                    d.batch, d.channels, d.l_in, self.l_out,
                    d.kl, d.stride_l, d.pad_l, d.dilation_l,
                    input_ptr, output_ptr, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_col2im_1d_bf16_run(
                    d.batch, d.channels, d.l_in, self.l_out,
                    d.kl, d.stride_l, d.pad_l, d.dilation_l,
                    input_ptr, output_ptr, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::Col2Im1dPlan: unexpected dtype after select()",
                ));
            }
        };
        map_im2col_status(status)
    }

    fn check_args(&self, args: &Col2Im1dArgs<'_, T>) -> Result<()> {
        let in_shape = [self.desc.batch, self.desc.channels * self.desc.kl, self.l_out];
        let out_shape = [self.desc.batch, self.desc.channels, self.desc.l_in];
        if args.input.shape != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Col2Im1dPlan: input shape != [N, C·kl, l_out]",
            ));
        }
        if args.output.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Col2Im1dPlan: output shape != [N, C, L_in]",
            ));
        }
        Ok(())
    }
}
