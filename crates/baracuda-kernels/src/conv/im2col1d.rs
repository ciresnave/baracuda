//! Im2Col1d — bespoke 1-D sliding-window unfold (Phase 19.3).
//!
//! 1-D analog of [`super::Im2ColPlan`]. Turns an NCL convolution
//! input into a column-shaped matrix where each column corresponds
//! to one output cell's input window — building block for
//! conv1d-via-GEMM lowering.
//!
//! Input shape: `[N, C, L_in]`.
//! Output shape: `[N, C·kl, l_out]` with
//!
//! ```text
//! l_out = (L_in + 2·pad_l - dilation_l·(kl-1) - 1) / stride_l + 1
//! ```
//!
//! Cells whose source coordinate lies in the pad region are written
//! as 0 (zero-pad convention).
//!
//! **Dtypes**: `f16, bf16, f32, f64`.
//!
//! Pair with [`super::Col2Im1dPlan`] for the inverse (used by Fuel's
//! conv-1d filter-gradient path).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ConvKind, Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::im2col::{build_im2col_sku, map_im2col_status};

/// Descriptor for `Im2Col1d`.
#[derive(Copy, Clone, Debug)]
pub struct Im2Col1dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input length `L_in`.
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

/// Args bundle for an `Im2Col1d` launch.
#[derive(Debug)]
pub struct Im2Col1dArgs<'a, T: Element> {
    /// Input activations `[N, C, L_in]` NCL contiguous.
    pub input: TensorRef<'a, T, 3>,
    /// Output column matrix `[N, C·kl, l_out]` contiguous.
    pub output: TensorMut<'a, T, 3>,
}

/// 1-D im2col plan (bespoke) — forward only.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`.
///
/// **Precision guarantee**: bit-exact, deterministic — pure scatter-
/// free copy (one thread per output cell). No accumulation.
///
/// **Workspace**: none.
#[derive(Debug)]
pub struct Im2Col1dPlan<T: Element> {
    desc: Im2Col1dDescriptor,
    l_out: i32,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> Im2Col1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Im2Col1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_im2col_1d::<T>(desc)?;
        let l_out = compute_im2col_1d_l_out(desc)?;
        let sku = build_im2col_sku::<T>(ConvKind::Im2Col1d);
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

    /// Computed `l_out` under the configured stride / pad / dilation.
    #[inline]
    pub fn output_length(&self) -> i32 {
        self.l_out
    }

    /// Run the 1-D im2col forward.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Im2Col1dArgs<'_, T>,
    ) -> Result<()> {
        crate::view_check::check_dense(&args.input)?;
        crate::view_check::check_dense(&args.output)?;
        check_args_1d(&self.desc, self.l_out, &args)?;
        let input_ptr = args.input.data().as_raw().0 as *const c_void;
        let output_ptr = args.output.data().as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw();
        let d = &self.desc;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_1d_f32_run(
                    d.batch,
                    d.channels,
                    d.l_in,
                    self.l_out,
                    d.kl,
                    d.stride_l,
                    d.pad_l,
                    d.dilation_l,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_1d_f64_run(
                    d.batch,
                    d.channels,
                    d.l_in,
                    self.l_out,
                    d.kl,
                    d.stride_l,
                    d.pad_l,
                    d.dilation_l,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_1d_f16_run(
                    d.batch,
                    d.channels,
                    d.l_in,
                    self.l_out,
                    d.kl,
                    d.stride_l,
                    d.pad_l,
                    d.dilation_l,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_1d_bf16_run(
                    d.batch,
                    d.channels,
                    d.l_in,
                    self.l_out,
                    d.kl,
                    d.stride_l,
                    d.pad_l,
                    d.dilation_l,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::Im2Col1dPlan: unexpected dtype after select()",
                ));
            }
        };
        map_im2col_status(status)
    }
}

// =============================================================================
// Shared helpers (also reused by the Col2Im1d sibling).
// =============================================================================

pub(super) fn validate_im2col_1d<T: Element>(d: &Im2Col1dDescriptor) -> Result<()> {
    if d.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::Im2Col1dPlan: descriptor.element != T::KIND",
        ));
    }
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::Im2Col1dPlan: dtype must be f32 / f64 / f16 / bf16",
        ));
    }
    if d.batch <= 0 || d.channels <= 0 || d.l_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: batch/channels/l_in must be > 0",
        ));
    }
    if d.kl <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: kl must be > 0",
        ));
    }
    if d.stride_l <= 0 || d.dilation_l <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: stride / dilation must be > 0",
        ));
    }
    if d.pad_l < 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: padding must be >= 0",
        ));
    }
    Ok(())
}

pub(super) fn compute_im2col_1d_l_out(d: &Im2Col1dDescriptor) -> Result<i32> {
    // Floor semantics, `0` when no window fits (KISS-OPS-6.13-0010).
    let l_out = crate::window_axis::out_extent(d.l_in, d.pad_l, d.kl, d.stride_l, d.dilation_l);
    if l_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: computed l_out <= 0",
        ));
    }
    // KISS-OPS-6.13-0011: every output window needs a real in-bounds tap.
    if !crate::window_axis::every_window_has_tap(d.l_in, d.pad_l, d.kl, d.stride_l, d.dilation_l) {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: an output window has no in-bounds tap \
             (KISS-OPS-6.13-0011) — padding / dilation / extent combination",
        ));
    }
    Ok(l_out)
}

fn check_args_1d<T: Element>(
    d: &Im2Col1dDescriptor,
    l_out: i32,
    args: &Im2Col1dArgs<'_, T>,
) -> Result<()> {
    let in_shape = [d.batch, d.channels, d.l_in];
    let out_shape = [d.batch, d.channels * d.kl, l_out];
    if args.input.shape() != in_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: input shape != [N, C, L_in]",
        ));
    }
    if args.output.shape() != out_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2Col1dPlan: output shape != [N, C·kl, l_out]",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod window_tests {
    use super::*;

    fn desc(l_in: i32, kl: i32, stride: i32, pad: i32, dil: i32) -> Im2Col1dDescriptor {
        Im2Col1dDescriptor {
            batch: 1,
            channels: 1,
            l_in,
            kl,
            stride_l: stride,
            pad_l: pad,
            dilation_l: dil,
            element: ElementKind::F32,
        }
    }

    #[test]
    fn declines_a_window_with_no_real_tap() {
        // KISS-OPS-6.13-0011 counterexample (n=1, k=2, d=2, s=1, p=1): one window, taps -1 and 1.
        assert!(compute_im2col_1d_l_out(&desc(1, 2, 1, 1, 2)).is_err());
        // Control: the same attributes over l_in = 4 hold three windows, each with a tap.
        assert_eq!(compute_im2col_1d_l_out(&desc(4, 2, 1, 1, 2)).unwrap(), 4);
    }

    #[test]
    fn declines_when_no_window_fits() {
        // l_in = 3, kl = 4, stride 4: the truncating form reported l_out = 1.
        assert!(compute_im2col_1d_l_out(&desc(3, 4, 4, 0, 1)).is_err());
        assert_eq!(compute_im2col_1d_l_out(&desc(4, 4, 4, 0, 1)).unwrap(), 1);
    }
}
