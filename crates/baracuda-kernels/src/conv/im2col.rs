//! Im2Col (2-D) — bespoke sliding-window unfold (Phase 19.3).
//!
//! `torch.nn.functional.unfold` equivalent for 2-D NCHW inputs. Turns
//! a 2-D convolution input into a column-shaped matrix where each
//! column corresponds to one output cell's input window — the
//! building block for conv-via-im2col-and-GEMM lowering.
//!
//! Input shape: `[N, C, H_in, W_in]`.
//! Output shape: `[N, C·kh·kw, h_out·w_out]` with
//!
//! ```text
//! h_out = (H_in + 2·pad_h - dilation_h·(kh-1) - 1) / stride_h + 1
//! w_out = (W_in + 2·pad_w - dilation_w·(kw-1) - 1) / stride_w + 1
//! ```
//!
//! Cells whose source coordinate lies in the pad region are written
//! as 0 (zero-pad convention, matches PyTorch's `unfold`).
//!
//! **Why bespoke**: cuDNN doesn't expose im2col as a public op. Fuel
//! needs this for their conv-via-GEMM fallback path (Conv2d ≡
//! im2col → GEMM → reshape) and for the conv-backward filter-gradient
//! when cuDNN's `cudnnConvolutionBackwardFilter` is unsuitable.
//!
//! **Dtypes**: `f16, bf16, f32, f64`.
//!
//! **No backward**: 2-D col2im is intentionally not provided here —
//! Fuel routes the conv-2d filter-gradient through cuDNN's
//! `cudnnConvolutionBackwardFilter` exposed by [`super::Conv2dPlan`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ConvKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for `Im2Col` (2-D).
///
/// Mirrors PyTorch's `nn.Unfold(kernel_size, dilation, padding, stride)`
/// shape. Output extents are computed via [`Im2ColPlan::output_dims`].
#[derive(Copy, Clone, Debug)]
pub struct Im2ColDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input height `H_in`.
    pub h_in: i32,
    /// Input width `W_in`.
    pub w_in: i32,
    /// Kernel height `kh`.
    pub kh: i32,
    /// Kernel width `kw`.
    pub kw: i32,
    /// Stride along the height axis.
    pub stride_h: i32,
    /// Stride along the width axis.
    pub stride_w: i32,
    /// Zero-padding rows on each side of the input height axis.
    pub pad_h: i32,
    /// Zero-padding columns on each side of the input width axis.
    pub pad_w: i32,
    /// Dilation along the height axis.
    pub dilation_h: i32,
    /// Dilation along the width axis.
    pub dilation_w: i32,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for an `Im2Col` (2-D) launch.
#[derive(Debug)]
pub struct Im2ColArgs<'a, T: Element> {
    /// Input activations `[N, C, H_in, W_in]` NCHW contiguous.
    pub input: TensorRef<'a, T, 4>,
    /// Output column matrix `[N, C·kh·kw, h_out·w_out]` contiguous.
    pub output: TensorMut<'a, T, 3>,
}

/// 2-D im2col plan (bespoke) — forward only.
///
/// **When to use**: any sliding-window unfold over an NCHW 2-D input
/// (e.g. lowering a 2-D conv into a single GEMM). Each output column
/// is one input window flattened in `(c, ki, kj)` row-major order;
/// columns are flattened in `(oh, ow)` row-major order.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`.
///
/// **Precision guarantee**: bit-exact, deterministic — pure scatter-
/// free copy (one thread per output cell, exactly one source per
/// cell). No accumulation, no atomicAdd.
///
/// **Workspace**: none.
#[derive(Debug)]
pub struct Im2ColPlan<T: Element> {
    desc: Im2ColDescriptor,
    h_out: i32,
    w_out: i32,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> Im2ColPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Im2ColDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::Im2ColPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::Im2ColPlan: dtype must be f32 / f64 / f16 / bf16",
            ));
        }
        let (h_out, w_out) = validate_geometry(desc)?;
        let sku = build_im2col_sku::<T>(ConvKind::Im2Col2d);
        Ok(Self {
            desc: *desc,
            h_out,
            w_out,
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

    /// Computed `(h_out, w_out)` output spatial extents.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32) {
        (self.h_out, self.w_out)
    }

    /// Run the im2col forward. Writes `output[n, c·kh·kw + ki·kw + kj,
    /// oh·w_out + ow] = input[n, c, oh·stride_h + ki·dilation_h -
    /// pad_h, ow·stride_w + kj·dilation_w - pad_w]` (with 0 for
    /// out-of-bounds reads).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Im2ColArgs<'_, T>,
    ) -> Result<()> {
        crate::view_check::check_dense(&args.input)?;
        crate::view_check::check_dense(&args.output)?;
        self.check_args(&args)?;
        let input_ptr = args.input.data().as_raw().0 as *const c_void;
        let output_ptr = args.output.data().as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw();
        let d = &self.desc;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_2d_f32_run(
                    d.batch,
                    d.channels,
                    d.h_in,
                    d.w_in,
                    self.h_out,
                    self.w_out,
                    d.kh,
                    d.kw,
                    d.stride_h,
                    d.stride_w,
                    d.pad_h,
                    d.pad_w,
                    d.dilation_h,
                    d.dilation_w,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_2d_f64_run(
                    d.batch,
                    d.channels,
                    d.h_in,
                    d.w_in,
                    self.h_out,
                    self.w_out,
                    d.kh,
                    d.kw,
                    d.stride_h,
                    d.stride_w,
                    d.pad_h,
                    d.pad_w,
                    d.dilation_h,
                    d.dilation_w,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_2d_f16_run(
                    d.batch,
                    d.channels,
                    d.h_in,
                    d.w_in,
                    self.h_out,
                    self.w_out,
                    d.kh,
                    d.kw,
                    d.stride_h,
                    d.stride_w,
                    d.pad_h,
                    d.pad_w,
                    d.dilation_h,
                    d.dilation_w,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_im2col_2d_bf16_run(
                    d.batch,
                    d.channels,
                    d.h_in,
                    d.w_in,
                    self.h_out,
                    self.w_out,
                    d.kh,
                    d.kw,
                    d.stride_h,
                    d.stride_w,
                    d.pad_h,
                    d.pad_w,
                    d.dilation_h,
                    d.dilation_w,
                    input_ptr,
                    output_ptr,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::Im2ColPlan: unexpected dtype after select()",
                ));
            }
        };
        map_im2col_status(status)
    }

    fn check_args(&self, args: &Im2ColArgs<'_, T>) -> Result<()> {
        let in_shape = [
            self.desc.batch,
            self.desc.channels,
            self.desc.h_in,
            self.desc.w_in,
        ];
        let col_rows = self.desc.channels * self.desc.kh * self.desc.kw;
        let spatial = self.h_out * self.w_out;
        let out_shape = [self.desc.batch, col_rows, spatial];
        if args.input.shape() != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Im2ColPlan: input shape != [N, C, H_in, W_in]",
            ));
        }
        if args.output.shape() != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Im2ColPlan: output shape != [N, C·kh·kw, h_out·w_out]",
            ));
        }
        Ok(())
    }
}

/// Geometry validation shared by [`Im2ColPlan::select`] and the host tests: positive
/// extents, no empty output, and (KISS-OPS-6.13-0011) a real in-bounds tap in every output
/// window on both axes. The attribute-only bound `2 * pad <= dilated extent` is not enough
/// once `dilation > 1`: `h_in = 1, kh = 2, dilation_h = 2, pad_h = 1` has one window whose taps
/// lie at -1 and 1.
fn validate_geometry(desc: &Im2ColDescriptor) -> Result<(i32, i32)> {
    if desc.batch <= 0 || desc.channels <= 0 || desc.h_in <= 0 || desc.w_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2ColPlan: input shape extents must be > 0",
        ));
    }
    if desc.kh <= 0 || desc.kw <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2ColPlan: kernel extents must be > 0",
        ));
    }
    if desc.stride_h <= 0 || desc.stride_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2ColPlan: stride must be > 0",
        ));
    }
    if desc.dilation_h <= 0 || desc.dilation_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2ColPlan: dilation must be > 0",
        ));
    }
    if desc.pad_h < 0 || desc.pad_w < 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2ColPlan: padding must be >= 0",
        ));
    }
    let (h_out, w_out) = compute_im2col_2d_dims(desc);
    if h_out <= 0 || w_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2ColPlan: computed output dims <= 0 — \
             padding / stride / dilation combination produces an empty output",
        ));
    }
    let tap =
        |n, pad, k, stride, dil| crate::window_axis::every_window_has_tap(n, pad, k, stride, dil);
    if !tap(
        desc.h_in,
        desc.pad_h,
        desc.kh,
        desc.stride_h,
        desc.dilation_h,
    ) || !tap(
        desc.w_in,
        desc.pad_w,
        desc.kw,
        desc.stride_w,
        desc.dilation_w,
    ) {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Im2ColPlan: an output window has no in-bounds tap \
             (KISS-OPS-6.13-0011) — padding / dilation / extent combination",
        ));
    }
    Ok((h_out, w_out))
}

/// Standard `(H_out, W_out)` formula — matches the PyTorch / cuDNN
/// convention used by [`super::Conv2dPlan`].
#[inline]
pub(super) fn compute_im2col_2d_dims(d: &Im2ColDescriptor) -> (i32, i32) {
    // Floor semantics, `0` when no window fits (KISS-OPS-6.13-0010).
    let h_out = crate::window_axis::out_extent(d.h_in, d.pad_h, d.kh, d.stride_h, d.dilation_h);
    let w_out = crate::window_axis::out_extent(d.w_in, d.pad_w, d.kw, d.stride_w, d.dilation_w);
    (h_out, w_out)
}

pub(super) fn build_im2col_sku<T: Element>(op: ConvKind) -> KernelSku {
    let math_precision = match T::KIND {
        ElementKind::F64 => MathPrecision::F64,
        ElementKind::F16 => MathPrecision::F16,
        ElementKind::Bf16 => MathPrecision::Bf16,
        _ => MathPrecision::F32,
    };
    let accumulator = match T::KIND {
        ElementKind::F64 => ElementKind::F64,
        _ => ElementKind::F32,
    };
    // im2col FW kernels are pure copies — bit-exact + deterministic.
    // The col2im 1-D BW uses atomicAdd scatter, so the BW plan
    // weakens `bit_stable_on_same_hardware` in its own SKU builder.
    let precision_guarantee = PrecisionGuarantee {
        math_precision,
        accumulator,
        bit_stable_on_same_hardware: !matches!(op, ConvKind::Col2Im1d),
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Convolution,
        op: op as u16,
        element: T::KIND,
        aux_element: None,
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}

/// Status-code mapper for the im2col family. Mirrors the lp_pool /
/// indexing / sort mappers.
pub(crate) fn map_im2col_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys::im2col reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys::im2col reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::KernelInternal(n)),
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;

    fn desc(h_in: i32, kh: i32, dil: i32, pad: i32) -> Im2ColDescriptor {
        Im2ColDescriptor {
            batch: 1,
            channels: 1,
            h_in,
            w_in: 8,
            kh,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: pad,
            pad_w: 0,
            dilation_h: dil,
            dilation_w: 1,
            element: ElementKind::F32,
        }
    }

    #[test]
    fn declines_a_window_with_no_real_tap() {
        // KISS-OPS-6.13-0011 counterexample: one window, taps at -1 and 1 over h_in = 1.
        let bad = desc(1, 2, 2, 1);
        assert_eq!(
            compute_im2col_2d_dims(&bad).0,
            1,
            "the extent alone is not the problem"
        );
        assert!(validate_geometry(&bad).is_err());
        // Control: over h_in = 4 every window has a tap, and the geometry validates.
        assert_eq!(validate_geometry(&desc(4, 2, 2, 1)).unwrap().0, 4);
    }

    #[test]
    fn declines_an_axis_where_no_window_fits() {
        // h_in = 3, kh = 5, stride 1, no pad: no window fits. (The truncating form also got
        // this one right; the shape it got wrong had a negative numerator smaller than the
        // stride, covered in `window_axis` and the 1-D test.)
        assert_eq!(compute_im2col_2d_dims(&desc(3, 5, 1, 0)).0, 0);
        assert!(validate_geometry(&desc(3, 5, 1, 0)).is_err());
    }
}
