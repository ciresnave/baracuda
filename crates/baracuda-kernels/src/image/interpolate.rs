//! `interpolate` FW plan — Category T trailblazer.
//!
//! Spatial resample of an NCHW input via bilinear interpolation.
//! Trailblazer mode: `Bilinear2d`. Other modes are reserved on
//! [`InterpolateMode`] and return `Unsupported`.
//!
//! Output shape: `[N, C, OH, OW]` from input `[N, C, IH, IW]`. The
//! coordinate mapping (per PyTorch ATen `UpSample.h`):
//!
//! - `align_corners=false` (PyTorch new-code default):
//!   `scale_h = scale_h.unwrap_or(IH/OH)^-1`, `src_y = (oh + 0.5) * scale_h - 0.5`
//! - `align_corners=true` (PyTorch `nn.Upsample(align_corners=True)`):
//!   `scale_h = scale_h.unwrap_or((IH-1)/(OH-1))^-1`, `src_y = oh * scale_h`
//!
//! `scale_h` / `scale_w` (when `Some`) are interpreted as PyTorch-style
//! SCALE values (output_size / input_size); the kernel uses `1/scale`
//! per output coordinate.
//!
//! Out-of-range samples are clamped to the input boundary (matches
//! PyTorch).
//!
//! Dtype coverage (Phase 21): `f32, f64, f16, bf16`. Half-precision
//! paths cast at load, accumulate in `f32`, cast at store.
//!
//! # Phase 21 breaking change
//!
//! [`InterpolateDescriptor`] gained `align_corners`, `scale_h`, and
//! `scale_w` fields. Pre-Phase-21 callers constructing the struct must
//! supply the new fields. The underlying FFI also took on three new
//! params (`align_corners: i32`, `scale_h_factor: f64`,
//! `scale_w_factor: f64`) — see `baracuda-kernels-sys` rustdoc.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Interpolation mode for [`InterpolatePlan`]. Only [`Self::Bilinear2d`]
/// is wired today; the other variants return `Unsupported`.
///
/// `#[non_exhaustive]` — additional interpolation modes (cubic
/// spline, lanczos, mitchell-netravali, …) may land in future
/// vision-domain phases. Match arms must include a `_ =>` catch-all.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum InterpolateMode {
    /// 2-D bilinear interpolation. Trailblazer.
    Bilinear2d,
    /// 2-D nearest-neighbor — reserved.
    Nearest2d,
    /// 2-D bicubic — reserved.
    Bicubic2d,
    /// 3-D trilinear — reserved.
    Trilinear3d,
    /// 1-D linear — reserved.
    Linear1d,
    /// 2-D area (adaptive average) — reserved.
    Area2d,
}

/// Descriptor for an `interpolate` op.
#[derive(Copy, Clone, Debug)]
pub struct InterpolateDescriptor {
    /// Batch.
    pub n: i32,
    /// Channels.
    pub c: i32,
    /// Input height.
    pub ih: i32,
    /// Input width.
    pub iw: i32,
    /// Output height.
    pub oh: i32,
    /// Output width.
    pub ow: i32,
    /// Interpolation mode.
    pub mode: InterpolateMode,
    /// Value element type. Must match `T::KIND`.
    pub element: ElementKind,
    /// Coordinate alignment mode. `false` matches PyTorch
    /// `F.interpolate` new-code default; `true` matches
    /// `nn.Upsample(align_corners=True)`.
    pub align_corners: bool,
    /// Per-axis SCALE override for height (output_size / input_size).
    /// `None` derives the scale from `(ih, oh)`; `Some(s)` overrides
    /// and the kernel uses `1.0 / s` per output coordinate. Matches
    /// PyTorch's `scale_factor` semantics.
    pub scale_h: Option<f64>,
    /// Per-axis SCALE override for width (output_size / input_size).
    /// `None` derives the scale from `(iw, ow)`; `Some(s)` overrides
    /// and the kernel uses `1.0 / s` per output coordinate. Matches
    /// PyTorch's `scale_factor` semantics.
    pub scale_w: Option<f64>,
}

/// Args bundle for an `interpolate` launch.
pub struct InterpolateArgs<'a, T: Element> {
    /// Input `[N, C, IH, IW]`. NCHW row-major contiguous.
    pub input: TensorRef<'a, T, 4>,
    /// Output `[N, C, OH, OW]`. NCHW row-major contiguous.
    pub output: TensorMut<'a, T, 4>,
}

/// `interpolate` plan.
///
/// Spatial resample of an NCHW input. PyTorch `F.interpolate`.
/// Coordinate mapping: `src = (dst + 0.5) * (src_size / dst_size) - 0.5`
/// (`align_corners=false`); corner samples clamp to the input
/// boundary.
///
/// **When to use**: forward 2-D bilinear resample. Pair with
/// [`InterpolateBackwardPlan`](crate::InterpolateBackwardPlan) for
/// autograd.
///
/// **Dtypes**: `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank-4 NCHW input `[N, C, IH, IW]`; output
/// `[N, C, OH, OW]`; all extents non-negative.
///
/// **Modes**: only `Bilinear2d` is wired in the trailblazer.
/// `Nearest2d` / `Bicubic2d` / `Trilinear3d` / `Linear1d` / `Area2d`
/// are reserved on the enum and return `Unsupported`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable on identical
/// hardware. No atomics on FW.
pub struct InterpolatePlan<T: Element> {
    desc: InterpolateDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> InterpolatePlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &InterpolateDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::InterpolatePlan: descriptor element != type parameter T",
            ));
        }
        if !matches!(desc.mode, InterpolateMode::Bilinear2d) {
            return Err(Error::Unsupported(
                "baracuda-kernels::InterpolatePlan: only Bilinear2d wired in trailblazer",
            ));
        }
        if desc.n < 0 || desc.c < 0 || desc.ih < 0 || desc.iw < 0 || desc.oh < 0 || desc.ow < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InterpolatePlan: all extents must be non-negative",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::InterpolatePlan: only `f32`, `f64`, `f16`, `bf16` wired",
            ));
        }
        // Validate scale factors (positive, finite) when present.
        if let Some(s) = desc.scale_h {
            if !s.is_finite() || s <= 0.0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::InterpolatePlan: scale_h must be positive and finite",
                ));
            }
        }
        if let Some(s) = desc.scale_w {
            if !s.is_finite() || s <= 0.0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::InterpolatePlan: scale_w must be positive and finite",
                ));
            }
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            // Half-precision paths accumulate in f32, then cast.
            accumulator: if matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
                ElementKind::F32
            } else {
                T::KIND
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Image,
            op: ImageKind::InterpolateBilinear2d as u16,
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
    pub fn can_implement(&self, args: &InterpolateArgs<'_, T>) -> Result<()> {
        if args.input.shape != [self.desc.n, self.desc.c, self.desc.ih, self.desc.iw] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InterpolatePlan: input shape mismatch",
            ));
        }
        if args.output.shape != [self.desc.n, self.desc.c, self.desc.oh, self.desc.ow] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::InterpolatePlan: output shape mismatch",
            ));
        }
        let in_numel = args.input.numel();
        let out_numel = args.output.numel();
        if (args.input.data.len() as i64) < in_numel {
            return Err(Error::BufferTooSmall {
                needed: in_numel as usize,
                got: args.input.data.len(),
            });
        }
        if (args.output.data.len() as i64) < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: args.output.data.len(),
            });
        }
        Ok(())
    }

    /// Workspace size (zero).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: InterpolateArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.output.numel() == 0 {
            return Ok(());
        }
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let output_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let ac: i32 = if self.desc.align_corners { 1 } else { 0 };
        // Sentinel: 0.0 = "derive from sizes" on the C side.
        let sh: f64 = self.desc.scale_h.unwrap_or(0.0);
        let sw: f64 = self.desc.scale_w.unwrap_or(0.0);
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_interpolate_bilinear_2d_f32_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    input_ptr, output_ptr,
                    core::ptr::null_mut(), 0,
                    ac, sh, sw,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_interpolate_bilinear_2d_f64_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    input_ptr, output_ptr,
                    core::ptr::null_mut(), 0,
                    ac, sh, sw,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_interpolate_bilinear_2d_f16_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    input_ptr, output_ptr,
                    core::ptr::null_mut(), 0,
                    ac, sh, sw,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_interpolate_bilinear_2d_bf16_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    input_ptr, output_ptr,
                    core::ptr::null_mut(), 0,
                    ac, sh, sw,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::InterpolatePlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
