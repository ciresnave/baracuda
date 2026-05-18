//! `grid_sample` FW plan — Category T (2-D).
//!
//! Sample an NCHW input at arbitrary normalized (x, y) coordinates
//! supplied via `grid: [N, OH, OW, 2]`. Trailblazer config matches
//! PyTorch defaults: `mode='bilinear'`, `padding_mode='zeros'`,
//! `align_corners=false`.
//!
//! Trailblazer dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a `grid_sample` op.
#[derive(Copy, Clone, Debug)]
pub struct GridSampleDescriptor {
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
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `grid_sample` launch.
pub struct GridSampleArgs<'a, T: Element> {
    /// Input `[N, C, IH, IW]`. NCHW row-major contiguous.
    pub input: TensorRef<'a, T, 4>,
    /// Sampling grid `[N, OH, OW, 2]` — (x, y) normalized in [-1, 1].
    pub grid: TensorRef<'a, T, 4>,
    /// Output `[N, C, OH, OW]`.
    pub output: TensorMut<'a, T, 4>,
}

/// `grid_sample` plan.
///
/// Sample an NCHW input at arbitrary normalized `(x, y)` coordinates
/// supplied via `grid: [N, OH, OW, 2]` (PyTorch `F.grid_sample`).
/// Trailblazer config matches PyTorch defaults: `mode='bilinear'`,
/// `padding_mode='zeros'`, `align_corners=false`.
///
/// **When to use**: forward 2-D grid sample. Pair with
/// [`GridSampleBackwardPlan`](crate::GridSampleBackwardPlan).
///
/// **Dtypes**: `{f32, f64}`.
///
/// **Shape limits**: rank-4 NCHW input; rank-4 grid `[N, OH, OW, 2]`;
/// output `[N, C, OH, OW]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct GridSamplePlan<T: Element> {
    desc: GridSampleDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> GridSamplePlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &GridSampleDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GridSamplePlan: descriptor element != T",
            ));
        }
        if desc.n < 0 || desc.c < 0 || desc.ih < 0 || desc.iw < 0 || desc.oh < 0 || desc.ow < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GridSamplePlan: all extents must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::GridSamplePlan: only `f32`, `f64` wired",
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
            op: ImageKind::GridSample2d as u16,
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
    pub fn can_implement(&self, args: &GridSampleArgs<'_, T>) -> Result<()> {
        if args.input.shape != [self.desc.n, self.desc.c, self.desc.ih, self.desc.iw] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GridSamplePlan: input shape mismatch",
            ));
        }
        if args.grid.shape != [self.desc.n, self.desc.oh, self.desc.ow, 2] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GridSamplePlan: grid shape must be [N, OH, OW, 2]",
            ));
        }
        if args.output.shape != [self.desc.n, self.desc.c, self.desc.oh, self.desc.ow] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GridSamplePlan: output shape mismatch",
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
        args: GridSampleArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.output.numel() == 0 {
            return Ok(());
        }
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let grid_ptr = args.grid.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_grid_sample_2d_f32_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    input_ptr, grid_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_grid_sample_2d_f64_run(
                    self.desc.n, self.desc.c, self.desc.ih, self.desc.iw,
                    self.desc.oh, self.desc.ow,
                    input_ptr, grid_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GridSamplePlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
