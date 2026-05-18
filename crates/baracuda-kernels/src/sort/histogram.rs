//! `histogram` plan — 1-D uniform-bin atomic-accumulating histogram.
//!
//! Trailblazer dtype coverage: `f32, f64` input → `i32` counts.
//! No BW (set-valued / non-differentiable).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SortKind, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a `histogram` op.
#[derive(Copy, Clone, Debug)]
pub struct HistogramDescriptor {
    /// Total input element count.
    pub numel: i64,
    /// Number of bins.
    pub num_bins: i32,
    /// Lower edge of the range (inclusive).
    pub lo: f64,
    /// Upper edge of the range (inclusive).
    pub hi: f64,
    /// Input element type.
    pub element: ElementKind,
}

/// Args bundle for a `histogram` launch.
pub struct HistogramArgs<'a, T: Element> {
    /// Input `[numel]` (interpreted as a flat 1-D buffer).
    pub input: TensorRef<'a, T, 1>,
    /// Output counts `[num_bins]` (i32). Launcher zeros it.
    pub output: TensorMut<'a, i32, 1>,
}

/// `histogram` plan.
pub struct HistogramPlan<T: Element> {
    desc: HistogramDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> HistogramPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &HistogramDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::HistogramPlan: descriptor element != type parameter T",
            ));
        }
        if desc.numel < 0 || desc.num_bins < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HistogramPlan: numel / num_bins must be non-negative",
            ));
        }
        if !(desc.hi > desc.lo) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HistogramPlan: hi must be > lo",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::HistogramPlan: today only f32 / f64 wired",
            ));
        }
        let sku = build_atomic_sku::<T>(SortKind::Histogram);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &HistogramArgs<'_, T>) -> Result<()> {
        if (args.input.shape[0] as i64) != self.desc.numel {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HistogramPlan: input shape[0] != descriptor numel",
            ));
        }
        if args.output.shape != [self.desc.num_bins] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HistogramPlan: output shape != [num_bins]",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes.
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
        args: HistogramArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.num_bins == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_histogram_f32_run(
                    self.desc.numel,
                    self.desc.num_bins,
                    self.desc.lo,
                    self.desc.hi,
                    in_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_histogram_f64_run(
                    self.desc.numel,
                    self.desc.num_bins,
                    self.desc.lo,
                    self.desc.hi,
                    in_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::HistogramPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}

/// Build SKU for histogram / bincount / unique-family atomic ops.
pub(crate) fn build_atomic_sku<T: Element>(op: SortKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if T::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: ElementKind::I32,
        bit_stable_on_same_hardware: true, // counts are deterministic
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Sorting,
        op: op as u16,
        element: T::KIND,
        aux_element: Some(ElementKind::I32),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
