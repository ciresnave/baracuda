//! `bincount` plan — count occurrences of each integer in `x`.
//!
//! Trailblazer dtype coverage: `i32, i64` input → `i32` counts.
//! No BW (set-valued / non-differentiable).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::histogram::build_atomic_sku;
use super::map_status;

/// Descriptor for a `bincount` op.
#[derive(Copy, Clone, Debug)]
pub struct BincountDescriptor {
    /// Total input element count.
    pub numel: i64,
    /// Number of bins. Caller pre-computes `max(max(x)+1, minlength)`.
    pub num_bins: i32,
    /// Input element type.
    pub element: ElementKind,
}

/// Args bundle for a `bincount` launch.
pub struct BincountArgs<'a, T: Element> {
    /// Input `[numel]`.
    pub input: TensorRef<'a, T, 1>,
    /// Output counts `[num_bins]`. Launcher zeros it.
    pub output: TensorMut<'a, i32, 1>,
}

/// `bincount` plan.
///
/// Counts occurrences of each integer in a flat input (PyTorch
/// `torch.bincount`).
///
/// **When to use**: forward bincount over int arrays. No BW.
///
/// **Dtypes**: input `{i32, i64}`; output always `i32` counts.
///
/// **Shape limits**: input flat `[numel]`; output `[num_bins]`.
/// Caller pre-computes `num_bins = max(max(x)+1, minlength)`.
/// Negative input values are skipped.
///
/// **Workspace**: none. Launcher zeros `output`.
///
/// **Precision guarantee**: **non-deterministic** — atomic
/// accumulation order varies. Counts are data-determined.
pub struct BincountPlan<T: Element> {
    desc: BincountDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> BincountPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &BincountDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BincountPlan: descriptor element != type parameter T",
            ));
        }
        if desc.numel < 0 || desc.num_bins < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BincountPlan: numel / num_bins must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::I32 | ElementKind::I64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BincountPlan: today only i32 / i64 input wired",
            ));
        }
        let sku = build_atomic_sku::<T>(SortKind::Bincount);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &BincountArgs<'_, T>) -> Result<()> {
        if (args.input.shape[0] as i64) != self.desc.numel {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BincountPlan: input shape[0] != descriptor numel",
            ));
        }
        if args.output.shape != [self.desc.num_bins] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BincountPlan: output shape != [num_bins]",
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
        args: BincountArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.num_bins == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_bincount_i32_run(
                    self.desc.numel,
                    self.desc.num_bins,
                    in_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_bincount_i64_run(
                    self.desc.numel,
                    self.desc.num_bins,
                    in_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BincountPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
