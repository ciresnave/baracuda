//! `searchsorted` plan — per-query binary search in a 1-D sorted array.
//!
//! `searchsorted(sorted_seq[L], values[N], right) -> output[N]` (i32).
//! `right == false` (default) returns `lower_bound`; `right == true`
//! returns `upper_bound`. PyTorch `torch.searchsorted`.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, i64`. No BW
//! (set-valued / non-differentiable).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SortKind, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a `searchsorted` op.
#[derive(Copy, Clone, Debug)]
pub struct SearchsortedDescriptor {
    /// Number of query values.
    pub num_queries: i64,
    /// Length of the sorted sequence.
    pub len_sorted: i32,
    /// `false` = lower_bound (default); `true` = upper_bound.
    pub right: bool,
    /// Element type of both sorted_seq and values.
    pub element: ElementKind,
}

/// Args bundle for a `searchsorted` launch.
pub struct SearchsortedArgs<'a, T: Element> {
    /// Sorted sequence `[len_sorted]`.
    pub sorted_seq: TensorRef<'a, T, 1>,
    /// Query values `[num_queries]`.
    pub values: TensorRef<'a, T, 1>,
    /// Output positions `[num_queries]` (i32).
    pub output: TensorMut<'a, i32, 1>,
}

/// `searchsorted` plan.
///
/// `searchsorted(sorted_seq[L], values[N], right) -> output[N]`
/// (PyTorch `torch.searchsorted`). `right == false` returns
/// `lower_bound`, `right == true` returns `upper_bound`.
///
/// **When to use**: per-query binary search into a sorted 1-D array.
/// Useful for histogram-binning / quantile-bucketing. No BW.
///
/// **Dtypes**: `{f32, f64, i32, i64}` for both sorted_seq and
/// values; output always `i32`.
///
/// **Shape limits**: sorted_seq `[len_sorted]`; values, output
/// `[num_queries]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. Pure binary
/// search.
pub struct SearchsortedPlan<T: Element> {
    desc: SearchsortedDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SearchsortedPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &SearchsortedDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SearchsortedPlan: descriptor element != type parameter T",
            ));
        }
        if desc.num_queries < 0 || desc.len_sorted < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SearchsortedPlan: num_queries / len_sorted must be \
                 non-negative",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32 | ElementKind::I64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::SearchsortedPlan: today only f32 / f64 / i32 / i64 wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if T::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: ElementKind::I32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Sorting,
            op: SortKind::Searchsorted as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I32),
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
    pub fn can_implement(&self, args: &SearchsortedArgs<'_, T>) -> Result<()> {
        if args.sorted_seq.shape != [self.desc.len_sorted] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SearchsortedPlan: sorted_seq shape != [len_sorted]",
            ));
        }
        if (args.values.shape[0] as i64) != self.desc.num_queries {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SearchsortedPlan: values shape != [num_queries]",
            ));
        }
        if (args.output.shape[0] as i64) != self.desc.num_queries {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SearchsortedPlan: output shape != [num_queries]",
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
        args: SearchsortedArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.num_queries == 0 {
            return Ok(());
        }
        let seq_ptr = args.sorted_seq.data.as_raw().0 as *const c_void;
        let val_ptr = args.values.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let right_flag = if self.desc.right { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_searchsorted_f32_run(
                    self.desc.num_queries,
                    self.desc.len_sorted,
                    right_flag,
                    seq_ptr,
                    val_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_searchsorted_f64_run(
                    self.desc.num_queries,
                    self.desc.len_sorted,
                    right_flag,
                    seq_ptr,
                    val_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_searchsorted_i32_run(
                    self.desc.num_queries,
                    self.desc.len_sorted,
                    right_flag,
                    seq_ptr,
                    val_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_searchsorted_i64_run(
                    self.desc.num_queries,
                    self.desc.len_sorted,
                    right_flag,
                    seq_ptr,
                    val_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SearchsortedPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
