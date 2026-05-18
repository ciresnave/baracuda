//! `unique_consecutive` plan — emit one cell per run-start in each row.
//!
//! Input must be pre-sorted (or the user only wants consecutive-equal
//! runs collapsed — the PyTorch `torch.unique_consecutive` semantics).
//!
//! Output is **NOT input-order** — slot assignment uses a per-row
//! atomic counter (block-race order). Callers that need input-order
//! output should issue a follow-up sort on `[batch, counter]` rows.
//! The per-row count is written to `counter[batch]` (a separate
//! tensor) — callers read it post-launch to learn the actual unique
//! count per row.
//!
//! Trailblazer dtype coverage: `f32, f64, i32`. Set-valued — no BW.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SortKind, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a `unique_consecutive` op.
#[derive(Copy, Clone, Debug)]
pub struct UniqueConsecutiveDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each input row.
    pub row_len: i32,
    /// Maximum number of unique values the output table can hold per
    /// row. Set to `row_len` for a worst-case-safe bound.
    pub max_unique: i32,
    /// Whether to emit per-run counts (`y_counts`). Today the kernel
    /// writes `1` per detected run-start regardless; this flag is
    /// reserved for future counts-aware variants.
    pub return_counts: bool,
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `unique_consecutive` launch.
pub struct UniqueConsecutiveArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Output values `[batch, max_unique]` (filled left-to-right per
    /// row up to the actual unique count).
    pub values: TensorMut<'a, T, 2>,
    /// Optional output per-cell counts `[batch, max_unique]`.
    pub counts: TensorMut<'a, i32, 2>,
    /// Per-row counter `[batch]` — post-launch holds the actual
    /// unique count per row.
    pub counter: TensorMut<'a, i32, 1>,
}

/// `unique_consecutive` plan.
pub struct UniqueConsecutivePlan<T: Element> {
    desc: UniqueConsecutiveDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UniqueConsecutivePlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &UniqueConsecutiveDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::UniqueConsecutivePlan: descriptor element != type parameter T",
            ));
        }
        if desc.batch < 0 || desc.row_len < 0 || desc.max_unique < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniqueConsecutivePlan: batch / row_len / max_unique \
                 must be non-negative",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::UniqueConsecutivePlan: today only f32 / f64 / i32 wired",
            ));
        }
        let sku = build_unique_sku::<T>(SortKind::UniqueConsecutive);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UniqueConsecutiveArgs<'_, T>) -> Result<()> {
        if args.input.shape != [self.desc.batch, self.desc.row_len] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniqueConsecutivePlan: input shape != [batch, row_len]",
            ));
        }
        if args.values.shape != [self.desc.batch, self.desc.max_unique] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniqueConsecutivePlan: values shape != [batch, max_unique]",
            ));
        }
        if args.counts.shape != [self.desc.batch, self.desc.max_unique] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniqueConsecutivePlan: counts shape != [batch, max_unique]",
            ));
        }
        if args.counter.shape != [self.desc.batch] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::UniqueConsecutivePlan: counter shape != [batch]",
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
        args: UniqueConsecutiveArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let vals_ptr = args.values.data.as_raw().0 as *mut c_void;
        let counts_ptr = args.counts.data.as_raw().0 as *mut c_void;
        let counter_ptr = args.counter.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unique_consecutive_f32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    self.desc.max_unique,
                    in_ptr,
                    vals_ptr,
                    counts_ptr,
                    counter_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unique_consecutive_f64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    self.desc.max_unique,
                    in_ptr,
                    vals_ptr,
                    counts_ptr,
                    counter_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unique_consecutive_i32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    self.desc.max_unique,
                    in_ptr,
                    vals_ptr,
                    counts_ptr,
                    counter_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UniqueConsecutivePlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}

/// Build SKU for unique-family ops — atomic-counter output is NOT
/// deterministic in slot order, so we tag accordingly.
pub(crate) fn build_unique_sku<T: Element>(op: SortKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if T::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: T::KIND,
        bit_stable_on_same_hardware: false,
        deterministic: false,
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
