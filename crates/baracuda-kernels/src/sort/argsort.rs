//! `argsort` plan — sorted indices only (no values output).
//!
//! `argsort(x, dim=last, descending)` returns just the i32 permutation
//! that would sort `x` along the last dim. PyTorch `torch.argsort`.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, i64`.
//! Non-differentiable (set-valued indices) — no BW.
//!
//! # Multi-block radix (Phase 40 — Fuel ask Gap 6b)
//!
//! `row_len ≤ 1024` is served by the block-bitonic kernel (no workspace
//! required). For `row_len > 1024` the plan transparently dispatches to
//! a CUB-segmented-radix-sort kernel which DOES require a workspace blob
//! (queried via [`ArgsortPlan::workspace_size`]). Callers that only ever
//! pass `row_len ≤ 1024` will see a `workspace_size()` of `0` and may
//! continue to pass [`Workspace::None`]. The dispatch happens internally
//! at `run()` time; the kernel SKU reflects whichever path was selected.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SortKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::sort::build_sku;

/// Descriptor for an `argsort` op.
#[derive(Copy, Clone, Debug)]
pub struct ArgsortDescriptor {
    /// Number of independent rows.
    pub batch: i32,
    /// Length of each row. `≤ 1024` uses the block-bitonic kernel;
    /// `> 1024` uses the multi-block CUB radix kernel (workspace
    /// required — call [`ArgsortPlan::workspace_size`]).
    pub row_len: i32,
    /// `true` = sort largest-first.
    pub descending: bool,
    /// Value element type (input).
    pub element: ElementKind,
}

/// Args bundle for an `argsort` launch.
pub struct ArgsortArgs<'a, T: Element> {
    /// Input `[batch, row_len]`.
    pub input: TensorRef<'a, T, 2>,
    /// Sorted indices output `[batch, row_len]`.
    pub indices: TensorMut<'a, i32, 2>,
}

/// `argsort` plan.
///
/// Returns only sorted indices along the last axis (PyTorch
/// `torch.argsort`). No values output, no BW (indices are
/// non-differentiable).
///
/// **When to use**: when only the permutation is needed (gather
/// downstream tensors via [`GatherPlan`](crate::GatherPlan) using
/// these indices). For sorted values + indices use
/// [`SortPlan`](crate::SortPlan).
///
/// **Dtypes**: input `{f32, f64, i32, i64}`; output always `i32`.
///
/// **Shape limits**: rank-2 `[batch, row_len]`; `row_len ≤ 1024`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable.
pub struct ArgsortPlan<T: Element> {
    desc: ArgsortDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> ArgsortPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ArgsortDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        // Local validator (broader than `validate_sort_desc`): allows
        // `row_len > 1024` because the multi-block radix path covers it.
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ArgsortPlan: descriptor element != type parameter T",
            ));
        }
        if desc.batch < 0 || desc.row_len < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgsortPlan: batch / row_len must be non-negative",
            ));
        }
        if !matches!(
            desc.element,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32 | ElementKind::I64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::ArgsortPlan: today only f32 / f64 / i32 / i64 wired",
            ));
        }
        let sku = build_sku::<T>(SortKind::Argsort);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &ArgsortArgs<'_, T>) -> Result<()> {
        let expected = [self.desc.batch, self.desc.row_len];
        if args.input.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgsortPlan: input shape != [batch, row_len]",
            ));
        }
        if args.indices.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgsortPlan: indices shape != [batch, row_len]",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes.
    ///
    /// `0` when `row_len ≤ 1024` (block-bitonic, in-SMEM). Non-zero
    /// when `row_len > 1024` (multi-block radix path needs scratch for
    /// CUB's `DeviceSegmentedRadixSort` plus keys/indices/offset
    /// buffers). The exact bytes depend on `(batch, row_len, T)`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        if self.desc.row_len <= 1024 {
            return 0;
        }
        let batch = self.desc.batch;
        let row_len = self.desc.row_len;
        if batch == 0 || row_len == 0 {
            return 0;
        }
        match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_workspace_size(
                    batch, row_len,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f64_big_workspace_size(
                    batch, row_len,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i32_big_workspace_size(
                    batch, row_len,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i64_big_workspace_size(
                    batch, row_len,
                )
            },
            _ => 0,
        }
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
        workspace: Workspace<'_>,
        args: ArgsortArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch == 0 || self.desc.row_len == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let idx_ptr = args.indices.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let desc_flag = if self.desc.descending { 1 } else { 0 };

        // Phase 40 dispatch: `row_len > 1024` → multi-block radix path
        // (requires non-empty workspace); otherwise block-bitonic
        // (workspace ignored).
        let use_big = self.desc.row_len > 1024;
        let (ws_ptr, ws_bytes) = if use_big {
            let needed = self.workspace_size();
            match workspace {
                Workspace::None => {
                    if needed == 0 {
                        (core::ptr::null_mut::<c_void>(), 0usize)
                    } else {
                        return Err(Error::WorkspaceTooSmall { needed, got: 0 });
                    }
                }
                Workspace::Borrowed(slice) => {
                    let got = slice.len();
                    if got < needed {
                        return Err(Error::WorkspaceTooSmall { needed, got });
                    }
                    (slice.as_raw().0 as *mut c_void, got)
                }
            }
        } else {
            // Bitonic path ignores workspace; pass null/0 for safety.
            let _ = workspace;
            (core::ptr::null_mut::<c_void>(), 0usize)
        };

        let status = match (T::KIND, use_big) {
            (ElementKind::F32, false) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::F64, false) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::I32, false) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i32_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::I64, false) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i64_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::F32, true) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            (ElementKind::F64, true) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_f64_big_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            (ElementKind::I32, true) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i32_big_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            (ElementKind::I64, true) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_argsort_i64_big_run(
                    self.desc.batch,
                    self.desc.row_len,
                    desc_flag,
                    in_ptr,
                    idx_ptr,
                    ws_ptr,
                    ws_bytes,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ArgsortPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
