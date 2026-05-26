//! `unsorted_segment_prod` plan — Category S, unsorted. Phase 25.
//!
//! `out[s, d] = Π_{n : segment_ids[n] == s} input[n, d]` with arbitrary
//! `segment_ids` ordering. The kernel fills `output` with `1.0` then
//! performs `atomicMul`-via-CAS into `output[seg[n], d]` per input cell.
//!
//! No native FP `atomicMul` exists; we implement it as an `atomicCAS`
//! retry loop on the underlying 32 / 64-bit slot. This is slower than
//! the additive variants but allowed per the OP-MATRIX (segment ops
//! contract).
//!
//! Non-deterministic — atomic ordering varies across launches.
//!
//! Dtype coverage: `f32, f64`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PrecisionGuarantee, SegmentKind, TensorMut,
    TensorRef, Workspace,
};

use super::map_status;
use super::segment_sum::{validate_desc, SegDescView};
use super::unsorted_segment_sum::{build_unsorted_sku, validate_unsorted_args};

/// Descriptor for an `unsorted_segment_prod` op.
#[derive(Copy, Clone, Debug)]
pub struct UnsortedSegmentProdDescriptor {
    /// Number of input rows.
    pub num_inputs: i32,
    /// Embedding / feature dim.
    pub embedding_dim: i32,
    /// Total number of segments.
    pub num_segments: i32,
    /// Value element type.
    pub element: ElementKind,
}

impl SegDescView for UnsortedSegmentProdDescriptor {
    #[inline]
    fn view(&self) -> (i32, i32, i32, ElementKind) {
        (
            self.num_inputs,
            self.embedding_dim,
            self.num_segments,
            self.element,
        )
    }
}

/// Args bundle for an `unsorted_segment_prod` launch.
pub struct UnsortedSegmentProdArgs<'a, T: Element> {
    /// Input `[N, D]`.
    pub input: TensorRef<'a, T, 2>,
    /// Segment ids `[N]`, i32, in any order.
    pub segment_ids: TensorRef<'a, i32, 1>,
    /// Output `[num_segments, D]`. Overwritten by the launch — kernel
    /// fills `1.0` before the atomic accumulation phase.
    pub output: TensorMut<'a, T, 2>,
}

/// `unsorted_segment_prod` plan. Phase 25.
///
/// `out[s, d] = Π input[n, d]` over `n : segment_ids[n] == s` (any
/// order). `atomicMul`-emulated CAS retry loop.
///
/// **When to use**: forward unsorted segment-product. For sorted IDs
/// prefer the deterministic
/// [`SegmentProdPlan`](crate::SegmentProdPlan). BW shares the
/// [`SegmentProdBackwardPlan`](crate::SegmentProdBackwardPlan) shape.
///
/// **Dtypes**: `{f32, f64}` only — `atomicCAS` slot widths are 32 / 64
/// bit. Empty segments emit `1.0` (multiplicative identity).
///
/// **Workspace**: none.
///
/// **Precision guarantee**: **non-deterministic** — atomic ordering
/// varies across launches.
pub struct UnsortedSegmentProdPlan<T: Element> {
    desc: UnsortedSegmentProdDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> UnsortedSegmentProdPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &UnsortedSegmentProdDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_desc(*desc, T::KIND, "UnsortedSegmentProdPlan")?;
        Ok(Self {
            desc: *desc,
            sku: build_unsorted_sku::<T>(SegmentKind::UnsortedSegmentProd),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &UnsortedSegmentProdArgs<'_, T>) -> Result<()> {
        validate_unsorted_args(
            self.desc.num_inputs,
            self.desc.embedding_dim,
            self.desc.num_segments,
            args.input.shape,
            args.segment_ids.shape,
            args.output.shape,
            "UnsortedSegmentProdPlan",
        )
    }

    /// Workspace size — zero.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel.
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
        args: UnsortedSegmentProdArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.num_segments as i64) * (self.desc.embedding_dim as i64);
        if total == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let id_ptr = args.segment_ids.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_prod_f32_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    in_ptr,
                    id_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_unsorted_segment_prod_f64_run(
                    self.desc.num_inputs,
                    self.desc.embedding_dim,
                    self.desc.num_segments,
                    in_ptr,
                    id_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::UnsortedSegmentProdPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
