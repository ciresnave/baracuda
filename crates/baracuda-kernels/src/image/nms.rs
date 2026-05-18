//! `nms` plan — Category T (non-max suppression).
//!
//! Box format: `(x1, y1, x2, y2)` with `x2 >= x1`, `y2 >= y1`. The
//! caller MUST supply `boxes` sorted by score, descending (matches
//! torchvision's `nms` contract). The kernel emits a boolean
//! `keep_mask` (`u8`, `0` / `1`) and a single-element `i32` count.
//!
//! No backward — NMS is a set-valued / non-differentiable op.
//!
//! Trailblazer dtype coverage: `f32, f64` (box coordinate dtype).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, ImageKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for `nms`.
#[derive(Copy, Clone, Debug)]
pub struct NmsDescriptor {
    /// Number of input boxes.
    pub num_boxes: i32,
    /// IoU suppression threshold. Pairs with IoU strictly greater
    /// than this are suppressed.
    pub iou_threshold: f32,
    /// Box-coordinate element type.
    pub element: ElementKind,
}

/// Args bundle for `nms`.
pub struct NmsArgs<'a, T: Element> {
    /// Boxes `[num_boxes, 4]` in `(x1, y1, x2, y2)`. Pre-sorted by
    /// score, descending.
    pub boxes: TensorRef<'a, T, 2>,
    /// Output keep mask `[num_boxes]`, u8 (`0` / `1`).
    pub keep_mask: TensorMut<'a, u8, 1>,
    /// Output count (single i32).
    pub count: TensorMut<'a, i32, 1>,
}

/// `nms` plan.
pub struct NmsPlan<T: Element> {
    desc: NmsDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> NmsPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &NmsDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::NmsPlan: descriptor element != T",
            ));
        }
        if desc.num_boxes < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NmsPlan: num_boxes must be non-negative",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::NmsPlan: only `f32`, `f64` wired",
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
            op: ImageKind::Nms as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::U8),
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
    pub fn can_implement(&self, args: &NmsArgs<'_, T>) -> Result<()> {
        if args.boxes.shape != [self.desc.num_boxes, 4] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NmsPlan: boxes must be [num_boxes, 4]",
            ));
        }
        if args.keep_mask.shape != [self.desc.num_boxes] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NmsPlan: keep_mask must be [num_boxes]",
            ));
        }
        if args.count.shape != [1] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::NmsPlan: count must be [1]",
            ));
        }
        Ok(())
    }

    /// Workspace (zero — kernel allocates its `killed` scratch in
    /// dynamic shared memory keyed off `num_boxes`).
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
        args: NmsArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let boxes_ptr = args.boxes.data.as_raw().0 as *const c_void;
        let mask_ptr = args.keep_mask.data.as_raw().0 as *mut c_void;
        let count_ptr = args.count.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nms_f32_run(
                    self.desc.num_boxes,
                    self.desc.iou_threshold,
                    boxes_ptr, mask_ptr, count_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_nms_f64_run(
                    self.desc.num_boxes,
                    self.desc.iou_threshold,
                    boxes_ptr, mask_ptr, count_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::NmsPlan::run reached unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
