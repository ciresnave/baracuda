//! Broadcast-reverse reduction plan.
//!
//! The autograd primitive that undoes a forward `BroadcastTo`: for
//! each output cell, reduce every input cell that broadcasts TO it.
//! The reduced dims are every dim where `output_shape[d] == 1` while
//! `input_shape[d] != 1` — an arbitrary *set* of axes collapses in a
//! single launch (contrast [`crate::ReducePlan`], which reduces one
//! `reduce_axis` per launch). Output keeps the input's rank with the
//! reduced dims at size 1 (keepdim convention).
//!
//! **Wired matrix**: `{Sum, Max, Min, Prod} × {f32, f16, bf16, f64}`
//! — 16 (op, dtype) cells over the Phase 31 / Phase 37
//! `baracuda_kernels_reduce_{sum,max,min,prod}_to_*` FFI symbols
//! (Phase 74 closes the facade gap — the symbols shipped without a
//! plan-level entry, which hid the capability from plan-surface
//! audits). The kernel template is shared (one thread per output
//! cell, sequential walk over the broadcast set); the per-op policy
//! supplies the identity + combine step.
//!
//! **Empty reduce sets** (any reduced `input_shape[d] == 0`): the
//! kernel writes the op's identity — `0` for Sum, `1` for Prod,
//! `-FLT_MAX` / `-DBL_MAX` for Max, `+FLT_MAX` / `+DBL_MAX` for Min.
//! For f32 / f64 outputs that is the most-extreme *finite* value;
//! for f16 / bf16 the f32 identity overflows the storage dtype on
//! the final narrowing store and lands as `∓inf`. See [`ReduceToOp`].
//!
//! **Layout**: the input may be arbitrarily strided (transposed /
//! sliced views — common in autograd traces); its strides pass
//! through to the kernel. The output MUST be contiguous over
//! `output_shape` (the kernel writes `dst[out_id]` by linear index;
//! validated in `can_implement`).
//!
//! **Workspace**: none — the per-output-cell kernel keeps the running
//! accumulator in registers.
//!
//! **Precision**: deterministic, bit-stable on the same hardware (no
//! atomic-add; sequential per-cell accumulation has a fixed order).
//! f16 / bf16 accumulate in f32 (Sum / Prod widen per the PyTorch
//! convention; Max / Min compare in f32, which is value-preserving);
//! f64 keeps everything in double.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ReduceToOp, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a broadcast-reverse reduction.
///
/// `input_shape` is the source extents; `output_shape` is the target
/// extents — same rank, with every reduced dim collapsed to size 1
/// (the caller left-pads `output_shape` with 1s if the forward
/// broadcast added leading dims). Per-dim constraint:
/// `output_shape[d] == 1 || output_shape[d] == input_shape[d]`.
#[derive(Copy, Clone, Debug)]
pub struct ReduceToDescriptor<const N: usize> {
    /// Which reduction to apply over each output cell's broadcast set.
    pub op: ReduceToOp,
    /// Input tensor shape.
    pub input_shape: [i32; N],
    /// Output tensor shape — `input_shape` with the reduced dims
    /// collapsed to 1.
    pub output_shape: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a broadcast-reverse reduction launch.
///
/// `x.shape` must match `desc.input_shape`; arbitrary (non-contiguous)
/// strides are fine — they pass through to the kernel. `y.shape` must
/// match `desc.output_shape` and `y` MUST be contiguous.
pub struct ReduceToArgs<'a, T: Element, const N: usize> {
    /// Input tensor — may be a strided (transposed / sliced) view.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — contiguous over `desc.output_shape`.
    pub y: TensorMut<'a, T, N>,
}

/// Broadcast-reverse reduction plan — see module docs for the wired
/// matrix, workspace, and precision guarantees.
///
/// `T: Element` is the element type (`f32` / `f64` / `f16` / `bf16`).
/// `const N: usize` is the tensor rank (input and output share it).
pub struct ReduceToPlan<T: Element, const N: usize> {
    desc: ReduceToDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ReduceToPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ReduceToDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReduceToPlan: descriptor element != type parameter T",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReduceToPlan: tensor rank > 8 not supported \
                 (kernel param block fixes MAX_RANK = 8)",
            ));
        }
        for d in 0..N {
            if desc.input_shape[d] < 0 || desc.output_shape[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ReduceToPlan: shape dims must be non-negative",
                ));
            }
            // Broadcast-reverse contract: every output dim is either
            // kept (== input dim) or reduced (== 1).
            if desc.output_shape[d] != 1 && desc.output_shape[d] != desc.input_shape[d] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ReduceToPlan: per-dim contract violated — \
                     output_shape[d] must be 1 (reduced) or equal input_shape[d] (kept)",
                ));
            }
        }

        // Supported matrix:
        //   {Sum, Max, Min, Prod} × {f32, f16, bf16, f64}   (16 cells)
        // The match arms in `run` remain the authoritative dispatch
        // table; the unreachable `_ =>` arm catches any future drift.
        let op_in_scope = matches!(
            desc.op,
            ReduceToOp::Sum | ReduceToOp::Max | ReduceToOp::Min | ReduceToOp::Prod
        );
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let supported = op_in_scope && dtype_in_scope;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReduceToPlan: supported matrix is \
                 {Sum, Max, Min, Prod} × {f32, f16, bf16, f64}",
            ));
        }

        // One thread per output cell, sequential walk over the
        // broadcast set in a fixed order — deterministic and
        // bit-stable on the same hardware. f32 / f16 / bf16 accumulate
        // in f32; f64 keeps everything in double (see module docs).
        let (math_precision, accumulator) = if T::KIND == ElementKind::F64 {
            (MathPrecision::F64, ElementKind::F64)
        } else {
            (MathPrecision::F32, ElementKind::F32)
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Reduction,
            op: desc.op as u16,
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
    pub fn can_implement(&self, args: &ReduceToArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReduceToPlan: X shape mismatch with descriptor input_shape",
            ));
        }
        if args.y.shape != self.desc.output_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReduceToPlan: Y shape mismatch with descriptor output_shape",
            ));
        }
        // The kernel writes `dst[out_id]` by linear index — the output
        // must be a plain contiguous allocation over output_shape. The
        // input may be arbitrarily strided.
        if !args.y.is_contiguous() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReduceToPlan: Y must be contiguous over output_shape \
                 (the kernel writes by linear output index)",
            ));
        }
        let y_numel = args.y.numel();
        let y_len = args.y.data.len() as i64;
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
            });
        }
        // Input bound: `x` may be an arbitrary strided view — including
        // stride-0 broadcast dims, where `numel` legitimately exceeds
        // the distinct storage extent — so the right bound is the
        // reachable SPAN `1 + Σ_d (shape[d]-1)·stride[d]`, not `numel`.
        // Negative strides can never index in-bounds (TensorRef has no
        // base offset; the data pointer IS the slice start).
        if args.x.numel() > 0 {
            let mut span: i64 = 1;
            for d in 0..N {
                let extent = self.desc.input_shape[d] as i64;
                if extent > 1 {
                    let stride = args.x.stride[d];
                    if stride < 0 {
                        return Err(Error::InvalidProblem(
                            "baracuda-kernels::ReduceToPlan: negative input strides walk \
                             before the buffer base (TensorRef has no base offset)",
                        ));
                    }
                    span += (extent - 1) * stride;
                }
            }
            let x_len = args.x.data.len() as i64;
            if x_len < span {
                return Err(Error::BufferTooSmall {
                    needed: span as usize,
                    got: x_len as usize,
                });
            }
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0` — the per-output-cell
    /// kernel keeps its accumulator in registers.
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
        args: ReduceToArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.y.numel() == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let input_shape = self.desc.input_shape;
        let input_stride = args.x.stride;
        let output_shape = self.desc.output_shape;
        let rank = N as i32;

        // Helper: every reduce-to FFI symbol shares the same parameter
        // shape (the kernel template is shared). The macro picks the
        // right symbol from (op, dtype).
        macro_rules! dispatch {
            ($sym:ident) => {{
                unsafe {
                    baracuda_kernels_sys::$sym(
                        x_ptr,
                        y_ptr,
                        input_shape.as_ptr(),
                        input_stride.as_ptr(),
                        rank,
                        output_shape.as_ptr(),
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }};
        }

        let status = match (self.desc.op, T::KIND) {
            // Sum
            (ReduceToOp::Sum, ElementKind::F32) => {
                dispatch!(baracuda_kernels_reduce_sum_to_f32_run)
            }
            (ReduceToOp::Sum, ElementKind::F16) => {
                dispatch!(baracuda_kernels_reduce_sum_to_f16_run)
            }
            (ReduceToOp::Sum, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_sum_to_bf16_run)
            }
            (ReduceToOp::Sum, ElementKind::F64) => {
                dispatch!(baracuda_kernels_reduce_sum_to_f64_run)
            }
            // Max
            (ReduceToOp::Max, ElementKind::F32) => {
                dispatch!(baracuda_kernels_reduce_max_to_f32_run)
            }
            (ReduceToOp::Max, ElementKind::F16) => {
                dispatch!(baracuda_kernels_reduce_max_to_f16_run)
            }
            (ReduceToOp::Max, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_max_to_bf16_run)
            }
            (ReduceToOp::Max, ElementKind::F64) => {
                dispatch!(baracuda_kernels_reduce_max_to_f64_run)
            }
            // Min
            (ReduceToOp::Min, ElementKind::F32) => {
                dispatch!(baracuda_kernels_reduce_min_to_f32_run)
            }
            (ReduceToOp::Min, ElementKind::F16) => {
                dispatch!(baracuda_kernels_reduce_min_to_f16_run)
            }
            (ReduceToOp::Min, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_min_to_bf16_run)
            }
            (ReduceToOp::Min, ElementKind::F64) => {
                dispatch!(baracuda_kernels_reduce_min_to_f64_run)
            }
            // Prod
            (ReduceToOp::Prod, ElementKind::F32) => {
                dispatch!(baracuda_kernels_reduce_prod_to_f32_run)
            }
            (ReduceToOp::Prod, ElementKind::F16) => {
                dispatch!(baracuda_kernels_reduce_prod_to_f16_run)
            }
            (ReduceToOp::Prod, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_prod_to_bf16_run)
            }
            (ReduceToOp::Prod, ElementKind::F64) => {
                dispatch!(baracuda_kernels_reduce_prod_to_f64_run)
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ReduceToPlan::run reached an unimplemented \
                     (op, dtype) pair — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
