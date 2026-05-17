//! Backward plan for single-axis reductions.
//!
//! Sibling of [`crate::ReducePlan`] for gradient computation. Today
//! only [`ReduceKind::Sum`] is wired — the Phase 4 reduction BW
//! trailblazer.
//!
//! **Sum BW** is the simplest reduction backward: the gradient broadcasts
//! `dy` across the reduced axis. With keepdim convention,
//! `dy.shape[reduce_axis] = 1` and we want
//! `dx[c] = dy[c with c[reduce_axis] = 0]` for every coord `c` in dx.
//!
//! Implementation: a strided-copy kernel that uses
//! `stride_dy[reduce_axis] = 0` so reading varies-coord-on-reduced-axis
//! collapses to the singleton dy slot. The Rust dispatcher constructs
//! this stride layout from the args' natural strides — the caller hands
//! in dy with whatever strides their contig allocator gave it, and the
//! plan overrides the reduce-axis stride to 0 before launch.
//!
//! Other reductions ([`ReduceKind::Mean`], `Max`, `Min`, `Prod`,
//! `Norm2`, ...) land in fanout. Mean BW is `Sum BW × (1/k)` where k
//! is the reduced extent (next sub-wave). Max/Min BW need to mask by
//! `(x == y)`; Prod BW needs `y / x` per cell. Each has its own
//! kernel template.
//!
//! Trailblazer constraints: contig-only on dx (the kernel writes
//! linearly into dx's coord space); arbitrary strides accepted on dy
//! but in practice the caller passes contig keepdim dy.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ReduceKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a single-axis reduction backward.
#[derive(Copy, Clone, Debug)]
pub struct ReduceBackwardDescriptor<const N: usize> {
    /// Which forward reduction this is the backward of.
    pub kind: ReduceKind,
    /// Shape of the forward input (= shape of dx).
    pub input_shape: [i32; N],
    /// Axis that was reduced. Must satisfy `0 <= reduce_axis < N`.
    pub reduce_axis: u8,
    /// Element type.
    pub element: ElementKind,
    /// Bessel correction for `Var` / `Std` BW only. `1` = sample
    /// variance (PyTorch default), `0` = population variance. Ignored
    /// by other reductions.
    pub correction: i32,
}

impl<const N: usize> ReduceBackwardDescriptor<N> {
    /// Compute the keepdim dy shape (input shape with reduce_axis = 1).
    pub fn dy_shape(&self) -> [i32; N] {
        let mut out = self.input_shape;
        out[self.reduce_axis as usize] = 1;
        out
    }
}

/// Args bundle for a reduction-backward launch.
///
/// `dy.shape` must equal the keepdim form (input shape with the reduced
/// axis collapsed to 1). `dx.shape` must equal `input_shape`. Both
/// fully contiguous (trailblazer constraint).
///
/// Save requirements vary by op:
/// - Sum, Mean: neither save needed; pass `x = None, y = None`.
/// - Max, Min: BOTH saves required — `x` is the forward input (full
///   shape), `y` is the forward output (keepdim shape). Gradient flows
///   to every position where `x[c] == y[c_reduced]` (split-across-ties
///   semantic; matches JAX, differs from PyTorch's first-index pick).
/// - Prod, Norm2 (future): same dual-save requirement.
pub struct ReduceBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — keepdim shape matching forward output.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input — full input shape. Required by Max/Min/
    /// Prod/Norm2; pass `None` for Sum/Mean.
    pub x: Option<TensorRef<'a, T, N>>,
    /// Saved forward output — keepdim shape (= dy.shape). Required by
    /// Max/Min/Prod/Norm2; pass `None` for Sum/Mean.
    pub y: Option<TensorRef<'a, T, N>>,
    /// Gradient w.r.t. the forward input — full input shape.
    pub dx: TensorMut<'a, T, N>,
}

/// Single-axis reduction backward plan.
pub struct ReduceBackwardPlan<T: Element, const N: usize> {
    desc: ReduceBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

#[inline]
fn op_needs_saves(kind: ReduceKind) -> bool {
    // Max/Min/Prod/Norm2/Std/LogSumExp reference both forward input
    // and forward output in their BW formulas. Var references only
    // saved x but takes a `y` slot for ABI uniformity with Std — we
    // still require a non-null `y` so callers stage both consistently.
    // Sum/Mean need neither.
    matches!(
        kind,
        ReduceKind::Max
            | ReduceKind::Min
            | ReduceKind::Prod
            | ReduceKind::Norm2
            | ReduceKind::Var
            | ReduceKind::Std
            | ReduceKind::LogSumExp
    )
}

impl<T: Element, const N: usize> ReduceBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &ReduceBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReduceBackwardPlan: descriptor element != T",
            ));
        }
        if (desc.reduce_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReduceBackwardPlan: reduce_axis out of range for rank N",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ReduceBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReduceBackwardPlan: tensor rank > 8 not supported \
                 (kernel param block fixes MAX_RANK = 8)",
            ));
        }

        // Wired today:
        //   `{Sum, Mean, Max, Min, Prod, Norm2, LogSumExp, Var, Std}
        //      × {f32, f16, bf16, f64}`
        // Max/Min use a single unified kernel (the routing logic is
        // identical: `x[c] == y[c_reduced]`). Prod, Norm2, and
        // LogSumExp each have their own dual-save kernel with a
        // different formula (LogSumExp computes `dy * exp(x - y)`).
        // Var / Std (Welford BW) are templated on T; internal
        // accumulation is f32 for f32/f16/bf16 and f64 for f64.
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let kind_in_scope = matches!(
            desc.kind,
            ReduceKind::Sum
                | ReduceKind::Mean
                | ReduceKind::Max
                | ReduceKind::Min
                | ReduceKind::Prod
                | ReduceKind::Norm2
                | ReduceKind::LogSumExp
                | ReduceKind::Var
                | ReduceKind::Std
        );
        let supported = kind_in_scope && dtype_in_fp_family;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReduceBackwardPlan: wired today: \
                 `{Sum, Mean, Max, Min, Prod, Norm2, LogSumExp, Var, Std} \
                  × {f32, f16, bf16, f64}`; \
                 other (kind, dtype) pairs land in later fanout",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Reduction,
            op: desc.kind as u16,
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
    pub fn can_implement(&self, args: &ReduceBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReduceBackwardPlan: dx shape must equal input_shape",
            ));
        }
        let expected_dy_shape = self.desc.dy_shape();
        if args.dy.shape != expected_dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReduceBackwardPlan: dy shape must equal input_shape \
                 with reduce_axis collapsed to 1 (keepdim form)",
            ));
        }
        if !args.dy.is_contiguous() || !args.dx.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReduceBackwardPlan: trailblazer requires contiguous \
                 dy / dx; strided fanout lands later",
            ));
        }
        let dx_numel = args.dx.numel();
        let dy_numel = args.dy.numel();
        if (args.dx.data.len() as i64) < dx_numel {
            return Err(Error::BufferTooSmall {
                needed: dx_numel as usize,
                got: args.dx.data.len(),
            });
        }
        if (args.dy.data.len() as i64) < dy_numel {
            return Err(Error::BufferTooSmall {
                needed: dy_numel as usize,
                got: args.dy.data.len(),
            });
        }
        // Max/Min require BOTH saved-x (forward input, full shape) and
        // saved-y (forward output, keepdim shape).
        if op_needs_saves(self.desc.kind) {
            let x = args.x.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::ReduceBackwardPlan: this op requires saved input `x`",
            ))?;
            let y = args.y.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::ReduceBackwardPlan: this op requires saved output `y`",
            ))?;
            if x.shape != self.desc.input_shape {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ReduceBackwardPlan: saved `x` shape must equal input_shape",
                ));
            }
            if y.shape != expected_dy_shape {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ReduceBackwardPlan: saved `y` shape must equal \
                     keepdim form (input_shape with reduce_axis = 1)",
                ));
            }
            if !x.is_contiguous() || !y.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ReduceBackwardPlan: saved x / y must be contiguous \
                     (strided fanout lands later)",
                ));
            }
            if (x.data.len() as i64) < dx_numel {
                return Err(Error::BufferTooSmall {
                    needed: dx_numel as usize,
                    got: x.data.len(),
                });
            }
            if (y.data.len() as i64) < dy_numel {
                return Err(Error::BufferTooSmall {
                    needed: dy_numel as usize,
                    got: y.data.len(),
                });
            }
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Kernel SKU identity.
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
        args: ReduceBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dx.numel();
        if numel == 0 {
            return Ok(());
        }
        // Construct the broadcast dy stride layout: take dy's natural
        // strides and zero out the reduced axis. The kernel walks the
        // full dx coord space; reading dy with stride 0 on the reduce
        // axis collapses every reduce-axis coord to the singleton dy
        // slot.
        let axis = self.desc.reduce_axis as usize;
        let mut stride_dy = args.dy.stride;
        stride_dy[axis] = 0;
        let shape = self.desc.input_shape;
        let stride_dx = args.dx.stride;
        let rank = N as i32;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (self.desc.kind, T::KIND) {
            (ReduceKind::Sum, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_sum_backward_f32_run(
                    numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Sum, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_sum_backward_f16_run(
                    numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Sum, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_sum_backward_bf16_run(
                    numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Sum, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_sum_backward_f64_run(
                    numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Max, _) | (ReduceKind::Min, _) => {
                // Both ops share one kernel: `x[c] == y[c_reduced]`
                // identifies recipient positions regardless of whether
                // y is a max or a min.
                let x = args.x.as_ref().expect("Max/Min BW require saved x");
                let y = args.y.as_ref().expect("Max/Min BW require saved y");
                let x_ptr = x.data.as_raw().0 as *const c_void;
                let y_ptr = y.data.as_raw().0 as *const c_void;
                let stride_x = x.stride;
                let mut stride_y = y.stride;
                stride_y[axis] = 0;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_max_min_backward_f32_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_max_min_backward_f16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_max_min_backward_bf16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_max_min_backward_f64_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::ReduceBackwardPlan::run: Max/Min BW reached an \
                         unimplemented dtype — select() should have caught this",
                    )),
                }
            }
            (ReduceKind::Mean, _) => {
                // `1/k` where k = reduced extent. Computed in f64 on the
                // host and cast to T inside the kernel.
                let extent = self.desc.input_shape[axis] as f64;
                if extent == 0.0 {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::ReduceBackwardPlan: Mean BW requires \
                         reduced extent > 0",
                    ));
                }
                let inv_extent = 1.0_f64 / extent;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_mean_backward_f32_run(
                            numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, dx_ptr, inv_extent,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_mean_backward_f16_run(
                            numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, dx_ptr, inv_extent,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_mean_backward_bf16_run(
                            numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, dx_ptr, inv_extent,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_mean_backward_f64_run(
                            numel, rank, shape.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, dx_ptr, inv_extent,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::ReduceBackwardPlan::run: Mean BW reached an \
                         unimplemented dtype — select() should have caught this",
                    )),
                }
            }
            (ReduceKind::Prod, _) => {
                // `dx[c] = dy[c_reduced] * y[c_reduced] / x[c]`. Dual-save.
                let x = args.x.as_ref().expect("Prod BW require saved x");
                let y = args.y.as_ref().expect("Prod BW require saved y");
                let x_ptr = x.data.as_raw().0 as *const c_void;
                let y_ptr = y.data.as_raw().0 as *const c_void;
                let stride_x = x.stride;
                let mut stride_y = y.stride;
                stride_y[axis] = 0;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_prod_backward_f32_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_prod_backward_f16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_prod_backward_bf16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_prod_backward_f64_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::ReduceBackwardPlan::run: Prod BW reached an \
                         unimplemented dtype — select() should have caught this",
                    )),
                }
            }
            (ReduceKind::Var, _) | (ReduceKind::Std, _) => {
                // Welford BW. `mean[c_reduced]` is recomputed inside the
                // kernel (single-pass sum/n on the saved-x reduce axis).
                // Var BW: `dx[c] = dy[c_reduced] * 2 * (x[c] - mean) / m`
                // Std BW: `dx[c] = dy[c_reduced] * (x[c] - mean) /
                //                  (m * y[c_reduced])`
                // where `m = max(n - correction, 1)`. Internal Welford
                // accumulator runs at f32 for f32/f16/bf16 and f64 for
                // f64 (see `WelfordAcc<T>` in the kernel header).
                let x = args
                    .x
                    .as_ref()
                    .expect("Var/Std BW require saved x");
                let y = args
                    .y
                    .as_ref()
                    .expect("Var/Std BW require saved y (Var ignores it; passed for ABI uniformity)");
                let x_ptr = x.data.as_raw().0 as *const c_void;
                let y_ptr = y.data.as_raw().0 as *const c_void;
                let stride_x = x.stride;
                let mut stride_y = y.stride;
                stride_y[axis] = 0;
                let reduce_axis_i32 = self.desc.reduce_axis as i32;
                let reduce_extent = self.desc.input_shape[axis];
                let reduce_stride_x = stride_x[axis];
                let correction = self.desc.correction;
                match (self.desc.kind, T::KIND) {
                    (ReduceKind::Var, ElementKind::F32) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_var_backward_f32_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    (ReduceKind::Var, ElementKind::F16) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_var_backward_f16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    (ReduceKind::Var, ElementKind::Bf16) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_var_backward_bf16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    (ReduceKind::Var, ElementKind::F64) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_var_backward_f64_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    (ReduceKind::Std, ElementKind::F32) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_std_backward_f32_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    (ReduceKind::Std, ElementKind::F16) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_std_backward_f16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    (ReduceKind::Std, ElementKind::Bf16) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_std_backward_bf16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    (ReduceKind::Std, ElementKind::F64) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_std_backward_f64_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            reduce_axis_i32, reduce_extent, reduce_stride_x, correction,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::ReduceBackwardPlan::run: Var/Std BW reached an \
                         unimplemented dtype — select() should have caught this",
                    )),
                }
            }
            (ReduceKind::Norm2, _) => {
                // `dx[c] = dy[c_reduced] * x[c] / y[c_reduced]`. Dual-save.
                let x = args.x.as_ref().expect("Norm2 BW require saved x");
                let y = args.y.as_ref().expect("Norm2 BW require saved y");
                let x_ptr = x.data.as_raw().0 as *const c_void;
                let y_ptr = y.data.as_raw().0 as *const c_void;
                let stride_x = x.stride;
                let mut stride_y = y.stride;
                stride_y[axis] = 0;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_norm2_backward_f32_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_norm2_backward_f16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_norm2_backward_bf16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_norm2_backward_f64_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::ReduceBackwardPlan::run: Norm2 BW reached an \
                         unimplemented dtype — select() should have caught this",
                    )),
                }
            }
            (ReduceKind::LogSumExp, _) => {
                // `dx[c] = dy[c_reduced] * exp(x[c] - y[c_reduced])`.
                // Dual-save. `y = lse(x) ≥ max(x) ≥ x[c]`, so the exp
                // arg is `≤ 0` and the result is bounded in `(0, 1]` —
                // no overflow possible at any dtype.
                let x = args.x.as_ref().expect("LogSumExp BW require saved x");
                let y = args.y.as_ref().expect("LogSumExp BW require saved y");
                let x_ptr = x.data.as_raw().0 as *const c_void;
                let y_ptr = y.data.as_raw().0 as *const c_void;
                let stride_x = x.stride;
                let mut stride_y = y.stride;
                stride_y[axis] = 0;
                match T::KIND {
                    ElementKind::F32 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_logsumexp_backward_f32_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_logsumexp_backward_f16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_logsumexp_backward_bf16_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    ElementKind::F64 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_reduce_logsumexp_backward_f64_run(
                            numel, rank, shape.as_ptr(),
                            stride_dy.as_ptr(), stride_x.as_ptr(),
                            stride_y.as_ptr(), stride_dx.as_ptr(),
                            dy_ptr, x_ptr, y_ptr, dx_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::ReduceBackwardPlan::run: LogSumExp BW reached an \
                         unimplemented dtype — select() should have caught this",
                    )),
                }
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ReduceBackwardPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
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
