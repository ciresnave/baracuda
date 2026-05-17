//! Single-axis reduction plan.
//!
//! Output shape == input shape with the reduced axis collapsed to size
//! 1 (keepdim convention). Wired for `{Sum, Mean, Max, Min, Prod,
//! Norm2, LogSumExp, Var, Std} × {f32, f16, bf16, f64}` — 36 (kind,
//! dtype) cells. The simple-reduce kernel template is shared (one
//! thread per output cell, sequential walk over the reduced axis);
//! each (op, dtype) has its own functor + FFI symbol. LogSumExp ships
//! a dedicated two-pass kernel (max, then sum-exp) under the same FFI
//! shape. Var / Std ship a Welford one-pass kernel templated on T;
//! internal accumulation is f32 for f32/f16/bf16 and f64 for f64.
//! Argmax / Argmin live in a separate plan shape because their output
//! dtype differs (index, not value). Any / All are reserved
//! discriminants for later fanout. Trace dispatches through
//! `TracePlan` (scalar output).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ReduceKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a single-axis reduction.
///
/// `input_shape` is the shape of the input tensor. `reduce_axis` is
/// the axis to reduce (0 ≤ `reduce_axis` < rank). Output shape is the
/// input shape with `[reduce_axis]` collapsed to size 1 (keepdim
/// convention — caller squeezes if they want).
#[derive(Copy, Clone, Debug)]
pub struct ReduceDescriptor<const N: usize> {
    /// Which reduction to apply (Sum / Mean / Max / ...).
    pub kind: ReduceKind,
    /// Input tensor shape.
    pub input_shape: [i32; N],
    /// Axis to reduce along. Must satisfy `0 <= reduce_axis < N`.
    pub reduce_axis: u8,
    /// Element type.
    pub element: ElementKind,
    /// Bessel correction for `Var` / `Std` only. `1` = sample
    /// variance (PyTorch default), `0` = population variance. Ignored
    /// by other reductions.
    pub correction: i32,
}

impl<const N: usize> ReduceDescriptor<N> {
    /// Compute the output shape (input shape with reduce axis = 1).
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = self.input_shape;
        out[self.reduce_axis as usize] = 1;
        out
    }
}

/// Args bundle for a reduction launch.
///
/// `x.shape` must match `desc.input_shape`. `y.shape` must match the
/// derived output shape. Output is conventionally contiguous; the
/// kernel accepts arbitrary strides.
pub struct ReduceArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — shape == input with reduced axis collapsed to 1.
    pub y: TensorMut<'a, T, N>,
}

/// Single-axis reduction plan.
///
/// `T: Element` is the element type (today: must be `f32`).
/// `const N: usize` is the tensor rank.
pub struct ReducePlan<T: Element, const N: usize> {
    desc: ReduceDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ReducePlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ReduceDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReducePlan: descriptor element != type parameter T",
            ));
        }
        if (desc.reduce_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReducePlan: reduce_axis must be < rank",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ReducePlan: input_shape dims must be non-negative",
                ));
            }
        }

        // Supported matrix:
        //   {Sum, Mean, Max, Min, Prod, Norm2, LogSumExp, Var, Std}
        //                            × {f32, f16, bf16, f64}   (36 cells)
        // Argmax/Argmin live in `ArgReducePlan` (i64 output); trace
        // lives in `TracePlan` (scalar output, both axes reduced). The
        // remaining reserved discriminants (Any / All) land in later
        // fanout.
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
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let supported = kind_in_scope && dtype_in_scope;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReducePlan: supported matrix is \
                 {Sum, Mean, Max, Min, Prod, Norm2, LogSumExp, Var, Std} × \
                 {f32, f16, bf16, f64}; other (kind, dtype) pairs land \
                 in later fanout (Argmax/Argmin via ArgReducePlan; trace \
                 via TracePlan)",
            ));
        }

        // The naive trailblazer kernel sums in input-order (one thread
        // per output cell, sequential over the reduced axis). Result
        // is deterministic and bit-stable for f32 on the same hardware.
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
    pub fn can_implement(&self, args: &ReduceArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReducePlan: X shape mismatch with descriptor input_shape",
            ));
        }
        let expected_out = self.desc.output_shape();
        if args.y.shape != expected_out {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ReducePlan: Y shape mismatch with derived output shape \
                 (input shape with reduce_axis collapsed to 1)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ReducePlan: tensor rank > 8 not supported",
            ));
        }
        let y_numel = args.y.numel();
        let x_numel = args.x.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
            });
        }
        if x_len < x_numel {
            return Err(Error::BufferTooSmall {
                needed: x_numel as usize,
                got: x_len as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0` for the naive trailblazer.
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
        args: ReduceArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_numel = args.y.numel();
        if output_numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let output_shape = self.desc.output_shape();
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;
        let reduce_axis = self.desc.reduce_axis as i32;
        let reduce_extent = self.desc.input_shape[self.desc.reduce_axis as usize];
        let reduce_stride_x = args.x.stride[self.desc.reduce_axis as usize];

        // Helper: every reduce FFI symbol shares the same parameter
        // shape (the kernel template is shared). The macro picks the
        // right symbol from (kind, dtype).
        macro_rules! dispatch {
            ($sym:ident) => {{
                unsafe {
                    baracuda_kernels_sys::$sym(
                        output_numel,
                        rank,
                        output_shape.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        reduce_axis,
                        reduce_extent,
                        reduce_stride_x,
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }};
        }

        let status = match (self.desc.kind, T::KIND) {
            // Sum
            (ReduceKind::Sum, ElementKind::F32) => dispatch!(baracuda_kernels_reduce_sum_f32_run),
            (ReduceKind::Sum, ElementKind::F16) => dispatch!(baracuda_kernels_reduce_sum_f16_run),
            (ReduceKind::Sum, ElementKind::Bf16) => dispatch!(baracuda_kernels_reduce_sum_bf16_run),
            (ReduceKind::Sum, ElementKind::F64) => dispatch!(baracuda_kernels_reduce_sum_f64_run),
            // Mean
            (ReduceKind::Mean, ElementKind::F32) => dispatch!(baracuda_kernels_reduce_mean_f32_run),
            (ReduceKind::Mean, ElementKind::F16) => dispatch!(baracuda_kernels_reduce_mean_f16_run),
            (ReduceKind::Mean, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_mean_bf16_run)
            }
            (ReduceKind::Mean, ElementKind::F64) => dispatch!(baracuda_kernels_reduce_mean_f64_run),
            // Max
            (ReduceKind::Max, ElementKind::F32) => dispatch!(baracuda_kernels_reduce_max_f32_run),
            (ReduceKind::Max, ElementKind::F16) => dispatch!(baracuda_kernels_reduce_max_f16_run),
            (ReduceKind::Max, ElementKind::Bf16) => dispatch!(baracuda_kernels_reduce_max_bf16_run),
            (ReduceKind::Max, ElementKind::F64) => dispatch!(baracuda_kernels_reduce_max_f64_run),
            // Min
            (ReduceKind::Min, ElementKind::F32) => dispatch!(baracuda_kernels_reduce_min_f32_run),
            (ReduceKind::Min, ElementKind::F16) => dispatch!(baracuda_kernels_reduce_min_f16_run),
            (ReduceKind::Min, ElementKind::Bf16) => dispatch!(baracuda_kernels_reduce_min_bf16_run),
            (ReduceKind::Min, ElementKind::F64) => dispatch!(baracuda_kernels_reduce_min_f64_run),
            // Prod
            (ReduceKind::Prod, ElementKind::F32) => dispatch!(baracuda_kernels_reduce_prod_f32_run),
            (ReduceKind::Prod, ElementKind::F16) => dispatch!(baracuda_kernels_reduce_prod_f16_run),
            (ReduceKind::Prod, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_prod_bf16_run)
            }
            (ReduceKind::Prod, ElementKind::F64) => dispatch!(baracuda_kernels_reduce_prod_f64_run),
            // Norm2 = sqrt(sum(x^2)) — shares the simple-reduce
            // parameter shape; finalize() does the sqrt.
            (ReduceKind::Norm2, ElementKind::F32) => {
                dispatch!(baracuda_kernels_reduce_norm2_f32_run)
            }
            (ReduceKind::Norm2, ElementKind::F16) => {
                dispatch!(baracuda_kernels_reduce_norm2_f16_run)
            }
            (ReduceKind::Norm2, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_norm2_bf16_run)
            }
            (ReduceKind::Norm2, ElementKind::F64) => {
                dispatch!(baracuda_kernels_reduce_norm2_f64_run)
            }
            // LogSumExp — `y = log(sum(exp(x - max))) + max` via a
            // two-pass kernel (max, then sum-exp). Same FFI parameter
            // shape as the simple-reduce family.
            (ReduceKind::LogSumExp, ElementKind::F32) => {
                dispatch!(baracuda_kernels_reduce_logsumexp_f32_run)
            }
            (ReduceKind::LogSumExp, ElementKind::F16) => {
                dispatch!(baracuda_kernels_reduce_logsumexp_f16_run)
            }
            (ReduceKind::LogSumExp, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_reduce_logsumexp_bf16_run)
            }
            (ReduceKind::LogSumExp, ElementKind::F64) => {
                dispatch!(baracuda_kernels_reduce_logsumexp_f64_run)
            }
            // Var / Std take an extra `correction` parameter and route
            // through the Welford-family FFI symbols. Welford state runs
            // at f32 for f32/f16/bf16 and f64 for f64 (handled by the
            // `WelfordAcc<T>` trait inside the kernel template).
            (ReduceKind::Var, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_var_f32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Var, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_var_f16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Var, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_var_bf16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Var, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_var_f64_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Std, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_std_f32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Std, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_std_f16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Std, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_std_bf16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ReduceKind::Std, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_reduce_std_f64_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    self.desc.correction,
                    x_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ReducePlan::run: this (kind, dtype) cell is not yet wired",
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
