//! Argmax / Argmin single-axis reduction.
//!
//! New plan shape from [`crate::ReducePlan`] because the output dtype
//! differs from the input dtype: input is `T: Element` (value), output
//! is `I: IndexOutputElement` (defaults to `i64` — PyTorch convention).
//!
//! **When to use**: forward argmax / argmin. No backward — `argmax` /
//! `argmin` are non-differentiable (gradient is zero almost everywhere).
//!
//! **Dtypes / shape**: `{Argmax, Argmin} × {f32, f16, bf16, f64}` value
//! input × `{u32, i32, i64}` index output; tensor rank `1..=8`; reduce
//! axis must be non-empty.
//!
//! **Tie-breaking**: returns the first-occurrence index along the
//! reduce axis (PyTorch convention).
//!
//! **Workspace**: none.
//!
//! **Precision**: deterministic, bit-stable on the same hardware (one-
//! thread-per-output-cell sequential scan over the reduce axis).
//!
//! Phase 12.2 (Fuel team feedback): output index dtype is now generic
//! over [`IndexOutputElement`] (`u32` / `i32` / `i64`). The legacy
//! default is `i64` so pre-Phase-12.2 callers compile unchanged; opt
//! into `u32` / `i32` via the third type parameter, e.g.
//! `ArgReducePlan::<f32, 3, u32>::select(...)`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, ArgReduceKind, BackendKind, Element, ElementKind, IndexOutputElement,
    IndexOutputKind, KernelSku, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee,
    TensorMut, TensorRef, Workspace,
};

/// Descriptor for an argmax / argmin axis reduction.
#[derive(Copy, Clone, Debug)]
pub struct ArgReduceDescriptor<const N: usize> {
    /// Which arg-reduction to apply.
    pub kind: ArgReduceKind,
    /// Input tensor shape.
    pub input_shape: [i32; N],
    /// Axis to reduce along.
    pub reduce_axis: u8,
    /// Input value element type.
    pub element: ElementKind,
}

impl<const N: usize> ArgReduceDescriptor<N> {
    /// Compute the output shape: input shape with reduce axis = 1.
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = self.input_shape;
        out[self.reduce_axis as usize] = 1;
        out
    }
}

/// Args bundle for an arg-reduction launch.
///
/// Note the asymmetric dtypes: `x` is the value dtype `T`, `y` is the
/// index dtype `I` (defaults to `i64` — PyTorch convention).
pub struct ArgReduceArgs<'a, T: Element, const N: usize, I: IndexOutputElement = i64> {
    /// Input.
    pub x: TensorRef<'a, T, N>,
    /// Output indices — shape matches input with reduce axis = 1. Type
    /// parameter `I` selects `u32`, `i32`, or `i64` (default).
    pub y: TensorMut<'a, I, N>,
}

/// Arg-reduce plan (argmax / argmin) — see module docs for dtypes,
/// tie-breaking, and precision.
///
/// `T: Element` is the value (input) dtype; `I: IndexOutputElement` is
/// the output index dtype (defaults to `i64`). `const N: usize` is the
/// tensor rank (1..=8).
///
/// The `I = i64` default preserves source-compat for pre-Phase-12.2
/// callers; new callers opt into narrower output dtypes via
/// `ArgReducePlan::<T, N, u32>::select(...)` or `<T, N, i32>`.
pub struct ArgReducePlan<T: Element, const N: usize, I: IndexOutputElement = i64> {
    desc: ArgReduceDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<(T, I)>,
}

impl<T: Element, const N: usize, I: IndexOutputElement> ArgReducePlan<T, N, I> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ArgReduceDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ArgReducePlan: descriptor element != type parameter T",
            ));
        }
        if (desc.reduce_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgReducePlan: reduce_axis must be < rank",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ArgReducePlan: input_shape dims must be non-negative",
                ));
            }
        }
        if desc.input_shape[desc.reduce_axis as usize] <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgReducePlan: cannot arg-reduce over an empty axis",
            ));
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ArgReducePlan: today only `f32`, `f16`, `bf16`, `f64` \
                 value dtypes are wired; other dtypes land in future fanout",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        // Distinguish the three output-dtype SKUs via `aux_element`.
        // `ElementKind` has I32 / I64 variants; for u32 we fall back to
        // `None` (the only output dtype without a matching ElementKind
        // — kernel selection is still uniquely keyed by `I::KIND` in
        // `run`, this tag is informational).
        let aux_element = match I::KIND {
            IndexOutputKind::U32 => None,
            IndexOutputKind::I32 => Some(ElementKind::I32),
            IndexOutputKind::I64 => Some(ElementKind::I64),
            // Defensive arm — `IndexOutputKind` is `#[non_exhaustive]`,
            // so unrecognized variants surface as a `None` aux tag
            // until a wired case is added.
            _ => None,
        };
        let sku = KernelSku {
            category: OpCategory::Reduction,
            op: desc.kind as u16,
            element: T::KIND,
            aux_element,
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
    pub fn can_implement(&self, args: &ArgReduceArgs<'_, T, N, I>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgReducePlan: X shape mismatch with descriptor",
            ));
        }
        let expected_out = self.desc.output_shape();
        if args.y.shape != expected_out {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ArgReducePlan: Y shape mismatch with derived output \
                 shape (input shape with reduce_axis collapsed to 1)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ArgReducePlan: tensor rank > 8 not supported",
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
        args: ArgReduceArgs<'_, T, N, I>,
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

        let status = match (self.desc.kind, T::KIND, I::KIND) {
            // -----------------------------------------------------------------
            // i64 output (legacy / default).
            // -----------------------------------------------------------------
            (ArgReduceKind::Argmax, ElementKind::F32, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F32, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::F16, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F16, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::Bf16, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_bf16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::Bf16, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_bf16_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::F64, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f64_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F64, IndexOutputKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f64_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -----------------------------------------------------------------
            // u32 output (Phase 12.2).
            // -----------------------------------------------------------------
            (ArgReduceKind::Argmax, ElementKind::F32, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f32_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F32, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f32_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::F16, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f16_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F16, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f16_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::Bf16, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_bf16_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::Bf16, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_bf16_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::F64, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f64_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F64, IndexOutputKind::U32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f64_u32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // -----------------------------------------------------------------
            // i32 output (Phase 12.2).
            // -----------------------------------------------------------------
            (ArgReduceKind::Argmax, ElementKind::F32, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f32_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F32, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f32_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::F16, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f16_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F16, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f16_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::Bf16, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_bf16_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::Bf16, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_bf16_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmax, ElementKind::F64, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmax_f64_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ArgReduceKind::Argmin, ElementKind::F64, IndexOutputKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_arg_reduce_argmin_f64_i32_run(
                    output_numel, rank, output_shape.as_ptr(),
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    reduce_axis, reduce_extent, reduce_stride_x,
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ArgReducePlan::run: only `{Argmax,Argmin} × \
                     {f32,f16,bf16,f64} × {u32,i32,i64}` wired today",
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
