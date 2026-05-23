//! `tril` backward plan (Phase 13.4).
//!
//! `d_input = tril(d_output, diagonal)` — the tril mask is its own
//! adjoint, so the BW reuses the FW kernel with the same arguments
//! pointed at the gradient buffers.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `tril` backward op.
///
/// Mirrors [`crate::TrilDescriptor`] — same `shape` and `diagonal`.
#[derive(Copy, Clone, Debug)]
pub struct TrilBackwardDescriptor<const N: usize> {
    /// Tensor shape (tril preserves shape).
    pub shape: [i32; N],
    /// Diagonal offset (same as the forward).
    pub diagonal: i32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Tril backward launch.
pub struct TrilBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — same shape as the forward output (== input).
    pub grad_output: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input — same shape as `grad_output`.
    pub grad_input: TensorMut<'a, T, N>,
}

/// `tril` backward plan.
///
/// `d_input = torch.tril(d_output, diagonal)`. Adjoint of
/// [`crate::TrilPlan`].
///
/// **When to use**: BW for [`TrilPlan`](crate::TrilPlan).
///
/// **Dtypes**: `{f16, bf16, f32, f64, i32, i64, Bool}`.
///
/// **Shape limits**: rank in `[2, 8]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact.
pub struct TrilBackwardPlan<T: Element, const N: usize> {
    desc: TrilBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> TrilBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &TrilBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TrilBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if N < 2 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TrilBackwardPlan: tensor rank must be >= 2",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::TrilBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::TrilBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if !dtype_in_scope(T::KIND) {
            return Err(Error::Unsupported(
                "baracuda-kernels::TrilBackwardPlan: dtype not wired; supported set is \
                 {f16, bf16, f32, f64, i32, i64, Bool}",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Tril as u16,
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
    pub fn can_implement(&self, args: &TrilBackwardArgs<'_, T, N>) -> Result<()> {
        if args.grad_output.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TrilBackwardPlan: grad_output shape mismatch with descriptor",
            ));
        }
        if args.grad_input.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TrilBackwardPlan: grad_input shape mismatch with descriptor",
            ));
        }
        let numel = args.grad_input.numel();
        let dy_len = args.grad_output.data.len() as i64;
        let dx_len = args.grad_input.data.len() as i64;
        if dy_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(dx_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0`.
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

    /// Launch — dispatches to the forward `tril_<dtype>_run` symbol with
    /// `grad_output` as the input and `grad_input` as the output.
    ///
    /// Dispatch policy mirrors [`crate::TrilPlan::run`]: canonical-contig
    /// fast path routes to the contig FFI; any non-canonical layout
    /// routes to the strided sibling (Phase 14.3).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: TrilBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.grad_input.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.grad_output.data.as_raw().0 as *const c_void;
        let dx_ptr = args.grad_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape = self.desc.shape;
        let rank = N as i32;
        let diagonal = self.desc.diagonal;

        let all_contig =
            args.grad_output.is_contiguous() && args.grad_input.is_contiguous();

        if !all_contig {
            let stride_x = args.grad_output.stride;
            let stride_y = args.grad_input.stride;
            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_tril_f16_strided_run(
                        dy_ptr, dx_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_tril_bf16_strided_run(
                        dy_ptr, dx_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_tril_f32_strided_run(
                        dy_ptr, dx_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::F64 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_tril_f64_strided_run(
                        dy_ptr, dx_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_tril_i32_strided_run(
                        dy_ptr, dx_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::I64 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_tril_i64_strided_run(
                        dy_ptr, dx_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::Bool => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_tril_bool_strided_run(
                        dy_ptr, dx_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::TrilBackwardPlan::run: dtype not wired (strided)",
                    ));
                }
            };
            return map_status(status);
        }

        let status = match T::KIND {
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_tril_f16_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_tril_bf16_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_tril_f32_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_tril_f64_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_tril_i32_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_tril_i64_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::Bool => unsafe {
                baracuda_kernels_sys::baracuda_kernels_tril_bool_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TrilBackwardPlan::run: dtype not wired",
                ));
            }
        };
        map_status(status)
    }
}

fn dtype_in_scope(k: ElementKind) -> bool {
    matches!(
        k,
        ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::F32
            | ElementKind::F64
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::Bool
    )
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
