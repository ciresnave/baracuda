//! `triu` plan — upper-triangular matrix mask (Phase 13.4).
//!
//! `output[..., i, j] = input[..., i, j] if j >= i + diagonal else 0`.
//! Operates on the last two dims of a rank-≥2 tensor; the batch prefix
//! (anything before the matrix axes) is masked independently with the
//! same `diagonal`. Output shape equals input shape.
//!
//! - `diagonal == 0`: main diagonal.
//! - `diagonal > 0`: shift the kept region UP (triu keeps more).
//! - `diagonal < 0`: shift the kept region DOWN (triu keeps less).
//!
//! Driven by Fuel team's CPU-only triu/tril gap. Pair with
//! [`TriuBackwardPlan`](crate::TriuBackwardPlan) — the BW applies the
//! same mask to the upstream gradient (`d_input = triu(d_output,
//! diagonal)`) so it reuses the FW launch symbol.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `triu` op.
///
/// `shape` is the logical tensor shape (input and output agree).
/// `diagonal` shifts the mask boundary — `0` is the main diagonal,
/// positive shifts up (more zeros below), negative shifts down (more
/// elements kept).
#[derive(Copy, Clone, Debug)]
pub struct TriuDescriptor<const N: usize> {
    /// Logical tensor shape. `N >= 2` enforced at `select` time.
    pub shape: [i32; N],
    /// Diagonal offset. `0` == main diagonal.
    pub diagonal: i32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Triu launch.
pub struct TriuArgs<'a, T: Element, const N: usize> {
    /// Input — same shape as output.
    pub input: TensorRef<'a, T, N>,
    /// Output — same shape as input. Off-diagonal positions are zeroed.
    pub output: TensorMut<'a, T, N>,
}

/// `triu` plan.
///
/// `y = torch.triu(x, diagonal)` — upper-triangular mask on the last
/// two dims of `x`.
///
/// **When to use**: forward triu. Pair with
/// [`TriuBackwardPlan`](crate::TriuBackwardPlan) — the BW is the same
/// mask applied to `d_output`.
///
/// **Dtypes**: `{f16, bf16, f32, f64, i32, i64, Bool}`.
///
/// **Shape limits**: rank in `[2, 8]`; last two dims (`M = shape[N-2]`,
/// `N_cols = shape[N-1]`) define the matrix; everything before is the
/// batch prefix.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact —
/// pure element select + zero, no arithmetic.
pub struct TriuPlan<T: Element, const N: usize> {
    desc: TriuDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> TriuPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &TriuDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TriuPlan: descriptor element != type parameter T",
            ));
        }
        if N < 2 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TriuPlan: tensor rank must be >= 2 \
                 (need at least an (M, N) matrix to mask)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::TriuPlan: tensor rank > 8 not supported",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::TriuPlan: shape dims must be non-negative",
                ));
            }
        }
        if !dtype_in_scope(T::KIND) {
            return Err(Error::Unsupported(
                "baracuda-kernels::TriuPlan: dtype not wired; supported set is \
                 {f16, bf16, f32, f64, i32, i64, Bool}",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pure select-or-zero — no arithmetic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Triu as u16,
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
    pub fn can_implement(&self, args: &TriuArgs<'_, T, N>) -> Result<()> {
        if args.input.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TriuPlan: input shape mismatch with descriptor",
            ));
        }
        if args.output.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TriuPlan: output shape mismatch with descriptor",
            ));
        }
        let numel = args.output.numel();
        let in_len = args.input.data.len() as i64;
        let out_len = args.output.data.len() as i64;
        if in_len < numel || out_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: in_len.min(out_len) as usize,
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

    /// Launch.
    ///
    /// Dispatch policy: if both `input` and `output` are canonical
    /// row-major contiguous, route to the contig fast path
    /// (`baracuda_kernels_triu_<dtype>_run`). Otherwise route to the
    /// strided sibling (`baracuda_kernels_triu_<dtype>_strided_run`,
    /// Phase 14.3) which threads per-axis signed strides for input
    /// and output through the kernel parameter block.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: TriuArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.output.numel();
        if numel == 0 {
            return Ok(());
        }
        let input_ptr = args.input.data.as_raw().0 as *const c_void;
        let output_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape = self.desc.shape;
        let rank = N as i32;
        let diagonal = self.desc.diagonal;

        // Canonical-contig fast path: both operands canonical row-major.
        // Any other layout (transposed, sliced, broadcast, flipped)
        // routes to the strided sibling.
        let all_contig = args.input.is_contiguous() && args.output.is_contiguous();

        if !all_contig {
            let stride_x = args.input.stride;
            let stride_y = args.output.stride;
            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_triu_f16_strided_run(
                        input_ptr, output_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_triu_bf16_strided_run(
                        input_ptr, output_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_triu_f32_strided_run(
                        input_ptr, output_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::F64 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_triu_f64_strided_run(
                        input_ptr, output_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::I32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_triu_i32_strided_run(
                        input_ptr, output_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::I64 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_triu_i64_strided_run(
                        input_ptr, output_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                ElementKind::Bool => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_triu_bool_strided_run(
                        input_ptr, output_ptr, shape.as_ptr(), rank,
                        stride_x.as_ptr(), stride_y.as_ptr(), diagonal, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::TriuPlan::run: dtype not wired (strided) \
                         (should have been rejected at select())",
                    ));
                }
            };
            return map_status(status);
        }

        let status = match T::KIND {
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_f16_run(
                    input_ptr, output_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_bf16_run(
                    input_ptr, output_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_f32_run(
                    input_ptr, output_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_f64_run(
                    input_ptr, output_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_i32_run(
                    input_ptr, output_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_i64_run(
                    input_ptr, output_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::Bool => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_bool_run(
                    input_ptr, output_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TriuPlan::run: dtype not wired \
                     (should have been rejected at select())",
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
