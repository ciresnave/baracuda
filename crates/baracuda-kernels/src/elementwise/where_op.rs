//! Heterogeneous-dtype ternary plan: `where(cond, a, b)`.
//!
//! Distinct from [`crate::TernaryPlan`] because the cond input has a
//! different dtype (`u8` — PyTorch / NumPy bool storage convention)
//! from the value inputs and output. `y = cond ? a : b` elementwise,
//! with full broadcast support on every operand including the cond.
//!
//! All 4 FP value dtypes wired: {f32, f16, bf16, f64} × {contig,
//! strided}. The op does no arithmetic — pure element selection —
//! so output is bit-exact against host reference regardless of dtype.
//!
//! Module name: `where_op` (rather than `where`) because `where` is a
//! Rust keyword.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `where` op.
///
/// `shape` is the output tensor shape. `element` is the **value** dtype
/// — cond is always `u8`. `element` must match the type parameter `T`
/// of the containing plan at `select` time.
///
/// No `kind` field because `where` is the only heterogeneous-dtype
/// ternary op in this Plan today. If future ops join (e.g., a
/// `masked_fill` variant), they get their own plan or a kind enum gets
/// introduced — choice deferred until that lands.
#[derive(Copy, Clone, Debug)]
pub struct WhereDescriptor<const N: usize> {
    /// Output tensor shape.
    pub shape: [i32; N],
    /// Value element type (a / b / y dtype; cond is always `u8`).
    pub element: ElementKind,
}

/// Args bundle for a `where` launch.
///
/// `cond` is `u8` (0 = false, non-zero = true). `a`, `b`, `y` share
/// dtype `T`. All four operands can broadcast independently to
/// `y.shape` via stride-0 axes.
pub struct WhereArgs<'a, T: Element, const N: usize> {
    /// Boolean mask. `0u8` selects `b`, any other value selects `a`.
    pub cond: TensorRef<'a, u8, N>,
    /// Value selected where `cond != 0`.
    pub a: TensorRef<'a, T, N>,
    /// Value selected where `cond == 0`.
    pub b: TensorRef<'a, T, N>,
    /// Output.
    pub y: TensorMut<'a, T, N>,
}

/// `where(cond, a, b)` plan with heterogeneous-dtype inputs.
///
/// `T: Element` is the value dtype (a / b / y). The cond is always `u8`.
/// `const N: usize` is the tensor rank.
pub struct WherePlan<T: Element, const N: usize> {
    desc: WhereDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> WherePlan<T, N> {
    /// Pick a kernel for `desc`. Returns [`Error::Unsupported`] if the
    /// value dtype isn't wired today.
    pub fn select(
        _stream: &Stream,
        desc: &WhereDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::WherePlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::WherePlan: shape dims must be non-negative",
                ));
            }
        }

        // All 4 FP value dtypes wired.
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::WherePlan: value dtype must be one of \
                 {F32, F16, Bf16, F64}",
            ));
        }

        // `where` is a pure select — no math, fully deterministic and
        // bit-stable on the same hardware. The MathPrecision tag
        // mirrors the value dtype by convention even though no
        // arithmetic happens.
        let (math_precision, accumulator) = match T::KIND {
            ElementKind::F16 => (MathPrecision::F16, ElementKind::F16),
            ElementKind::Bf16 => (MathPrecision::Bf16, ElementKind::Bf16),
            ElementKind::F64 => (MathPrecision::F64, ElementKind::F64),
            _ => (MathPrecision::F32, ElementKind::F32),
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::TernaryElementwise,
            // `op` discriminant: `TernaryKind::Where` lives in the
            // shared op enum but is intentionally not routed via
            // `TernaryPlan` — we tag the SKU with its discriminant
            // value (4) so telemetry / autotuner-cache keys
            // distinguish this from same-dtype ternary ops.
            op: 4,
            element: T::KIND,
            // `aux_element` captures cond's dtype — but ElementKind
            // doesn't have a `U8` variant today, so leave None and
            // rely on the `Where`-specific op discriminant.
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

    /// Validate that this plan can launch with `args`.
    pub fn can_implement(&self, args: &WhereArgs<'_, T, N>) -> Result<()> {
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::WherePlan: Y shape mismatch with descriptor",
            ));
        }

        // Per-axis broadcast compatibility check for all four operands.
        for d in 0..N {
            let y_dim = self.desc.shape[d];
            let checks = [
                (args.cond.shape[d], args.cond.stride[d]),
                (args.a.shape[d], args.a.stride[d]),
                (args.b.shape[d], args.b.stride[d]),
            ];
            for (op_dim, op_stride) in checks {
                if op_dim != y_dim && !(op_dim == 1 && op_stride == 0) {
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::WherePlan: input axis not broadcast-compatible \
                         with output (require shape[d] == y.shape[d], OR \
                         shape[d] == 1 AND stride[d] == 0)",
                    ));
                }
            }
        }

        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::WherePlan: tensor rank > 8 not supported",
            ));
        }

        let y_numel = args.y.numel();
        let cond_numel = args.cond.numel();
        let a_numel = args.a.numel();
        let b_numel = args.b.numel();
        let cond_len = args.cond.data.len() as i64;
        let a_len = args.a.data.len() as i64;
        let b_len = args.b.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
            });
        }
        if cond_len < cond_numel {
            return Err(Error::BufferTooSmall {
                needed: cond_numel as usize,
                got: cond_len as usize,
            });
        }
        if a_len < a_numel {
            return Err(Error::BufferTooSmall {
                needed: a_numel as usize,
                got: a_len as usize,
            });
        }
        if b_len < b_numel {
            return Err(Error::BufferTooSmall {
                needed: b_numel as usize,
                got: b_len as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0` for the trailblazer.
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
        args: WhereArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.y.numel();
        if numel == 0 {
            return Ok(());
        }
        let cond_ptr = args.cond.data.as_raw().0 as *const c_void;
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let all_contig_same_shape = args.cond.shape == args.y.shape
            && args.a.shape == args.y.shape
            && args.b.shape == args.y.shape
            && args.cond.is_contiguous()
            && args.a.is_contiguous()
            && args.b.is_contiguous()
            && args.y.is_contiguous();

        if !all_contig_same_shape {
            return self.run_strided(
                stream_ptr, cond_ptr, a_ptr, b_ptr, y_ptr, numel, &args,
            );
        }

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_f32_run(
                    numel,
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_f16_run(
                    numel,
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_bf16_run(
                    numel,
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_f64_run(
                    numel,
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::WherePlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }

    /// Strided / broadcast kernel path.
    fn run_strided(
        &self,
        stream_ptr: *mut c_void,
        cond_ptr: *const c_void,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        y_ptr: *mut c_void,
        numel: i64,
        args: &WhereArgs<'_, T, N>,
    ) -> Result<()> {
        let shape = args.y.shape;
        let stride_cond = args.cond.stride;
        let stride_a = args.a.stride;
        let stride_b = args.b.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_f32_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_cond.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_f16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_cond.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_bf16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_cond.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_where_f64_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_cond.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    cond_ptr,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::WherePlan: strided path reached unimplemented dtype \
                     — select() should have caught this",
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
