//! Ternary elementwise plan.
//!
//! 3→1 sibling of [`crate::BinaryPlan`]. Same-dtype-input, same-dtype-
//! output ops with three inputs (a, b, c) and one output (y).
//!
//! Wired matrix: {[`TernaryKind::Clamp`], [`TernaryKind::Fma`],
//! [`TernaryKind::Addcmul`], [`TernaryKind::Addcdiv`]} × {f32, f16,
//! bf16, f64} = 16 (kind, dtype) cells, each with both the contig fast
//! path and the strided / broadcast path (32 launchers total).
//!
//! Addcmul / Addcdiv read [`TernaryDescriptor::scale`] (PyTorch's
//! `value` parameter); Clamp / Fma ignore it. The dispatcher routes
//! parameterized ops through a separate FFI family that threads the
//! `scale` parameter through to the kernel.
//!
//! Reserved-but-deferred: [`TernaryKind::Where`] needs a
//! heterogeneous-dtype plan shape (its bool cond input is dtype `u8`,
//! not `T`) — see [`crate::WherePlan`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, TernaryKind, Workspace,
};

/// Descriptor for a ternary elementwise op.
///
/// `scale` is used by parameterized ops (`Addcmul`, `Addcdiv`) — set to
/// the `value` multiplier from PyTorch's `torch.addcmul(c, a, b,
/// value=k)` convention. Ignored by unparameterized ops (`Clamp`,
/// `Fma`). Default `1.0`.
#[derive(Copy, Clone, Debug)]
pub struct TernaryDescriptor<const N: usize> {
    /// Which ternary op to apply.
    pub kind: TernaryKind,
    /// Output tensor shape.
    pub shape: [i32; N],
    /// Element type (shared across a, b, c, y).
    pub element: ElementKind,
    /// Scalar multiplier for parameterized ops (`Addcmul`, `Addcdiv`).
    /// Unused by `Clamp` / `Fma` — pass `1.0` for those.
    pub scale: f32,
}

/// Args bundle for a ternary elementwise launch.
///
/// All four operands share dtype `T`. Each input may be broadcast to
/// `y.shape` via stride-0 axes (typical use case for `clamp(x, lo, hi)`
/// where `lo` and `hi` are scalars: pass them as rank-N tensors with
/// `shape[d] = 1` and `stride[d] = 0` on every axis).
pub struct TernaryArgs<'a, T: Element, const N: usize> {
    /// First input. For `clamp`, this is `x` (the value to clamp).
    pub a: TensorRef<'a, T, N>,
    /// Second input. For `clamp`, this is `lo` (the lower bound).
    pub b: TensorRef<'a, T, N>,
    /// Third input. For `clamp`, this is `hi` (the upper bound).
    pub c: TensorRef<'a, T, N>,
    /// Output.
    pub y: TensorMut<'a, T, N>,
}

/// Ternary elementwise plan.
///
/// `T: Element` is the kernel's element type (today: must be `f32`).
/// `const N: usize` is the tensor rank.
pub struct TernaryPlan<T: Element, const N: usize> {
    desc: TernaryDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> TernaryPlan<T, N> {
    /// Pick a kernel for `desc`. Returns [`Error::Unsupported`] if the
    /// `(kind, T::KIND)` pair isn't wired today.
    pub fn select(
        _stream: &Stream,
        desc: &TernaryDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TernaryPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::TernaryPlan: shape dims must be non-negative",
                ));
            }
        }

        // Wired matrix: {Clamp, Fma, Addcmul, Addcdiv} × {f32, f16,
        // bf16, f64}. Addcmul / Addcdiv read `desc.scale`; Clamp / Fma
        // ignore it. Where stays reserved-but-deferred — it requires a
        // separate heterogeneous-dtype plan ([`crate::WherePlan`]).
        let kind_in_scope = matches!(
            desc.kind,
            TernaryKind::Clamp | TernaryKind::Fma | TernaryKind::Addcmul | TernaryKind::Addcdiv
        );
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let supported = kind_in_scope && dtype_in_scope;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::TernaryPlan: this (kind, dtype) cell is not yet \
                 wired; see the dispatcher's kind / dtype scope for the supported set. \
                 Note: `Where` requires a separate heterogeneous-dtype plan \
                 (`crate::WherePlan`).",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::TernaryElementwise,
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

    /// Validate that this plan can launch with `args`.
    ///
    /// Per-axis broadcast compatibility: each input's `shape[d]` must
    /// match `y.shape[d]` or be 1 with `stride[d] == 0`.
    pub fn can_implement(&self, args: &TernaryArgs<'_, T, N>) -> Result<()> {
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TernaryPlan: Y shape mismatch with descriptor",
            ));
        }

        for d in 0..N {
            let y_dim = self.desc.shape[d];
            for (name, (op_dim, op_stride)) in [
                ("A", (args.a.shape[d], args.a.stride[d])),
                ("B", (args.b.shape[d], args.b.stride[d])),
                ("C", (args.c.shape[d], args.c.stride[d])),
            ] {
                if op_dim != y_dim && !(op_dim == 1 && op_stride == 0) {
                    let _ = name; // Error variant takes a static string;
                    // log the bad operand via a single shared message.
                    return Err(Error::InvalidProblem(
                        "baracuda-kernels::TernaryPlan: input axis not broadcast-compatible \
                         with output (require shape[d] == y.shape[d], OR \
                         shape[d] == 1 AND stride[d] == 0)",
                    ));
                }
            }
        }

        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::TernaryPlan: tensor rank > 8 not supported",
            ));
        }

        let y_numel = args.y.numel();
        let a_numel = args.a.numel();
        let b_numel = args.b.numel();
        let c_numel = args.c.numel();
        let a_len = args.a.data.len() as i64;
        let b_len = args.b.data.len() as i64;
        let c_len = args.c.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
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
        if c_len < c_numel {
            return Err(Error::BufferTooSmall {
                needed: c_numel as usize,
                got: c_len as usize,
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
        args: TernaryArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.y.numel();
        if numel == 0 {
            return Ok(());
        }
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let c_ptr = args.c.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let all_contig_same_shape = args.a.shape == args.y.shape
            && args.b.shape == args.y.shape
            && args.c.shape == args.y.shape
            && args.a.is_contiguous()
            && args.b.is_contiguous()
            && args.c.is_contiguous()
            && args.y.is_contiguous();

        if !all_contig_same_shape {
            return self.run_strided(stream_ptr, a_ptr, b_ptr, c_ptr, y_ptr, numel, &args);
        }

        let status = match (self.desc.kind, T::KIND) {
            // --- Clamp --------------------------------------------------
            (TernaryKind::Clamp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f32_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_bf16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f64_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Fma ----------------------------------------------------
            (TernaryKind::Fma, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_f32_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_f16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_bf16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_f64_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Addcmul (reads desc.scale) ------------------------------
            (TernaryKind::Addcmul, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_f32_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_f16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_bf16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_f64_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Addcdiv (reads desc.scale) ------------------------------
            (TernaryKind::Addcdiv, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_f32_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_f16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_bf16_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_f64_run(
                    numel, a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TernaryPlan::run reached an unimplemented \
                     (kind, dtype) — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }

    /// Strided / broadcast kernel path.
    fn run_strided(
        &self,
        stream_ptr: *mut c_void,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        c_ptr: *const c_void,
        y_ptr: *mut c_void,
        numel: i64,
        args: &TernaryArgs<'_, T, N>,
    ) -> Result<()> {
        let shape = args.y.shape;
        let stride_a = args.a.stride;
        let stride_b = args.b.stride;
        let stride_c = args.c.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        let status = match (self.desc.kind, T::KIND) {
            // --- Clamp --------------------------------------------------
            (TernaryKind::Clamp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Clamp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Fma ----------------------------------------------------
            (TernaryKind::Fma, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Fma, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_fma_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Addcmul (reads desc.scale) ------------------------------
            (TernaryKind::Addcmul, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcmul, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcmul_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Addcdiv (reads desc.scale) ------------------------------
            (TernaryKind::Addcdiv, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (TernaryKind::Addcdiv, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ternary_addcdiv_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(),
                    stride_c.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, c_ptr, y_ptr, self.desc.scale,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TernaryPlan::run_strided reached an \
                     unimplemented (kind, dtype) pair — select() should have caught this",
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
