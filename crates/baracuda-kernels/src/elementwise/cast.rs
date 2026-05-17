//! Cast plan — heterogeneous dtype conversion (`y = (TOut) x`).
//!
//! Distinct from the same-dtype [`crate::UnaryPlan`] family because the
//! input and output element types differ. The plan is generic over
//! both `TIn: Element` and `TOut: Element`, with [`select`] dispatching
//! on the runtime `(input_element, output_element)` pair.
//!
//! Today wired (at the Plan-layer dispatch): every cross-dtype pair in
//! `{f32, f64, f16, bf16, i32, i64} × {same}`. The kernels in
//! `baracuda-kernels-sys` cover a broader set (also `u8` / `i8`
//! endpoints) — those would route via the [`IntElement`] family with a
//! distinct plan shape, deferred. Bool is not wired today either —
//! its truthiness convention (`x != 0 → 1`) would need a dedicated
//! kernel rather than a pure `static_cast`. FP8 endpoints land in a
//! future Phase-2 FP8 fanout. The kernel itself is contig-only —
//! baracuda's plan layer materializes strided views upstream.
//!
//! [`IntElement`]: baracuda_kernels_types::IntElement

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};

/// Descriptor for a dtype cast.
///
/// `numel` is the total number of elements (both input and output have
/// the same number — cast doesn't change shape). `input_element` must
/// match `TIn::KIND` and `output_element` must match `TOut::KIND` at
/// `select` time.
#[derive(Copy, Clone, Debug)]
pub struct CastDescriptor {
    /// Number of elements in both input and output.
    pub numel: i32,
    /// Input element type.
    pub input_element: ElementKind,
    /// Output element type.
    pub output_element: ElementKind,
}

/// Args bundle for a cast launch. Both `input` and `output` are
/// rank-1 contiguous views over `numel` elements.
pub struct CastArgs<'a, TIn: Element, TOut: Element> {
    /// Input — `TIn` element type.
    pub input: TensorRef<'a, TIn, 1>,
    /// Output — `TOut` element type.
    pub output: TensorMut<'a, TOut, 1>,
}

/// Cast plan.
///
/// `TIn` is the input element type. `TOut` is the output element type.
pub struct CastPlan<TIn: Element, TOut: Element> {
    desc: CastDescriptor,
    sku: KernelSku,
    _marker_in: PhantomData<TIn>,
    _marker_out: PhantomData<TOut>,
}

impl<TIn: Element, TOut: Element> CastPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &CastDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CastPlan: descriptor input_element != type parameter TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::CastPlan: descriptor output_element != type parameter TOut",
            ));
        }
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CastPlan: numel must be non-negative",
            ));
        }
        if !pair_in_scope(TIn::KIND, TOut::KIND) {
            return Err(Error::Unsupported(
                "baracuda-kernels::CastPlan: this (TIn, TOut) pair is not wired today; \
                 supported set is {f32, f64, f16, bf16, i32, i64} × {same}",
            ));
        }

        // Cast is a pure copy + numeric conversion — no fused math.
        // Precision guarantee reflects the f32 detour for half-precision
        // endpoints but is a no-op (bit-stable copy) for same-dtype
        // pairs and for purely integer cross-casts.
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::UnaryElementwise,
            op: UnaryKind::Cast as u16,
            element: TIn::KIND,
            aux_element: Some(TOut::KIND),
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker_in: PhantomData,
            _marker_out: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &CastArgs<'_, TIn, TOut>) -> Result<()> {
        let expected = self.desc.numel as i64;
        if args.input.numel() != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CastPlan: input numel mismatch with descriptor",
            ));
        }
        if args.output.numel() != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CastPlan: output numel mismatch with descriptor",
            ));
        }
        if (args.input.data.len() as i64) < expected {
            return Err(Error::BufferTooSmall {
                needed: expected as usize,
                got: args.input.data.len(),
            });
        }
        if (args.output.data.len() as i64) < expected {
            return Err(Error::BufferTooSmall {
                needed: expected as usize,
                got: args.output.data.len(),
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
        args: CastArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let y_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        // Dispatch table — each cell calls the matching
        // `cast_<sin>_<sout>` FFI. The `match` is on the runtime kinds
        // (which `select` already proved consistent with TIn / TOut);
        // unreachable arm is the "select bug" guard.
        let status = match (TIn::KIND, TOut::KIND) {
            // f32 -> *
            (ElementKind::F32, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_f64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_i32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_i64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // f64 -> *
            (ElementKind::F64, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f64_f32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F64, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f64_f64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F64, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f64_f16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F64, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f64_bf16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F64, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f64_i32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F64, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f64_i64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            // f16 -> *
            (ElementKind::F16, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_f32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F16, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_f64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F16, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_f16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F16, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_bf16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F16, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_i32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::F16, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_i64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            // bf16 -> *
            (ElementKind::Bf16, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_f32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::Bf16, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_f64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::Bf16, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_f16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::Bf16, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_bf16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::Bf16, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_i32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::Bf16, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_i64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            // i32 -> *
            (ElementKind::I32, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_f32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I32, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_f64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I32, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_f16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I32, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_bf16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I32, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_i32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I32, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_i64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            // i64 -> *
            (ElementKind::I64, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_f32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I64, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_f64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I64, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_f16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I64, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_bf16_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I64, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_i32_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            (ElementKind::I64, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_i64_run(numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr)
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CastPlan::run reached an unimplemented \
                     (TIn, TOut) pair — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

/// Supported (TIn, TOut) pairs — same set the dispatch table covers.
fn pair_in_scope(input: ElementKind, output: ElementKind) -> bool {
    fn allowed(k: ElementKind) -> bool {
        matches!(
            k,
            ElementKind::F32
                | ElementKind::F64
                | ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::I32
                | ElementKind::I64
        )
    }
    allowed(input) && allowed(output)
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
