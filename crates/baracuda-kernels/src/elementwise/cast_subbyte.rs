//! Sub-byte / non-arithmetic cast plan — sibling of [`crate::CastPlan`].
//!
//! Phase 13.3 adds cast paths for dtypes that the classic `CastPlan`
//! doesn't cover because their `Element`-trait status is different OR
//! their conversion semantics aren't a plain `static_cast<TOut>(x)`:
//!
//! - **`Bool` ↔ `T`** — `Bool` IS an `Element`, but the per-element
//!   conversion is **0/non-zero truthiness**, not arithmetic. T → Bool
//!   produces strictly 0 or 1; Bool → T produces 0.0 or 1.0 (the
//!   conversion normalizes whatever non-zero byte the caller stored).
//!
//! - **`Fp8E4M3` / `Fp8E5M2` ↔ {`f32`, `f16`, `bf16`}** — Fp8 lives on
//!   the [`FpElement`] sibling trait, not [`Element`]. Conversions
//!   route through `f32` via NVIDIA's `__nv_cvt_*_fp8` intrinsics with
//!   `SATFINITE` semantics (the wide → narrow direction clamps `|x|` to
//!   the format's max-finite — 448 for E4M3, 57344 for E5M2 — instead
//!   of producing infinities). Same convention as the Phase 2 Fp8 GEMM
//!   epilogue.
//!
//! - **`S4` / `U4` ↔ {`i32`, `i64`, `f32`}** — S4 / U4 are
//!   nibble-packed (one byte = two elements; low nibble = even index,
//!   high nibble = odd index). The wide → narrow direction (**PACK**)
//!   saturates to `[-8, 7]` for S4 or `[0, 15]` for U4 before
//!   nibble-masking. The narrow → wide direction (**UNPACK**) sign-
//!   extends (S4) or zero-extends (U4). `numel` is the element count
//!   and must be even; the packed buffer holds `numel / 2` bytes.
//!
//! ## Why a sibling plan instead of widening `CastPlan`'s trait bound?
//!
//! `CastPlan<TIn: Element, TOut: Element>` is parameterized on the
//! `Element` trait. The sub-byte dtypes (`S4`, `U4`, `Fp8E4M3`,
//! `Fp8E5M2`) live on `IntElement` / `FpElement` siblings. Widening
//! `CastPlan`'s bound to a common ancestor (e.g. `DeviceRepr + Copy +
//! 'static` as `ContiguizePlan` does) would force every existing caller
//! to either add a new bound or absorb a regression. The Phase 13.3
//! sibling plan keeps `CastPlan` source-compatible.
//!
//! `S8` and `U8` ARE covered by both plans — the existing `CastPlan`
//! reaches them via the `i8` / `u8` FFI symbols already present in
//! `cast.cu`. `CastSubBytePlan` doesn't re-wire those; if the caller
//! has S8/U8 endpoints they should still use `CastPlan` (which keeps
//! the full {fp + int + i8 + u8} × {same} matrix).
//!
//! No backward — Cast is treated as identity by autograd (same
//! convention as `CastPlan`).
//!
//! [`FpElement`]: baracuda_kernels_types::FpElement

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, KernelSku, MathPrecision, OpCategory, PlanPreference,
    PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};
use baracuda_types::DeviceRepr;

/// Descriptor for a sub-byte cast.
///
/// `numel` is the element count for both input and output (cast doesn't
/// change shape). For S4 / U4 endpoints `numel` MUST be even — the
/// packed buffer is `numel / 2` bytes, and odd numels would leave a
/// dangling half-byte.
///
/// `input_element` and `output_element` are the runtime dtype tags;
/// they must match `TIn` / `TOut`'s underlying width (the plan
/// cross-checks via `sizeof::<T>()` at select time).
#[derive(Copy, Clone, Debug)]
pub struct CastSubByteDescriptor {
    /// Number of elements in both input and output.
    pub numel: i32,
    /// Input element type.
    pub input_element: ElementKind,
    /// Output element type.
    pub output_element: ElementKind,
}

/// Args bundle for a [`CastSubBytePlan`] launch.
///
/// `input` and `output` are rank-1 contiguous views. For S4 / U4
/// endpoints the `data` buffer is the **packed byte storage** (length =
/// `numel / 2` bytes), not the element view; the caller types it as
/// `TensorRef<'a, S4, 1>` and the plan handles the `numel / 2` byte
/// math internally.
pub struct CastSubByteArgs<'a, TIn: DeviceRepr + Copy + 'static, TOut: DeviceRepr + Copy + 'static>
{
    /// Input — `TIn` element type.
    pub input: TensorRef<'a, TIn, 1>,
    /// Output — `TOut` element type.
    pub output: TensorMut<'a, TOut, 1>,
}

/// Sub-byte cast plan.
///
/// `TIn` is the input element type, `TOut` is the output element type.
/// Both are bounded by [`baracuda_types::DeviceRepr`] only (not
/// [`baracuda_kernels_types::Element`]) so the dtype set can include
/// `S4`, `U4`, `Fp8E4M3`, `Fp8E5M2`, and `Bool` alongside the classic
/// fp / int element types.
///
/// **Coverage**: see the [crate-level module docs](self) for the full
/// supported (`TIn`, `TOut`) pair list. A select-time check rejects
/// any pair outside the explicit table with [`Error::Unsupported`].
///
/// **Workspace**: none.
pub struct CastSubBytePlan<
    TIn: DeviceRepr + Copy + 'static,
    TOut: DeviceRepr + Copy + 'static,
> {
    desc: CastSubByteDescriptor,
    sku: KernelSku,
    _marker_in: PhantomData<TIn>,
    _marker_out: PhantomData<TOut>,
}

impl<TIn: DeviceRepr + Copy + 'static, TOut: DeviceRepr + Copy + 'static>
    CastSubBytePlan<TIn, TOut>
{
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &CastSubByteDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if !type_size_matches_kind::<TIn>(desc.input_element) {
            return Err(Error::Unsupported(
                "baracuda-kernels::CastSubBytePlan: sizeof::<TIn>() does not match \
                 descriptor input_element width",
            ));
        }
        if !type_size_matches_kind::<TOut>(desc.output_element) {
            return Err(Error::Unsupported(
                "baracuda-kernels::CastSubBytePlan: sizeof::<TOut>() does not match \
                 descriptor output_element width",
            ));
        }
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CastSubBytePlan: numel must be non-negative",
            ));
        }
        // S4 / U4 endpoints require even numel (nibble packing).
        let inv = matches!(
            desc.input_element,
            ElementKind::S4 | ElementKind::U4
        );
        let outv = matches!(
            desc.output_element,
            ElementKind::S4 | ElementKind::U4
        );
        if (inv || outv) && (desc.numel % 2 != 0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::CastSubBytePlan: S4 / U4 endpoints require even numel \
                 (packed buffer is numel/2 bytes)",
            ));
        }
        if !pair_in_scope(desc.input_element, desc.output_element) {
            return Err(Error::Unsupported(
                "baracuda-kernels::CastSubBytePlan: (input, output) pair not in scope \
                 for Phase 13.3 — see module docs for the wired set",
            ));
        }

        // Conversions either go through f32 (Fp8 / nibble→f32) or are
        // pure integer extension / truthiness. Either way the precision
        // contract is bit-stable on the same hardware (no
        // tensor-core reduction, no thread-cooperation, no race).
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::UnaryElementwise,
            op: UnaryKind::Cast as u16,
            element: desc.input_element,
            aux_element: Some(desc.output_element),
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

    /// Validate args. Checks numel agreement and buffer sizing (the S4 /
    /// U4 packed-buffer accounting collapses to `numel / 2` bytes,
    /// which is `numel / 2` packed-slot elements at the buffer layer).
    pub fn can_implement(&self, args: &CastSubByteArgs<'_, TIn, TOut>) -> Result<()> {
        let expected = self.desc.numel as i64;
        // For S4 / U4 endpoints, the TensorRef's `numel()` returns the
        // packed-slot count (one slot per byte = two logical elements).
        // We accept either the logical-element count OR the packed-slot
        // count here — the contract is "the buffer has enough storage
        // for `numel` logical elements".
        let in_packed = matches!(self.desc.input_element, ElementKind::S4 | ElementKind::U4);
        let out_packed = matches!(self.desc.output_element, ElementKind::S4 | ElementKind::U4);

        let needed_in = if in_packed { (expected + 1) / 2 } else { expected };
        let needed_out = if out_packed { (expected + 1) / 2 } else { expected };

        if (args.input.data.len() as i64) < needed_in {
            return Err(Error::BufferTooSmall {
                needed: needed_in as usize,
                got: args.input.data.len(),
            });
        }
        if (args.output.data.len() as i64) < needed_out {
            return Err(Error::BufferTooSmall {
                needed: needed_out as usize,
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
        args: CastSubByteArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let y_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (self.desc.input_element, self.desc.output_element) {
            // ---- Bool -> T ----
            (ElementKind::Bool, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bool_i32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bool, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bool_i64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bool, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bool_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bool, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bool_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bool, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bool_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ---- T -> Bool ----
            (ElementKind::I32, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_bool_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::I64, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_bool_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_bool_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_bool_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_bool_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ---- Fp8E4M3 <-> {f32, f16, bf16} ----
            (ElementKind::Fp8E4M3, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_fp8e4m3_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_fp8e4m3_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E4M3, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_fp8e4m3_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::Fp8E4M3) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_fp8e4m3_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::Fp8E4M3) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_fp8e4m3_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::Fp8E4M3) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_fp8e4m3_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ---- Fp8E5M2 <-> {f32, f16, bf16} ----
            (ElementKind::Fp8E5M2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_fp8e5m2_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_fp8e5m2_f16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Fp8E5M2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_fp8e5m2_bf16_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::Fp8E5M2) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_fp8e5m2_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::Fp8E5M2) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f16_fp8e5m2_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::Fp8E5M2) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_bf16_fp8e5m2_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ---- S4 <-> {i32, i64, f32} ----
            (ElementKind::S4, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_s4_i32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::S4, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_s4_i64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::S4, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_s4_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::I32, ElementKind::S4) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_s4_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::I64, ElementKind::S4) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_s4_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::S4) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_s4_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ---- U4 <-> {i32, i64, f32} ----
            (ElementKind::U4, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_u4_i32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::U4, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_u4_i64_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::U4, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_u4_f32_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::I32, ElementKind::U4) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i32_u4_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::I64, ElementKind::U4) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_i64_u4_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::U4) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_cast_f32_u4_run(
                    numel, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::CastSubBytePlan::run reached an unimplemented \
                     (input, output) pair — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

/// Coverage check — returns `true` if the (input, output) pair is wired
/// in the Phase 13.3 dispatch table.
fn pair_in_scope(input: ElementKind, output: ElementKind) -> bool {
    use ElementKind::*;
    match (input, output) {
        // Bool ↔ {i32, i64, f32, f16, bf16}.
        (Bool, I32) | (Bool, I64) | (Bool, F32) | (Bool, F16) | (Bool, Bf16) => true,
        (I32, Bool) | (I64, Bool) | (F32, Bool) | (F16, Bool) | (Bf16, Bool) => true,
        // Fp8E4M3 ↔ {f32, f16, bf16}.
        (Fp8E4M3, F32) | (Fp8E4M3, F16) | (Fp8E4M3, Bf16) => true,
        (F32, Fp8E4M3) | (F16, Fp8E4M3) | (Bf16, Fp8E4M3) => true,
        // Fp8E5M2 ↔ {f32, f16, bf16}.
        (Fp8E5M2, F32) | (Fp8E5M2, F16) | (Fp8E5M2, Bf16) => true,
        (F32, Fp8E5M2) | (F16, Fp8E5M2) | (Bf16, Fp8E5M2) => true,
        // S4 ↔ {i32, i64, f32}.
        (S4, I32) | (S4, I64) | (S4, F32) => true,
        (I32, S4) | (I64, S4) | (F32, S4) => true,
        // U4 ↔ {i32, i64, f32}.
        (U4, I32) | (U4, I64) | (U4, F32) => true,
        (I32, U4) | (I64, U4) | (F32, U4) => true,
        _ => false,
    }
}

/// Cross-check `sizeof::<T>()` against the byte width implied by
/// `kind`. Same helper used by `ContiguizePlan::select`.
fn type_size_matches_kind<T>(kind: ElementKind) -> bool {
    let want = match kind {
        ElementKind::Bool
        | ElementKind::S8
        | ElementKind::U8
        | ElementKind::Fp8E4M3
        | ElementKind::Fp8E5M2
        | ElementKind::S4
        | ElementKind::U4 => 1,
        ElementKind::F16 | ElementKind::Bf16 => 2,
        ElementKind::F32 | ElementKind::F32Strict | ElementKind::I32 => 4,
        ElementKind::F64 | ElementKind::I64 | ElementKind::Complex32 => 8,
        ElementKind::Complex64 => 16,
        ElementKind::Bin => return false,
    };
    core::mem::size_of::<T>() == want
}

fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem \
             (S4 / U4 require even numel — check descriptor)",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
