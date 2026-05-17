//! Affine plan — fused `y[i] = a * x[i] + b` with scalar `a`, `b`.
//!
//! Phase 3 fanout from `fuel-cuda-kernels/affine.cu`. Same-dtype-input
//! / same-dtype-output but carries two scalar parameters (`a`, `b`),
//! so it gets its own plan shape instead of routing through the unified
//! [`crate::UnaryPlan`].
//!
//! Today wired across `{f32, f64, f16, bf16, i32, i64}` — every
//! [`Element`]-implementing numeric scalar in the unified Plan layer.
//! `u8` / `i8` kernels also ship in `baracuda-kernels-sys` but those
//! types live on the `IntElement` family with its own (deferred) plan
//! shape. f16 / bf16 compute through f32 internally; `a` / `b` cross
//! the FFI as `f32` for those dtypes (matching the rest of the
//! elementwise family's f32-accumulator precision-guarantee contract).
//! The kernel is contig-only — baracuda's plan layer materializes
//! strided views upstream.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};
use half::{bf16, f16};

/// Descriptor for an affine op.
///
/// `a` and `b` are the scalar multiplier / bias. Both share the
/// output element type (no cross-dtype scalar parameters). `element`
/// must match `T::KIND` at `select` time.
#[derive(Copy, Clone, Debug)]
pub struct AffineDescriptor<T: Element> {
    /// Number of elements in input and output.
    pub numel: i32,
    /// Multiplier — same dtype as input / output.
    pub a: T,
    /// Additive bias — same dtype as input / output.
    pub b: T,
    /// Input / output element type. Must equal `T::KIND`.
    pub element: ElementKind,
}

/// Args bundle for an affine launch.
pub struct AffineArgs<'a, T: Element> {
    /// Input tensor — rank-1 contiguous view.
    pub input: TensorRef<'a, T, 1>,
    /// Output tensor — rank-1 contiguous view, same numel as input.
    pub output: TensorMut<'a, T, 1>,
}

/// Affine plan.
pub struct AffinePlan<T: Element> {
    desc: AffineDescriptor<T>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AffinePlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &AffineDescriptor<T>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::AffinePlan: descriptor element != type parameter T",
            ));
        }
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AffinePlan: numel must be non-negative",
            ));
        }
        if !dtype_in_scope(T::KIND) {
            return Err(Error::Unsupported(
                "baracuda-kernels::AffinePlan: dtype not wired today; supported set is \
                 {f32, f64, f16, bf16, i32, i64}",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::UnaryElementwise,
            op: UnaryKind::Affine as u16,
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
    pub fn can_implement(&self, args: &AffineArgs<'_, T>) -> Result<()> {
        let expected = self.desc.numel as i64;
        if args.input.numel() != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AffinePlan: input numel mismatch with descriptor",
            ));
        }
        if args.output.numel() != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::AffinePlan: output numel mismatch with descriptor",
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
        args: AffineArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let y_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        // SAFETY: each match arm only fires when `T::KIND` equals the
        // matched ElementKind. The `transmute_copy` of `desc.a` /
        // `desc.b` preserves the bit pattern across monomorphized
        // layouts of the same logical type. f16 / bf16 are upcast to
        // f32 before crossing the FFI.
        let status = unsafe {
            match T::KIND {
                ElementKind::F32 => {
                    let a: f32 = core::mem::transmute_copy(&self.desc.a);
                    let b: f32 = core::mem::transmute_copy(&self.desc.b);
                    baracuda_kernels_sys::baracuda_kernels_affine_f32_run(
                        numel, x_ptr, y_ptr, a, b, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::F64 => {
                    let a: f64 = core::mem::transmute_copy(&self.desc.a);
                    let b: f64 = core::mem::transmute_copy(&self.desc.b);
                    baracuda_kernels_sys::baracuda_kernels_affine_f64_run(
                        numel, x_ptr, y_ptr, a, b, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::I32 => {
                    let a: i32 = core::mem::transmute_copy(&self.desc.a);
                    let b: i32 = core::mem::transmute_copy(&self.desc.b);
                    baracuda_kernels_sys::baracuda_kernels_affine_i32_run(
                        numel, x_ptr, y_ptr, a, b, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::I64 => {
                    let a: i64 = core::mem::transmute_copy(&self.desc.a);
                    let b: i64 = core::mem::transmute_copy(&self.desc.b);
                    baracuda_kernels_sys::baracuda_kernels_affine_i64_run(
                        numel, x_ptr, y_ptr, a, b, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::F16 => {
                    let a: f16 = core::mem::transmute_copy(&self.desc.a);
                    let b: f16 = core::mem::transmute_copy(&self.desc.b);
                    baracuda_kernels_sys::baracuda_kernels_affine_f16_run(
                        numel, x_ptr, y_ptr, a.to_f32(), b.to_f32(),
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::Bf16 => {
                    let a: bf16 = core::mem::transmute_copy(&self.desc.a);
                    let b: bf16 = core::mem::transmute_copy(&self.desc.b);
                    baracuda_kernels_sys::baracuda_kernels_affine_bf16_run(
                        numel, x_ptr, y_ptr, a.to_f32(), b.to_f32(),
                        core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                _ => {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::AffinePlan::run reached an unimplemented dtype \
                         — select() should have caught this",
                    ));
                }
            }
        };
        map_status(status)
    }
}

fn dtype_in_scope(k: ElementKind) -> bool {
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
