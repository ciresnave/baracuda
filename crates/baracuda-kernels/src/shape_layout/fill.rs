//! Fill plan — `y[i] = value` for all `i`.
//!
//! Phase 3 fanout from `fuel-cuda-kernels/fill.cu`. Lives under the
//! shape-layout family because its descriptor produces an output
//! tensor with no input dependency — same family slot as `torch.full`.
//!
//! Today wired across `{f32, f64, f16, bf16, i32, i64}` — every
//! [`Element`]-implementing numeric scalar baracuda exposes through
//! the unified Plan layer. `u8` / `i8` kernels also ship in
//! `baracuda-kernels-sys` but those types live on the `IntElement`
//! family with its own (deferred) plan shape. f16 / bf16 transport
//! their `value` over the FFI as a raw `u16` bit pattern; the
//! safe-plan layer below performs the bit cast.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, Workspace,
};
use half::{bf16, f16};

/// Descriptor for a fill op.
///
/// `value` is consumed in-place by the launcher (no descriptor-time
/// dtype conversion). `element` must match `T::KIND` at `select` time.
#[derive(Copy, Clone, Debug)]
pub struct FillDescriptor<T: Element> {
    /// Number of elements to write.
    pub numel: i32,
    /// Scalar to broadcast across the output. Same dtype as the output
    /// tensor (no internal conversion).
    pub value: T,
    /// Output element type. Must equal `T::KIND`.
    pub element: ElementKind,
}

/// Args bundle for a fill launch.
pub struct FillArgs<'a, T: Element> {
    /// Output tensor — rank-1 contiguous view over `numel` elements.
    pub output: TensorMut<'a, T, 1>,
}

/// Fill plan.
pub struct FillPlan<T: Element> {
    desc: FillDescriptor<T>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FillPlan<T> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &FillDescriptor<T>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FillPlan: descriptor element != type parameter T",
            ));
        }
        if desc.numel < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FillPlan: numel must be non-negative",
            ));
        }
        if !dtype_in_scope(T::KIND) {
            return Err(Error::Unsupported(
                "baracuda-kernels::FillPlan: dtype not wired today; supported set is \
                 {f32, f64, f16, bf16, i32, i64}",
            ));
        }

        // Pure copy — no arithmetic.
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Fill as u16,
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
    pub fn can_implement(&self, args: &FillArgs<'_, T>) -> Result<()> {
        let expected = self.desc.numel as i64;
        if args.output.numel() != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FillPlan: output numel mismatch with descriptor",
            ));
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
        args: FillArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = self.desc.numel as i64;
        if numel == 0 {
            return Ok(());
        }
        let y_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        // Dispatch by runtime element kind. The descriptor's `value`
        // is already typed as `T` at the Rust level — we just need to
        // pick the right FFI per dtype. For f16 / bf16 the value
        // crosses the FFI as a u16 bit pattern.
        //
        // SAFETY: each match arm only fires when `T::KIND` equals the
        // matched ElementKind, by the construction of `T: Element`.
        // The `transmute_copy` calls preserve the bit pattern between
        // monomorphized layouts of the same logical type.
        let status = unsafe {
            match T::KIND {
                ElementKind::F32 => {
                    let v: f32 = core::mem::transmute_copy(&self.desc.value);
                    baracuda_kernels_sys::baracuda_kernels_fill_f32_run(
                        numel, y_ptr, v, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::F64 => {
                    let v: f64 = core::mem::transmute_copy(&self.desc.value);
                    baracuda_kernels_sys::baracuda_kernels_fill_f64_run(
                        numel, y_ptr, v, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::I32 => {
                    let v: i32 = core::mem::transmute_copy(&self.desc.value);
                    baracuda_kernels_sys::baracuda_kernels_fill_i32_run(
                        numel, y_ptr, v, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::I64 => {
                    let v: i64 = core::mem::transmute_copy(&self.desc.value);
                    baracuda_kernels_sys::baracuda_kernels_fill_i64_run(
                        numel, y_ptr, v, core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::F16 => {
                    let v: f16 = core::mem::transmute_copy(&self.desc.value);
                    baracuda_kernels_sys::baracuda_kernels_fill_f16_run(
                        numel, y_ptr, v.to_bits(), core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                ElementKind::Bf16 => {
                    let v: bf16 = core::mem::transmute_copy(&self.desc.value);
                    baracuda_kernels_sys::baracuda_kernels_fill_bf16_run(
                        numel, y_ptr, v.to_bits(), core::ptr::null_mut(), 0, stream_ptr,
                    )
                }
                _ => {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::FillPlan::run reached an unimplemented dtype \
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
