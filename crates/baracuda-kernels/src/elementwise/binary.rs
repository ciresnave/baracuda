//! Binary elementwise plan.
//!
//! Phase 3 trailblazer surface for the baracuda-kernels elementwise op
//! family (category C from the comprehensive plan). Mirrors the shape
//! of [`crate::IntGemmPlan`] (descriptor + args + select/can_implement/
//! run/sku/precision_guarantee) but for arbitrary-rank tensors with no
//! GEMM-style accumulator / epilogue chain.
//!
//! Today only the `Add` op on `f32` over fully-contiguous tensors of
//! matching shape is wired — this is the Phase 3 trailblazer SKU. Other
//! binary ops ([`BinaryKind::Sub`], `Mul`, `Div`, …) and other dtypes
//! (`f16`, `bf16`, `f64`, integer family) join in fanout sessions; the
//! `Add` instantiation in `baracuda-kernels-sys` is the template
//! pattern they follow.
//!
//! Broadcasting is supported: operands with `stride[d] = 0` on a
//! broadcast axis route through a strided kernel path that handles
//! arbitrary per-axis stride (broadcast, transposed views, arbitrary
//! strided slices). The dispatcher picks contig vs strided at run
//! time based on `is_contiguous()` of all three operands.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, BinaryKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a binary elementwise op.
///
/// `shape` describes the **output** tensor shape (== both input shapes
/// after the caller-side rank-normalization convention — see the crate
/// docs for the broadcasting contract). `element` must match `T::KIND`
/// at `select` time.
#[derive(Copy, Clone, Debug)]
pub struct BinaryDescriptor<const N: usize> {
    /// Which binary op to apply.
    pub kind: BinaryKind,
    /// Output tensor shape (`= a.shape = b.shape` for the contig case).
    pub shape: [i32; N],
    /// Primary element type. Must match the type parameter `T` of the
    /// containing plan.
    pub element: ElementKind,
}

/// Args bundle for a binary elementwise launch.
///
/// Lifetime `'a` and rank `N` are shared across all three tensors; the
/// element type `T` is shared too (heterogeneous-dtype ops like
/// `compare(f32, f32) -> bool` use a future `BinaryWithOutDtypePlan`,
/// not this one).
pub struct BinaryArgs<'a, T: Element, const N: usize> {
    /// First input.
    pub a: TensorRef<'a, T, N>,
    /// Second input.
    pub b: TensorRef<'a, T, N>,
    /// Output. Aliasing with either input is allowed (in-place add).
    pub y: TensorMut<'a, T, N>,
}

/// Binary elementwise plan.
///
/// `T: Element` is the kernel's element type (today: must be `f32`).
/// `const N: usize` is the tensor rank — fixed at compile time to keep
/// the descriptor heap-free and the rank invariants type-checked.
pub struct BinaryPlan<T: Element, const N: usize> {
    desc: BinaryDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BinaryPlan<T, N> {
    /// Pick a kernel for `desc`. Returns [`Error::Unsupported`] if the
    /// `(kind, T::KIND)` pair isn't wired today.
    pub fn select(
        _stream: &Stream,
        desc: &BinaryDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryPlan: shape dims must be non-negative",
                ));
            }
        }

        // Trailblazer + op-fanout + dtype-fanout matrix: {Add, Sub, Mul,
        // Div, Pow, Atan2, Hypot} × {F32, F16, Bf16, F64}. Other (kind,
        // dtype) combinations are reserved discriminants today (Eq /
        // comparison family, Lerp, ... and the integer dtype family).
        //
        // Lerp is reserved-but-deferred: it takes a scalar `weight: f32`
        // alongside its two tensor inputs, which doesn't fit this Plan's
        // `BinaryArgs<a, b, y>` shape. A parameterized-binary plan shape
        // (analogous to the ternary `addcmul`/`addcdiv` family) will
        // host Lerp in a later milestone.
        let supported = matches!(
            (desc.kind, T::KIND),
            (BinaryKind::Add, ElementKind::F32)
                | (BinaryKind::Add, ElementKind::F16)
                | (BinaryKind::Add, ElementKind::Bf16)
                | (BinaryKind::Add, ElementKind::F64)
                | (BinaryKind::Sub, ElementKind::F32)
                | (BinaryKind::Sub, ElementKind::F16)
                | (BinaryKind::Sub, ElementKind::Bf16)
                | (BinaryKind::Sub, ElementKind::F64)
                | (BinaryKind::Mul, ElementKind::F32)
                | (BinaryKind::Mul, ElementKind::F16)
                | (BinaryKind::Mul, ElementKind::Bf16)
                | (BinaryKind::Mul, ElementKind::F64)
                | (BinaryKind::Div, ElementKind::F32)
                | (BinaryKind::Div, ElementKind::F16)
                | (BinaryKind::Div, ElementKind::Bf16)
                | (BinaryKind::Div, ElementKind::F64)
                | (BinaryKind::Pow, ElementKind::F32)
                | (BinaryKind::Pow, ElementKind::F16)
                | (BinaryKind::Pow, ElementKind::Bf16)
                | (BinaryKind::Pow, ElementKind::F64)
                | (BinaryKind::Atan2, ElementKind::F32)
                | (BinaryKind::Atan2, ElementKind::F16)
                | (BinaryKind::Atan2, ElementKind::Bf16)
                | (BinaryKind::Atan2, ElementKind::F64)
                | (BinaryKind::Hypot, ElementKind::F32)
                | (BinaryKind::Hypot, ElementKind::F16)
                | (BinaryKind::Hypot, ElementKind::Bf16)
                | (BinaryKind::Hypot, ElementKind::F64)
                | (BinaryKind::Copysign, ElementKind::F32)
                | (BinaryKind::Copysign, ElementKind::F16)
                | (BinaryKind::Copysign, ElementKind::Bf16)
                | (BinaryKind::Copysign, ElementKind::F64)
                | (BinaryKind::Nextafter, ElementKind::F32)
                | (BinaryKind::Nextafter, ElementKind::F16)
                | (BinaryKind::Nextafter, ElementKind::Bf16)
                | (BinaryKind::Nextafter, ElementKind::F64)
                | (BinaryKind::Fmin, ElementKind::F32)
                | (BinaryKind::Fmin, ElementKind::F16)
                | (BinaryKind::Fmin, ElementKind::Bf16)
                | (BinaryKind::Fmin, ElementKind::F64)
                | (BinaryKind::Fmax, ElementKind::F32)
                | (BinaryKind::Fmax, ElementKind::F16)
                | (BinaryKind::Fmax, ElementKind::Bf16)
                | (BinaryKind::Fmax, ElementKind::F64)
                | (BinaryKind::Maximum, ElementKind::F32)
                | (BinaryKind::Maximum, ElementKind::F16)
                | (BinaryKind::Maximum, ElementKind::Bf16)
                | (BinaryKind::Maximum, ElementKind::F64)
                | (BinaryKind::Minimum, ElementKind::F32)
                | (BinaryKind::Minimum, ElementKind::F16)
                | (BinaryKind::Minimum, ElementKind::Bf16)
                | (BinaryKind::Minimum, ElementKind::F64)
                | (BinaryKind::FloorDivide, ElementKind::F32)
                | (BinaryKind::FloorDivide, ElementKind::F16)
                | (BinaryKind::FloorDivide, ElementKind::Bf16)
                | (BinaryKind::FloorDivide, ElementKind::F64)
                | (BinaryKind::Mod, ElementKind::F32)
                | (BinaryKind::Mod, ElementKind::F16)
                | (BinaryKind::Mod, ElementKind::Bf16)
                | (BinaryKind::Mod, ElementKind::F64)
                | (BinaryKind::Remainder, ElementKind::F32)
                | (BinaryKind::Remainder, ElementKind::F16)
                | (BinaryKind::Remainder, ElementKind::Bf16)
                | (BinaryKind::Remainder, ElementKind::F64)
                // Phase 3.3 integer + bool fanout. Five bitwise ops
                // across {i32, i64} + three logical ops across Bool.
                // Contig only — strided / broadcast deferred.
                | (BinaryKind::BitwiseAnd, ElementKind::I32)
                | (BinaryKind::BitwiseAnd, ElementKind::I64)
                | (BinaryKind::BitwiseOr, ElementKind::I32)
                | (BinaryKind::BitwiseOr, ElementKind::I64)
                | (BinaryKind::BitwiseXor, ElementKind::I32)
                | (BinaryKind::BitwiseXor, ElementKind::I64)
                | (BinaryKind::BitwiseLeftShift, ElementKind::I32)
                | (BinaryKind::BitwiseLeftShift, ElementKind::I64)
                | (BinaryKind::BitwiseRightShift, ElementKind::I32)
                | (BinaryKind::BitwiseRightShift, ElementKind::I64)
                | (BinaryKind::LogicalAnd, ElementKind::Bool)
                | (BinaryKind::LogicalOr, ElementKind::Bool)
                | (BinaryKind::LogicalXor, ElementKind::Bool)
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryPlan: today only \
                 `{Add,Sub,Mul,Div,Pow,Atan2,Hypot,Copysign,Nextafter,Fmin,Fmax,\
                 Maximum,Minimum,FloorDivide,Mod,Remainder}` \
                 × `{f32, f16, bf16, f64}` + Phase 3.3 integer / bool fanout \
                 (`{BitwiseAnd,BitwiseOr,BitwiseXor,BitwiseLeftShift,\
                 BitwiseRightShift}` × `{i32, i64}` and \
                 `{LogicalAnd,LogicalOr,LogicalXor}` × Bool — contig only); \
                 other (kind, dtype) pairs land in fanout sessions. Lerp is \
                 reserved-but-deferred pending a parameterized-binary plan \
                 shape.",
            ));
        }

        // The chosen kernel is arch-agnostic SIMT (CUDA cores, no tensor
        // cores). PrecisionGuarantee mirrors what `F32Strict` GEMM
        // reports: full IEEE 754 binary32, bit-stable on the same
        // hardware, deterministic across runs (no atomic accumulation,
        // no warp reduction, no random tile schedule).
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::BinaryElementwise,
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
    /// Accepts both fully-contiguous operands (which take the contig
    /// fast path) and broadcast / strided operands (which take the
    /// strided path). For each axis `d`, each input operand must
    /// satisfy `shape[d] == y.shape[d]` (no broadcast on that axis) or
    /// `shape[d] == 1 && stride[d] == 0` (broadcast on that axis). The
    /// output must be exactly `desc.shape` and is conventionally
    /// contiguous, though the strided kernel accepts arbitrary `y`
    /// strides too.
    pub fn can_implement(&self, args: &BinaryArgs<'_, T, N>) -> Result<()> {
        // Output must match the descriptor exactly. No broadcast on the
        // output side — `y` is the destination of the broadcast, not
        // a participant in it.
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryPlan: Y shape mismatch with descriptor",
            ));
        }

        // Per-axis broadcast compatibility check. An input axis must
        // either match the output's axis exactly, or be 1 with stride 0.
        for d in 0..N {
            let y_dim = self.desc.shape[d];
            let a_dim = args.a.shape[d];
            let b_dim = args.b.shape[d];
            if a_dim != y_dim && !(a_dim == 1 && args.a.stride[d] == 0) {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryPlan: A axis is not broadcast-compatible \
                     with output (require shape[d] == y.shape[d], OR \
                     shape[d] == 1 AND stride[d] == 0)",
                ));
            }
            if b_dim != y_dim && !(b_dim == 1 && args.b.stride[d] == 0) {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryPlan: B axis is not broadcast-compatible \
                     with output",
                ));
            }
        }

        // Strided kernel handles up to MAX_RANK = 8 axes. Reject
        // larger ranks here so callers see a clean Unsupported instead
        // of silent truncation in the kernel.
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryPlan: tensor rank > 8 not supported \
                 (kernel param block fixes MAX_RANK = 8)",
            ));
        }

        // Buffer sizing: the output must cover its own numel; each
        // input must cover at least the largest gmem offset reachable
        // by its strides. For broadcast (stride 0) the reachable
        // offset is 0 along that axis — the simplest safe bound is
        // `numel(input) = product(input.shape)`, treating broadcast
        // dims as size-1.
        let y_numel = args.y.numel();
        let a_numel = args.a.numel();
        let b_numel = args.b.numel();
        let a_len = args.a.data.len() as i64;
        let b_len = args.b.data.len() as i64;
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
        Ok(())
    }

    /// Workspace size in bytes. Always `0` for the trailblazer SKU.
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
        args: BinaryArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.y.numel();
        if numel == 0 {
            return Ok(());
        }
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        // Contig fast path requires all three operands to be fully
        // contiguous AND have identical shape (no broadcast). Any other
        // case (broadcast, transposed, strided slice) goes through the
        // strided kernel.
        let all_contig_same_shape = args.a.shape == args.y.shape
            && args.b.shape == args.y.shape
            && args.a.is_contiguous()
            && args.b.is_contiguous()
            && args.y.is_contiguous();

        if !all_contig_same_shape {
            return self.run_strided(stream_ptr, a_ptr, b_ptr, y_ptr, numel, &args);
        }

        let status = match (self.desc.kind, T::KIND) {
            (BinaryKind::Add, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_f32_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_f32_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_f32_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_f32_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_f16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_bf16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_f64_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_f16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_bf16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_f64_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_f16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_bf16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_f64_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_f16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_bf16_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_f64_run(
                    numel,
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_f32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_f16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_f64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // ---- Phase 3.3 integer + bool fanout (contig only) ----
            (BinaryKind::BitwiseAnd, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_and_i32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseAnd, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_and_i64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseOr, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_or_i32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseOr, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_or_i64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseXor, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_xor_i32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseXor, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_xor_i64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseLeftShift, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_left_shift_i32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseLeftShift, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_left_shift_i64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseRightShift, ElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_right_shift_i32_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::BitwiseRightShift, ElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_bitwise_right_shift_i64_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::LogicalAnd, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_logical_and_bool_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::LogicalOr, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_logical_or_bool_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::LogicalXor, ElementKind::Bool) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_logical_xor_bool_run(
                    numel, a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
                ))
            }
        };
        map_status(status)
    }
}

impl<T: Element, const N: usize> BinaryPlan<T, N> {
    /// Launch the strided / broadcast kernel path.
    ///
    /// Called by [`Self::run`] when at least one operand isn't
    /// contiguous (broadcast, transposed view, arbitrary strided
    /// slice). The kernel reads each output coord c via the per-operand
    /// strides — a stride of 0 along axis d collapses that axis to
    /// element 0, which is the broadcast semantic.
    fn run_strided(
        &self,
        stream_ptr: *mut c_void,
        a_ptr: *const c_void,
        b_ptr: *const c_void,
        y_ptr: *mut c_void,
        numel: i64,
        args: &BinaryArgs<'_, T, N>,
    ) -> Result<()> {
        // Output shape (== descriptor shape) drives the kernel's coord
        // loop. Stride arrays come from each operand.
        let shape = args.y.shape;
        let stride_a = args.a.stride;
        let stride_b = args.b.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        let status = match (self.desc.kind, T::KIND) {
            (BinaryKind::Add, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_f32_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_f16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_bf16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Add, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_add_f64_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_f32_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_f16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_bf16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Sub, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_sub_f64_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_f32_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_f16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_bf16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Mul, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mul_f64_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_f32_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_f16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_bf16_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Div, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_div_f64_strided_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    stride_a.as_ptr(),
                    stride_b.as_ptr(),
                    stride_y.as_ptr(),
                    a_ptr,
                    b_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Pow, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_pow_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Atan2, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_atan2_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Hypot, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_hypot_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Copysign, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_copysign_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Nextafter, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_nextafter_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmin, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmin_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Fmax, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_fmax_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Maximum, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_maximum_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Minimum, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_minimum_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::FloorDivide, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_floor_divide_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Mod, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_mod_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Remainder, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_remainder_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryPlan::run_strided reached an \
                     unimplemented (kind, dtype) pair — select() should \
                     have caught this",
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
