//! Binary comparison plan.
//!
//! Sibling of [`crate::BinaryPlan`] for ops where the output dtype
//! differs from the input dtype: comparisons produce `u8` (PyTorch /
//! NumPy bool storage convention: 0 = false, 1 = true) regardless of
//! the input element type.
//!
//! Fully wired matrix: {Eq, Ne, Gt, Ge, Lt, Le} × {f32, f16, bf16,
//! f64} = 24 (kind, dtype) cells, each with both the contig fast path
//! and the strided / broadcast path (48 launchers total). The
//! dispatcher's supported check reduces to a straight cross product
//! `kind_in_scope && dtype_in_scope`; the match arms in
//! `run` / `run_strided` remain the authoritative dispatch table.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, BinaryCmpKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a binary comparison op.
///
/// `shape` is the OUTPUT tensor shape. `element` is the INPUT dtype —
/// the output is always `u8`. `element` must match the type parameter
/// `T` of the containing plan at `select` time.
#[derive(Copy, Clone, Debug)]
pub struct BinaryCmpDescriptor<const N: usize> {
    /// Which comparison op to apply.
    pub kind: BinaryCmpKind,
    /// Output tensor shape.
    pub shape: [i32; N],
    /// Input element type (output is always `u8`).
    pub element: ElementKind,
}

/// Args bundle for a binary comparison launch.
///
/// Inputs are `T`; output is `u8` (0 / 1). Aliasing `y` with `a` or `b`
/// is unsafe because `y` has a different element size than the inputs;
/// the kernel does NOT alias-check.
pub struct BinaryCmpArgs<'a, T: Element, const N: usize> {
    /// First input.
    pub a: TensorRef<'a, T, N>,
    /// Second input.
    pub b: TensorRef<'a, T, N>,
    /// Output. `u8` storage: 0 = false, 1 = true.
    pub y: TensorMut<'a, u8, N>,
}

/// Binary comparison plan.
///
/// `T: Element` is the input element type (today: must be `f32`).
/// Output is always `u8`. `const N: usize` is the tensor rank.
pub struct BinaryCmpPlan<T: Element, const N: usize> {
    desc: BinaryCmpDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BinaryCmpPlan<T, N> {
    /// Pick a kernel for `desc`. Returns [`Error::Unsupported`] if the
    /// `(kind, T::KIND)` pair isn't wired today.
    pub fn select(
        _stream: &Stream,
        desc: &BinaryCmpDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryCmpPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryCmpPlan: shape dims must be non-negative",
                ));
            }
        }

        // Supported matrix: all 6 BinaryCmpKind variants across the
        // 4 FP dtypes. The match arms in `run` / `run_strided` remain
        // the authoritative dispatch table; the unreachable `_ =>` arm
        // catches any future drift if new variants are added upstream.
        let kind_in_scope = matches!(
            desc.kind,
            BinaryCmpKind::Eq
                | BinaryCmpKind::Ne
                | BinaryCmpKind::Gt
                | BinaryCmpKind::Ge
                | BinaryCmpKind::Lt
                | BinaryCmpKind::Le
        );
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let supported = kind_in_scope && dtype_in_scope;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryCmpPlan: this (kind, dtype) cell is not yet \
                 wired; see the dispatcher's kind / dtype scope for the supported set",
            ));
        }

        // Comparisons are bit-stable + deterministic — no math, no
        // ordering ambiguity. Output dtype is u8 but we tag the SKU's
        // primary `element` as the INPUT dtype (matches the input
        // tensor's dtype, drives the kernel selection); `aux_element`
        // captures the output dtype.
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
            // u8 output isn't an ElementKind variant today; encode as
            // None and rely on the kind tag to disambiguate from
            // same-dtype binary ops.
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
    /// Accepts contig and strided / broadcast operands. Same broadcast
    /// rules as [`crate::BinaryPlan::can_implement`].
    pub fn can_implement(&self, args: &BinaryCmpArgs<'_, T, N>) -> Result<()> {
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryCmpPlan: Y shape mismatch with descriptor",
            ));
        }

        for d in 0..N {
            let y_dim = self.desc.shape[d];
            let a_dim = args.a.shape[d];
            let b_dim = args.b.shape[d];
            if a_dim != y_dim && !(a_dim == 1 && args.a.stride[d] == 0) {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryCmpPlan: A axis not broadcast-compatible with output",
                ));
            }
            if b_dim != y_dim && !(b_dim == 1 && args.b.stride[d] == 0) {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryCmpPlan: B axis not broadcast-compatible with output",
                ));
            }
        }

        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryCmpPlan: tensor rank > 8 not supported \
                 (kernel param block fixes MAX_RANK = 8)",
            ));
        }

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
        args: BinaryCmpArgs<'_, T, N>,
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

        let all_contig_same_shape = args.a.shape == args.y.shape
            && args.b.shape == args.y.shape
            && args.a.is_contiguous()
            && args.b.is_contiguous()
            && args.y.is_contiguous();

        if !all_contig_same_shape {
            return self.run_strided(stream_ptr, a_ptr, b_ptr, y_ptr, numel, &args);
        }

        let status = match (self.desc.kind, T::KIND) {
            // --- Eq -----------------------------------------------------
            (BinaryCmpKind::Eq, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_f32_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Eq, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_f16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Eq, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Eq, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_f64_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Ne -----------------------------------------------------
            (BinaryCmpKind::Ne, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_f32_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ne, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_f16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ne, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ne, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_f64_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Gt -----------------------------------------------------
            (BinaryCmpKind::Gt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_f32_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Gt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_f16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Gt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Gt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_f64_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Ge -----------------------------------------------------
            (BinaryCmpKind::Ge, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_f32_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ge, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_f16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ge, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ge, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_f64_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Lt -----------------------------------------------------
            (BinaryCmpKind::Lt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_f32_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Lt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_f16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Lt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Lt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_f64_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Le -----------------------------------------------------
            (BinaryCmpKind::Le, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_f32_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Le, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_f16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Le, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Le, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_f64_run(
                    numel, a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryCmpPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
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
        y_ptr: *mut c_void,
        numel: i64,
        args: &BinaryCmpArgs<'_, T, N>,
    ) -> Result<()> {
        let shape = args.y.shape;
        let stride_a = args.a.stride;
        let stride_b = args.b.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        let status = match (self.desc.kind, T::KIND) {
            // --- Eq -----------------------------------------------------
            (BinaryCmpKind::Eq, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Eq, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Eq, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Eq, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_eq_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Ne -----------------------------------------------------
            (BinaryCmpKind::Ne, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ne, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ne, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ne, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ne_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Gt -----------------------------------------------------
            (BinaryCmpKind::Gt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Gt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Gt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Gt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_gt_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Ge -----------------------------------------------------
            (BinaryCmpKind::Ge, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ge, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ge, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Ge, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_ge_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Lt -----------------------------------------------------
            (BinaryCmpKind::Lt, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Lt, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Lt, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Lt, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_lt_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            // --- Le -----------------------------------------------------
            (BinaryCmpKind::Le, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_f32_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Le, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_f16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Le, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_bf16_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryCmpKind::Le, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_cmp_le_f64_strided_run(
                    numel, rank, shape.as_ptr(),
                    stride_a.as_ptr(), stride_b.as_ptr(), stride_y.as_ptr(),
                    a_ptr, b_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryCmpPlan::run_strided reached an \
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
