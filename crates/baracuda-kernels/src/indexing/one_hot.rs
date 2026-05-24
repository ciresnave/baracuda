//! `one_hot` plan — Category L (non-differentiable).
//!
//! `out[indices..., c] = 1 if c == src[indices...] else 0`. PyTorch
//! `torch.nn.functional.one_hot`.
//!
//! Class-index input dtype is generic over [`IndexElement`] (`i32` or
//! `i64`); output dtype is configurable and selected at the plan layer
//! (`f32, f64, i32, bool`). The output rank is `input_rank + 1` — the
//! new last axis has extent `num_classes`. Out-of-range src values yield
//! an all-zero row.
//!
//! The kernel assumes contiguous output (row-major, last axis =
//! `num_classes`); the caller passes a flat numel for the kernel to
//! sweep. No backward — class indices are non-differentiable.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexElement, IndexElementKind, IndexingKind,
    KernelSku, MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef,
    Workspace,
};

use super::gather::map_status;

/// Descriptor for a `one_hot` op.
///
/// `N` is the output rank — the input is rank `N-1` (the new last axis
/// of extent `num_classes` is appended).
#[derive(Copy, Clone, Debug)]
pub struct OneHotDescriptor<const N: usize> {
    /// Output tensor shape. Last axis must equal `num_classes`.
    pub out_shape: [i32; N],
    /// Number of one-hot classes (== output's last-axis extent).
    pub num_classes: i32,
    /// Output element type.
    pub element: ElementKind,
}

/// Args bundle for a `one_hot` launch.
///
/// The input is a rank-`N-1` tensor of class indices (`i32` or `i64`);
/// we expose it here as a 1-D flat view (callers reshape via
/// `TensorRef`'s strides since the kernel walks it via
/// `batch_idx = i / num_classes`).
///
/// Phase 15.2: the index tensor is generic over `I: IndexElement` (`i32`
/// or `i64`). The plan dispatches to the matching `_i64idx_` FFI symbol
/// when `I == i64` at launch time. `I` defaults to `i32` for
/// source-compat with pre-Phase-15.2 callers.
pub struct OneHotArgs<'a, T: Element, const N: usize, I: IndexElement = i32> {
    /// Class-index input. Treated as a 1-D contiguous array of
    /// `out_numel / num_classes` cells by the kernel. Type parameter
    /// `I` selects `i32` (legacy) or `i64` (PyTorch default).
    pub src: TensorRef<'a, I, 1>,
    /// One-hot output. Contiguous, row-major, last axis = num_classes.
    pub out: TensorMut<'a, T, N>,
}

/// `one_hot` plan.
///
/// `out[indices..., c] = 1 if c == src[indices...] else 0`
/// (PyTorch `torch.nn.functional.one_hot`).
///
/// **When to use**: forward one-hot encoding of class indices. No
/// backward — class indices are non-differentiable.
///
/// **Dtypes**: input index tensor `{i32, i64}`; output
/// `{f32, f64, i32, bool}`.
///
/// **Shape limits**: output rank `N ∈ [1, 8]`; `num_classes > 0`;
/// `out_shape[N-1] == num_classes`. Output is row-major contiguous.
/// Out-of-range src values yield an all-zero row.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. Pure
/// equality + store, no arithmetic.
pub struct OneHotPlan<T: Element, const N: usize> {
    desc: OneHotDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> OneHotPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &OneHotDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::OneHotPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::OneHotPlan: output rank must be >= 1",
            ));
        }
        if desc.num_classes <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::OneHotPlan: num_classes must be positive",
            ));
        }
        if desc.out_shape[N - 1] != desc.num_classes {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::OneHotPlan: out_shape last axis must equal num_classes",
            ));
        }
        for &d in desc.out_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::OneHotPlan: out_shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32 | ElementKind::Bool
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::OneHotPlan: today only output dtypes \
                 `f32`, `f64`, `i32`, `bool` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::OneHot as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::I32),
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
    ///
    /// Phase 15.2: generic over `I: IndexElement` so both i32 (legacy)
    /// and i64 (PyTorch default) class-index buffers are accepted.
    pub fn can_implement<I: IndexElement>(&self, args: &OneHotArgs<'_, T, N, I>) -> Result<()> {
        if args.out.shape != self.desc.out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::OneHotPlan: out shape mismatch with descriptor",
            ));
        }
        let out_numel = args.out.numel();
        if self.desc.num_classes == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::OneHotPlan: num_classes must be > 0",
            ));
        }
        let expected_src_numel = out_numel / self.desc.num_classes as i64;
        if args.src.numel() != expected_src_numel {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::OneHotPlan: src numel must equal out_numel / num_classes",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::OneHotPlan: output rank > 8 not supported",
            ));
        }
        let out_len = args.out.data.len() as i64;
        let src_len = args.src.data.len() as i64;
        if out_len < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: out_len as usize,
            });
        }
        if src_len < expected_src_numel {
            return Err(Error::BufferTooSmall {
                needed: expected_src_numel as usize,
                got: src_len as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
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
    ///
    /// Phase 15.2: generic over `I: IndexElement`. Dispatches to the
    /// matching `_i64idx_` FFI symbol when
    /// `I::KIND == IndexElementKind::I64`.
    pub fn run<I: IndexElement>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: OneHotArgs<'_, T, N, I>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let out_numel = args.out.numel();
        if out_numel == 0 {
            return Ok(());
        }
        let src_ptr = args.src.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (T::KIND, I::KIND) {
            (ElementKind::F32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_f32_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_f64_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::I32, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_i32_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::Bool, IndexElementKind::I32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_bool_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::F32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_i64idx_f32_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::F64, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_i64idx_f64_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::I32, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_i64idx_i32_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (ElementKind::Bool, IndexElementKind::I64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_one_hot_i64idx_bool_run(
                    out_numel,
                    self.desc.num_classes,
                    src_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::OneHotPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
