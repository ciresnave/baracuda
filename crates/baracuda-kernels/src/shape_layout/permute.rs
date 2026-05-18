//! Materialized `permute` plan — `y = x.permute(dims)`.
//!
//! Output axis `d` is input axis `dims[d]`, so `output[d] =
//! input[dims[d]]`. Use this when the caller needs a CONTIGUOUS
//! permuted output; for in-place strided view manipulation (no kernel
//! launch), callers can construct a `TensorRef` with reshuffled strides
//! directly.
//!
//! Today `f32`, `f16`, `bf16`, and `f64` are wired.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `permute` op.
///
/// `dims` is a permutation of `[0, 1, ..., N-1]`. The output's axis
/// `d` corresponds to the input's axis `dims[d]`. Output shape is
/// derived: `output[d] = input_shape[dims[d]]`.
#[derive(Copy, Clone, Debug)]
pub struct PermuteDescriptor<const N: usize> {
    /// Input tensor shape.
    pub input_shape: [i32; N],
    /// Permutation: `dims[d]` is the input axis index that becomes
    /// output axis `d`. Must be a permutation of `[0, N)`.
    pub dims: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> PermuteDescriptor<N> {
    /// Compute the output shape from input shape and permutation.
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = [0i32; N];
        for d in 0..N {
            out[d] = self.input_shape[self.dims[d] as usize];
        }
        out
    }
}

/// Args bundle for a Permute launch.
pub struct PermuteArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — shape matches `desc.output_shape()`.
    pub y: TensorMut<'a, T, N>,
}

/// Materialized `permute` plan.
///
/// `y = x.permute(dims)` — output axis `d` is input axis `dims[d]`
/// (PyTorch `torch.permute`).
///
/// **When to use**: when the caller needs a CONTIGUOUS permuted
/// output. For zero-cost strided views (no kernel launch), construct
/// a `TensorRef` with reshuffled strides directly. Pair with
/// [`PermuteBackwardPlan`](crate::PermuteBackwardPlan).
///
/// **Dtypes**: `{f32, f64, f16, bf16}`. Pure copy, arithmetic-free
/// so bit-exact at every dtype.
///
/// **Shape limits**: rank in `[1, 8]`; `dims` must be a permutation
/// of `[0, N)`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact.
pub struct PermutePlan<T: Element, const N: usize> {
    desc: PermuteDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> PermutePlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &PermuteDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PermutePlan: descriptor element != type parameter T",
            ));
        }
        // Validate `dims` is a permutation of [0, N).
        let mut seen = [false; 8];
        for d in 0..N {
            let v = desc.dims[d];
            if v < 0 || (v as usize) >= N {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PermutePlan: dims values must be in [0, N)",
                ));
            }
            if seen[v as usize] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PermutePlan: dims must be a permutation (no duplicates)",
                ));
            }
            seen[v as usize] = true;
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PermutePlan: input_shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::PermutePlan: today only `f32`, `f16`, `bf16`, `f64` \
                 are wired; other dtypes land in future fanout",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Permute as u16,
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
    pub fn can_implement(&self, args: &PermuteArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PermutePlan: X shape mismatch with descriptor input_shape",
            ));
        }
        let expected_out = self.desc.output_shape();
        if args.y.shape != expected_out {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PermutePlan: Y shape mismatch with derived output shape \
                 (output[d] = input_shape[dims[d]])",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::PermutePlan: tensor rank > 8 not supported",
            ));
        }
        let x_numel = args.x.numel();
        let y_numel = args.y.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if x_len < x_numel {
            return Err(Error::BufferTooSmall {
                needed: x_numel as usize,
                got: x_len as usize,
            });
        }
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
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

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: PermuteArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let input_numel = args.x.numel();
        if input_numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let input_shape = self.desc.input_shape;
        let dims = self.desc.dims;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_permute_f32_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    dims.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_permute_f16_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    dims.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_permute_bf16_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    dims.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_permute_f64_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    dims.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PermutePlan::run: only `f32`, `f16`, `bf16`, `f64` \
                     wired today",
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
