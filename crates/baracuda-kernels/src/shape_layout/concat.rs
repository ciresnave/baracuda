//! 2-input `concat` plan — `y = cat(a, b, dim=k)`.
//!
//! Variable-arity (N-input) concat needs a separate plan shape with
//! device-side packing of N pointers + N stride arrays — deferred to a
//! later session. For real ML use cases the 2-input variant covers the
//! majority (residual joins, key/value concat in attention KV-cache,
//! etc.).
//!
//! Today `f32`, `f16`, `bf16`, and `f64` are wired (one INSTANTIATE per
//! dtype in the .cu, one FFI block per dtype here, one match arm in
//! `run`).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a 2-input concat op.
///
/// `a_shape` and `b_shape` must match on every axis except `concat_dim`.
/// Output shape is `a_shape` with `[concat_dim]` set to
/// `a_shape[concat_dim] + b_shape[concat_dim]`.
#[derive(Copy, Clone, Debug)]
pub struct ConcatDescriptor<const N: usize> {
    /// Shape of the first input.
    pub a_shape: [i32; N],
    /// Shape of the second input.
    pub b_shape: [i32; N],
    /// Axis to concatenate along. Must satisfy `0 <= concat_dim < N`.
    pub concat_dim: u8,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> ConcatDescriptor<N> {
    /// Compute the output shape (a_shape with `[concat_dim]` summed).
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = self.a_shape;
        let d = self.concat_dim as usize;
        out[d] = self.a_shape[d] + self.b_shape[d];
        out
    }
}

/// Args bundle for a Concat launch.
pub struct ConcatArgs<'a, T: Element, const N: usize> {
    /// First input. `a.shape` must equal `desc.a_shape`.
    pub a: TensorRef<'a, T, N>,
    /// Second input. `b.shape` must equal `desc.b_shape`.
    pub b: TensorRef<'a, T, N>,
    /// Output — shape matches `desc.output_shape()`.
    pub y: TensorMut<'a, T, N>,
}

/// 2-input `concat` plan.
///
/// `y = cat(a, b, dim=concat_dim)` (PyTorch `torch.cat`).
///
/// **When to use**: residual joins, KV-cache concat in attention, any
/// 2-input concatenation. Variable-arity (N-input) concat needs a
/// separate plan with device-side pointer arrays — deferred. Pair
/// with [`ConcatBackwardPlan`](crate::ConcatBackwardPlan).
///
/// **Dtypes**: `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank in `[1, 8]`; `a_shape` and `b_shape` must
/// match on every axis except `concat_dim`; `concat_dim ∈ [0, N)`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact
/// (pure load + store).
pub struct ConcatPlan<T: Element, const N: usize> {
    desc: ConcatDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ConcatPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ConcatDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConcatPlan: descriptor element != type parameter T",
            ));
        }
        if (desc.concat_dim as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatPlan: concat_dim must be < rank",
            ));
        }
        // Shapes must match on every axis except concat_dim.
        let cd = desc.concat_dim as usize;
        for d in 0..N {
            if desc.a_shape[d] < 0 || desc.b_shape[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ConcatPlan: input shape dims must be non-negative",
                ));
            }
            if d != cd && desc.a_shape[d] != desc.b_shape[d] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ConcatPlan: input shapes must match on every \
                     axis except concat_dim",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConcatPlan: today only `f32`, `f16`, `bf16`, `f64` \
                 are wired; other dtypes land in future fanout",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Concat does no arithmetic — pure element copy.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Concat as u16,
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
    pub fn can_implement(&self, args: &ConcatArgs<'_, T, N>) -> Result<()> {
        if args.a.shape != self.desc.a_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatPlan: A shape mismatch with descriptor a_shape",
            ));
        }
        if args.b.shape != self.desc.b_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatPlan: B shape mismatch with descriptor b_shape",
            ));
        }
        let expected_out = self.desc.output_shape();
        if args.y.shape != expected_out {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatPlan: Y shape mismatch with derived output \
                 shape (= a_shape with [concat_dim] = a + b extents)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConcatPlan: tensor rank > 8 not supported",
            ));
        }
        let a_numel = args.a.numel();
        let b_numel = args.b.numel();
        let y_numel = args.y.numel();
        let a_len = args.a.data.len() as i64;
        let b_len = args.b.data.len() as i64;
        let y_len = args.y.data.len() as i64;
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
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
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
        args: ConcatArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_numel = args.y.numel();
        if output_numel == 0 {
            return Ok(());
        }
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let output_shape = self.desc.output_shape();
        let stride_a = args.a.stride;
        let stride_b = args.b.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;
        let concat_dim = self.desc.concat_dim as i32;
        let split_offset = self.desc.a_shape[self.desc.concat_dim as usize];

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_f32_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
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
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_f16_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
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
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_bf16_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
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
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_f64_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
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
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ConcatPlan::run: only `f32`, `f16`, `bf16`, `f64` \
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
