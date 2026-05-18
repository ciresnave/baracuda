//! 2-input `concat` backward plan — Category N (Phase 3 BW).
//!
//! Backward of `y = cat(a, b, dim=k)`:
//!   `da = dy[..., :split_offset, ...]`
//!   `db = dy[..., split_offset:, ...]`
//! along axis `k`, where `split_offset = a.shape[k]`. Each `dy` cell
//! maps to exactly one of `da` or `db` — pure inverse routing, no
//! summation, bit-exact across every wired dtype.
//!
//! Today only `f32`, `f16`, `bf16`, and `f64` are wired (mirrors the
//! forward [`crate::ConcatPlan`]). One launcher per dtype.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a 2-input concat backward op.
///
/// `output_shape` is the forward output shape (= `dy.shape`).
/// `concat_dim` selects the axis being split.  `split_offset` is the
/// forward `a.shape[concat_dim]` — `da.shape[concat_dim] = split_offset`
/// and `db.shape[concat_dim] = output_shape[concat_dim] - split_offset`.
#[derive(Copy, Clone, Debug)]
pub struct ConcatBackwardDescriptor<const N: usize> {
    /// Forward output shape (= dy shape).
    pub output_shape: [i32; N],
    /// Axis to split along. Must satisfy `0 <= concat_dim < N`.
    pub concat_dim: u8,
    /// First-half extent along `concat_dim` — i.e. `a.shape[concat_dim]`.
    /// `da.shape[concat_dim] = split_offset` and
    /// `db.shape[concat_dim] = output_shape[concat_dim] - split_offset`.
    pub split_offset: i32,
    /// Element type of dy, da and db.
    pub element: ElementKind,
}

impl<const N: usize> ConcatBackwardDescriptor<N> {
    /// Compute the `da` shape (= output_shape with `[concat_dim] =
    /// split_offset`).
    pub fn da_shape(&self) -> [i32; N] {
        let mut out = self.output_shape;
        out[self.concat_dim as usize] = self.split_offset;
        out
    }

    /// Compute the `db` shape (= output_shape with `[concat_dim] =
    /// output_shape[concat_dim] - split_offset`).
    pub fn db_shape(&self) -> [i32; N] {
        let mut out = self.output_shape;
        let d = self.concat_dim as usize;
        out[d] = self.output_shape[d] - self.split_offset;
        out
    }
}

/// Args bundle for a Concat2 backward launch.
///
/// `dy.shape` must match `desc.output_shape`. `da.shape` and `db.shape`
/// must match `desc.da_shape()` / `desc.db_shape()` respectively. No
/// saved forward tensors are needed — the BW formula is a pure copy.
pub struct ConcatBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — full forward output shape.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the first forward input.
    pub da: TensorMut<'a, T, N>,
    /// Gradient w.r.t. the second forward input.
    pub db: TensorMut<'a, T, N>,
}

/// 2-input `concat` backward plan.
///
/// Adjoint of [`crate::ConcatPlan`]: split `dy` along `concat_dim`
/// into `da` and `db`. Pure slice — no arithmetic.
///
/// **When to use**: BW for [`ConcatPlan`](crate::ConcatPlan).
///
/// **Dtypes**: `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank in `[1, 8]`; `dy` has the FW output shape;
/// `da`, `db` have the FW input shapes.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact.
pub struct ConcatBackwardPlan<T: Element, const N: usize> {
    desc: ConcatBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ConcatBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &ConcatBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConcatBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConcatBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        if (desc.concat_dim as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatBackwardPlan: concat_dim must be < rank",
            ));
        }
        for d in 0..N {
            if desc.output_shape[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ConcatBackwardPlan: output_shape dims must be \
                     non-negative",
                ));
            }
        }
        let cd = desc.concat_dim as usize;
        if desc.split_offset < 0 || desc.split_offset > desc.output_shape[cd] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatBackwardPlan: split_offset must satisfy \
                 0 <= split_offset <= output_shape[concat_dim]",
            ));
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConcatBackwardPlan: today only `f32`, `f16`, `bf16`, \
                 `f64` are wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pure copy — no arithmetic.
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
    pub fn can_implement(&self, args: &ConcatBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.output_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatBackwardPlan: dy shape mismatch with descriptor \
                 output_shape",
            ));
        }
        let expected_da = self.desc.da_shape();
        let expected_db = self.desc.db_shape();
        if args.da.shape != expected_da {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatBackwardPlan: da shape mismatch with derived \
                 da shape (= output_shape with [concat_dim] = split_offset)",
            ));
        }
        if args.db.shape != expected_db {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConcatBackwardPlan: db shape mismatch with derived \
                 db shape (= output_shape with [concat_dim] = \
                 output_shape[concat_dim] - split_offset)",
            ));
        }
        let dy_numel = args.dy.numel();
        let da_numel = args.da.numel();
        let db_numel = args.db.numel();
        let dy_len = args.dy.data.len() as i64;
        let da_len = args.da.data.len() as i64;
        let db_len = args.db.data.len() as i64;
        if dy_len < dy_numel {
            return Err(Error::BufferTooSmall {
                needed: dy_numel as usize,
                got: dy_len as usize,
            });
        }
        if da_len < da_numel {
            return Err(Error::BufferTooSmall {
                needed: da_numel as usize,
                got: da_len as usize,
            });
        }
        if db_len < db_numel {
            return Err(Error::BufferTooSmall {
                needed: db_numel as usize,
                got: db_len as usize,
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
        args: ConcatBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_numel = args.dy.numel();
        if output_numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let da_ptr = args.da.data.as_raw().0 as *mut c_void;
        let db_ptr = args.db.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let output_shape = self.desc.output_shape;
        let stride_dy = args.dy.stride;
        let stride_da = args.da.stride;
        let stride_db = args.db.stride;
        let rank = N as i32;
        let concat_dim = self.desc.concat_dim as i32;
        let split_offset = self.desc.split_offset;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_backward_f32_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
                    stride_dy.as_ptr(),
                    stride_da.as_ptr(),
                    stride_db.as_ptr(),
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_backward_f16_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
                    stride_dy.as_ptr(),
                    stride_da.as_ptr(),
                    stride_db.as_ptr(),
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_backward_bf16_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
                    stride_dy.as_ptr(),
                    stride_da.as_ptr(),
                    stride_db.as_ptr(),
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_concat2_backward_f64_run(
                    output_numel,
                    rank,
                    output_shape.as_ptr(),
                    concat_dim,
                    split_offset,
                    stride_dy.as_ptr(),
                    stride_da.as_ptr(),
                    stride_db.as_ptr(),
                    dy_ptr,
                    da_ptr,
                    db_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ConcatBackwardPlan::run: only f32/f16/bf16/f64 \
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
