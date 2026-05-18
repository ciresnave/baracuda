//! `pad` backward plan — Category N (Phase 3 BW).
//!
//! Backward of `y = pad(x, pad_low, pad_high, mode=Constant, value=v)`:
//! `dx = dy[pad_low : pad_low + input_shape]` per axis — a pure slice.
//! The gradient at every pad-region cell is identically zero (the
//! forward wrote a constant there, which has no dependence on `x`), so
//! those cells of `dy` are discarded. Bit-exact across all wired
//! dtypes — pure element copy, no arithmetic.
//!
//! Today only [`PadMode::Constant`] is wired in BW. The non-constant
//! modes (Reflect / Replicate / Circular) require a scatter-add
//! backward — multiple pad cells in `dy` can map to the same input
//! coord in `dx` — and ship in a later fanout.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory, PadMode,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `pad` backward op.
///
/// Mirrors [`crate::PadDescriptor`] for the params the BW actually
/// needs: `input_shape` (= dx shape), `pad_low` (slice offset), and
/// `pad_high` (only used to derive dy shape for the args check).
/// `value` is irrelevant for the BW — the constant gradient is zero by
/// construction. `mode` must be `Constant` today.
#[derive(Copy, Clone, Debug)]
pub struct PadBackwardDescriptor<const N: usize> {
    /// Padding mode that was used in the forward. Today only
    /// `Constant` is wired in BW.
    pub mode: PadMode,
    /// Input tensor shape (= dx shape).
    pub input_shape: [i32; N],
    /// Forward `pad_low` per axis. Non-negative. Becomes the slice
    /// offset in the BW.
    pub pad_low: [i32; N],
    /// Forward `pad_high` per axis. Non-negative. Used to derive the
    /// expected dy shape.
    pub pad_high: [i32; N],
    /// Element type of dy and dx.
    pub element: ElementKind,
}

impl<const N: usize> PadBackwardDescriptor<N> {
    /// Compute the dy shape (= forward output shape) from
    /// `input_shape + pad_low + pad_high` per axis.
    pub fn dy_shape(&self) -> [i32; N] {
        let mut out = [0i32; N];
        for d in 0..N {
            out[d] = self.input_shape[d] + self.pad_low[d] + self.pad_high[d];
        }
        out
    }
}

/// Args bundle for a Pad backward launch.
///
/// `dy.shape` must match the forward output shape (`input_shape +
/// pad_low + pad_high` per axis). `dx.shape` must match
/// `desc.input_shape`. No saved forward tensors are needed — the BW
/// formula is a pure slice.
pub struct PadBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — full forward output shape.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input — input shape.
    pub dx: TensorMut<'a, T, N>,
}

/// `pad` backward plan.
///
/// Adjoint of [`crate::PadPlan`] in `Constant` mode:
/// `dx = dy[pad_low : pad_low + input_shape]` per axis — a pure
/// slice. Pad-region cells of `dy` are discarded (their FW values
/// were a constant, independent of `x`).
///
/// **When to use**: BW for [`PadPlan`](crate::PadPlan) in
/// `Constant` mode.
///
/// **Dtypes**: `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank in `[1, 8]`; `dy` has the FW output shape;
/// `dx` has `input_shape`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact.
///
/// **Known limitations**: BW for `Reflect` / `Replicate` /
/// `Circular` modes is not yet wired — those need a scatter-add
/// (multiple pad cells can map to the same input coord). `select()`
/// returns `Unsupported` for non-`Constant` modes today.
pub struct PadBackwardPlan<T: Element, const N: usize> {
    desc: PadBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> PadBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &PadBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PadBackwardPlan: descriptor element != type parameter T",
            ));
        }
        for d in 0..N {
            if desc.input_shape[d] < 0 || desc.pad_low[d] < 0 || desc.pad_high[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PadBackwardPlan: input_shape / pad_low / pad_high \
                     must be non-negative",
                ));
            }
        }

        // Today only Constant-mode BW is wired (pure slice). Reflect /
        // Replicate / Circular BWs need scatter-add (multiple dy cells
        // can map to the same dx coord) and ship in a later fanout.
        if !matches!(desc.mode, PadMode::Constant) {
            return Err(Error::Unsupported(
                "baracuda-kernels::PadBackwardPlan: today only PadMode::Constant is wired; \
                 Reflect / Replicate / Circular BWs need scatter-add and land in fanout",
            ));
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::PadBackwardPlan: today only `f32`, `f16`, `bf16`, `f64` \
                 are wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pure slice — no arithmetic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Pad as u16,
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
    pub fn can_implement(&self, args: &PadBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PadBackwardPlan: dx shape mismatch with descriptor input_shape",
            ));
        }
        let expected_dy = self.desc.dy_shape();
        if args.dy.shape != expected_dy {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PadBackwardPlan: dy shape mismatch with derived output shape \
                 (= input_shape + pad_low + pad_high per axis)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::PadBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let dx_numel = args.dx.numel();
        let dy_numel = args.dy.numel();
        if (args.dx.data.len() as i64) < dx_numel {
            return Err(Error::BufferTooSmall {
                needed: dx_numel as usize,
                got: args.dx.data.len(),
            });
        }
        if (args.dy.data.len() as i64) < dy_numel {
            return Err(Error::BufferTooSmall {
                needed: dy_numel as usize,
                got: args.dy.data.len(),
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
        args: PadBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let input_numel = args.dx.numel();
        if input_numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let input_shape = self.desc.input_shape;
        let pad_low = self.desc.pad_low;
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pad_constant_backward_f32_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pad_constant_backward_f16_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pad_constant_backward_bf16_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pad_constant_backward_f64_run(
                    input_numel,
                    rank,
                    input_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PadBackwardPlan::run: only f32/f16/bf16/f64 wired today",
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
