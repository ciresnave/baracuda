//! `pad` plan — Category N entry point.
//!
//! Output shape per-axis is `input_shape[d] + pad_low[d] + pad_high[d]`
//! (the FIRST Phase 3 plan where output shape differs from input). The
//! kernel iterates output cells, computes input coord per axis via
//! subtraction of `pad_low`, and either copies the input cell or writes
//! the configured pad value.
//!
//! All four [`PadMode`]s ({Constant, Reflect, Replicate, Circular})
//! are wired for `{f32, f16, bf16, f64}` — 16 (mode, dtype) cells.
//! The descriptor's `value` field is consumed only by `Constant` mode;
//! the other modes derive pad-region values from the input itself
//! (mirror, clamp, or cyclic wrap respectively).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory, PadMode,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

/// Descriptor for a constant-pad op.
///
/// `input_shape` is the shape of the input tensor. `pad_low[d]` and
/// `pad_high[d]` are the pad amounts on each side of axis `d`. Output
/// shape is `input_shape[d] + pad_low[d] + pad_high[d]` per axis.
/// `value` is the constant used in the pad region (for `Constant`
/// mode). `element` must match `T::KIND` at `select` time.
#[derive(Copy, Clone, Debug)]
pub struct PadDescriptor<const N: usize> {
    /// Padding mode — one of [`PadMode::Constant`] / [`PadMode::Reflect`]
    /// / [`PadMode::Replicate`] / [`PadMode::Circular`]. All four are
    /// wired for every supported dtype.
    pub mode: PadMode,
    /// Input tensor shape.
    pub input_shape: [i32; N],
    /// Pad amount on the low side of each axis. Non-negative.
    pub pad_low: [i32; N],
    /// Pad amount on the high side of each axis. Non-negative.
    pub pad_high: [i32; N],
    /// Constant value used in the pad region for `Constant` mode.
    pub value: f32,
    /// Element type of input and output.
    pub element: ElementKind,
}

impl<const N: usize> PadDescriptor<N> {
    /// Compute the output shape from input shape + pad amounts.
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = [0i32; N];
        for d in 0..N {
            out[d] = self.input_shape[d] + self.pad_low[d] + self.pad_high[d];
        }
        out
    }
}

/// Args bundle for a Pad launch.
///
/// `x.shape` must match `desc.input_shape`. `y.shape` must match the
/// output shape derived from descriptor (`input_shape + pad_low +
/// pad_high` per axis). Both can be strided views — the kernel walks
/// per-axis strides.
pub struct PadArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — larger than input by the configured pad amounts.
    pub y: TensorMut<'a, T, N>,
}

/// `pad` plan.
///
/// `y = F.pad(x, pad_low, pad_high, mode, value)` — per-axis low /
/// high padding (PyTorch `torch.nn.functional.pad`).
///
/// **When to use**: forward pad. Pair with
/// [`PadBackwardPlan`](crate::PadBackwardPlan) for autograd
/// (slice-back of `Constant` mode; the other modes have
/// scatter-add BWs not yet wired — see below).
///
/// **Dtypes**: `{f32, f64, f16, bf16}` — 16 (mode, dtype) cells.
///
/// **Modes**: all four [`PadMode`] variants — `Constant`, `Reflect`,
/// `Replicate`, `Circular`. `value` is consumed only by `Constant`;
/// the others derive pad-region values from the input.
///
/// **Shape limits**: rank in `[1, 8]`; `pad_low[d]`, `pad_high[d]`
/// non-negative; output shape per axis is
/// `input_shape[d] + pad_low[d] + pad_high[d]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact (no
/// arithmetic — pure index + copy / value-write).
pub struct PadPlan<T: Element, const N: usize> {
    desc: PadDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> PadPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &PadDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PadPlan: descriptor element != type parameter T",
            ));
        }
        for d in 0..N {
            if desc.input_shape[d] < 0 || desc.pad_low[d] < 0 || desc.pad_high[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PadPlan: input_shape / pad_low / pad_high \
                     must be non-negative",
                ));
            }
        }

        // Full Pad matrix today: 4 modes × {f32, f16, bf16, f64}.
        // Reflect / Replicate / Circular do not consume the
        // descriptor's `value` field — pad-region values are derived
        // from the input itself.
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let mode_in_scope = matches!(
            desc.mode,
            PadMode::Constant | PadMode::Reflect | PadMode::Replicate | PadMode::Circular
        );
        if !(dtype_in_scope && mode_in_scope) {
            return Err(Error::Unsupported(
                "baracuda-kernels::PadPlan: supported matrix is \
                 {Constant, Reflect, Replicate, Circular} × {f32, f16, bf16, f64}",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pad does no arithmetic — pure copy + constant fill.
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
    pub fn can_implement(&self, args: &PadArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PadPlan: X shape mismatch with descriptor input_shape",
            ));
        }
        let expected_out = self.desc.output_shape();
        if args.y.shape != expected_out {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PadPlan: Y shape mismatch with derived output shape \
                 (= input_shape + pad_low + pad_high per axis)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::PadPlan: tensor rank > 8 not supported",
            ));
        }
        let y_numel = args.y.numel();
        let x_numel = args.x.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
            });
        }
        if x_len < x_numel {
            return Err(Error::BufferTooSmall {
                needed: x_numel as usize,
                got: x_len as usize,
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
        args: PadArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_numel = args.y.numel();
        if output_numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let input_shape = self.desc.input_shape;
        let output_shape = self.desc.output_shape();
        let pad_low = self.desc.pad_low;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        // Non-constant pad modes share a parameter shape (no `value`).
        macro_rules! dispatch_mode {
            ($sym:ident) => {{
                unsafe {
                    baracuda_kernels_sys::$sym(
                        output_numel,
                        rank,
                        input_shape.as_ptr(),
                        output_shape.as_ptr(),
                        pad_low.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }};
        }

        let status = match (self.desc.mode, T::KIND) {
            (PadMode::Constant, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pad_constant_f32_run(
                    output_numel,
                    rank,
                    input_shape.as_ptr(),
                    output_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    self.desc.value,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (PadMode::Constant, ElementKind::F16) => unsafe {
                // Convert the descriptor's f32 value to f16, then pass
                // the 16-bit pattern by value — ABI-compatible with the
                // C side's `__half value` parameter on Windows x64
                // (small POD struct → register).
                let value_bits = f16::from_f32(self.desc.value).to_bits();
                baracuda_kernels_sys::baracuda_kernels_pad_constant_f16_run(
                    output_numel,
                    rank,
                    input_shape.as_ptr(),
                    output_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    value_bits,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (PadMode::Constant, ElementKind::Bf16) => unsafe {
                let value_bits = bf16::from_f32(self.desc.value).to_bits();
                baracuda_kernels_sys::baracuda_kernels_pad_constant_bf16_run(
                    output_numel,
                    rank,
                    input_shape.as_ptr(),
                    output_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    value_bits,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            (PadMode::Constant, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_pad_constant_f64_run(
                    output_numel,
                    rank,
                    input_shape.as_ptr(),
                    output_shape.as_ptr(),
                    pad_low.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    self.desc.value as f64,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            // Reflect — mirror across boundary.
            (PadMode::Reflect, ElementKind::F32) => {
                dispatch_mode!(baracuda_kernels_pad_reflect_f32_run)
            }
            (PadMode::Reflect, ElementKind::F16) => {
                dispatch_mode!(baracuda_kernels_pad_reflect_f16_run)
            }
            (PadMode::Reflect, ElementKind::Bf16) => {
                dispatch_mode!(baracuda_kernels_pad_reflect_bf16_run)
            }
            (PadMode::Reflect, ElementKind::F64) => {
                dispatch_mode!(baracuda_kernels_pad_reflect_f64_run)
            }
            // Replicate — clamp to edge.
            (PadMode::Replicate, ElementKind::F32) => {
                dispatch_mode!(baracuda_kernels_pad_replicate_f32_run)
            }
            (PadMode::Replicate, ElementKind::F16) => {
                dispatch_mode!(baracuda_kernels_pad_replicate_f16_run)
            }
            (PadMode::Replicate, ElementKind::Bf16) => {
                dispatch_mode!(baracuda_kernels_pad_replicate_bf16_run)
            }
            (PadMode::Replicate, ElementKind::F64) => {
                dispatch_mode!(baracuda_kernels_pad_replicate_f64_run)
            }
            // Circular — cyclic wrap.
            (PadMode::Circular, ElementKind::F32) => {
                dispatch_mode!(baracuda_kernels_pad_circular_f32_run)
            }
            (PadMode::Circular, ElementKind::F16) => {
                dispatch_mode!(baracuda_kernels_pad_circular_f16_run)
            }
            (PadMode::Circular, ElementKind::Bf16) => {
                dispatch_mode!(baracuda_kernels_pad_circular_bf16_run)
            }
            (PadMode::Circular, ElementKind::F64) => {
                dispatch_mode!(baracuda_kernels_pad_circular_f64_run)
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PadPlan::run: this (mode, dtype) cell is not wired",
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
