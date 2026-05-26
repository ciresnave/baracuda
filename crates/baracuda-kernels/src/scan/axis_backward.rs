//! Single-axis scan backward plan.
//!
//! Backward pass for the prefix-scan family.
//!
//! - `Cumsum BW`: `dx = cumsum(dy, reverse = !fw.reverse)` — that is,
//!   the FW kernel applied to `dy` with the scan direction flipped. No
//!   new CUDA kernel required.
//! - `Cumprod BW`: `dx[j] = Σ_{i in suffix} dy[i] * y[i] / x[j]`. Needs
//!   the saved FW input `x` and the saved FW output `y`.
//! - `Cummax / Cummin BW`: gradient flows to first-occurrence
//!   argmax/argmin position. Needs the saved FW input `x` only —
//!   running winner is recomputed by the kernel from `x`.
//! - `LogCumsumExp BW`: `dx[k] = Σ_{i ∈ range(k)} dy[i] * exp(x[k] - y[i])`.
//!   Needs both saved FW input `x` and saved FW output `y`.
//!
//! Today wired: `{Cumsum, Cumprod, Cummax, Cummin, LogCumsumExp} ×
//! {f32, f16, bf16, f64}`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ScanKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a single-axis scan BW op.
///
/// Mirrors [`crate::ScanDescriptor`] — same `input_shape`, `scan_axis`,
/// and `reverse` as the FW. The BW kernel handles direction internally.
#[derive(Copy, Clone, Debug)]
pub struct ScanBackwardDescriptor<const N: usize> {
    /// Which forward scan kind this is the backward of.
    pub kind: ScanKind,
    /// Tensor shape (shared by dy / dx and the optional x / y saves).
    pub input_shape: [i32; N],
    /// Forward scan axis.
    pub scan_axis: u8,
    /// Forward direction flag.
    pub reverse: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a scan BW launch.
///
/// `x` and `y` are saved forward tensors; whether they must be supplied
/// is op-dependent:
///
/// | op            | needs x | needs y |
/// |---------------|---------|---------|
/// | Cumsum        |    no   |    no   |
/// | Cumprod       |   yes   |   yes   |
/// | Cummax        |   yes   |    no   |
/// | Cummin        |   yes   |    no   |
/// | LogCumsumExp  |   yes   |   yes   |
///
/// Pass `None` for unused slots. The plan's `can_implement` validates
/// the op-specific requirement.
pub struct ScanBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — same shape as the forward output.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input — same shape.
    pub dx: TensorMut<'a, T, N>,
    /// Saved forward input. Required for Cumprod / Cummax / Cummin BW.
    pub x: Option<TensorRef<'a, T, N>>,
    /// Saved forward output. Required for Cumprod BW.
    pub y: Option<TensorRef<'a, T, N>>,
}

/// True iff the scan kind's BW requires the saved forward input `x`.
#[inline]
fn op_needs_saved_x(kind: ScanKind) -> bool {
    matches!(
        kind,
        ScanKind::Cumprod | ScanKind::Cummax | ScanKind::Cummin | ScanKind::LogCumsumExp
    )
}

/// True iff the scan kind's BW requires the saved forward output `y`.
#[inline]
fn op_needs_saved_y(kind: ScanKind) -> bool {
    matches!(kind, ScanKind::Cumprod | ScanKind::LogCumsumExp)
}

/// Single-axis scan backward plan.
pub struct ScanBackwardPlan<T: Element, const N: usize> {
    desc: ScanBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ScanBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &ScanBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScanBackwardPlan: descriptor element != T",
            ));
        }
        if (desc.scan_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScanBackwardPlan: scan_axis out of range for rank N",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ScanBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScanBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let kind_supported = matches!(
            desc.kind,
            ScanKind::Cumsum
                | ScanKind::Cumprod
                | ScanKind::Cummax
                | ScanKind::Cummin
                | ScanKind::LogCumsumExp
        );
        if !kind_supported || !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScanBackwardPlan: wired today: \
                 `{Cumsum, Cumprod, Cummax, Cummin, LogCumsumExp} × {f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Scan,
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

    /// Validate args.
    pub fn can_implement(&self, args: &ScanBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScanBackwardPlan: dy shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScanBackwardPlan: dx shape mismatch",
            ));
        }
        let numel = args.dx.numel();
        let dy_len = args.dy.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        if dy_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(dx_len) as usize,
            });
        }
        // Op-specific saved-tensor checks.
        if op_needs_saved_x(self.desc.kind) {
            let x = args.x.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::ScanBackwardPlan: Cumprod / Cummax / Cummin BW \
                 require args.x (saved forward input)",
            ))?;
            if x.shape != self.desc.input_shape {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ScanBackwardPlan: args.x shape mismatch",
                ));
            }
            if (x.data.len() as i64) < numel {
                return Err(Error::BufferTooSmall {
                    needed: numel as usize,
                    got: x.data.len(),
                });
            }
        }
        if op_needs_saved_y(self.desc.kind) {
            let y = args.y.as_ref().ok_or(Error::InvalidProblem(
                "baracuda-kernels::ScanBackwardPlan: Cumprod BW requires args.y \
                 (saved forward output)",
            ))?;
            if y.shape != self.desc.input_shape {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ScanBackwardPlan: args.y shape mismatch",
                ));
            }
            if (y.data.len() as i64) < numel {
                return Err(Error::BufferTooSmall {
                    needed: numel as usize,
                    got: y.data.len(),
                });
            }
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }
    /// Kernel SKU identity.
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
        args: ScanBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dx.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let axis = self.desc.scan_axis as usize;
        let shape = self.desc.input_shape;
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;
        let scan_extent = shape[axis];
        let reverse_flag = if self.desc.reverse { 1i32 } else { 0 };

        match self.desc.kind {
            ScanKind::Cumsum => {
                // BW = FW with reverse flipped. Dispatch to the FW
                // kernel; dy → x, dx → y.
                let scan_stride_dy = stride_dy[axis];
                let cumsum_reverse = if self.desc.reverse { 0i32 } else { 1 };
                macro_rules! dispatch_cumsum {
                    ($sym:ident) => {
                        unsafe {
                            baracuda_kernels_sys::$sym(
                                numel,
                                rank,
                                shape.as_ptr(),
                                stride_dy.as_ptr(),
                                stride_dx.as_ptr(),
                                axis as i32,
                                scan_extent,
                                scan_stride_dy,
                                cumsum_reverse,
                                dy_ptr,
                                dx_ptr,
                                core::ptr::null_mut(),
                                0,
                                stream_ptr,
                            )
                        }
                    };
                }
                let status = match T::KIND {
                    ElementKind::F32 => dispatch_cumsum!(baracuda_kernels_scan_cumsum_f32_run),
                    ElementKind::F16 => dispatch_cumsum!(baracuda_kernels_scan_cumsum_f16_run),
                    ElementKind::Bf16 => dispatch_cumsum!(baracuda_kernels_scan_cumsum_bf16_run),
                    ElementKind::F64 => dispatch_cumsum!(baracuda_kernels_scan_cumsum_f64_run),
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::ScanBackwardPlan::run unsupported dtype for Cumsum",
                        ));
                    }
                };
                map_status(status)
            }
            ScanKind::Cumprod => {
                let x_ref = args.x.expect("Cumprod BW requires saved x — validated above");
                let y_ref = args.y.expect("Cumprod BW requires saved y — validated above");
                let stride_x = x_ref.stride;
                let stride_y = y_ref.stride;
                let x_ptr = x_ref.data.as_raw().0 as *const c_void;
                let y_ptr = y_ref.data.as_raw().0 as *const c_void;
                macro_rules! dispatch_cumprod_bw {
                    ($sym:ident) => {
                        unsafe {
                            baracuda_kernels_sys::$sym(
                                numel,
                                rank,
                                shape.as_ptr(),
                                stride_dy.as_ptr(),
                                stride_x.as_ptr(),
                                stride_y.as_ptr(),
                                stride_dx.as_ptr(),
                                axis as i32,
                                scan_extent,
                                reverse_flag,
                                dy_ptr,
                                x_ptr,
                                y_ptr,
                                dx_ptr,
                                core::ptr::null_mut(),
                                0,
                                stream_ptr,
                            )
                        }
                    };
                }
                let status = match T::KIND {
                    ElementKind::F32 => {
                        dispatch_cumprod_bw!(baracuda_kernels_scan_cumprod_backward_f32_run)
                    }
                    ElementKind::F16 => {
                        dispatch_cumprod_bw!(baracuda_kernels_scan_cumprod_backward_f16_run)
                    }
                    ElementKind::Bf16 => {
                        dispatch_cumprod_bw!(baracuda_kernels_scan_cumprod_backward_bf16_run)
                    }
                    ElementKind::F64 => {
                        dispatch_cumprod_bw!(baracuda_kernels_scan_cumprod_backward_f64_run)
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::ScanBackwardPlan::run unsupported dtype for Cumprod",
                        ));
                    }
                };
                map_status(status)
            }
            ScanKind::LogCumsumExp => {
                let x_ref = args
                    .x
                    .expect("LogCumsumExp BW requires saved x — validated above");
                let y_ref = args
                    .y
                    .expect("LogCumsumExp BW requires saved y — validated above");
                let stride_x = x_ref.stride;
                let stride_y = y_ref.stride;
                let x_ptr = x_ref.data.as_raw().0 as *const c_void;
                let y_ptr = y_ref.data.as_raw().0 as *const c_void;
                macro_rules! dispatch_lcse_bw {
                    ($sym:ident) => {
                        unsafe {
                            baracuda_kernels_sys::$sym(
                                numel,
                                rank,
                                shape.as_ptr(),
                                stride_dy.as_ptr(),
                                stride_x.as_ptr(),
                                stride_y.as_ptr(),
                                stride_dx.as_ptr(),
                                axis as i32,
                                scan_extent,
                                reverse_flag,
                                dy_ptr,
                                x_ptr,
                                y_ptr,
                                dx_ptr,
                                core::ptr::null_mut(),
                                0,
                                stream_ptr,
                            )
                        }
                    };
                }
                let status = match T::KIND {
                    ElementKind::F32 => {
                        dispatch_lcse_bw!(baracuda_kernels_scan_log_cumsum_exp_backward_f32_run)
                    }
                    ElementKind::F16 => {
                        dispatch_lcse_bw!(baracuda_kernels_scan_log_cumsum_exp_backward_f16_run)
                    }
                    ElementKind::Bf16 => {
                        dispatch_lcse_bw!(baracuda_kernels_scan_log_cumsum_exp_backward_bf16_run)
                    }
                    ElementKind::F64 => {
                        dispatch_lcse_bw!(baracuda_kernels_scan_log_cumsum_exp_backward_f64_run)
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::ScanBackwardPlan::run unsupported dtype for LogCumsumExp",
                        ));
                    }
                };
                map_status(status)
            }
            ScanKind::Cummax | ScanKind::Cummin => {
                let x_ref = args
                    .x
                    .expect("Cummax/Cummin BW requires saved x — validated above");
                let stride_x = x_ref.stride;
                let x_ptr = x_ref.data.as_raw().0 as *const c_void;
                macro_rules! dispatch_extrema_bw {
                    ($sym:ident) => {
                        unsafe {
                            baracuda_kernels_sys::$sym(
                                numel,
                                rank,
                                shape.as_ptr(),
                                stride_dy.as_ptr(),
                                stride_x.as_ptr(),
                                stride_dx.as_ptr(),
                                axis as i32,
                                scan_extent,
                                reverse_flag,
                                dy_ptr,
                                x_ptr,
                                dx_ptr,
                                core::ptr::null_mut(),
                                0,
                                stream_ptr,
                            )
                        }
                    };
                }
                let status = match (self.desc.kind, T::KIND) {
                    (ScanKind::Cummax, ElementKind::F32) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummax_backward_f32_run)
                    }
                    (ScanKind::Cummax, ElementKind::F16) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummax_backward_f16_run)
                    }
                    (ScanKind::Cummax, ElementKind::Bf16) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummax_backward_bf16_run)
                    }
                    (ScanKind::Cummax, ElementKind::F64) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummax_backward_f64_run)
                    }
                    (ScanKind::Cummin, ElementKind::F32) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummin_backward_f32_run)
                    }
                    (ScanKind::Cummin, ElementKind::F16) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummin_backward_f16_run)
                    }
                    (ScanKind::Cummin, ElementKind::Bf16) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummin_backward_bf16_run)
                    }
                    (ScanKind::Cummin, ElementKind::F64) => {
                        dispatch_extrema_bw!(baracuda_kernels_scan_cummin_backward_f64_run)
                    }
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels::ScanBackwardPlan::run reached an unimplemented \
                             (kind, dtype) pair for Cummax/Cummin",
                        ));
                    }
                };
                map_status(status)
            }
            // Defensive arm — `ScanKind` is `#[non_exhaustive]`, so a
            // newly-added variant surfaces here as an explicit
            // `Unsupported` until the kernel dispatch is wired.
            _ => Err(Error::Unsupported(
                "baracuda-kernels::ScanBackwardPlan::run reached an unimplemented ScanKind variant",
            )),
        }
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
