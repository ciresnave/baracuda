//! Single-axis scan forward plan.
//!
//! Length-preserving prefix scan along a single axis. Output shape ==
//! input shape; the scan axis is *not* collapsed (unlike a reduction).
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

/// Descriptor for a single-axis scan op.
#[derive(Copy, Clone, Debug)]
pub struct ScanDescriptor<const N: usize> {
    /// Which scan kind to apply.
    pub kind: ScanKind,
    /// Tensor shape — input and output share it.
    pub input_shape: [i32; N],
    /// Axis along which the scan accumulates. Must be in `[0, N)`.
    pub scan_axis: u8,
    /// `true` → scan from the end of the axis toward the start; `false`
    /// → standard forward scan (PyTorch default).
    pub reverse: bool,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a scan launch.
pub struct ScanArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — same shape as input.
    pub y: TensorMut<'a, T, N>,
}

/// Single-axis scan plan.
pub struct ScanPlan<T: Element, const N: usize> {
    desc: ScanDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> ScanPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &ScanDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScanPlan: descriptor element != T",
            ));
        }
        if (desc.scan_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScanPlan: scan_axis out of range for rank N",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::ScanPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScanPlan: tensor rank > 8 not supported \
                 (kernel param block fixes MAX_RANK = 8)",
            ));
        }

        // Wired today: `{Cumsum, Cumprod, Cummax, Cummin, LogCumsumExp}
        // × {f32, f16, bf16, f64}`.
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
        let supported = kind_supported && dtype_in_fp_family;
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::ScanPlan: wired today: \
                 `{Cumsum, Cumprod, Cummax, Cummin, LogCumsumExp} × {f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Bit-stable across runs (deterministic single-thread-per-cell
            // accumulator; same input → same output).
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
    pub fn can_implement(&self, args: &ScanArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScanPlan: x shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ScanPlan: y shape mismatch (scans are \
                 length-preserving — y.shape must equal x.shape)",
            ));
        }
        let numel = args.x.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if x_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x_len.min(y_len) as usize,
            });
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
        args: ScanArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let axis = self.desc.scan_axis as usize;
        let shape = self.desc.input_shape;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;
        let scan_extent = shape[axis];
        let scan_stride_x = stride_x[axis];
        let reverse = if self.desc.reverse { 1i32 } else { 0 };

        macro_rules! dispatch {
            ($sym:ident) => {
                unsafe {
                    baracuda_kernels_sys::$sym(
                        numel,
                        rank,
                        shape.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        axis as i32,
                        scan_extent,
                        scan_stride_x,
                        reverse,
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            };
        }

        let status = match (self.desc.kind, T::KIND) {
            (ScanKind::Cumsum, ElementKind::F32) => dispatch!(baracuda_kernels_scan_cumsum_f32_run),
            (ScanKind::Cumsum, ElementKind::F16) => dispatch!(baracuda_kernels_scan_cumsum_f16_run),
            (ScanKind::Cumsum, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_scan_cumsum_bf16_run)
            }
            (ScanKind::Cumsum, ElementKind::F64) => dispatch!(baracuda_kernels_scan_cumsum_f64_run),
            (ScanKind::Cumprod, ElementKind::F32) => {
                dispatch!(baracuda_kernels_scan_cumprod_f32_run)
            }
            (ScanKind::Cumprod, ElementKind::F16) => {
                dispatch!(baracuda_kernels_scan_cumprod_f16_run)
            }
            (ScanKind::Cumprod, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_scan_cumprod_bf16_run)
            }
            (ScanKind::Cumprod, ElementKind::F64) => {
                dispatch!(baracuda_kernels_scan_cumprod_f64_run)
            }
            (ScanKind::Cummax, ElementKind::F32) => dispatch!(baracuda_kernels_scan_cummax_f32_run),
            (ScanKind::Cummax, ElementKind::F16) => dispatch!(baracuda_kernels_scan_cummax_f16_run),
            (ScanKind::Cummax, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_scan_cummax_bf16_run)
            }
            (ScanKind::Cummax, ElementKind::F64) => dispatch!(baracuda_kernels_scan_cummax_f64_run),
            (ScanKind::Cummin, ElementKind::F32) => dispatch!(baracuda_kernels_scan_cummin_f32_run),
            (ScanKind::Cummin, ElementKind::F16) => dispatch!(baracuda_kernels_scan_cummin_f16_run),
            (ScanKind::Cummin, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_scan_cummin_bf16_run)
            }
            (ScanKind::Cummin, ElementKind::F64) => dispatch!(baracuda_kernels_scan_cummin_f64_run),
            (ScanKind::LogCumsumExp, ElementKind::F32) => {
                dispatch!(baracuda_kernels_scan_log_cumsum_exp_f32_run)
            }
            (ScanKind::LogCumsumExp, ElementKind::F16) => {
                dispatch!(baracuda_kernels_scan_log_cumsum_exp_f16_run)
            }
            (ScanKind::LogCumsumExp, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_scan_log_cumsum_exp_bf16_run)
            }
            (ScanKind::LogCumsumExp, ElementKind::F64) => {
                dispatch!(baracuda_kernels_scan_log_cumsum_exp_f64_run)
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::ScanPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
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
