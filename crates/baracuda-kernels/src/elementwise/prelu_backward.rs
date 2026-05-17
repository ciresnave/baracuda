//! PReLU BW plan — Milestone 5.3.
//!
//! Two kernels:
//!   - per-cell `dx[..., c, ...] = dy if x > 0 else weight[c] * dy`
//!   - per-channel `dweight[c] = Σ over non-channel cells of (dy · x where x < 0)`
//!     via deterministic warp-shuffle reduction (no atomicAdd).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, UnaryKind, Workspace,
};

/// Descriptor for a PReLU BW op.
#[derive(Copy, Clone, Debug)]
pub struct PReluBackwardDescriptor<const N: usize> {
    /// Input shape.
    pub input_shape: [i32; N],
    /// Channel axis (`-1` for scalar weight).
    pub channel_axis: i8,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a PReLU BW launch.
pub struct PReluBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// x saved from FW.
    pub x: TensorRef<'a, T, N>,
    /// weight saved from FW.
    pub weight: TensorRef<'a, T, 1>,
    /// Output gradient w.r.t. x.
    pub dx: TensorMut<'a, T, N>,
    /// Output gradient w.r.t. weight — shape `[C]` (per-channel) or `[1]`
    /// (scalar).
    pub dweight: TensorMut<'a, T, 1>,
}

/// PReLU backward plan.
pub struct PReluBackwardPlan<T: Element, const N: usize> {
    desc: PReluBackwardDescriptor<N>,
    sku: KernelSku,
    channel_stride: i64,
    channel_extent: i32,
    scalar_weight: bool,
    _marker: PhantomData<T>,
}

fn check_dtype<T: Element>() -> Result<()> {
    let ok = matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
    );
    if !ok {
        return Err(Error::Unsupported(
            "baracuda-kernels::PReluBackwardPlan: only {f32, f16, bf16, f64} wired",
        ));
    }
    Ok(())
}

impl<T: Element, const N: usize> PReluBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &PReluBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PReluBackwardPlan: descriptor element != T",
            ));
        }
        check_dtype::<T>()?;
        let rank = N as i8;
        let scalar_weight = desc.channel_axis < 0;
        if !scalar_weight && (desc.channel_axis >= rank) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PReluBackwardPlan: channel_axis out of range",
            ));
        }
        let (channel_stride, channel_extent) = if scalar_weight {
            (1i64, 1i32)
        } else {
            let axis = desc.channel_axis as usize;
            let extent = desc.input_shape[axis];
            let mut stride: i64 = 1;
            for d in (axis + 1)..N {
                stride = stride.saturating_mul(desc.input_shape[d] as i64);
            }
            (stride, extent)
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: if T::KIND == ElementKind::F64 {
                ElementKind::F64
            } else {
                ElementKind::F32
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::UnaryElementwise,
            op: UnaryKind::PReLU as u16,
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
            channel_stride,
            channel_extent,
            scalar_weight,
            _marker: PhantomData,
        })
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
        args: PReluBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        if args.x.shape != self.desc.input_shape
            || args.dy.shape != self.desc.input_shape
            || args.dx.shape != self.desc.input_shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PReluBackwardPlan: tensor shape mismatch",
            ));
        }
        let expected_weight = if self.scalar_weight { 1 } else { self.channel_extent };
        if args.weight.shape[0] != expected_weight || args.dweight.shape[0] != expected_weight {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PReluBackwardPlan: weight shape mismatch",
            ));
        }
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let weight_ptr = args.weight.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let dweight_ptr = args.dweight.data.as_raw().0 as *mut c_void;
        let scalar_flag: i32 = if self.scalar_weight { 1 } else { 0 };
        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_prelu_backward_f32_run(
                    numel,
                    self.channel_stride,
                    self.channel_extent,
                    scalar_flag,
                    dy_ptr,
                    x_ptr,
                    weight_ptr,
                    dx_ptr,
                    dweight_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_prelu_backward_f16_run(
                    numel,
                    self.channel_stride,
                    self.channel_extent,
                    scalar_flag,
                    dy_ptr,
                    x_ptr,
                    weight_ptr,
                    dx_ptr,
                    dweight_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_prelu_backward_bf16_run(
                    numel,
                    self.channel_stride,
                    self.channel_extent,
                    scalar_flag,
                    dy_ptr,
                    x_ptr,
                    weight_ptr,
                    dx_ptr,
                    dweight_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_prelu_backward_f64_run(
                    numel,
                    self.channel_stride,
                    self.channel_extent,
                    scalar_flag,
                    dy_ptr,
                    x_ptr,
                    weight_ptr,
                    dx_ptr,
                    dweight_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::PReluBackwardPlan::run unwired dtype",
                ));
            }
        };
        match status {
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
}
