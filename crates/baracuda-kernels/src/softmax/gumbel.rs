//! GumbelSoftmax forward plan — stochastic softmax with reparameterized
//! Gumbel noise.
//!
//! `y = softmax((x + g) / τ)` where `g[k] = -log(-log(u[k]))` with
//! `u[k] ~ Uniform(0, 1)`, drawn through cuRAND with the descriptor's
//! seed. Optional `hard` mode emits a one-hot tensor at the row's noisy
//! argmax (straight-through gradient lives in autograd; the saved
//! `y_soft` is what BW differentiates).
//!
//! Wired today: `T ∈ {f32, f16, bf16, f64}`.
//!
//! Workspace: `numel * sizeof(f32)` bytes for the cuRAND uniform-rand
//! buffer.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    curandCreateGenerator, curandDestroyGenerator, curandGenerateUniform, curandGenerator_t,
    curandSetPseudoRandomGeneratorSeed, curandSetStream,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SoftmaxKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a GumbelSoftmax forward op.
#[derive(Copy, Clone, Debug)]
pub struct GumbelSoftmaxDescriptor<const N: usize> {
    /// Tensor shape (input and output share it).
    pub input_shape: [i32; N],
    /// Axis along which to compute softmax. Must be in `[0, N)`.
    pub softmax_axis: u8,
    /// Temperature `τ`. Must be `> 0`. Lower → sharper, higher → more uniform.
    pub temperature: f32,
    /// Hard mode: when true, emit a one-hot at the row's noisy argmax.
    /// Straight-through gradient flows back through the soft form in
    /// autograd; the kernel still computes (and the plan still saves)
    /// the soft output for BW.
    pub hard: bool,
    /// Deterministic seed for the noise sample.
    pub seed: u64,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a GumbelSoftmax forward launch.
pub struct GumbelSoftmaxArgs<'a, T: Element, const N: usize> {
    /// Input logits.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — same shape as input.
    pub y: TensorMut<'a, T, N>,
}

/// GumbelSoftmax forward plan.
///
/// Owns a cuRAND generator (lazy, `!Sync`) — same lifetime / threading
/// model as [`crate::random::RandomPlan`].
pub struct GumbelSoftmaxPlan<T: Element, const N: usize> {
    desc: GumbelSoftmaxDescriptor<N>,
    sku: KernelSku,
    generator: Cell<curandGenerator_t>,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GumbelSoftmaxPlan<T, N> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &GumbelSoftmaxDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GumbelSoftmaxPlan: descriptor element != T",
            ));
        }
        if (desc.softmax_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GumbelSoftmaxPlan: softmax_axis out of range for rank N",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::GumbelSoftmaxPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::GumbelSoftmaxPlan: tensor rank > 8 not supported",
            ));
        }
        if !(desc.temperature > 0.0) || !desc.temperature.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GumbelSoftmaxPlan: temperature must be > 0 and finite",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::GumbelSoftmaxPlan: wired today: {f32, f16, bf16, f64}",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: match T::KIND {
                ElementKind::F64 => ElementKind::F64,
                _ => ElementKind::F32,
            },
            // Bit-stable per stream given fixed seed.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Softmax,
            op: SoftmaxKind::GumbelSoftmax as u16,
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
            generator: Cell::new(core::ptr::null_mut()),
            _marker: PhantomData,
        })
    }

    /// Validate args.
    pub fn can_implement(&self, args: &GumbelSoftmaxArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GumbelSoftmaxPlan: x shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GumbelSoftmaxPlan: y shape mismatch",
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

    /// Workspace size — `numel * sizeof(f32)` bytes for the cuRAND
    /// uniform-rand buffer.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        let numel: i64 = self.desc.input_shape.iter().map(|&d| d as i64).product();
        (numel.max(0) as usize) * core::mem::size_of::<f32>()
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

    fn ensure_generator(&self) -> Result<curandGenerator_t> {
        let g = self.generator.get();
        if !g.is_null() {
            return Ok(g);
        }
        let mut handle: curandGenerator_t = core::ptr::null_mut();
        let status = unsafe { curandCreateGenerator(&mut handle as *mut _, 100) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe { curandSetPseudoRandomGeneratorSeed(handle, self.desc.seed) };
        if status != 0 {
            unsafe {
                let _ = curandDestroyGenerator(handle);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.generator.set(handle);
        Ok(handle)
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: GumbelSoftmaxArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }

        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                return Err(Error::WorkspaceTooSmall { needed, got: 0 });
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };

        let stream_ptr = stream.as_raw() as *mut c_void;
        let gen_handle = self.ensure_generator()?;
        let status = unsafe { curandSetStream(gen_handle, stream_ptr) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let rand_ptr = ws_ptr as *mut f32;
        let status = unsafe { curandGenerateUniform(gen_handle, rand_ptr, numel as usize) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }

        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;

        let axis = self.desc.softmax_axis as usize;
        let shape = self.desc.input_shape;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;
        let extent = shape[axis];
        let stride_x_axis = stride_x[axis];
        let stride_y_axis = stride_y[axis];
        let inv_tau = 1.0_f32 / self.desc.temperature;
        let hard = if self.desc.hard { 1_i32 } else { 0_i32 };

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
                        extent,
                        stride_x_axis,
                        stride_y_axis,
                        inv_tau,
                        hard,
                        x_ptr,
                        rand_ptr as *const c_void,
                        y_ptr,
                        core::ptr::null_mut(),
                        ws_bytes,
                        stream_ptr,
                    )
                }
            };
        }
        let status = match T::KIND {
            ElementKind::F32 => dispatch!(baracuda_kernels_gumbel_softmax_f32_run),
            ElementKind::F16 => dispatch!(baracuda_kernels_gumbel_softmax_f16_run),
            ElementKind::Bf16 => dispatch!(baracuda_kernels_gumbel_softmax_bf16_run),
            ElementKind::F64 => dispatch!(baracuda_kernels_gumbel_softmax_f64_run),
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GumbelSoftmaxPlan::run reached an unimplemented \
                     dtype — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

impl<T: Element, const N: usize> Drop for GumbelSoftmaxPlan<T, N> {
    fn drop(&mut self) {
        let g = self.generator.get();
        if !g.is_null() {
            unsafe {
                let _ = curandDestroyGenerator(g);
            }
            self.generator.set(core::ptr::null_mut());
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
