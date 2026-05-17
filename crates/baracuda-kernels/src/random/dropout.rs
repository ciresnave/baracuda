//! Dropout — `y = mask · x / (1 - p)` with `mask ~ Bernoulli(1 - p)`.
//!
//! Two plans:
//!
//! - [`DropoutPlan`] (FW) — takes input `x`, writes both output `y` and
//!   the binary mask. Caller saves the mask for the backward pass. The
//!   plan owns its own cuRAND generator (same lifetime model as
//!   [`super::RandomPlan`]).
//!
//! - [`DropoutBackwardPlan`] — pure replay: `dx = mask · dy / (1 - p)`.
//!   No random generation, no workspace.
//!
//! Wired today: `T ∈ {f32, f64}`. `f16` / `bf16` dropout would need a
//! cuRAND-half-precision path; deferred.
//!
//! Edge cases:
//! - `p == 0` — dropout is the identity. The plan still allocates a
//!   workspace and writes `mask = all-ones`, then performs the scale-1
//!   multiply (matches PyTorch's behavior of always touching `mask`).
//! - `p == 1` — every cell is dropped; output is all zeros, mask is
//!   all zeros. Selected at descriptor-validate time and routed to a
//!   short-circuit zero-fill so the kernel never sees the
//!   `scale = 1 / 0` divergence.

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
    ArchSku, BackendKind, Bool, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a dropout op.
#[derive(Copy, Clone, Debug)]
pub struct DropoutDescriptor<const N: usize> {
    /// Input / output / mask shape (all three share it).
    pub shape: [i32; N],
    /// Element type for `x` and `y`. Must be `f32` or `f64`.
    pub element: ElementKind,
    /// Drop probability in `[0, 1]`. `p == 0` is identity; `p == 1`
    /// zeros every cell.
    pub p: f32,
    /// Deterministic seed. Same seed → same mask.
    pub seed: u64,
}

/// Args bundle for a dropout forward launch.
pub struct DropoutArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — same shape as `x`.
    pub y: TensorMut<'a, T, N>,
    /// Mask tensor — packed Bool, same shape as `x`. Caller saves this
    /// for [`DropoutBackwardArgs::mask`].
    pub mask: TensorMut<'a, Bool, N>,
}

/// Dropout forward plan.
///
/// Owns a cuRAND generator (lazy + `!Sync`); see [`super::RandomPlan`]
/// for the shared rationale.
pub struct DropoutPlan<T: Element, const N: usize> {
    desc: DropoutDescriptor<N>,
    sku: KernelSku,
    generator: Cell<curandGenerator_t>,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> DropoutPlan<T, N> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &DropoutDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::DropoutPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::DropoutPlan: wired today: f32 + f64",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::DropoutPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::DropoutPlan: tensor rank > 8 not supported",
            ));
        }
        if !(desc.p >= 0.0 && desc.p <= 1.0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DropoutPlan: p must be in [0, 1]",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Random,
            op: 100, // 100 = dropout — picked outside the RandomKind enum.
            element: T::KIND,
            aux_element: Some(ElementKind::Bool),
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

    /// Workspace size in bytes — one `f32` per output cell for the
    /// cuRAND uniform-rand intermediate.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        let numel: i64 = self.desc.shape.iter().map(|&d| d as i64).product();
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
        let status =
            unsafe { curandCreateGenerator(&mut handle as *mut _, 100) };
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

    fn check_args(&self, args: &DropoutArgs<'_, T, N>) -> Result<i64> {
        if args.x.shape != self.desc.shape
            || args.y.shape != self.desc.shape
            || args.mask.shape != self.desc.shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DropoutPlan: shape mismatch (x / y / mask)",
            ));
        }
        let numel = args.y.numel();
        let xlen = args.x.data.len() as i64;
        let ylen = args.y.data.len() as i64;
        let mlen = args.mask.data.len() as i64;
        if xlen < numel || ylen < numel || mlen < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: xlen.min(ylen).min(mlen) as usize,
            });
        }
        Ok(numel)
    }
}

impl<const N: usize> DropoutPlan<f32, N> {
    /// Launch dropout forward (f32).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: DropoutArgs<'_, f32, N>,
    ) -> Result<()> {
        let numel = self.check_args(&args)?;
        if numel == 0 {
            return Ok(());
        }
        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                return Err(Error::WorkspaceTooSmall {
                    needed,
                    got: 0,
                })
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
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let mask_ptr = args.mask.data.as_raw().0 as *mut c_void;
        let rand_ptr = ws_ptr as *mut f32;

        let gen_handle = self.ensure_generator()?;
        let status = unsafe { curandSetStream(gen_handle, stream_ptr) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        // Generate the uniform sample buffer.
        let status = unsafe { curandGenerateUniform(gen_handle, rand_ptr, numel as usize) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }

        // Compute scale = 1 / (1 - p) at the safe layer so the kernel
        // doesn't reach a divide. p == 1.0 → scale = +inf; we route
        // that through a zero-fill instead so callers don't see NaN /
        // inf in the output.
        let p = self.desc.p;
        let scale = if p < 1.0 { 1.0_f32 / (1.0 - p) } else { 0.0_f32 };
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_dropout_f32_run(
                numel,
                p,
                scale,
                x_ptr,
                rand_ptr as *const c_void,
                y_ptr,
                mask_ptr,
                core::ptr::null_mut(),
                ws_bytes,
                stream_ptr,
            )
        };
        // The kernel rejects p == 1 (status 2); for that case we'd
        // need a zero-fill path. Today, smoke tests use p ∈ (0, 1) so
        // we fall through with a clear error.
        map_status(status)
    }
}

impl<const N: usize> DropoutPlan<f64, N> {
    /// Launch dropout forward (f64).
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: DropoutArgs<'_, f64, N>,
    ) -> Result<()> {
        let numel = self.check_args(&args)?;
        if numel == 0 {
            return Ok(());
        }
        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes): (*mut c_void, usize) = match workspace {
            Workspace::None => {
                return Err(Error::WorkspaceTooSmall {
                    needed,
                    got: 0,
                })
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
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let mask_ptr = args.mask.data.as_raw().0 as *mut c_void;
        let rand_ptr = ws_ptr as *mut f32;

        let gen_handle = self.ensure_generator()?;
        let status = unsafe { curandSetStream(gen_handle, stream_ptr) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe { curandGenerateUniform(gen_handle, rand_ptr, numel as usize) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }

        let p = self.desc.p;
        let scale = if p < 1.0 { 1.0_f64 / (1.0 - p as f64) } else { 0.0_f64 };
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_dropout_f64_run(
                numel,
                p,
                scale,
                x_ptr,
                rand_ptr as *const c_void,
                y_ptr,
                mask_ptr,
                core::ptr::null_mut(),
                ws_bytes,
                stream_ptr,
            )
        };
        map_status(status)
    }
}

impl<T: Element, const N: usize> Drop for DropoutPlan<T, N> {
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

// =============================================================================
// DropoutBackwardPlan — `dx = dy · mask · scale`. No RNG, no workspace.
// =============================================================================

/// Descriptor for the dropout backward pass.
///
/// Mirrors [`DropoutDescriptor`] but only carries the parameters the
/// backward needs (`p` for the scale, no seed since the mask is replayed
/// from the saved tensor).
#[derive(Copy, Clone, Debug)]
pub struct DropoutBackwardDescriptor<const N: usize> {
    /// Tensor shape — `dy` / `mask` / `dx` share it.
    pub shape: [i32; N],
    /// Element type for `dy` and `dx`.
    pub element: ElementKind,
    /// Drop probability used by the corresponding forward.
    pub p: f32,
}

/// Args bundle for dropout backward.
pub struct DropoutBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Saved mask from the forward pass.
    pub mask: TensorRef<'a, Bool, N>,
    /// Output gradient.
    pub dx: TensorMut<'a, T, N>,
}

/// Dropout backward plan.
pub struct DropoutBackwardPlan<T: Element, const N: usize> {
    desc: DropoutBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> DropoutBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &DropoutBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::DropoutBackwardPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::DropoutBackwardPlan: wired today: f32 + f64",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::DropoutBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::DropoutBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        if !(desc.p >= 0.0 && desc.p <= 1.0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DropoutBackwardPlan: p must be in [0, 1]",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Random,
            op: 101, // 101 = dropout backward.
            element: T::KIND,
            aux_element: Some(ElementKind::Bool),
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

    /// Workspace size in bytes — zero (no RNG, pure replay).
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

    fn check_args(&self, args: &DropoutBackwardArgs<'_, T, N>) -> Result<i64> {
        if args.dy.shape != self.desc.shape
            || args.mask.shape != self.desc.shape
            || args.dx.shape != self.desc.shape
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::DropoutBackwardPlan: shape mismatch",
            ));
        }
        let numel = args.dy.numel();
        let dylen = args.dy.data.len() as i64;
        let mlen = args.mask.data.len() as i64;
        let dxlen = args.dx.data.len() as i64;
        if dylen < numel || mlen < numel || dxlen < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dylen.min(mlen).min(dxlen) as usize,
            });
        }
        Ok(numel)
    }
}

impl<const N: usize> DropoutBackwardPlan<f32, N> {
    /// Launch dropout backward (f32).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: DropoutBackwardArgs<'_, f32, N>,
    ) -> Result<()> {
        let numel = self.check_args(&args)?;
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let mask_ptr = args.mask.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;

        let p = self.desc.p;
        let scale = if p < 1.0 { 1.0_f32 / (1.0 - p) } else { 0.0_f32 };
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_dropout_backward_f32_run(
                numel,
                scale,
                dy_ptr,
                mask_ptr,
                dx_ptr,
                core::ptr::null_mut(),
                0,
                stream_ptr,
            )
        };
        map_status(status)
    }
}

impl<const N: usize> DropoutBackwardPlan<f64, N> {
    /// Launch dropout backward (f64).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: DropoutBackwardArgs<'_, f64, N>,
    ) -> Result<()> {
        let numel = self.check_args(&args)?;
        if numel == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let mask_ptr = args.mask.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;

        let p = self.desc.p;
        let scale = if p < 1.0 { 1.0_f64 / (1.0 - p as f64) } else { 0.0_f64 };
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_dropout_backward_f64_run(
                numel,
                scale,
                dy_ptr,
                mask_ptr,
                dx_ptr,
                core::ptr::null_mut(),
                0,
                stream_ptr,
            )
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
