//! Manifold-Constrained Hyper-Connections (Phase 43, Tier 1).
//!
//! Drop-in replacement for the bare `y = x + sublayer(x)` residual
//! connection in transformer blocks. Mixes `n` parallel residual
//! streams (`x_expanded[B, n, C]`) through a small Sinkhorn-Knopp-
//! normalized `(n × n)` matrix `M`, then adds the post-stream
//! contribution `H_post[i] * RMSNorm(aggregate)`.
//!
//! Backed by the vendored mHC.cu (Andre Slavescu, MIT) — see
//! `crates/baracuda-kernels-sys/vendor/mhc/` for license and provenance.
//! Paper: DeepSeek-AI, *Manifold-Constrained Hyper-Connections*,
//! arXiv:2512.24880.
//!
//! ## Tier 1 scope
//!
//! - **Static-H FW** (Tier 1). Dynamic-H FW still lives behind the same
//!   vendored `MHCLayer` class and is not wired.
//! - **Static-H backward** (Tier 2): [`HyperConnectionPlan::select_training`] +
//!   [`HyperConnectionPlan::run_backward`]. Checked against central differences
//!   of an f64 reference on `n = 4`; the Sinkhorn adjoint is baracuda's own
//!   because the vendored one is wrong (see `mhc_launcher.cu`). Limited to
//!   `sinkhorn_iters <= 20`.
//! - **Activation dtype `T` in `{f32, f16, bf16}`** (Tier 3). The vendored
//!   path is f32 activations with a bf16 RMSNorm gamma, and it is unchanged.
//!   For `T = f16 / bf16` the plan widens `x_expanded` to an f32 scratch
//!   buffer it owns, runs the f32 path, and narrows `out` back, using
//!   [`CastPlan`](crate::elementwise::CastPlan). The result equals the f32 path on the widened inputs,
//!   rounded once to `T` on the way out; the cast is a scratch conversion,
//!   not a change to the numerics. See [`HyperConnectionPlan::scratch_bytes`]
//!   for the scratch size and the per-call traffic.
//! - **n ≤ 32**. Above 32 the upstream switches to a cuBLAS-Lt
//!   tensor-core mixing kernel that has not yet been validated in
//!   the C-ABI shim.
//!
//! ## Stateful plan
//!
//! Unlike most `*Plan` types in this crate, `HyperConnectionPlan`
//! owns a non-trivial native handle (an `MHCLayer*`) that allocates
//! ~`B*n*C*sizeof(float)` bytes of device-side scratch on construction.
//! `Drop` destroys the handle. Reuse the plan across many forward
//! calls to amortize the alloc cost — that's the whole point of the
//! stateful design upstream.

#[cfg(feature = "mhc")]
use core::ffi::c_void;
use core::marker::PhantomData;
#[cfg(feature = "mhc")]
use std::sync::Mutex;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
#[cfg(feature = "mhc")]
use baracuda_driver::{Context, DeviceBuffer, DeviceSlice, DeviceSliceMut};
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

#[cfg(feature = "mhc")]
use crate::elementwise::{CastArgs, CastDescriptor, CastPlan};

/// Descriptor for a `HyperConnectionPlan` (static-H FW).
#[derive(Copy, Clone, Debug)]
pub struct HyperConnectionDescriptor {
    /// Batch size — outer dim of `x_expanded`.
    pub batch: i32,
    /// Hidden dim — innermost dim of `x_expanded` / `out` / RMSNorm.
    pub hidden_dim: i32,
    /// Number of parallel residual streams (`n` in the paper). Must
    /// be in `1..=32`.
    pub n_streams: i32,
    /// Sinkhorn-Knopp iteration count. Paper uses 20; the kernel
    /// rejects anything outside `1..=1000`.
    pub sinkhorn_iters: i32,
    /// Epsilon added to the RMSNorm denominator and used as the
    /// Sinkhorn divide-by-zero guard. Paper uses `1e-5`.
    pub eps: f32,
    /// Element type for `x_expanded` / `out`: `F32`, `F16` or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a `HyperConnectionPlan` launch.
#[derive(Debug)]
pub struct HyperConnectionArgs<'a, T: Element> {
    /// Residual-stream input — `[B, n, C]` row-major contiguous.
    pub x_expanded: TensorRef<'a, T, 3>,
    /// RMSNorm gamma — `[C]` bf16. **Always bf16 regardless of `T`**
    /// (matches upstream `floatX` typedef).
    pub rmsnorm_weight: TensorRef<'a, half::bf16, 1>,
    /// Pre-mixing logits — `[n]` f32. The kernel passes them through
    /// sigmoid internally.
    pub h_pre: TensorRef<'a, f32, 1>,
    /// Post-mixing logits — `[n]` f32. The kernel passes them
    /// through `2 * sigmoid(.)` internally.
    pub h_post: TensorRef<'a, f32, 1>,
    /// Pre-Sinkhorn residual mixing matrix — `[n, n]` f32. The
    /// kernel passes it through Sinkhorn-Knopp iteration to project
    /// onto the doubly-stochastic manifold before mixing.
    pub h_res: TensorRef<'a, f32, 2>,
    /// Output — `[B, n, C]` row-major contiguous, same dtype as
    /// input.
    pub out: TensorMut<'a, T, 3>,
}

/// Args bundle for [`HyperConnectionPlan::run_backward`].
///
/// All tensors are f32 (`T`), including `d_rmsnorm_weight`, which is the
/// gradient of a bf16 weight.
#[derive(Debug)]
pub struct HyperConnectionBackwardArgs<'a, T: Element> {
    /// Upstream gradient of `out` — `[B, n, C]`.
    pub d_out: TensorRef<'a, T, 3>,
    /// Gradient w.r.t. `x_expanded` — `[B, n, C]`, fully overwritten.
    pub d_x_expanded: TensorMut<'a, T, 3>,
    /// Gradient w.r.t. the RMSNorm gamma — `[C]`, fully overwritten.
    pub d_rmsnorm_weight: TensorMut<'a, T, 1>,
    /// Gradient w.r.t. the pre-sigmoid `h_pre` logits — `[n]`.
    pub d_h_pre: TensorMut<'a, T, 1>,
    /// Gradient w.r.t. the pre-sigmoid `h_post` logits — `[n]`.
    pub d_h_post: TensorMut<'a, T, 1>,
    /// Gradient w.r.t. the pre-exp `h_res` — `[n, n]`.
    pub d_h_res: TensorMut<'a, T, 2>,
}

/// Hyper-Connection forward plan (static-H, bf16 weights, Tier 1).
///
/// **Formula** (with `M = Sinkhorn-Knopp(softmax_or_exp(H_res))`,
/// `s_pre = sigmoid(H_pre)`, `s_post = 2 * sigmoid(H_post)`,
/// `y_agg[b, c] = Σ_i s_pre[i] * x_expanded[b, i, c]`,
/// `y_norm = RMSNorm(y_agg)`):
///
/// `out[b, i, c] = Σ_j M[i, j] * x_expanded[b, j, c] + s_post[i] * y_norm[b, c]`
///
/// **When to use**: replace the bare `x + sublayer(x)` residual in a
/// transformer block when training a fresh model — mHC reports
/// improved training stability + downstream task scores in
/// DeepSeek-AI's experiments.
///
/// **Dtypes**: `T` is `f32`, `f16` or `bf16`. The `rmsnorm_weight` is always
/// `bf16` regardless of `T`; `h_pre` / `h_post` / `h_res` are always `f32`.
///
/// **State**: this plan owns a native `MHCLayer*` handle with
/// ~`B*n*C*sizeof(float)` bytes of GPU scratch. Reuse across many
/// `run()` calls; construction is heavy.
#[derive(Debug)]
pub struct HyperConnectionPlan<T: Element> {
    desc: HyperConnectionDescriptor,
    sku: KernelSku,
    #[cfg(feature = "mhc")]
    handle: *mut c_void,
    /// True when the handle was created with gradient buffers (`select_training`).
    training: bool,
    /// f32 scratch plus cast plans; `Some` only for `T = f16 / bf16`. The lock
    /// serialises runs on one plan, which the native handle needs anyway.
    #[cfg(feature = "mhc")]
    half: Option<Mutex<HalfPath<T>>>,
    _marker: PhantomData<T>,
}

/// Widen / narrow machinery for `T = f16 / bf16` (see the module docs).
#[cfg(feature = "mhc")]
#[derive(Debug)]
struct HalfPath<T: Element> {
    x32: DeviceBuffer<f32>,
    out32: DeviceBuffer<f32>,
    to_f32: CastPlan<T, f32>,
    from_f32: CastPlan<f32, T>,
    backward: Option<HalfBackward<T>>,
}

/// Extra f32 scratch for the gradients of a half-precision training plan.
#[cfg(feature = "mhc")]
#[derive(Debug)]
struct HalfBackward<T: Element> {
    d_out32: DeviceBuffer<f32>,
    d_x32: DeviceBuffer<f32>,
    d_gamma32: DeviceBuffer<f32>,
    d_h_pre32: DeviceBuffer<f32>,
    d_h_post32: DeviceBuffer<f32>,
    d_h_res32: DeviceBuffer<f32>,
    gamma_from_f32: CastPlan<f32, T>,
    h_vec_from_f32: CastPlan<f32, T>,
    h_res_from_f32: CastPlan<f32, T>,
}

/// Run one contiguous cast of `numel` elements through `plan`.
#[cfg(feature = "mhc")]
fn cast_run<A: Element, B: Element>(
    plan: &CastPlan<A, B>,
    stream: &Stream,
    input: DeviceSlice<'_, A>,
    output: DeviceSliceMut<'_, B>,
    numel: i32,
) -> Result<()> {
    let input = crate::view_check::valid(TensorRef::contiguous(input, [numel]))?;
    let output = crate::view_check::valid(TensorMut::contiguous(output, [numel]))?;
    plan.run(stream, Workspace::None, CastArgs { input, output })
}

/// Select a cast plan over `numel` elements.
#[cfg(feature = "mhc")]
fn cast_plan<A: Element, B: Element>(stream: &Stream, numel: i32) -> Result<CastPlan<A, B>> {
    let desc = CastDescriptor {
        numel,
        input_element: A::KIND,
        output_element: B::KIND,
    };
    CastPlan::select(stream, &desc, PlanPreference::default())
}

#[cfg(feature = "mhc")]
impl<T: Element> HalfPath<T> {
    /// Scratch and cast plans for a stream of `e` elements; the gradient scratch is
    /// added when `training`.
    fn new(
        stream: &Stream,
        desc: &HyperConnectionDescriptor,
        e: usize,
        training: bool,
    ) -> Result<Self> {
        let ctx = stream.context();
        let backward = if training {
            Some(HalfBackward::new(stream, desc, e)?)
        } else {
            None
        };
        Ok(Self {
            x32: scratch_f32(ctx, e)?,
            out32: scratch_f32(ctx, e)?,
            to_f32: cast_plan(stream, e as i32)?,
            from_f32: cast_plan(stream, e as i32)?,
            backward,
        })
    }
}

#[cfg(feature = "mhc")]
impl<T: Element> HalfBackward<T> {
    fn new(stream: &Stream, desc: &HyperConnectionDescriptor, e: usize) -> Result<Self> {
        let ctx = stream.context();
        let (c, n) = (desc.hidden_dim, desc.n_streams);
        Ok(Self {
            d_out32: scratch_f32(ctx, e)?,
            d_x32: scratch_f32(ctx, e)?,
            d_gamma32: scratch_f32(ctx, c as usize)?,
            d_h_pre32: scratch_f32(ctx, n as usize)?,
            d_h_post32: scratch_f32(ctx, n as usize)?,
            d_h_res32: scratch_f32(ctx, (n * n) as usize)?,
            gamma_from_f32: cast_plan(stream, c)?,
            h_vec_from_f32: cast_plan(stream, n)?,
            h_res_from_f32: cast_plan(stream, n * n)?,
        })
    }

    /// Raw f32 pointers for one native backward call over this scratch.
    fn ptrs(&mut self) -> BackwardPtrs {
        BackwardPtrs {
            d_out: self.d_out32.as_slice().as_raw().0 as *const c_void,
            d_x: self.d_x32.as_slice_mut().as_raw().0 as *mut c_void,
            d_gamma: self.d_gamma32.as_slice_mut().as_raw().0 as *mut c_void,
            d_h_pre: self.d_h_pre32.as_slice_mut().as_raw().0 as *mut c_void,
            d_h_post: self.d_h_post32.as_slice_mut().as_raw().0 as *mut c_void,
            d_h_res: self.d_h_res32.as_slice_mut().as_raw().0 as *mut c_void,
        }
    }

    /// Narrow the four small gradients (gamma, `h_pre`, `h_post`, `h_res`) into the
    /// caller's tensors.
    fn narrow_small(
        &self,
        stream: &Stream,
        c: i32,
        n: i32,
        args: &mut HyperConnectionBackwardArgs<'_, T>,
    ) -> Result<()> {
        let dg = args.d_rmsnorm_weight.data_mut().slice_mut(0..c as usize);
        cast_run(
            &self.gamma_from_f32,
            stream,
            self.d_gamma32.as_slice(),
            dg,
            c,
        )?;
        let dpre = args.d_h_pre.data_mut().slice_mut(0..n as usize);
        cast_run(
            &self.h_vec_from_f32,
            stream,
            self.d_h_pre32.as_slice(),
            dpre,
            n,
        )?;
        let dpost = args.d_h_post.data_mut().slice_mut(0..n as usize);
        cast_run(
            &self.h_vec_from_f32,
            stream,
            self.d_h_post32.as_slice(),
            dpost,
            n,
        )?;
        let dres = args.d_h_res.data_mut().slice_mut(0..(n * n) as usize);
        cast_run(
            &self.h_res_from_f32,
            stream,
            self.d_h_res32.as_slice(),
            dres,
            n * n,
        )
    }
}

/// Lock the plan-owned scratch (a poisoned lock means a prior run panicked mid-cast).
#[cfg(feature = "mhc")]
fn lock_scratch<T: Element>(
    half: &Mutex<HalfPath<T>>,
) -> Result<std::sync::MutexGuard<'_, HalfPath<T>>> {
    half.lock().map_err(|_| {
        Error::InvalidProblem("baracuda-kernels::HyperConnectionPlan: scratch lock poisoned")
    })
}

/// Zero-initialised f32 scratch of `len` elements on `ctx`.
#[cfg(feature = "mhc")]
fn scratch_f32(ctx: &Context, len: usize) -> Result<DeviceBuffer<f32>> {
    DeviceBuffer::zeros(ctx, len).map_err(|_| {
        Error::InvalidProblem("baracuda-kernels::HyperConnectionPlan: f32 scratch alloc failed")
    })
}

// The handle wraps device memory owned by the current process; safe
// to send between threads as long as construction / Drop happen on
// the same one. We don't expose any &mut-on-shared-handle API, so
// the contract is purely "owner thread + caller stream".
unsafe impl<T: Element> Send for HyperConnectionPlan<T> {}
unsafe impl<T: Element> Sync for HyperConnectionPlan<T> {}

impl<T: Element> HyperConnectionPlan<T> {
    /// Construct a plan for the given descriptor. Allocates the
    /// internal `MHCLayer` scratch on the current CUDA context.
    /// Returns `Err(Error::Unsupported)` if the `mhc` feature is off
    /// or the descriptor is outside the Tier-1 SKU matrix.
    pub fn select(
        stream: &Stream,
        desc: &HyperConnectionDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        Self::select_impl(stream, desc, pref, false)
    }

    /// Like [`Self::select`], but the plan also allocates the gradient buffers
    /// so [`Self::run_backward`] can follow a [`Self::run`].
    pub fn select_training(
        stream: &Stream,
        desc: &HyperConnectionDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        Self::select_impl(stream, desc, pref, true)
    }

    fn is_half() -> bool {
        matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16)
    }

    /// `batch * n_streams * hidden_dim`, the element count of `x_expanded` / `out`.
    fn stream_elems(desc: &HyperConnectionDescriptor) -> i64 {
        i64::from(desc.batch) * i64::from(desc.n_streams) * i64::from(desc.hidden_dim)
    }

    /// Bytes of f32 scratch the plan allocates for `desc` (the `f16 / bf16`
    /// widen / narrow path), host-side and device-free. `0` for `f32`, which
    /// runs the vendored path in place.
    ///
    /// With `E = batch * n_streams * hidden_dim`, `C = hidden_dim`, `n = n_streams`:
    /// the forward scratch is `2 * E` f32 (widened input, f32 output); a training
    /// plan adds `2 * E + C + 2 * n + n * n` f32 (widened `d_out`, `d_x_expanded`,
    /// and the gradients of gamma, `h_pre`, `h_post`, `h_res`).
    ///
    /// Extra device traffic per forward call, on top of the f32 path, with `T`
    /// 2 bytes wide: widen reads `2 * E` and writes `4 * E` bytes, narrow reads
    /// `4 * E` and writes `2 * E`; `12 * E` bytes in total. Backward adds the
    /// same `12 * E` for `d_out` and `d_x_expanded`, plus `O(C + n^2)` for the
    /// small gradients.
    pub fn scratch_bytes(desc: &HyperConnectionDescriptor, training: bool) -> usize {
        if !matches!(desc.element, ElementKind::F16 | ElementKind::Bf16) {
            return 0;
        }
        let e = Self::stream_elems(desc).max(0) as usize;
        let (c, n) = (
            desc.hidden_dim.max(0) as usize,
            desc.n_streams.max(0) as usize,
        );
        let forward = 2 * e;
        let backward = if training {
            2 * e + c + 2 * n + n * n
        } else {
            0
        };
        (forward + backward) * core::mem::size_of::<f32>()
    }

    fn validate_descriptor(desc: &HyperConnectionDescriptor, training: bool) -> Result<()> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: descriptor element != T",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: wired element types are `{f32, f16, bf16}`",
            ));
        }
        Self::validate_shape(desc)?;
        if Self::is_half() && Self::stream_elems(desc) > i64::from(i32::MAX) {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: f16 / bf16 activations need \
                 batch * n_streams * hidden_dim <= i32::MAX (the cast kernels count in i32)",
            ));
        }
        Self::validate_solver(desc, training)
    }

    fn validate_shape(desc: &HyperConnectionDescriptor) -> Result<()> {
        if desc.batch <= 0 || desc.hidden_dim <= 0 || desc.n_streams <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: batch / hidden_dim / n_streams must be positive",
            ));
        }
        if desc.n_streams >= 32 {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: n_streams >= 32 not yet supported \
                 (would activate the cuBLAS-Lt tensor-core mix path)",
            ));
        }
        if desc.hidden_dim < desc.n_streams {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: hidden_dim < n_streams (need at least \
                 one channel per stream for the aggregate kernel)",
            ));
        }
        Ok(())
    }

    fn validate_solver(desc: &HyperConnectionDescriptor, training: bool) -> Result<()> {
        if training && desc.sinkhorn_iters > 20 {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: training plans support sinkhorn_iters <= 20",
            ));
        }
        if desc.sinkhorn_iters <= 0 || desc.sinkhorn_iters > 1000 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: sinkhorn_iters must be in 1..=1000",
            ));
        }
        if !(desc.eps.is_finite() && desc.eps > 0.0 && desc.eps < 1.0) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: eps must be finite and in (0, 1)",
            ));
        }
        Ok(())
    }

    fn build_sku() -> KernelSku {
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Sinkhorn-Knopp + stream-mix are deterministic per launch;
            // upstream kernels avoid atomicAdd on the FW path. Two
            // back-to-back launches at the same shape produce bit-equal
            // outputs on the same hardware.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::HyperConnection as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::Bf16), // rmsnorm_weight dtype
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        }
    }

    #[cfg(feature = "mhc")]
    fn create_handle(desc: &HyperConnectionDescriptor, training: bool) -> Result<*mut c_void> {
        // Pre-flight C-side validation (mirrors create's range
        // checks). Lets us return InvalidArg / Unsupported without
        // attempting an alloc that would just fail.
        let probe = unsafe {
            baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_can_implement(
                desc.batch,
                desc.hidden_dim,
                desc.n_streams,
            )
        };
        super::map_status(probe)?;

        let create = if training {
            baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_create_training
        } else {
            baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_create
        };
        // SAFETY: plain value arguments; the function returns an owned handle or null.
        let handle = unsafe {
            create(
                desc.batch,
                desc.hidden_dim,
                desc.n_streams,
                desc.sinkhorn_iters,
                desc.eps,
            )
        };
        if handle.is_null() {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: native MHCLayer init failed",
            ));
        }
        Ok(handle)
    }

    /// f32 scratch and cast plans for `T = f16 / bf16`; `None` for `f32`.
    #[cfg(feature = "mhc")]
    fn build_half(
        stream: &Stream,
        desc: &HyperConnectionDescriptor,
        training: bool,
    ) -> Result<Option<Mutex<HalfPath<T>>>> {
        if !Self::is_half() {
            return Ok(None);
        }
        let e = Self::stream_elems(desc) as usize;
        Ok(Some(Mutex::new(HalfPath::new(stream, desc, e, training)?)))
    }

    fn select_impl(
        stream: &Stream,
        desc: &HyperConnectionDescriptor,
        _pref: PlanPreference,
        training: bool,
    ) -> Result<Self> {
        Self::validate_descriptor(desc, training)?;

        let sku = Self::build_sku();

        #[cfg(feature = "mhc")]
        {
            // Scratch first: a failure here must not leak the native handle.
            let half = Self::build_half(stream, desc, training)?;
            let handle = Self::create_handle(desc, training)?;
            Ok(Self {
                desc: *desc,
                sku,
                handle,
                training,
                half,
                _marker: PhantomData,
            })
        }
        #[cfg(not(feature = "mhc"))]
        {
            let _ = (sku, stream); // silence unused warnings when the feature is off
            Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: build with the `mhc` cargo feature",
            ))
        }
    }

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &HyperConnectionArgs<'_, T>) -> Result<()> {
        crate::view_check::check_fits(&args.x_expanded)?;
        crate::view_check::check_fits(&args.rmsnorm_weight)?;
        crate::view_check::check_fits(&args.h_pre)?;
        crate::view_check::check_fits(&args.h_post)?;
        crate::view_check::check_fits(&args.h_res)?;
        crate::view_check::check_fits(&args.out)?;
        let b = self.desc.batch;
        let n = self.desc.n_streams;
        let c = self.desc.hidden_dim;

        if args.x_expanded.shape() != [b, n, c] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: x_expanded shape mismatch with [B, n, C]",
            ));
        }
        if args.out.shape() != [b, n, c] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: out shape mismatch with [B, n, C]",
            ));
        }
        if args.rmsnorm_weight.shape() != [c] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: rmsnorm_weight shape mismatch with [C]",
            ));
        }
        if args.h_pre.shape() != [n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: h_pre shape mismatch with [n]",
            ));
        }
        if args.h_post.shape() != [n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: h_post shape mismatch with [n]",
            ));
        }
        if args.h_res.shape() != [n, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: h_res shape mismatch with [n, n]",
            ));
        }

        // The upstream kernels assume row-major contiguous layout for
        // x_expanded / out. We reject non-contiguous strides — adding
        // strided support is a kernel rewrite.
        let bnc = (b as i64) * (n as i64) * (c as i64);
        if (args.x_expanded.data().len() as i64) < bnc {
            return Err(Error::BufferTooSmall {
                needed: bnc as usize,
                got: args.x_expanded.data().len(),
            });
        }
        if (args.out.data().len() as i64) < bnc {
            return Err(Error::BufferTooSmall {
                needed: bnc as usize,
                got: args.out.data().len(),
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always zero — internal scratch lives
    /// in the native handle (allocated at `select` time).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel SKU this plan dispatches to.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees — deterministic, bit-stable on the same
    /// hardware (no atomicAdd on the FW path).
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch the kernel against `args`.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        mut args: HyperConnectionArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(feature = "mhc")]
        {
            match &self.half {
                None => {
                    let x = args.x_expanded.data().as_raw().0 as *const c_void;
                    let out = args.out.data().as_raw().0 as *mut c_void;
                    self.native_run(stream, x, out, &args)
                }
                Some(half) => self.run_half(stream, half, &mut args),
            }
        }
        #[cfg(not(feature = "mhc"))]
        {
            let _ = (stream, &mut args);
            Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: build with the `mhc` cargo feature",
            ))
        }
    }

    /// The vendored f32 forward on raw `x` / `out` pointers (`[B, n, C]` f32);
    /// the weights are taken from `args`.
    #[cfg(feature = "mhc")]
    fn native_run(
        &self,
        stream: &Stream,
        x: *const c_void,
        out: *mut c_void,
        args: &HyperConnectionArgs<'_, T>,
    ) -> Result<()> {
        // SAFETY: `x` / `out` are live f32 `[B, n, C]` device buffers (the caller's
        // views, checked by `can_implement`, or this plan's scratch); the weights are
        // checked views; the handle is owned by `self`.
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_run(
                self.handle,
                x,
                args.rmsnorm_weight.data().as_raw().0 as *const c_void,
                args.h_pre.data().as_raw().0 as *const c_void,
                args.h_post.data().as_raw().0 as *const c_void,
                args.h_res.data().as_raw().0 as *const c_void,
                out,
                self.desc.batch,
                self.desc.hidden_dim,
                self.desc.n_streams,
                core::ptr::null_mut(),
                0,
                stream.as_raw(),
            )
        };
        super::map_status(status)
    }

    /// `T = f16 / bf16`: widen `x_expanded` into f32 scratch, run the f32 path,
    /// narrow the result into `out`.
    #[cfg(feature = "mhc")]
    fn run_half(
        &self,
        stream: &Stream,
        half: &Mutex<HalfPath<T>>,
        args: &mut HyperConnectionArgs<'_, T>,
    ) -> Result<()> {
        let mut guard = lock_scratch(half)?;
        let h = &mut *guard;
        let e = Self::stream_elems(&self.desc) as usize;
        let numel = e as i32;
        cast_run(
            &h.to_f32,
            stream,
            args.x_expanded.data(),
            h.x32.as_slice_mut(),
            numel,
        )?;
        let x = h.x32.as_slice().as_raw().0 as *const c_void;
        let out = h.out32.as_slice_mut().as_raw().0 as *mut c_void;
        self.native_run(stream, x, out, args)?;
        let narrow_to = args.out.data_mut().slice_mut(0..e);
        cast_run(&h.from_f32, stream, h.out32.as_slice(), narrow_to, numel)
    }
}

impl<T: Element> HyperConnectionPlan<T> {
    fn check_backward_args(&self, args: &HyperConnectionBackwardArgs<'_, T>) -> Result<()> {
        Self::check_backward_fits(args)?;
        let (b, n, c) = (self.desc.batch, self.desc.n_streams, self.desc.hidden_dim);
        let shapes_ok = args.d_out.shape() == [b, n, c]
            && args.d_x_expanded.shape() == [b, n, c]
            && args.d_rmsnorm_weight.shape() == [c]
            && args.d_h_pre.shape() == [n]
            && args.d_h_post.shape() == [n]
            && args.d_h_res.shape() == [n, n];
        if !shapes_ok {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: backward tensor shapes disagree with [B, n, C] / [C] / [n] / [n, n]",
            ));
        }
        Ok(())
    }

    fn check_backward_fits(args: &HyperConnectionBackwardArgs<'_, T>) -> Result<()> {
        crate::view_check::check_fits(&args.d_out)?;
        crate::view_check::check_fits(&args.d_x_expanded)?;
        crate::view_check::check_fits(&args.d_rmsnorm_weight)?;
        crate::view_check::check_fits(&args.d_h_pre)?;
        crate::view_check::check_fits(&args.d_h_post)?;
        crate::view_check::check_fits(&args.d_h_res)
    }

    /// Static-H backward (Tier 2). The plan must come from
    /// [`Self::select_training`], and this must follow a [`Self::run`] on the SAME
    /// plan: the activations are read from the plan's internal buffers, so
    /// another `run` in between (or concurrent use) changes the result.
    pub fn run_backward(
        &self,
        stream: &Stream,
        mut args: HyperConnectionBackwardArgs<'_, T>,
    ) -> Result<()> {
        if !self.training {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: run_backward needs select_training",
            ));
        }
        self.check_backward_args(&args)?;
        #[cfg(feature = "mhc")]
        {
            match &self.half {
                None => {
                    let ptrs = BackwardPtrs {
                        d_out: args.d_out.data().as_raw().0 as *const c_void,
                        d_x: args.d_x_expanded.data().as_raw().0 as *mut c_void,
                        d_gamma: args.d_rmsnorm_weight.data().as_raw().0 as *mut c_void,
                        d_h_pre: args.d_h_pre.data().as_raw().0 as *mut c_void,
                        d_h_post: args.d_h_post.data().as_raw().0 as *mut c_void,
                        d_h_res: args.d_h_res.data().as_raw().0 as *mut c_void,
                    };
                    self.native_backward(stream, &ptrs)
                }
                Some(half) => self.run_backward_half(stream, half, &mut args),
            }
        }
        #[cfg(not(feature = "mhc"))]
        {
            let _ = (stream, &mut args);
            Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: build with the `mhc` cargo feature",
            ))
        }
    }

    /// The vendored f32 backward on raw pointers (all f32).
    #[cfg(feature = "mhc")]
    fn native_backward(&self, stream: &Stream, p: &BackwardPtrs) -> Result<()> {
        let (b, n, c) = (self.desc.batch, self.desc.n_streams, self.desc.hidden_dim);
        // SAFETY: every pointer is a live f32 device allocation of the checked shape and
        // length (the caller's views, or this plan's scratch); the handle is owned by `self`.
        let status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_backward(
                self.handle,
                p.d_out,
                p.d_x,
                p.d_gamma,
                p.d_h_pre,
                p.d_h_post,
                p.d_h_res,
                b,
                c,
                n,
                stream.as_raw(),
            )
        };
        super::map_status(status)
    }

    /// `T = f16 / bf16` backward: widen `d_out`, run the f32 backward into f32
    /// scratch, narrow each gradient into the caller's tensors.
    #[cfg(feature = "mhc")]
    fn run_backward_half(
        &self,
        stream: &Stream,
        half: &Mutex<HalfPath<T>>,
        args: &mut HyperConnectionBackwardArgs<'_, T>,
    ) -> Result<()> {
        let mut guard = lock_scratch(half)?;
        let HalfPath {
            to_f32,
            from_f32,
            backward,
            ..
        } = &mut *guard;
        let hb = backward.as_mut().ok_or(Error::Unsupported(
            "baracuda-kernels::HyperConnectionPlan: half backward scratch missing",
        ))?;
        let e = Self::stream_elems(&self.desc) as usize;
        let (c, n) = (self.desc.hidden_dim, self.desc.n_streams);
        cast_run(
            to_f32,
            stream,
            args.d_out.data(),
            hb.d_out32.as_slice_mut(),
            e as i32,
        )?;
        let ptrs = hb.ptrs();
        self.native_backward(stream, &ptrs)?;
        let dx = args.d_x_expanded.data_mut().slice_mut(0..e);
        cast_run(from_f32, stream, hb.d_x32.as_slice(), dx, e as i32)?;
        hb.narrow_small(stream, c, n, args)
    }
}

/// Raw device pointers for one native backward call (all f32).
#[cfg(feature = "mhc")]
struct BackwardPtrs {
    d_out: *const c_void,
    d_x: *mut c_void,
    d_gamma: *mut c_void,
    d_h_pre: *mut c_void,
    d_h_post: *mut c_void,
    d_h_res: *mut c_void,
}

impl<T: Element> Drop for HyperConnectionPlan<T> {
    fn drop(&mut self) {
        #[cfg(feature = "mhc")]
        {
            if !self.handle.is_null() {
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_destroy(
                        self.handle,
                    );
                }
                self.handle = core::ptr::null_mut();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Device-free checks: element-type admission and the scratch-size arithmetic.
    use super::*;
    use half::{bf16, f16};

    fn desc(element: ElementKind, batch: i32, n: i32, c: i32) -> HyperConnectionDescriptor {
        HyperConnectionDescriptor {
            batch,
            hidden_dim: c,
            n_streams: n,
            sinkhorn_iters: 20,
            eps: 1e-5,
            element,
        }
    }

    fn admits<T: Element>(d: &HyperConnectionDescriptor, training: bool) -> bool {
        HyperConnectionPlan::<T>::validate_descriptor(d, training).is_ok()
    }

    #[test]
    fn f32_f16_bf16_are_admitted() {
        assert!(admits::<f32>(&desc(ElementKind::F32, 2, 4, 64), false));
        assert!(admits::<f16>(&desc(ElementKind::F16, 2, 4, 64), false));
        assert!(admits::<bf16>(&desc(ElementKind::Bf16, 2, 4, 64), false));
        assert!(admits::<f16>(&desc(ElementKind::F16, 2, 4, 64), true));
        assert!(admits::<bf16>(&desc(ElementKind::Bf16, 2, 4, 64), true));
    }

    #[test]
    fn other_element_types_and_mismatches_are_rejected() {
        // Control for the admissions above: the same shape with an unwired T fails.
        assert!(!admits::<f64>(&desc(ElementKind::F64, 2, 4, 64), false));
        // descriptor.element must equal T.
        assert!(!admits::<f16>(&desc(ElementKind::Bf16, 2, 4, 64), false));
        assert!(!admits::<f32>(&desc(ElementKind::F16, 2, 4, 64), false));
    }

    #[test]
    fn half_stream_count_is_capped_at_i32_max() {
        // 2^15 * 4 * 2^15 = 2^32 elements: over the i32 cast count.
        let big = desc(ElementKind::F16, 1 << 15, 4, 1 << 15);
        assert!(!admits::<f16>(&big, false));
        // Control: the f32 path does not go through the i32 casts, and the same
        // shape just under the cap is admitted for f16.
        assert!(admits::<f32>(
            &desc(ElementKind::F32, 1 << 15, 4, 1 << 15),
            false
        ));
        assert!(admits::<f16>(
            &desc(ElementKind::F16, 1 << 10, 4, 1 << 10),
            false
        ));
    }

    #[test]
    fn scratch_bytes_matches_the_documented_formula() {
        let (b, n, c) = (3, 4, 64);
        let e = (b * n * c) as usize;
        // f32 runs in place: no scratch.
        assert_eq!(
            HyperConnectionPlan::<f32>::scratch_bytes(&desc(ElementKind::F32, b, n, c), true),
            0
        );
        for (kind, name) in [(ElementKind::F16, "f16"), (ElementKind::Bf16, "bf16")] {
            let d = desc(kind, b, n, c);
            let fwd = HyperConnectionPlan::<f16>::scratch_bytes(&d, false);
            assert_eq!(fwd, 2 * e * 4, "{name} forward scratch");
            let (cc, nn) = (c as usize, n as usize);
            let train = HyperConnectionPlan::<f16>::scratch_bytes(&d, true);
            assert_eq!(
                train,
                (4 * e + cc + 2 * nn + nn * nn) * 4,
                "{name} training scratch"
            );
            assert!(train > fwd);
        }
    }

    #[test]
    fn scratch_bytes_known_value() {
        // B=2, n=4, C=8: E = 64. Forward 2*64 f32 = 512 B. Training adds
        // 2*64 + 8 + 2*4 + 16 = 160 f32 = 640 B.
        let d = desc(ElementKind::Bf16, 2, 4, 8);
        assert_eq!(HyperConnectionPlan::<bf16>::scratch_bytes(&d, false), 512);
        assert_eq!(
            HyperConnectionPlan::<bf16>::scratch_bytes(&d, true),
            512 + 640
        );
    }
}
