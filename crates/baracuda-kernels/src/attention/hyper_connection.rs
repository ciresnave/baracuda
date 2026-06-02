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
//! - **Static-H FW only**. Dynamic-H FW and the BW pass live behind
//!   the same vendored `MHCLayer` class and ship in Tier 2.
//! - **bf16 weights / f32 activations**. The upstream `floatX` is
//!   hardcoded to `nv_bfloat16`; f16 / f32 paths require additional
//!   convert kernels in the launcher and ship in Tier 3.
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

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

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
    /// Element type for `x_expanded` / `out` (f32 in Tier 1).
    pub element: ElementKind,
}

/// Args bundle for a `HyperConnectionPlan` launch.
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
/// **Dtypes**: `f32` only in Tier 1. The `rmsnorm_weight` is always
/// `bf16` regardless of `T`.
///
/// **State**: this plan owns a native `MHCLayer*` handle with
/// ~`B*n*C*sizeof(float)` bytes of GPU scratch. Reuse across many
/// `run()` calls; construction is heavy.
pub struct HyperConnectionPlan<T: Element> {
    desc: HyperConnectionDescriptor,
    sku: KernelSku,
    #[cfg(feature = "mhc")]
    handle: *mut c_void,
    _marker: PhantomData<T>,
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
        _stream: &Stream,
        desc: &HyperConnectionDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: descriptor element != T",
            ));
        }
        // Tier 1: f32 only.
        if !matches!(T::KIND, ElementKind::F32) {
            return Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: Tier 1 wired today: `{f32}` only \
                 (f16 / bf16 deferred to Tier 3)",
            ));
        }
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
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::HyperConnection as u16,
            element: T::KIND,
            aux_element: Some(ElementKind::Bf16), // rmsnorm_weight dtype
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };

        #[cfg(feature = "mhc")]
        {
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

            let handle = unsafe {
                baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_create(
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
            Ok(Self {
                desc: *desc,
                sku,
                handle,
                _marker: PhantomData,
            })
        }
        #[cfg(not(feature = "mhc"))]
        {
            let _ = sku; // silence unused warning when feature is off
            Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: build with the `mhc` cargo feature",
            ))
        }
    }

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &HyperConnectionArgs<'_, T>) -> Result<()> {
        let b = self.desc.batch;
        let n = self.desc.n_streams;
        let c = self.desc.hidden_dim;

        if args.x_expanded.shape != [b, n, c] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: x_expanded shape mismatch with [B, n, C]",
            ));
        }
        if args.out.shape != [b, n, c] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: out shape mismatch with [B, n, C]",
            ));
        }
        if args.rmsnorm_weight.shape != [c] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: rmsnorm_weight shape mismatch with [C]",
            ));
        }
        if args.h_pre.shape != [n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: h_pre shape mismatch with [n]",
            ));
        }
        if args.h_post.shape != [n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: h_post shape mismatch with [n]",
            ));
        }
        if args.h_res.shape != [n, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::HyperConnectionPlan: h_res shape mismatch with [n, n]",
            ));
        }

        // The upstream kernels assume row-major contiguous layout for
        // x_expanded / out. We reject non-contiguous strides — adding
        // strided support is a kernel rewrite.
        let bnc = (b as i64) * (n as i64) * (c as i64);
        if (args.x_expanded.data.len() as i64) < bnc {
            return Err(Error::BufferTooSmall {
                needed: bnc as usize,
                got: args.x_expanded.data.len(),
            });
        }
        if (args.out.data.len() as i64) < bnc {
            return Err(Error::BufferTooSmall {
                needed: bnc as usize,
                got: args.out.data.len(),
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
        args: HyperConnectionArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(feature = "mhc")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let status = unsafe {
                baracuda_kernels_sys::baracuda_kernels_mhc_layer_static_bf16_run(
                    self.handle,
                    args.x_expanded.data.as_raw().0 as *const c_void,
                    args.rmsnorm_weight.data.as_raw().0 as *const c_void,
                    args.h_pre.data.as_raw().0 as *const c_void,
                    args.h_post.data.as_raw().0 as *const c_void,
                    args.h_res.data.as_raw().0 as *const c_void,
                    args.out.data.as_raw().0 as *mut c_void,
                    self.desc.batch,
                    self.desc.hidden_dim,
                    self.desc.n_streams,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            };
            super::map_status(status)
        }
        #[cfg(not(feature = "mhc"))]
        {
            let _ = stream;
            Err(Error::Unsupported(
                "baracuda-kernels::HyperConnectionPlan: build with the `mhc` cargo feature",
            ))
        }
    }
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
