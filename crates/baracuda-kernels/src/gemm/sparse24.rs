//! 2:4 Structured Sparsity GEMM forward plan (Phase 54).
//!
//! Clean-room hand-port of facebookresearch/xFormers `sparse24/`
//! algorithmic reference (BSD-3-Clause). See
//! [`crates/baracuda-kernels-sys/vendor/xformers/VENDOR.md`] for the
//! attribution + cherry-pick scope documentation.
//!
//! ## 2:4 pattern
//!
//! In every 4 consecutive weight cells, at most 2 are non-zero.
//! Compressed weight format:
//!
//! | tensor | shape | dtype |
//! |--------|-------|-------|
//! | `W_compressed` | `[M, K/2]` | `T` (the GEMM dtype) |
//! | `W_metadata`   | `[M, K/8]` | `u16` (2 4-groups per uint16; one byte per 4-group) |
//!
//! The dense weight reconstruction is:
//!
//! ```text
//! for m in 0..M:
//!     for k_group in 0..K/4:
//!         meta_byte = (k_group & 1 == 0) ? metadata[m, k_group/2] & 0xFF
//!                                        : (metadata[m, k_group/2] >> 8) & 0xFF
//!         pos0 = meta_byte & 0x3
//!         pos1 = (meta_byte >> 2) & 0x3
//!         w_dense[m, k_group*4 + pos0] = compressed[m, k_group*2 + 0]
//!         w_dense[m, k_group*4 + pos1] = compressed[m, k_group*2 + 1]
//!         (other 2 positions in the 4-group are 0)
//! ```
//!
//! Output:
//!
//! ```text
//! Y[N, M] = X[N, K] @ W_dense^T
//! ```
//!
//! (Following PyTorch/xFormers convention — weight is `[out_features,
//! in_features]`; `X @ W^T` is the canonical Linear-layer dispatch.)
//!
//! ## Tier-1 implementation strategy
//!
//! **Inflate-then-dense-matmul**: launch an inflation kernel that
//! reconstructs `W_dense` in a caller-supplied workspace buffer
//! (`M * K * sizeof(T)` bytes), then run a reference dense GEMM. This
//! is **correctness first**; the sparse-tensor-core (`mma.sp.sync.aligned`)
//! hardware speedup is deferred to Tier 2 alongside cuSPARSELt
//! integration.
//!
//! The Tier-1 path is **not faster than dense cuBLAS** — it's slower
//! at most shapes because the reference matmul is a naive triple-loop
//! kernel (no tensor cores). The API + compression format are the
//! Phase 54 deliverable; performance lands with the Tier-2
//! cuSPARSELt-or-PTX backend.
//!
//! ## Constraints
//!
//! - `K` must be a multiple of 8.
//! - Wired dtypes: `{f32, f16, bf16}`.
//!
//! ## Workspace
//!
//! `M * K * sizeof(T)` bytes for the inflated dense W. Query via
//! [`GemmSparse24Plan::workspace_size`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a 2:4 sparse GEMM forward op.
#[derive(Copy, Clone, Debug)]
pub struct GemmSparse24Descriptor {
    /// Batch / sequence dimension (`N` rows of the X / Y tensors).
    pub n: i32,
    /// Output feature dimension (`M` — number of W rows; cols of Y).
    pub m: i32,
    /// Input feature dimension (`K` — cols of X; cols of W). Must be
    /// a multiple of 8.
    pub k: i32,
    /// Element type — must match the plan's type parameter.
    pub element: ElementKind,
}

/// Args bundle for a 2:4 sparse GEMM launch.
pub struct GemmSparse24Args<'a, T: Element> {
    /// Compressed weight — `[M, K/2]`, row-major contiguous.
    pub w_compressed: TensorRef<'a, T, 2>,
    /// Metadata — `[M, K/8]` `u16`, row-major contiguous. Each
    /// `u16` carries 2 4-groups (one per byte; low byte first).
    /// Per-byte encoding: bits `[0:1]` = pos0, bits `[2:3]` = pos1.
    pub w_metadata: TensorRef<'a, u16, 2>,
    /// Activation — `[N, K]`, row-major contiguous.
    pub x: TensorRef<'a, T, 2>,
    /// Output — `[N, M]`, row-major contiguous.
    pub y: TensorMut<'a, T, 2>,
}

/// 2:4 structured-sparsity GEMM forward plan.
///
/// **When to use**: post-pruning inference where weights have been
/// offline-compressed to the 2:4 format (e.g. via xFormers'
/// `sparse24.compress` utility or NVIDIA's `apex.contrib.sparsity`).
///
/// **Dtypes**: `f32`, `f16`, `bf16`.
///
/// **Workspace**: `M * K * sizeof(T)` bytes — required for the inflated
/// dense `W` reconstruction.
///
/// **Tier-1 caveat**: the trailblazer inflates-then-dense-matmuls;
/// performance is NOT competitive with cuBLAS at this stage. The
/// API + compression format are what Phase 54 delivers; the
/// sparse-tensor-core speedup lands in Tier 2.
pub struct GemmSparse24Plan<T: Element> {
    desc: GemmSparse24Descriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> GemmSparse24Plan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &GemmSparse24Descriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GemmSparse24Plan: descriptor element != T",
            ));
        }
        if desc.n < 0 || desc.m < 0 || desc.k < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemmSparse24Plan: N, M, K must be non-negative",
            ));
        }
        if (desc.k & 7) != 0 {
            return Err(Error::Unsupported(
                "baracuda-kernels::GemmSparse24Plan: K must be a multiple of 8",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::GemmSparse24Plan: wired today: `{f32, f16, bf16}`",
            ));
        }

        // Pre-flight C-side validation.
        #[cfg(feature = "xformers_sparse24")]
        {
            let probe = unsafe {
                match T::KIND {
                    ElementKind::F32 =>
                        baracuda_kernels_sys::baracuda_kernels_gemm_f32_sparse24_gemm_can_implement(
                            desc.n, desc.m, desc.k,
                        ),
                    ElementKind::F16 =>
                        baracuda_kernels_sys::baracuda_kernels_gemm_f16_sparse24_gemm_can_implement(
                            desc.n, desc.m, desc.k,
                        ),
                    ElementKind::Bf16 =>
                        baracuda_kernels_sys::baracuda_kernels_gemm_bf16_sparse24_gemm_can_implement(
                            desc.n, desc.m, desc.k,
                        ),
                    _ => 3,
                }
            };
            super::super::attention::map_status_pub(probe)?;
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Reference GEMM is deterministic per-cell (no atomicAdd).
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Gemm,
            op: 0,
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

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &GemmSparse24Args<'_, T>) -> Result<()> {
        if args.w_compressed.shape != [self.desc.m, self.desc.k / 2] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemmSparse24Plan: w_compressed shape must be [M, K/2]",
            ));
        }
        if args.w_metadata.shape != [self.desc.m, self.desc.k / 8] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemmSparse24Plan: w_metadata shape must be [M, K/8]",
            ));
        }
        if args.x.shape != [self.desc.n, self.desc.k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemmSparse24Plan: x shape must be [N, K]",
            ));
        }
        if args.y.shape != [self.desc.n, self.desc.m] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GemmSparse24Plan: y shape must be [N, M]",
            ));
        }
        if !args.w_compressed.is_contiguous()
            || !args.w_metadata.is_contiguous()
            || !args.x.is_contiguous()
            || !args.y.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::GemmSparse24Plan: all tensors must be contiguous in Tier 1",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes — `M * K * sizeof(T)` for the inflated
    /// dense W.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        (self.desc.m as usize)
            * (self.desc.k as usize)
            * core::mem::size_of::<T>()
    }

    /// SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Run the inflate-then-matmul reference path.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: GemmSparse24Args<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.y.numel() == 0 {
            return Ok(());
        }
        #[cfg(feature = "xformers_sparse24")]
        {
            let needed = self.workspace_size();
            let (ws_ptr, ws_bytes) = match workspace {
                Workspace::None => {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: 0,
                    });
                }
                Workspace::Borrowed(bytes) => {
                    let got = bytes.len();
                    if got < needed {
                        return Err(Error::WorkspaceTooSmall {
                            needed,
                            got,
                        });
                    }
                    (bytes.as_raw().0 as *mut c_void, got as u64)
                }
            };
            let stream_ptr = stream.as_raw() as *mut c_void;
            let x_ptr = args.x.data.as_raw().0 as *const c_void;
            let wc_ptr = args.w_compressed.data.as_raw().0 as *const c_void;
            let wm_ptr = args.w_metadata.data.as_raw().0 as *const c_void;
            let y_ptr = args.y.data.as_raw().0 as *mut c_void;
            let status = unsafe {
                match T::KIND {
                    ElementKind::F32 =>
                        baracuda_kernels_sys::baracuda_kernels_gemm_f32_sparse24_gemm_run(
                            self.desc.n, self.desc.m, self.desc.k,
                            x_ptr, wc_ptr, wm_ptr, y_ptr,
                            ws_ptr, ws_bytes, stream_ptr,
                        ),
                    ElementKind::F16 =>
                        baracuda_kernels_sys::baracuda_kernels_gemm_f16_sparse24_gemm_run(
                            self.desc.n, self.desc.m, self.desc.k,
                            x_ptr, wc_ptr, wm_ptr, y_ptr,
                            ws_ptr, ws_bytes, stream_ptr,
                        ),
                    ElementKind::Bf16 =>
                        baracuda_kernels_sys::baracuda_kernels_gemm_bf16_sparse24_gemm_run(
                            self.desc.n, self.desc.m, self.desc.k,
                            x_ptr, wc_ptr, wm_ptr, y_ptr,
                            ws_ptr, ws_bytes, stream_ptr,
                        ),
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::GemmSparse24Plan::run reached an unimplemented dtype",
                    )),
                }
            };
            super::super::attention::map_status_pub(status)
        }
        #[cfg(not(feature = "xformers_sparse24"))]
        {
            let _ = (stream, workspace);
            Err(Error::Unsupported(
                "baracuda-kernels::GemmSparse24Plan: build with the `xformers_sparse24` cargo feature",
            ))
        }
    }
}
