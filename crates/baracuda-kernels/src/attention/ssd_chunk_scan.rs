//! Mamba-2 SSD chunk-scan — Phase 50 (gated behind the `mamba` feature).
//!
//! State-Space Duality (SSD) reformulates Mamba's selective SSM
//! recurrence onto a chunk-scan + GEMM decomposition. This makes the
//! Mamba-2 family (Mamba-2 8B, Codestral-Mamba, Falcon-Mamba, Zamba2)
//! servable by baracuda with roughly half the kernel surface of
//! Mamba-1's `selective_scan` — most of the chunk-state matmuls reuse
//! baracuda's existing CUTLASS / cuBLAS GEMM.
//!
//! ## Algorithm summary
//!
//! Mamba-2's selective SSM with scalar `A` per head:
//!
//! ```text
//!     decay[t]    = exp(dt[t] · A_h)                         // scalar
//!     h[t, d, n]  = decay[t] · h[t-1, d, n]
//!                   + dt[t] · B[t, n] · x[t, d]              // outer product
//!     y[t, d]     = sum_n C[t, n] · h[t, d, n]
//! ```
//!
//! ## Shape contract (rank-4 / rank-3, contiguous, row-major)
//!
//! | tensor | shape                                  |
//! |--------|----------------------------------------|
//! | `x`    | `[B, L, H, D]`                         |
//! | `dt`   | `[B, L, H]`                            |
//! | `A`    | `[H]`                                  |
//! | `B`    | `[B, L, H, N]`                         |
//! | `C`    | `[B, L, H, N]`                         |
//! | `y`    | `[B, L, H, D]`                         |
//!
//! Where:
//! - `B` = batch
//! - `L` = sequence length
//! - `H` = SSM heads
//! - `D` = head dim (typically 64 or 128)
//! - `N` = state dim (typically 64 or 128)
//!
//! ## Trailblazer scope
//!
//! - **Dtypes**: `f32`, `f16`, `bf16` (matches upstream — f64 not
//!   provided by `mamba_ssm` Triton reference).
//! - **State residency**: `head_dim * state_dim` floats must fit in
//!   the bespoke kernel's SMEM. FW caps `D, N ≤ 256` (rejected by
//!   `select` otherwise); BW caps `D, N ≤ 64` (tighter SMEM budget
//!   because BW keeps both `dh` and recorded states accessible).
//!   Larger problems are a follow-up that pushes state to gmem with
//!   `cp.async` double buffering.
//! - **Layout**: contiguous only (no strided / no varlen / no paged
//!   state).
//! - **Chunk-aware perf**: the FFI accepts `chunk_size` but the
//!   trailblazer kernel ignores it (sequential per-(b, h)
//!   recurrence). The chunk-scan decomposition is a perf optimization
//!   that produces bit-identical outputs; a future phase will route
//!   shaped problems through baracuda's GEMM stack for the intra/
//!   inter chunk matmul portions.
//!
//! ## Numerical guarantees
//!
//! - FW: deterministic, bit-stable on same hardware. Each output cell
//!   `y[t, d]` is written by exactly one block; no atomicAdd.
//! - BW: dx / dB / dC / ddt are deterministic (written by exactly one
//!   block per cell). `dA` is accumulated across `b` blocks via
//!   `atomicAdd` — order-dependent across batch samples. f16 / bf16
//!   atomicAdd is provided natively on sm_80+.
//!
//! ## Workspace
//!
//! - FW: zero.
//! - BW: `B * H * L * D * N` f32 cells (recorded `h[t]` states from
//!   pass 1, consumed by pass 2). Query via
//!   [`SsdChunkScanBackwardPlan::workspace_size`] before launch.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a Mamba-2 SSD chunk-scan FW op.
#[derive(Copy, Clone, Debug)]
pub struct SsdChunkScanDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Sequence length (`L`).
    pub seq_len: i32,
    /// Number of SSM heads (`H`).
    pub num_heads: i32,
    /// Per-head dim (`D`, typically 64 or 128).
    pub head_dim: i32,
    /// State dim (`N`, typically 64 or 128).
    pub state_dim: i32,
    /// Chunk size for the chunk-scan decomposition (typically 64 or
    /// 128). The trailblazer kernel ignores this and runs the
    /// sequential recurrence; a future perf kernel will use it.
    pub chunk_size: i32,
    /// Element dtype — must match the plan's type parameter. One of
    /// `F32`, `F16`, `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a Mamba-2 SSD chunk-scan FW launch.
pub struct SsdChunkScanArgs<'a, T: Element> {
    /// SSM input projections — shape `[B, L, H, D]`, contiguous.
    pub x: TensorRef<'a, T, 4>,
    /// Per-time, per-head step sizes — shape `[B, L, H]`, contiguous.
    pub dt: TensorRef<'a, T, 3>,
    /// Per-head SSM eigenvalue — shape `[H]`, contiguous. Typically
    /// negative (so `decay = exp(dt·A) ∈ (0, 1]`).
    pub a: TensorRef<'a, T, 1>,
    /// Input-side state projection — shape `[B, L, H, N]`, contiguous.
    pub b: TensorRef<'a, T, 4>,
    /// Output-side state readout — shape `[B, L, H, N]`, contiguous.
    pub c: TensorRef<'a, T, 4>,
    /// Output — shape `[B, L, H, D]`, contiguous.
    pub y: TensorMut<'a, T, 4>,
}

/// Mamba-2 SSD chunk-scan forward plan.
///
/// See module docs for the shape contract and algorithm summary.
pub struct SsdChunkScanPlan<T: Element> {
    desc: SsdChunkScanDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SsdChunkScanPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SsdChunkScanDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0
            || desc.seq_len < 0
            || desc.num_heads < 0
            || desc.head_dim < 0
            || desc.state_dim < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SsdChunkScanPlan: extents must be non-negative",
            ));
        }
        if desc.chunk_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SsdChunkScanPlan: chunk_size must be positive",
            ));
        }
        if desc.head_dim > 256 || desc.state_dim > 256 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanPlan: head_dim and state_dim must be <= 256 in the trailblazer",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanPlan: wired today: `{f32, f16, bf16}`",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::SsdChunkScan as u16,
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
    pub fn can_implement(&self, args: &SsdChunkScanArgs<'_, T>) -> Result<()> {
        let shape_x = [
            self.desc.batch_size,
            self.desc.seq_len,
            self.desc.num_heads,
            self.desc.head_dim,
        ];
        let shape_dt = [self.desc.batch_size, self.desc.seq_len, self.desc.num_heads];
        let shape_a = [self.desc.num_heads];
        let shape_bn = [
            self.desc.batch_size,
            self.desc.seq_len,
            self.desc.num_heads,
            self.desc.state_dim,
        ];
        if args.x.shape != shape_x || args.y.shape != shape_x {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SsdChunkScanPlan: x / y shape must be [B, L, H, D]",
            ));
        }
        if args.dt.shape != shape_dt {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SsdChunkScanPlan: dt shape must be [B, L, H]",
            ));
        }
        if args.a.shape != shape_a {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SsdChunkScanPlan: A shape must be [H]",
            ));
        }
        if args.b.shape != shape_bn || args.c.shape != shape_bn {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SsdChunkScanPlan: B / C shape must be [B, L, H, N]",
            ));
        }
        if !args.x.is_contiguous()
            || !args.dt.is_contiguous()
            || !args.a.is_contiguous()
            || !args.b.is_contiguous()
            || !args.c.is_contiguous()
            || !args.y.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanPlan: trailblazer requires contiguous tensors",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes — zero for FW.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
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

    /// Launch the FW kernel on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: SsdChunkScanArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch_size == 0
            || self.desc.seq_len == 0
            || self.desc.num_heads == 0
        {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr  = args.x.data.as_raw().0 as *const c_void;
        let dt_ptr = args.dt.data.as_raw().0 as *const c_void;
        let a_ptr  = args.a.data.as_raw().0 as *const c_void;
        let b_ptr  = args.b.data.as_raw().0 as *const c_void;
        let c_ptr  = args.c.data.as_raw().0 as *const c_void;
        let y_ptr  = args.y.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ssd_chunk_scan_f32_run(
                    self.desc.batch_size, self.desc.seq_len, self.desc.num_heads,
                    self.desc.head_dim, self.desc.state_dim, self.desc.chunk_size,
                    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ssd_chunk_scan_f16_run(
                    self.desc.batch_size, self.desc.seq_len, self.desc.num_heads,
                    self.desc.head_dim, self.desc.state_dim, self.desc.chunk_size,
                    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ssd_chunk_scan_bf16_run(
                    self.desc.batch_size, self.desc.seq_len, self.desc.num_heads,
                    self.desc.head_dim, self.desc.state_dim, self.desc.chunk_size,
                    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, y_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanPlan: dtype not wired",
            )),
        };
        map_status(status)
    }
}

// ============================================================================
// BACKWARD
// ============================================================================

/// Descriptor for a Mamba-2 SSD chunk-scan BW op.
///
/// Same shape parameters as the FW descriptor.
#[derive(Copy, Clone, Debug)]
pub struct SsdChunkScanBackwardDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Sequence length (`L`).
    pub seq_len: i32,
    /// Number of SSM heads (`H`).
    pub num_heads: i32,
    /// Per-head dim (`D`).
    pub head_dim: i32,
    /// State dim (`N`).
    pub state_dim: i32,
    /// Chunk size (forwarded to the FFI; ignored by the trailblazer
    /// kernel — see [`SsdChunkScanDescriptor::chunk_size`]).
    pub chunk_size: i32,
    /// Element dtype.
    pub element: ElementKind,
}

/// Args bundle for a Mamba-2 SSD chunk-scan BW launch.
///
/// All gradient outputs (`dx`, `dB`, `dC`, `ddt`, `dA`) must be
/// **zero-initialized by the caller** before launch — `dA` is
/// atomic-accumulated across batch samples within the kernel.
pub struct SsdChunkScanBackwardArgs<'a, T: Element> {
    /// SSM input projections (saved) — shape `[B, L, H, D]`.
    pub x: TensorRef<'a, T, 4>,
    /// Per-time step sizes (saved) — shape `[B, L, H]`.
    pub dt: TensorRef<'a, T, 3>,
    /// Per-head A (saved) — shape `[H]`.
    pub a: TensorRef<'a, T, 1>,
    /// B projection (saved) — shape `[B, L, H, N]`.
    pub b: TensorRef<'a, T, 4>,
    /// C projection (saved) — shape `[B, L, H, N]`.
    pub c: TensorRef<'a, T, 4>,
    /// Output gradient — shape `[B, L, H, D]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Output: `dx` — shape `[B, L, H, D]`.
    pub dx: TensorMut<'a, T, 4>,
    /// Output: `dB` — shape `[B, L, H, N]`.
    pub d_b: TensorMut<'a, T, 4>,
    /// Output: `dC` — shape `[B, L, H, N]`.
    pub d_c: TensorMut<'a, T, 4>,
    /// Output: `ddt` — shape `[B, L, H]`.
    pub d_dt: TensorMut<'a, T, 3>,
    /// Output: `dA` — shape `[H]`. **Must be zero-initialized**;
    /// kernel uses `atomicAdd`.
    pub d_a: TensorMut<'a, T, 1>,
}

/// Mamba-2 SSD chunk-scan backward plan.
pub struct SsdChunkScanBackwardPlan<T: Element> {
    desc: SsdChunkScanBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SsdChunkScanBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SsdChunkScanBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanBackwardPlan: descriptor element != T",
            ));
        }
        if desc.head_dim > 64 || desc.state_dim > 64 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanBackwardPlan: BW caps head_dim/state_dim at 64 (SMEM budget)",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanBackwardPlan: wired today: `{f32, f16, bf16}`",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // dA uses atomicAdd → not deterministic at f16/bf16 (FP
            // atomicAdd is order-dependent).
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::SsdChunkScan as u16,
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

    /// Workspace size in bytes — `B * H * L * D * N * 4` (f32 recorded
    /// states from BW pass 1, consumed by pass 2).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_ssd_chunk_scan_workspace_bytes(
                self.desc.batch_size, self.desc.seq_len, self.desc.num_heads,
                self.desc.head_dim, self.desc.state_dim,
                self.desc.chunk_size, 0,
            )
        }
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

    /// Launch the BW kernel pair on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: SsdChunkScanBackwardArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.batch_size == 0
            || self.desc.seq_len == 0
            || self.desc.num_heads == 0
        {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::Borrowed(buf) => (
                buf.as_raw().0 as *mut c_void,
                buf.len(),
            ),
            Workspace::None => (core::ptr::null_mut(), 0usize),
        };
        if ws_bytes < self.workspace_size() {
            return Err(Error::WorkspaceTooSmall {
                needed: self.workspace_size(),
                got: ws_bytes,
            });
        }

        let x_ptr  = args.x.data.as_raw().0 as *const c_void;
        let dt_ptr = args.dt.data.as_raw().0 as *const c_void;
        let a_ptr  = args.a.data.as_raw().0 as *const c_void;
        let b_ptr  = args.b.data.as_raw().0 as *const c_void;
        let c_ptr  = args.c.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let db_ptr = args.d_b.data.as_raw().0 as *mut c_void;
        let dc_ptr = args.d_c.data.as_raw().0 as *mut c_void;
        let ddt_ptr = args.d_dt.data.as_raw().0 as *mut c_void;
        let da_ptr = args.d_a.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ssd_chunk_scan_f32_backward_run(
                    self.desc.batch_size, self.desc.seq_len, self.desc.num_heads,
                    self.desc.head_dim, self.desc.state_dim, self.desc.chunk_size,
                    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, dy_ptr,
                    dx_ptr, db_ptr, dc_ptr, ddt_ptr, da_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ssd_chunk_scan_f16_backward_run(
                    self.desc.batch_size, self.desc.seq_len, self.desc.num_heads,
                    self.desc.head_dim, self.desc.state_dim, self.desc.chunk_size,
                    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, dy_ptr,
                    dx_ptr, db_ptr, dc_ptr, ddt_ptr, da_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ssd_chunk_scan_bf16_backward_run(
                    self.desc.batch_size, self.desc.seq_len, self.desc.num_heads,
                    self.desc.head_dim, self.desc.state_dim, self.desc.chunk_size,
                    x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, dy_ptr,
                    dx_ptr, db_ptr, dc_ptr, ddt_ptr, da_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "baracuda-kernels::SsdChunkScanBackwardPlan: dtype not wired",
            )),
        };
        map_status(status)
    }
}
