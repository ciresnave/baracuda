//! Mamba-1 selective_scan — Phase 50b (gated behind the `mamba` feature).
//!
//! Completes the state-space LLM coverage that Phase 50 explicitly
//! deferred. Phase 50 shipped Mamba-2's SSD chunk-scan formulation; this
//! module ships the original Mamba-1 selective_scan kernel that powers
//! every Mamba-1-class shipping model (Mamba-7B, Falcon-Mamba,
//! Codestral-Mamba). Hand-port of `state-spaces/mamba`
//! `csrc/selective_scan/` under Apache-2.0; see
//! `crates/baracuda-kernels-sys/vendor/mamba/VENDOR.md` for attribution.
//!
//! ## Algorithm summary
//!
//! Per-channel selective state-space scan with a per-`(d, n)` state
//! matrix `A`, time-varying `B[t, n]` / `C[t, n]` modulation, scalar
//! per-channel `delta[t, d]`, optional skip-connection `D[d] * u[t, d]`,
//! optional SiLU-gated tail `silu(z[t, d])`, optional `delta_bias[d]` +
//! optional `softplus(delta + bias)` mapping:
//!
//! ```text
//!     delta_eff = softplus?(delta[t, d] + delta_bias[d])
//!     dA        = exp(delta_eff · A[d, n])               // per (d, n)
//!     dBu       = delta_eff · B[t, n] · u[t, d]
//!     h[d, n]   = dA · h[d, n] + dBu                     // state update
//!     y[t, d]   = sum_n h[d, n] · C[t, n]
//!     y[t, d]  += D[d] · u[t, d]                         // if D given
//!     y[t, d]  *= silu(z[t, d])                          // if z given
//! ```
//!
//! ## Shape contract (contiguous, row-major)
//!
//! | tensor       | shape          | optional |
//! |--------------|----------------|----------|
//! | `u`          | `[B, L, D]`    |          |
//! | `delta`      | `[B, L, D]`    |          |
//! | `a`          | `[D, N]`       |          |
//! | `b`          | `[B, L, N]`    |          |
//! | `c`          | `[B, L, N]`    |          |
//! | `d`         (skip) | `[D]`     | yes      |
//! | `z`          | `[B, L, D]`    | yes      |
//! | `delta_bias` | `[D]`          | yes      |
//! | `y`          | `[B, L, D]`    |          |
//! | `last_state` | `[B, D, N]`    | yes      |
//!
//! Where:
//! - `B` = batch
//! - `L` = sequence length
//! - `D` = channel dim (upstream "dim")
//! - `N` = state dim (upstream "dstate", typically 16 in Mamba-1)
//!
//! ## Trailblazer scope
//!
//! - **Dtypes**: `f32`, `f16`, `bf16`. Complex variants reserved in
//!   upstream but no shipping LLM uses them — deferred.
//! - **State residency**: `N ≤ 256`. Each block keeps its `[N]` state
//!   vector in SMEM and walks `L` sequentially. One block per `(b, d)`.
//! - **Layout**: contiguous only (no strided / no varlen / no paged
//!   state).
//! - **No variable-length sequences** (`cu_seqlens`) — deferred.
//!
//! ## Numerical guarantees
//!
//! - FW: deterministic, bit-stable on same hardware. Each `y[b, t, d]`
//!   is written by exactly one block; no atomicAdd.
//! - BW: `du`, `dC`, `ddelta`, `dz` are deterministic. `dA`, `dD`,
//!   `ddelta_bias` are accumulated across `(b, t)` via `atomicAdd`;
//!   `dB`, `dC` are accumulated across `d` via `atomicAdd`. f16 / bf16
//!   atomicAdd is provided natively on sm_80+.
//!
//! ## Workspace
//!
//! - FW: zero.
//! - BW: `B * D * L * N * sizeof(T)` (recorded `h[t]` states from BW
//!   pass 1, consumed by pass 2). Query via
//!   [`SelectiveScanBackwardPlan::workspace_size`] before launch.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a Mamba-1 selective_scan FW op.
#[derive(Copy, Clone, Debug)]
pub struct SelectiveScanDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Sequence length (`L`).
    pub seq_len: i32,
    /// Channel dim (`D`, upstream `dim`).
    pub dim: i32,
    /// State dim (`N`, upstream `dstate`, typically 16).
    pub dstate: i32,
    /// Apply softplus to `delta + delta_bias?` before the recurrence
    /// (`true` matches the Mamba-1 reference default).
    pub delta_softplus: bool,
    /// Element dtype — must match the plan's type parameter. One of
    /// `F32`, `F16`, `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a Mamba-1 selective_scan FW launch.
pub struct SelectiveScanArgs<'a, T: Element> {
    /// Input — shape `[B, L, D]`, contiguous.
    pub u: TensorRef<'a, T, 3>,
    /// Time-step — shape `[B, L, D]`, contiguous.
    pub delta: TensorRef<'a, T, 3>,
    /// Per-channel state matrix — shape `[D, N]`, contiguous.
    pub a: TensorRef<'a, T, 2>,
    /// Time-varying input-side projection — shape `[B, L, N]`,
    /// contiguous.
    pub b: TensorRef<'a, T, 3>,
    /// Time-varying output-side projection — shape `[B, L, N]`,
    /// contiguous.
    pub c: TensorRef<'a, T, 3>,
    /// Optional skip-connection vector — shape `[D]`, contiguous.
    pub d_skip: Option<TensorRef<'a, T, 1>>,
    /// Optional SiLU-gated tail — shape `[B, L, D]`, contiguous.
    pub z: Option<TensorRef<'a, T, 3>>,
    /// Optional per-channel `delta` bias — shape `[D]`, contiguous.
    pub delta_bias: Option<TensorRef<'a, T, 1>>,
    /// Output — shape `[B, L, D]`, contiguous.
    pub y: TensorMut<'a, T, 3>,
    /// Optional saved-out last state — shape `[B, D, N]`, contiguous.
    pub last_state: Option<TensorMut<'a, T, 3>>,
}

/// Mamba-1 selective_scan forward plan.
///
/// See module docs for the shape contract and algorithm summary.
pub struct SelectiveScanPlan<T: Element> {
    desc: SelectiveScanDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SelectiveScanPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SelectiveScanDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0 || desc.seq_len < 0 || desc.dim < 0 || desc.dstate < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanPlan: extents must be non-negative",
            ));
        }
        if desc.dstate > 256 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanPlan: dstate must be <= 256 in the trailblazer",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanPlan: wired today: `{f32, f16, bf16}`",
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
            op: AttentionKind::SelectiveScan as u16,
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
    pub fn can_implement(&self, args: &SelectiveScanArgs<'_, T>) -> Result<()> {
        let shape_udy = [self.desc.batch_size, self.desc.seq_len, self.desc.dim];
        let shape_a = [self.desc.dim, self.desc.dstate];
        let shape_bc = [self.desc.batch_size, self.desc.seq_len, self.desc.dstate];
        if args.u.shape != shape_udy
            || args.delta.shape != shape_udy
            || args.y.shape != shape_udy
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanPlan: u/delta/y shape must be [B, L, D]",
            ));
        }
        if args.a.shape != shape_a {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanPlan: A shape must be [D, N]",
            ));
        }
        if args.b.shape != shape_bc || args.c.shape != shape_bc {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanPlan: B / C shape must be [B, L, N]",
            ));
        }
        if let Some(ds) = &args.d_skip {
            if ds.shape != [self.desc.dim] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SelectiveScanPlan: D (skip) shape must be [D]",
                ));
            }
            if !ds.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SelectiveScanPlan: D (skip) must be contiguous",
                ));
            }
        }
        if let Some(z) = &args.z {
            if z.shape != shape_udy {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SelectiveScanPlan: z shape must be [B, L, D]",
                ));
            }
            if !z.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SelectiveScanPlan: z must be contiguous",
                ));
            }
        }
        if let Some(db) = &args.delta_bias {
            if db.shape != [self.desc.dim] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SelectiveScanPlan: delta_bias shape must be [D]",
                ));
            }
            if !db.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SelectiveScanPlan: delta_bias must be contiguous",
                ));
            }
        }
        if let Some(ls) = &args.last_state {
            if ls.shape != [self.desc.batch_size, self.desc.dim, self.desc.dstate] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SelectiveScanPlan: last_state shape must be [B, D, N]",
                ));
            }
            if !ls.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SelectiveScanPlan: last_state must be contiguous",
                ));
            }
        }
        if !args.u.is_contiguous()
            || !args.delta.is_contiguous()
            || !args.a.is_contiguous()
            || !args.b.is_contiguous()
            || !args.c.is_contiguous()
            || !args.y.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanPlan: trailblazer requires contiguous tensors",
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
        mut args: SelectiveScanArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if self.desc.batch_size == 0 || self.desc.seq_len == 0 || self.desc.dim == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let u_ptr = args.u.data.as_raw().0 as *const c_void;
        let delta_ptr = args.delta.data.as_raw().0 as *const c_void;
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let c_ptr = args.c.data.as_raw().0 as *const c_void;
        let d_ptr = args.d_skip.as_ref()
            .map(|d| d.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let z_ptr = args.z.as_ref()
            .map(|z| z.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let db_ptr = args.delta_bias.as_ref()
            .map(|db| db.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let ls_ptr = args.last_state.as_mut()
            .map(|ls| ls.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let dsp = if self.desc.delta_softplus { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_selective_scan_f32_run(
                    self.desc.batch_size, self.desc.seq_len,
                    self.desc.dim, self.desc.dstate, dsp,
                    u_ptr, delta_ptr, a_ptr, b_ptr, c_ptr,
                    d_ptr, z_ptr, db_ptr,
                    y_ptr, ls_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_selective_scan_f16_run(
                    self.desc.batch_size, self.desc.seq_len,
                    self.desc.dim, self.desc.dstate, dsp,
                    u_ptr, delta_ptr, a_ptr, b_ptr, c_ptr,
                    d_ptr, z_ptr, db_ptr,
                    y_ptr, ls_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_selective_scan_bf16_run(
                    self.desc.batch_size, self.desc.seq_len,
                    self.desc.dim, self.desc.dstate, dsp,
                    u_ptr, delta_ptr, a_ptr, b_ptr, c_ptr,
                    d_ptr, z_ptr, db_ptr,
                    y_ptr, ls_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanPlan: dtype not wired",
            )),
        };
        map_status(status)
    }
}

// ============================================================================
// BACKWARD
// ============================================================================

/// Descriptor for a Mamba-1 selective_scan BW op.
///
/// Same shape parameters as the FW descriptor plus the optional-tensor
/// presence flags (caller must declare which optional inputs were given
/// at FW time so the BW kernel can decide whether the matching output
/// gradient should be written + atomic-zeroed).
#[derive(Copy, Clone, Debug)]
pub struct SelectiveScanBackwardDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Sequence length (`L`).
    pub seq_len: i32,
    /// Channel dim (`D`).
    pub dim: i32,
    /// State dim (`N`).
    pub dstate: i32,
    /// Whether FW used softplus mapping on `delta + delta_bias?`.
    pub delta_softplus: bool,
    /// Element dtype.
    pub element: ElementKind,
}

/// Args bundle for a Mamba-1 selective_scan BW launch.
///
/// **Caller-zero contract**: `d_a`, `d_b`, `d_c`, and (when their FW
/// counterparts were given) `d_d`, `d_delta_bias` MUST be zero-
/// initialized before launch — these are accumulated via `atomicAdd`
/// inside the kernel. `du`, `d_delta`, `dz` are deterministic (one
/// writer per cell) and may carry any prior value.
pub struct SelectiveScanBackwardArgs<'a, T: Element> {
    /// Input (saved) — shape `[B, L, D]`.
    pub u: TensorRef<'a, T, 3>,
    /// Time-step (saved) — shape `[B, L, D]`.
    pub delta: TensorRef<'a, T, 3>,
    /// Per-channel state matrix (saved) — shape `[D, N]`.
    pub a: TensorRef<'a, T, 2>,
    /// Time-varying input-side projection (saved) — shape `[B, L, N]`.
    pub b: TensorRef<'a, T, 3>,
    /// Time-varying output-side projection (saved) — shape `[B, L, N]`.
    pub c: TensorRef<'a, T, 3>,
    /// Optional skip-connection (saved) — shape `[D]`.
    pub d_skip: Option<TensorRef<'a, T, 1>>,
    /// Optional gating (saved) — shape `[B, L, D]`.
    pub z: Option<TensorRef<'a, T, 3>>,
    /// Optional delta bias (saved) — shape `[D]`.
    pub delta_bias: Option<TensorRef<'a, T, 1>>,
    /// Output gradient — shape `[B, L, D]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Output: `du` — shape `[B, L, D]`.
    pub du: TensorMut<'a, T, 3>,
    /// Output: `dB` — shape `[B, L, N]`. **Must be zero-initialized**;
    /// kernel uses `atomicAdd` over the `d` axis.
    pub d_b: TensorMut<'a, T, 3>,
    /// Output: `dC` — shape `[B, L, N]`. **Must be zero-initialized**;
    /// kernel uses `atomicAdd` over the `d` axis.
    pub d_c: TensorMut<'a, T, 3>,
    /// Output: `ddelta` — shape `[B, L, D]`.
    pub d_delta: TensorMut<'a, T, 3>,
    /// Output: `dA` — shape `[D, N]`. **Must be zero-initialized**;
    /// kernel uses `atomicAdd` over `(b, t)`.
    pub d_a: TensorMut<'a, T, 2>,
    /// Output: `dD` — shape `[D]`. Required iff `d_skip` is given.
    /// **Must be zero-initialized**; kernel uses `atomicAdd`.
    pub d_d: Option<TensorMut<'a, T, 1>>,
    /// Output: `dz` — shape `[B, L, D]`. Required iff `z` is given.
    /// Deterministic.
    pub dz: Option<TensorMut<'a, T, 3>>,
    /// Output: `d_delta_bias` — shape `[D]`. Required iff `delta_bias`
    /// is given. **Must be zero-initialized**; kernel uses `atomicAdd`.
    pub d_delta_bias: Option<TensorMut<'a, T, 1>>,
}

/// Mamba-1 selective_scan backward plan.
pub struct SelectiveScanBackwardPlan<T: Element> {
    desc: SelectiveScanBackwardDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> SelectiveScanBackwardPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SelectiveScanBackwardDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanBackwardPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0 || desc.seq_len < 0 || desc.dim < 0 || desc.dstate < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanBackwardPlan: extents must be non-negative",
            ));
        }
        if desc.dstate > 256 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanBackwardPlan: dstate must be <= 256 in the trailblazer",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanBackwardPlan: wired today: `{f32, f16, bf16}`",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // dA / dB / dC / dD / d_delta_bias use atomicAdd — not
            // deterministic at any FP dtype (order-dependent).
            bit_stable_on_same_hardware: false,
            deterministic: false,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::SelectiveScan as u16,
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

    /// Workspace size in bytes — `B * D * L * N * sizeof(T)` (recorded
    /// `h[t]` states from pass 1, consumed by pass 2).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        let dtype_id: i32 = match T::KIND {
            ElementKind::F32 => 0,
            ElementKind::F16 => 1,
            ElementKind::Bf16 => 2,
            _ => return 0,
        };
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_selective_scan_workspace_bytes(
                self.desc.batch_size, self.desc.seq_len,
                self.desc.dim, self.desc.dstate, dtype_id,
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

    /// Launch the BW kernel pair (pass-1 record + pass-2 reverse) on
    /// the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        mut args: SelectiveScanBackwardArgs<'_, T>,
    ) -> Result<()> {
        if self.desc.batch_size == 0 || self.desc.seq_len == 0 || self.desc.dim == 0 {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::Borrowed(buf) => (buf.as_raw().0 as *mut c_void, buf.len()),
            Workspace::None => (core::ptr::null_mut(), 0usize),
        };
        let need = self.workspace_size();
        if ws_bytes < need {
            return Err(Error::WorkspaceTooSmall { needed: need, got: ws_bytes });
        }

        // Optional-input presence checks must be consistent between
        // FW inputs and BW gradient outputs.
        if args.d_skip.is_some() != args.d_d.is_some() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanBackwardPlan: d_skip and d_d must be both given or both omitted",
            ));
        }
        if args.z.is_some() != args.dz.is_some() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanBackwardPlan: z and dz must be both given or both omitted",
            ));
        }
        if args.delta_bias.is_some() != args.d_delta_bias.is_some() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SelectiveScanBackwardPlan: delta_bias and d_delta_bias must be both given or both omitted",
            ));
        }

        let u_ptr = args.u.data.as_raw().0 as *const c_void;
        let delta_ptr = args.delta.data.as_raw().0 as *const c_void;
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let c_ptr = args.c.data.as_raw().0 as *const c_void;
        let d_in_ptr = args.d_skip.as_ref()
            .map(|d| d.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let z_ptr = args.z.as_ref()
            .map(|z| z.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let db_ptr = args.delta_bias.as_ref()
            .map(|db| db.data.as_raw().0 as *const c_void)
            .unwrap_or(core::ptr::null());
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let du_ptr = args.du.data.as_raw().0 as *mut c_void;
        let dB_ptr = args.d_b.data.as_raw().0 as *mut c_void;
        let dC_ptr = args.d_c.data.as_raw().0 as *mut c_void;
        let ddelta_ptr = args.d_delta.data.as_raw().0 as *mut c_void;
        let dA_ptr = args.d_a.data.as_raw().0 as *mut c_void;
        let dD_ptr = args.d_d.as_mut()
            .map(|d| d.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let dz_ptr = args.dz.as_mut()
            .map(|z| z.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let ddb_ptr = args.d_delta_bias.as_mut()
            .map(|db| db.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let dsp = if self.desc.delta_softplus { 1 } else { 0 };

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_selective_scan_f32_backward_run(
                    self.desc.batch_size, self.desc.seq_len,
                    self.desc.dim, self.desc.dstate, dsp,
                    u_ptr, delta_ptr, a_ptr, b_ptr, c_ptr,
                    d_in_ptr, z_ptr, db_ptr,
                    dy_ptr,
                    du_ptr, dB_ptr, dC_ptr, ddelta_ptr,
                    dA_ptr, dD_ptr, dz_ptr, ddb_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_selective_scan_f16_backward_run(
                    self.desc.batch_size, self.desc.seq_len,
                    self.desc.dim, self.desc.dstate, dsp,
                    u_ptr, delta_ptr, a_ptr, b_ptr, c_ptr,
                    d_in_ptr, z_ptr, db_ptr,
                    dy_ptr,
                    du_ptr, dB_ptr, dC_ptr, ddelta_ptr,
                    dA_ptr, dD_ptr, dz_ptr, ddb_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_selective_scan_bf16_backward_run(
                    self.desc.batch_size, self.desc.seq_len,
                    self.desc.dim, self.desc.dstate, dsp,
                    u_ptr, delta_ptr, a_ptr, b_ptr, c_ptr,
                    d_in_ptr, z_ptr, db_ptr,
                    dy_ptr,
                    du_ptr, dB_ptr, dC_ptr, ddelta_ptr,
                    dA_ptr, dD_ptr, dz_ptr, ddb_ptr,
                    ws_ptr, ws_bytes, stream_ptr,
                )
            },
            _ => return Err(Error::Unsupported(
                "baracuda-kernels::SelectiveScanBackwardPlan: dtype not wired",
            )),
        };
        map_status(status)
    }
}
