//! Unified integer GEMM plan.
//!
//! Dispatches by [`LayoutSku`]:
//!
//! - [`LayoutSku::Rcr`] → `baracuda-cutlass`'s CUTLASS-based int8
//!   kernels (CUTLASS 4.2.0 `Mma` template instantiations,
//!   `LinearCombinationClamp` / `LinearCombinationBiasElementwise`
//!   epilogues, the full alpha.15 surface).
//! - [`LayoutSku::Rrr`] → bespoke `mma.sync.m16n8k32.row.col.satfinite`
//!   kernels in `baracuda-kernels-sys`. RRR coverage starts with
//!   `S8 × Identity` (the SKU that motivates this crate) and grows out
//!   the rest of the 18-SKU matrix in subsequent commits — see
//!   ~/.claude/plans/baracuda-kernels-comprehensive.md §5.
//!
//! The dispatch enum is private (`Backend`); callers see a single Plan
//! type with one `select` / `run` contract. The chosen backend is
//! reflected in [`IntGemmPlan::sku`] (via [`GemmSku::layout`]) for
//! telemetry and autotuner cache keys.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BiasElement, ElementKind, EpilogueKind, IntElement, LayoutSku, PlanPreference,
    PrecisionGuarantee, S8, Workspace,
};

// Public re-exports of the descriptor / args structs. These live in
// `baracuda-cutlass` today (their definition is genuinely shared
// between the cutlass-direct and the unified-facade entry points); we
// re-export under our crate's path so callers only ever import from
// `baracuda_kernels`.
pub use baracuda_cutlass::{GemmSku, IntGemmArgs, IntGemmDescriptor};

/// Unified integer GEMM plan.
///
/// `T: IntElement` is the kernel element type ([`S8`] / [`U8`]).
/// `BT: BiasElement` is the bias broadcast type (`f32` / `i32`),
/// meaningful only for the `Bias*` [`EpilogueKind`] variants.
pub struct IntGemmPlan<T: IntElement, BT: BiasElement = f32> {
    desc: IntGemmDescriptor,
    sku: GemmSku,
    backend: Backend<T, BT>,
}

/// Private dispatch state.
enum Backend<T: IntElement, BT: BiasElement> {
    /// RCR layout → delegate to the CUTLASS plan in `baracuda-cutlass`.
    Cutlass(baracuda_cutlass::IntGemmPlan<T, BT>),
    /// RRR layout → bespoke `mma.sync` kernel in `baracuda-kernels-sys`.
    /// The `PhantomData` keeps the element types bound so dispatch can
    /// match on `T::KIND` / `BT::KIND` at run time.
    Bespoke(BespokeRrr<T, BT>),
}

struct BespokeRrr<T: IntElement, BT: BiasElement> {
    _element: PhantomData<T>,
    _bias_element: PhantomData<BT>,
}

impl<T: IntElement, BT: BiasElement> IntGemmPlan<T, BT> {
    /// Pick an int-GEMM kernel for `desc`.
    pub fn select(
        stream: &Stream,
        desc: &IntGemmDescriptor,
        pref: PlanPreference,
    ) -> Result<Self> {
        match desc.layout {
            LayoutSku::Rcr => {
                // Defer entirely to the CUTLASS plan layer. It already
                // does the full descriptor / arch / sku check chain and
                // wraps the underlying kernel. Pull its sku back out so
                // our public sku() returns a stable value across both
                // backends.
                let inner = baracuda_cutlass::IntGemmPlan::<T, BT>::select(stream, desc, pref)?;
                let sku = inner.sku();
                Ok(Self {
                    desc: *desc,
                    sku,
                    backend: Backend::Cutlass(inner),
                })
            }
            LayoutSku::Rrr => {
                if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
                    return Err(Error::InvalidProblem(
                        "int GEMM problem must have positive M, N, K",
                    ));
                }
                // Today: only `S8 × Identity` is implemented. The other
                // 17 SKUs land in subsequent commits — return
                // `Unsupported` with a precise reason so callers can
                // gate on it.
                if T::KIND != ElementKind::S8 {
                    return Err(Error::Unsupported(
                        "baracuda-kernels: int8 RRR bespoke kernels: \
                         only S8 is implemented today (U8 / int4 / bin \
                         follow in later commits)",
                    ));
                }
                if !matches!(desc.epilogue, EpilogueKind::Identity) {
                    return Err(Error::Unsupported(
                        "baracuda-kernels: int8 RRR bespoke kernels: \
                         only the Identity epilogue is implemented today \
                         (bias / activation variants follow in later commits)",
                    ));
                }
                let sku = GemmSku {
                    arch: ArchSku::Sm80,
                    layout: desc.layout,
                    epilogue: desc.epilogue,
                    element: T::KIND,
                    // Identity carries no bias-element tag — matches the
                    // float-family convention; see the GemmSku docstring.
                    bias_element: None,
                };
                Ok(Self {
                    desc: *desc,
                    sku,
                    backend: Backend::Bespoke(BespokeRrr {
                        _element: PhantomData,
                        _bias_element: PhantomData,
                    }),
                })
            }
        }
    }

    /// Validate that this plan can launch with `args`.
    pub fn can_implement(&self, args: &IntGemmArgs<'_, T, BT>) -> Result<()> {
        match &self.backend {
            Backend::Cutlass(inner) => inner.can_implement(args),
            Backend::Bespoke(_) => {
                // Shape sanity — the kernel itself is robust against
                // pointer misalignment (byte-granular gmem loads), so
                // no alignment check beyond shape.
                if self.desc.m <= 0 || self.desc.n <= 0 || self.desc.k <= 0 {
                    return Err(Error::InvalidProblem(
                        "int GEMM problem must have positive M, N, K",
                    ));
                }
                if args.a.rows != self.desc.m || args.a.cols != self.desc.k {
                    return Err(Error::InvalidProblem(
                        "A shape mismatch with descriptor (M, K)",
                    ));
                }
                if args.b.rows != self.desc.k || args.b.cols != self.desc.n {
                    return Err(Error::InvalidProblem(
                        "B shape mismatch with descriptor (K, N) (row-major)",
                    ));
                }
                if args.d.rows != self.desc.m || args.d.cols != self.desc.n {
                    return Err(Error::InvalidProblem(
                        "D shape mismatch with descriptor (M, N)",
                    ));
                }
                Ok(())
            }
        }
    }

    /// Workspace size in bytes.
    pub fn workspace_size(&self) -> usize {
        match &self.backend {
            Backend::Cutlass(inner) => inner.workspace_size(),
            // The first bespoke RRR SKU does all its work in smem +
            // registers; no caller scratch needed.
            Backend::Bespoke(_) => 0,
        }
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> GemmSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    ///
    /// Identical for both backends today: the int8 path is bit-stable
    /// on the same hardware (`OpMultiplyAddSaturate` is deterministic;
    /// the SIMT epilogue scalar math is bit-stable too).
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee()
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: IntGemmArgs<'_, T, BT>,
    ) -> Result<()> {
        match &self.backend {
            Backend::Cutlass(inner) => inner.run(stream, workspace, args),
            Backend::Bespoke(_) => {
                // Sanity: workspace is always None for bespoke SKUs today.
                let _ = workspace; // silence unused-binding warning

                let a_ptr = args.a.data.as_raw().0 as *const c_void;
                let b_ptr = args.b.data.as_raw().0 as *const c_void;
                let d_ptr = args.d.data.as_raw().0 as *mut c_void;
                let (c_ptr, ldc) = match &args.c {
                    Some(c) => (c.data.as_raw().0 as *const c_void, c.ld),
                    None => (core::ptr::null(), 0i64),
                };

                let stream_ptr = stream.as_raw() as *mut c_void;

                // Dispatch — only S8 × Identity is implemented today.
                let status = match (T::KIND, self.sku.epilogue) {
                    (ElementKind::S8, EpilogueKind::Identity) => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_gemm_s8_rrr_sm80_run(
                            self.desc.m, self.desc.n, self.desc.k,
                            a_ptr, args.a.ld,
                            b_ptr, args.b.ld,
                            c_ptr, ldc,
                            d_ptr, args.d.ld,
                            args.alpha, args.beta,
                            core::ptr::null_mut(), 0,
                            stream_ptr,
                        )
                    },
                    // `select` should have rejected anything else, but
                    // surface as a clean error rather than panic if a
                    // future descriptor variant slips through.
                    _ => {
                        return Err(Error::Unsupported(
                            "baracuda-kernels: int8 RRR bespoke kernel dispatcher \
                             reached an unimplemented (element, epilogue) pair",
                        ));
                    }
                };

                map_bespoke_status(status)
            }
        }
    }
}

fn map_bespoke_status(code: i32) -> Result<()> {
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

// Type-level guard that the `S8` re-export is in scope; kept for clarity
// in case a future commit adds compile-time element-specific dispatch
// branches that need to refer to `S8` by path.
#[allow(dead_code)]
fn _hold_s8_in_scope() {
    let _ = S8(0);
}
