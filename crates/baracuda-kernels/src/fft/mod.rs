//! FFT op family — Milestone 6.4 (Category Fft).
//!
//! Wraps cuFFT's 1-D APIs for the four canonical PyTorch / JAX FFTs
//! plus the two bespoke index-permutation helpers (`fftshift` /
//! `ifftshift` — cuFFT has no native shift):
//!
//! - [`FftPlan`] — `FFT` / `IFFT` (complex-to-complex). Carries an
//!   `inverse: bool` on the descriptor so the same plan shape covers
//!   both directions. Generic on `T: Element` parameterized over
//!   [`Complex32`] / [`Complex64`].
//! - [`RfftPlan`] — `RFFT` (real-to-complex). Input is real `[batch,
//!   n]`, output is complex `[batch, n/2 + 1]` (Hermitian-half).
//!   Generic on the real type `f32` / `f64`.
//! - [`IrfftPlan`] — `IRFFT` (complex-to-real). Input is complex
//!   `[batch, n/2 + 1]`, output is real `[batch, n]`. Generic on the
//!   real type. The output length `n` is a descriptor parameter (can't
//!   be inferred from the Hermitian-half input shape).
//! - [`FftShiftPlan`] — `fftshift` / `ifftshift` (bespoke kernel,
//!   pure index permutation). Element-width generic (4 / 8 / 16-byte
//!   cells) so it covers any of `f32`, `f64`, [`Complex32`],
//!   [`Complex64`].
//!
//! ## Dtype coverage
//!
//! `f32` (single precision, via `cufftExec{C2C,R2C,C2R}`) and `f64`
//! (double precision, via `cufftExec{Z2Z,D2Z,Z2D}`) only. cuFFT's main
//! API does not expose `f16` / `bf16` for native transforms — callers
//! needing reduced precision must cast on either side.
//!
//! Spectrum-domain tensors use [`Complex32`] / [`Complex64`] —
//! `#[repr(C)]` newtype wrappers around a pair of FP fields,
//! ABI-compatible with cuFFT's `cufftComplex` / `cufftDoubleComplex`.
//!
//! ## Normalization convention
//!
//! Forward transforms are unnormalized (matches both cuFFT's native
//! behavior and PyTorch's `norm="backward"` default).
//!
//! Inverse transforms are normalized by `1/N` to match PyTorch's
//! `norm="backward"`. cuFFT itself returns `N · IFFT(x)` — the plan
//! layer multiplies the output by `1/N` after the inverse exec via the
//! bespoke `baracuda_kernels_scale_inplace_*` kernels. The `N` used is
//! the **signal length** (the descriptor's `n` field), not the total
//! complex-cell count, which differs for IRFFT where the Hermitian-half
//! input has fewer cells than the real output.
//!
//! ## Handle ownership
//!
//! Each plan lazily owns one `cufftHandle` in a `Cell<>`. cuFFT
//! handles are unusual among CUDA libraries — they're integer IDs, not
//! pointers — so the sentinel for "not yet created" is `-1` rather
//! than `null`. The handle is created via `cufftPlan1d` on first
//! `run`, bound to the caller's stream via `cufftSetStream` on every
//! launch (so the plan is reusable across streams), and destroyed in
//! `Drop` via `cufftDestroy`. cuFFT handles are not thread-safe; the
//! plan is `!Sync` / `!Send` via the `Cell<cufftHandle>` it holds.
//!
//! cuFFT plans manage their own internal workspace — no caller-
//! supplied workspace is required for the basic 1-D APIs. The plans
//! report `workspace_size() == 0` accordingly.
//!
//! ## Scope
//!
//! 1-D FFTs (`FftPlan`, `RfftPlan`, `IrfftPlan`, `FftShiftPlan`) are
//! the primary Milestone 6.4 surface. Multi-dimensional FFTs
//! (Milestone 6.8) ship as siblings:
//!
//! - [`FftNdPlan`] — ND C2C (rank `1..=3`, trailblazer).
//! - [`RfftNdPlan`] / [`IrfftNdPlan`] — ND R2C / C2R (Hermitian-half on
//!   the last transformed axis only).
//!
//! The ND plans wrap `cufftPlanMany` with the default "tight" layout
//! (`inembed = onembed = null`) and require the transformed axes to be
//! a contiguous suffix of the logical operand.
//!
//! ND `fftshift` / `ifftshift` is wired as [`FftShiftNdPlan`] — a
//! bespoke index-permutation kernel that shifts a caller-selected
//! subset of axes (`[FFTSHIFT_ND_MAX_SHIFT_AXES]` capacity) over a
//! tensor of rank up to [`FFTSHIFT_ND_MAX_RANK`]. The 1-D
//! [`FftShiftPlan`] remains the fast path for single-axis shifts.

pub mod fft;
pub mod fftn;
pub mod fftshift;
pub mod fftshift_nd;
pub mod rfft;
pub mod rfftn;

pub use fft::{FftArgs, FftDescriptor, FftPlan};
pub use fftn::{FftNdArgs, FftNdDescriptor, FftNdPlan};
pub use fftshift::{FftShiftArgs, FftShiftDescriptor, FftShiftPlan};
pub use fftshift_nd::{
    FftShiftNdArgs, FftShiftNdDescriptor, FftShiftNdPlan, FFTSHIFT_ND_MAX_RANK,
    FFTSHIFT_ND_MAX_SHIFT_AXES,
};
pub use rfft::{IrfftArgs, IrfftDescriptor, IrfftPlan, RfftArgs, RfftDescriptor, RfftPlan};
pub use rfftn::{IrfftNdArgs, IrfftNdDescriptor, IrfftNdPlan, RfftNdArgs, RfftNdDescriptor, RfftNdPlan};
