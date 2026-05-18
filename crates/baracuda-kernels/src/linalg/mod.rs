//! Dense linear algebra op family — Phase 6 (Category Linalg).
//!
//! Wraps cuSOLVER's dense API plus a few bespoke kernels for batched-QR
//! variants that cuSOLVER does not surface. The family covers:
//!
//! ### Factorizations
//! - [`CholeskyPlan`] — `A = L · L^T` (SPD), non-batched + batched.
//! - [`LuPlan`] — `P · A = L · U` (partial pivoting); non-batched today
//!   (`batch_size == 1` only — cuSOLVER's dense `getrf` is non-batched,
//!   cuBLAS batched LU is a deferred follow-up).
//! - [`QrPlan`] — `A = Q · R`; 2-D only (cuSOLVER has no batched `geqrf`).
//! - [`BatchedQrPlan`] — batched-QR via **cuBLAS** `geqrfBatched`,
//!   packed output, `f32` / `f64` / `Complex32` / `Complex64`.
//! - [`BatchedQrMaterializePlan`] — bespoke kernel that unpacks
//!   [`BatchedQrPlan`]'s output into dense `Q [B, M, M]` + `R [B, K, N]`.
//! - [`SvdPlan`] — `A = U · diag(S) · V^T`. 2-D only (`gesvd`,
//!   bidiag-QR). `full_matrices` toggles full vs thin shapes.
//! - [`BatchedSvdPlan`] — Jacobi-batched (`gesvdjBatched`), square-only.
//! - [`BatchedSvdaPlan`] — rectangular approximate-SVD
//!   (`gesvdaStridedBatched`) with rank-truncation.
//!
//! ### Eigendecompositions
//! - [`EighPlan`] — `A · v = λ · v` (symmetric / Hermitian), real eigvals.
//! - [`EigPlan`] — general non-symmetric `A · v = λ · v` via `Xgeev`;
//!   real input → real packed eigvals (`wr` / `wi`), complex input →
//!   complex eigvals.
//!
//! ### Solvers / inverse / least-squares
//! - [`SolvePlan`] — `A · X = B` via `getrf` + `getrs`.
//! - [`InversePlan`] — `A^{-1}` via `getrf` + `getrs` over identity RHS.
//! - [`LstSqPlan`] — `min ‖A·x - b‖²` via `_gels` (iterative) with
//!   optional QR (`geqrf` + `ormqr` + `trsm`) fallback.
//!
//! ### Householder application
//! - [`BatchedOrmqrPlan`] — reflector-by-reflector apply (GEMV-rates;
//!   wins for tiny matrices). Real `op ∈ {N, T}`, complex `op ∈ {N, C}`.
//!   `side ∈ {Left, Right}`.
//! - [`BatchedOrmqrWyPlan`] — WY-blocked apply via cuBLAS strided-batched
//!   GEMM (GEMM-rates; wins for `M, N > ~16`). `side = Left` only.
//!
//! ## Dtype coverage
//!
//! Most plans support `f32` + `f64` only — cuSOLVER's dense API does
//! **not** expose `f16` / `bf16` for these factorizations. Complex
//! (`Complex32` / `Complex64`) is wired for [`EighPlan`], [`EigPlan`],
//! [`BatchedQrPlan`], [`BatchedOrmqrPlan`]. See per-plan docs for the
//! authoritative dtype list.
//!
//! ## Row-major / column-major adapter
//!
//! cuSOLVER is column-major (LAPACK convention). PyTorch and the rest
//! of baracuda are row-major. The plan layer handles the bridge:
//!
//! - **Symmetric ops (Cholesky)**: a row-major lower-triangular factor
//!   `L` over storage `S` is bit-identical to a column-major upper-
//!   triangular factor `U` over the same storage `S` (because `L^T = U`
//!   when re-interpreting row-major as column-major). So
//!   `CholeskyDescriptor { lower: true }` (row-major input) maps to
//!   `uplo = CUBLAS_FILL_MODE_UPPER` when handing the matrix to cuSOLVER.
//!
//! - **Non-symmetric ops (LU / QR / SVD)**: the row-major `[M, N]`
//!   matrix `A` is interpreted as the column-major `[N, M]` matrix
//!   `A^T`. cuSOLVER factors `A^T = L'U'` (LU) or `Q' R'` (QR). For
//!   plans that surface separate output tensors (`Q`, `R`, `U`, `V^T`),
//!   the caller-facing tensors document this transpose semantics — the
//!   reconstructed `Q · R` (interpreted row-major) factors the input
//!   row-major matrix bit-for-bit only after the appropriate transpose.
//!   For LU's in-place output, callers similarly see the column-major
//!   factor in their row-major buffer.
//!
//! The smoke tests anchor the convention by reconstructing the input
//! matrix from the factors using the *same* row-major / column-major
//! interpretation throughout — the algebra works out regardless of
//! which storage convention is on the wire, as long as it's consistent
//! end-to-end.
//!
//! ## Handle + workspace ownership
//!
//! Each plan lazily owns one `cusolverDnHandle_t` in a `Cell<>` (created
//! on first `run`; bound to the caller's stream on every launch so the
//! plan is reusable across streams). The handle is destroyed in `Drop`.
//! cuSOLVER handles are not thread-safe — the plan is `!Sync` / `!Send`
//! by virtue of the `Cell<cusolverDnHandle_t>` it holds.
//!
//! Workspace is **caller-provided** (`Workspace::Borrowed`). The plan
//! reports the required byte count through `workspace_size()`, which
//! reflects the upper bound from the cuSOLVER `_bufferSize` queries.
//! Because `_bufferSize` requires a live handle (which the plan does
//! not own at `select` time), the bytes-needed query is performed
//! lazily on first `run` and cached in a `Cell<usize>`. The
//! `workspace_size()` accessor returns 0 before the first `run` and
//! the true cached size afterwards — callers that need the size before
//! launching can call the `query_workspace_size(stream)` helper.
//!
//! Batched ops (`*potrfBatched`, `*getrfBatched`) do not take a
//! workspace argument — cuSOLVER allocates internally — so the plan
//! reports `0` for batched-only configurations.

pub mod cholesky;
pub mod eig;
pub mod eigh;
pub mod inverse;
pub mod lstsq;
pub mod lu;
pub mod ormqr_batched;
pub mod ormqr_batched_wy;
pub mod qr;
pub mod qr_batched;
pub mod qr_batched_materialize;
pub mod solve;
pub mod svd;
pub mod svd_batched;
pub mod svda_batched;

pub use cholesky::{CholeskyArgs, CholeskyDescriptor, CholeskyPlan};
pub use eig::{EigArgs, EigDescriptor, EigPlan};
pub use eigh::{EighArgs, EighDescriptor, EighPlan};
pub use inverse::{InverseArgs, InverseDescriptor, InversePlan};
pub use lstsq::{LstSqArgs, LstSqDescriptor, LstSqPlan};
pub use lu::{LuArgs, LuDescriptor, LuPlan};
pub use ormqr_batched::{
    BatchedOrmqrArgs, BatchedOrmqrDescriptor, BatchedOrmqrOp, BatchedOrmqrPlan, BatchedOrmqrSide,
};
pub use ormqr_batched_wy::{
    BatchedOrmqrWyArgs, BatchedOrmqrWyDescriptor, BatchedOrmqrWyPlan, WY_NB,
};
pub use qr::{QrArgs, QrDescriptor, QrPlan};
pub use qr_batched::{BatchedQrArgs, BatchedQrDescriptor, BatchedQrPlan};
pub use qr_batched_materialize::{
    BatchedQrMaterializeArgs, BatchedQrMaterializeDescriptor, BatchedQrMaterializePlan,
};
pub use solve::{SolveArgs, SolveDescriptor, SolvePlan};
pub use svd::{SvdArgs, SvdDescriptor, SvdPlan};
pub use svd_batched::{BatchedSvdArgs, BatchedSvdDescriptor, BatchedSvdPlan};
pub use svda_batched::{BatchedSvdaArgs, BatchedSvdaDescriptor, BatchedSvdaPlan};
