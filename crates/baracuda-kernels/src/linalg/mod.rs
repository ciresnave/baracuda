//! Dense linear algebra op family — Milestone 6.3 (Category Linalg).
//!
//! Wraps cuSOLVER's dense API. Four canonical PyTorch / JAX
//! factorizations land in this milestone:
//!
//! - [`CholeskyPlan`] — `A = L · L^T` (symmetric positive-definite),
//!   batched. Lower or upper fill mode.
//! - [`LuPlan`] — `P · A = L · U` (general; partial pivoting), batched
//!   for square inputs. Non-batched `getrf` handles rectangular `[M, N]`.
//! - [`QrPlan`] — `A = Q · R`. 2-D only (cuSOLVER's dense API has no
//!   batched `geqrf`). Full `Q` (`[M, M]`) materialized via `ormqr` on
//!   identity.
//! - [`SvdPlan`] — `A = U · diag(S) · V^T`. 2-D only. `full_matrices`
//!   selects between full (`U: [M, M]`, `V^T: [N, N]`) and thin
//!   (`U: [M, K]`, `V^T: [K, N]`) shapes where `K = min(M, N)`.
//!
//! ## Dtype coverage
//!
//! `f32` + `f64` only. cuSOLVER's dense API does **not** expose `f16` /
//! `bf16` for these factorizations — callers that need mixed-precision
//! linalg must cast on either side.
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
