//! WY-blocked batched-`ormqr` — applies Householder-encoded `Q` (or
//! `Q^T`) from a [`super::qr_batched::BatchedQrPlan`] packed output to a
//! stack of right-hand-side matrices at GEMM-rates.
//!
//! Sibling to [`super::ormqr_batched::BatchedOrmqrPlan`] (Milestone 6.14)
//! which applies reflectors one-at-a-time (GEMV-rates). The WY variant
//! groups `nb` consecutive reflectors into a single block reflector
//!
//! ```text
//! H_0 · H_1 · ... · H_{nb-1} = I - V · T · V^T
//! ```
//!
//! and applies it via three cuBLAS strided-batched GEMMs per block:
//! `W := V^T·C`, `W := T·W`, `C := C - V·W`. For `M, N > ~16` this wins
//! decisively over the reflector-by-reflector kernel; for tiny matrices
//! the reflector kernel still wins. Both plans share the same descriptor
//! shape — callers pick by problem size.
//!
//! **Scope (trailblazer)** — mirrors [`super::ormqr_batched`]:
//! - `side = Left` only.
//! - `op ∈ {N, T}`.
//! - `dtype ∈ {f32, f64}`.
//!
//! **Algorithm — LAPACK DLARFT** (per-block T-build):
//!
//! For a block of `nb` reflectors starting at `block_start`:
//! ```text
//! T[0, 0] = -tau_0
//! For k = 1 .. nb-1:
//!     t[j] = Σ_{r=block_start+k}^{M-1} V[r, j] · V[r, k]   (j < k)
//!     T[0..k, k] = T[0..k, 0..k] · (-tau_k · t[0..k])
//!     T[k, k] = -tau_k
//! ```
//!
//! For `op = T`, iteration over blocks is reversed (apply `H_0` first,
//! then `H_1`, …, `H_{K-1}`); within each block the cumulative product
//! `H_0·...·H_{nb-1}` already equals `Q_block^T`'s transpose, so the
//! same V/T encode both directions and the per-block math is identical.
//!
//! **V extraction**: cuBLAS GEMM cannot consume an "implicit-1" packed-A
//! matrix; the plan materializes a dense `V [B, M, nb]` per block via a
//! small bespoke kernel (sets the implicit 1 at each diagonal, zeros
//! above, copies the strict lower below). Caller workspace funds the V
//! scratch.
//!
//! **Workspace layout** (column-major, per the rest of the linalg
//! family):
//! - `T scratch`:   `B * num_blocks * nb * nb` elements
//! - `V scratch`:   `B * M * nb`               elements
//! - `W scratch`:   `B * nb * N`               elements (used twice
//!                  per block: first as `V^T·C`, then as `T·W`).
//! - `W2 scratch`:  `B * nb * N`               elements (output of
//!                  `T·W`, then read by the rank-`nb` update).
//!
//! All four buffers come from the caller-provided workspace, byte-
//! contiguous in the order above.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_batched_ormqr_wy_build_t_f32_run,
    baracuda_kernels_batched_ormqr_wy_build_t_f64_run,
    baracuda_kernels_batched_ormqr_wy_extract_v_f32_run,
    baracuda_kernels_batched_ormqr_wy_extract_v_f64_run, cublasCreate_v2, cublasDestroy_v2,
    cublasDgemmStridedBatched, cublasHandle_t, cublasSetStream_v2, cublasSgemmStridedBatched,
    CUBLAS_OP_N, CUBLAS_OP_T,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, LinalgKind, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, Workspace,
};

use super::cholesky::unpack_workspace;
use super::ormqr_batched::{BatchedOrmqrOp, BatchedOrmqrSide};

/// WY block size — number of reflectors fused per block-reflector. Must
/// match the value in `baracuda_batched_ormqr_wy.cuh`.
pub const WY_NB: i32 = 32;

/// Descriptor for a WY-blocked batched-`ormqr` op. Shape-compatible with
/// [`super::ormqr_batched::BatchedOrmqrDescriptor`] so callers can swap
/// based on problem size.
#[derive(Copy, Clone, Debug)]
pub struct BatchedOrmqrWyDescriptor {
    /// Row count `M` of each `C` matrix (and of each `A_packed`).
    pub m: i32,
    /// Column count `N` of each `C` matrix.
    pub n: i32,
    /// Number of Householder reflectors `K` in each `A_packed`.
    pub k: i32,
    /// Number of independent slots in the batch.
    pub batch_size: i32,
    /// Side of the multiplication. Trailblazer accepts only
    /// [`BatchedOrmqrSide::Left`].
    pub side: BatchedOrmqrSide,
    /// Op tag — [`BatchedOrmqrOp::N`] (apply `Q`) or
    /// [`BatchedOrmqrOp::T`] (apply `Q^T`).
    pub op: BatchedOrmqrOp,
    /// Element type. Must be `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for a WY-blocked batched-`ormqr` launch.
///
/// `a` and `tau` are the *unmodified* outputs of
/// [`super::qr_batched::BatchedQrPlan::run`]; `c` is the right-hand
/// side stack, **overwritten in place** with the result.
///
/// Note: `a` is taken as `TensorMut` for API symmetry with `BatchedQrPlan`
/// and to allow the WY plan to materialize V into caller workspace
/// (which is `TensorMut` shape).
pub struct BatchedOrmqrWyArgs<'a, T: Element> {
    /// `geqrf`-packed input: `[batch, M, K]` column-major (per slot,
    /// strict lower triangle holds the Householder reflectors).
    /// Read-only — taken as `TensorMut` for parity with the args shape
    /// used elsewhere in the linalg family.
    pub a: TensorMut<'a, T, 3>,
    /// `geqrf` Householder scalars: `[batch, K]`.
    pub tau: TensorMut<'a, T, 2>,
    /// Right-hand-side matrix stack `[batch, M, N]` column-major.
    /// Overwritten in place with the result.
    pub c: TensorMut<'a, T, 3>,
}

/// WY-blocked batched-`ormqr` plan — apply Householder `Q` at
/// GEMM-rates.
///
/// Groups `WY_NB = 32` consecutive reflectors into a single block-
/// reflector `H_0 · ... · H_{nb-1} = I - V · T · V^T`, then applies
/// via three cuBLAS strided-batched GEMMs per block. Sibling to
/// [`super::BatchedOrmqrPlan`] (reflector-by-reflector, GEMV-rates) —
/// choose by problem size: WY wins for `M, N > ~16`.
///
/// **When to use**: batched `Q · C` apply for medium / large
/// matrices. Pair with [`super::BatchedQrPlan`] which produces the
/// packed inputs.
///
/// **Dtypes**: `f32`, `f64` only (the trailblazer scope; complex is a
/// deferred follow-up).
///
/// **Constraints**: `side = Left` only; `op ∈ {N, T}`.
///
/// **Storage**: column-major end-to-end.
///
/// **Workspace**: scratch for `V` and the per-block `T` matrices (size
/// computed via [`Self::workspace_size`]).
///
/// **Precision guarantee**: deterministic per launch but not
/// bit-stable (cuBLAS reduction order).
///
/// Owns a lazy cuBLAS handle (`!Sync` / `!Send`); destroyed on `Drop`.
pub struct BatchedOrmqrWyPlan<T: Element> {
    desc: BatchedOrmqrWyDescriptor,
    sku: KernelSku,
    handle: Cell<cublasHandle_t>,
    workspace_bytes: Cell<usize>,
    num_blocks: i32,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchedOrmqrWyPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &BatchedOrmqrWyDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedOrmqrWyPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedOrmqrWyPlan: WY kernel wired for f32 + f64 only",
            ));
        }
        if !matches!(desc.side, BatchedOrmqrSide::Left) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedOrmqrWyPlan: side = Right is deferred",
            ));
        }
        if matches!(desc.op, BatchedOrmqrOp::C) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BatchedOrmqrWyPlan: op = C (conjugate transpose) is \
                 deferred — trailblazer is real-dtype (f32/f64) only",
            ));
        }
        if desc.m <= 0 || desc.n <= 0 || desc.k <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrWyPlan: M, N, K must be > 0",
            ));
        }
        if desc.batch_size <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrWyPlan: batch_size must be > 0",
            ));
        }
        if desc.k > desc.m {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrWyPlan: K must be <= M (ormqr contract)",
            ));
        }

        let nb = WY_NB;
        let num_blocks = (desc.k + nb - 1) / nb;

        // Workspace layout (in elements, multiplied by sizeof(T) for bytes):
        //   T:  B * num_blocks * nb * nb
        //   V:  B * M * nb
        //   W:  B * nb * N
        //   W2: B * nb * N
        let elem = core::mem::size_of::<T>();
        let b = desc.batch_size as usize;
        let m = desc.m as usize;
        let n = desc.n as usize;
        let nbu = nb as usize;
        let nbb = num_blocks as usize;
        let t_elems = b * nbb * nbu * nbu;
        let v_elems = b * m * nbu;
        let w_elems = b * nbu * n;
        let w2_elems = b * nbu * n;
        let ws_bytes = (t_elems + v_elems + w_elems + w2_elems) * elem;

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
            category: OpCategory::Linalg,
            op: LinalgKind::BatchedOrmqrWy as u16,
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
            handle: Cell::new(core::ptr::null_mut()),
            workspace_bytes: Cell::new(ws_bytes),
            num_blocks,
            _marker: PhantomData,
        })
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

    /// Workspace size in bytes — `T + V + 2·W` scratch buffers (see
    /// module docs).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        self.workspace_bytes.get()
    }

    /// Workspace requirement is known at `select` time — this returns
    /// the cached size for API uniformity with the rest of the linalg
    /// family.
    pub fn query_workspace_size(&self, _stream: &Stream) -> Result<usize> {
        Ok(self.workspace_bytes.get())
    }

    fn ensure_handle(&self) -> Result<cublasHandle_t> {
        let h = self.handle.get();
        if !h.is_null() {
            return Ok(h);
        }
        let mut handle: cublasHandle_t = core::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.handle.set(handle);
        Ok(handle)
    }

    fn bind_stream(&self, h: cublasHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cublasSetStream_v2(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    fn check_args(&self, args: &BatchedOrmqrWyArgs<'_, T>) -> Result<()> {
        let b = self.desc.batch_size;
        let m = self.desc.m;
        let n = self.desc.n;
        let k = self.desc.k;
        if args.a.shape != [b, m, k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrWyPlan: A shape != [batch, M, K]",
            ));
        }
        if args.tau.shape != [b, k] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrWyPlan: tau shape != [batch, K]",
            ));
        }
        if args.c.shape != [b, m, n] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BatchedOrmqrWyPlan: C shape != [batch, M, N]",
            ));
        }
        Ok(())
    }
}

/// Status-code → Result translation. Shared shape with the rest of the
/// linalg bespoke plans.
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

macro_rules! impl_batched_ormqr_wy_run {
    ($T:ty, $build_t:ident, $extract_v:ident, $gemm_strided:ident) => {
        impl BatchedOrmqrWyPlan<$T> {
            /// Run the WY-blocked batched-`ormqr`.
            pub fn run(
                &self,
                stream: &Stream,
                workspace: Workspace<'_>,
                args: BatchedOrmqrWyArgs<'_, $T>,
            ) -> Result<()> {
                self.check_args(&args)?;
                let h = self.ensure_handle()?;
                self.bind_stream(h, stream)?;

                let b = self.desc.batch_size;
                let m = self.desc.m;
                let n = self.desc.n;
                let k = self.desc.k;
                let nb = WY_NB;
                let num_blocks = self.num_blocks;

                let needed = self.workspace_bytes.get();
                let (ws_ptr, ws_bytes) = unpack_workspace(workspace, needed)?;
                if ws_bytes < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: ws_bytes,
                    });
                }

                // Carve workspace into T / V / W / W2.
                let elem = core::mem::size_of::<$T>();
                let bu = b as usize;
                let mu = m as usize;
                let nu = n as usize;
                let nbu = nb as usize;
                let nbb = num_blocks as usize;
                let t_elems = bu * nbb * nbu * nbu;
                let v_elems = bu * mu * nbu;
                let w_elems = bu * nbu * nu;
                let w2_elems = bu * nbu * nu;

                let t_ptr = ws_ptr as *mut u8;
                let v_ptr = unsafe { t_ptr.add(t_elems * elem) };
                let w_ptr = unsafe { v_ptr.add(v_elems * elem) };
                let w2_ptr = unsafe { w_ptr.add(w_elems * elem) };
                // sanity: stays within `needed`.
                debug_assert_eq!(
                    needed,
                    (t_elems + v_elems + w_elems + w2_elems) * elem
                );
                let _ = w2_elems; // used implicitly via ws-bytes accounting above

                let a_ptr_v = args.a.data.as_raw().0 as *const c_void;
                let tau_ptr_v = args.tau.data.as_raw().0 as *const c_void;
                let c_ptr = args.c.data.as_raw().0 as *mut $T;
                let stream_ptr = stream.as_raw() as *mut c_void;

                // ----- Step 1: build T for every block in one launch ---------
                let status = unsafe {
                    $build_t(
                        b,
                        m,
                        k,
                        nb,
                        num_blocks,
                        a_ptr_v,
                        tau_ptr_v,
                        t_ptr as *mut c_void,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                };
                map_status(status)?;

                // ----- Step 2: per-block apply ---------------------------------
                // For op = N (apply Q = H_{K-1} · ... · H_0), iterate
                // blocks from last to first (matching the per-reflector
                // order). For op = T (apply Q^T = H_0 · ... · H_{K-1}),
                // iterate blocks from first to last.
                //
                // Within each block, the math is the same — applying
                // (I - V·T·V^T) on the left of C reduces to three GEMMs:
                //   W  := V^T · C       (nb × N)
                //   W2 := T · W         (nb × N)
                //   C  := C - V · W2    (M × N)
                //
                // For op = T we'd want (I - V·T^T·V^T) instead, but
                // cuBLAS GEMM handles the implicit transpose via the
                // `transa` arg at no perf cost — see the third GEMM
                // below.
                let block_indices: Vec<i32> = match self.desc.op {
                    BatchedOrmqrOp::N => (0..num_blocks).rev().collect(),
                    BatchedOrmqrOp::T => (0..num_blocks).collect(),
                    BatchedOrmqrOp::C => {
                        // Rejected at `select`; defensively return an
                        // Unsupported error here as well so a misuse via
                        // direct construction can't reach the GEMM dispatch.
                        return Err(Error::Unsupported(
                            "baracuda-kernels::BatchedOrmqrWyPlan: op = C is deferred",
                        ));
                    }
                };

                for blk in block_indices {
                    let block_start = blk * nb;
                    let block_k = if block_start + nb < k {
                        nb
                    } else {
                        k - block_start
                    };
                    if block_k <= 0 {
                        continue;
                    }

                    // Materialize V[B, M, nb] for this block.
                    let status = unsafe {
                        $extract_v(
                            b,
                            m,
                            k,
                            nb,
                            block_start,
                            block_k,
                            a_ptr_v,
                            v_ptr as *mut c_void,
                            core::ptr::null_mut(),
                            0,
                            stream_ptr,
                        )
                    };
                    map_status(status)?;

                    // T for this block is at offset (blk * nb * nb) per slot
                    // (slot stride = num_blocks * nb * nb).
                    let t_block_offset_elems = (blk as i64) * (nb as i64) * (nb as i64);
                    let t_block_ptr = unsafe {
                        (t_ptr as *mut $T).offset(t_block_offset_elems as isize)
                    };
                    let t_slot_stride: i64 = (num_blocks as i64) * (nb as i64) * (nb as i64);

                    let v_typed = v_ptr as *const $T;
                    let v_slot_stride: i64 = (m as i64) * (nb as i64);
                    let w_typed = w_ptr as *mut $T;
                    let w_slot_stride: i64 = (nb as i64) * (n as i64);
                    let w2_typed = w2_ptr as *mut $T;
                    let w2_slot_stride: i64 = (nb as i64) * (n as i64);
                    let c_slot_stride: i64 = (m as i64) * (n as i64);

                    let one: $T = 1 as $T;
                    let zero: $T = 0 as $T;
                    let neg_one: $T = -(1 as $T);

                    // --------- GEMM 1: W := V^T · C    (nb × N)
                    //   op(A) = V^T  → transa = T,  A = V [M, nb],  lda = M
                    //   op(B) = C    → transb = N,  B = C [M, N],   ldb = M
                    //   C_out = W [nb, N],  ldc = nb
                    let status = unsafe {
                        $gemm_strided(
                            h,
                            CUBLAS_OP_T,
                            CUBLAS_OP_N,
                            nb, n, m,
                            &one as *const $T,
                            v_typed, m, v_slot_stride,
                            c_ptr as *const $T, m, c_slot_stride,
                            &zero as *const $T,
                            w_typed, nb, w_slot_stride,
                            b,
                        )
                    };
                    if status != 0 {
                        return Err(Error::CutlassInternal(-status));
                    }

                    // --------- GEMM 2: W2 := op(T) · W    (nb × N)
                    //   For op = N (apply Q): we want H_0·...·H_{nb-1} =
                    //     I - V·T·V^T  ⇒  use T as-is (transa = N).
                    //   For op = T (apply Q^T): we want H_{nb-1}·...·H_0 =
                    //     I - V·T^T·V^T  ⇒  use T transposed (transa = T).
                    let trans_t = match self.desc.op {
                        BatchedOrmqrOp::N => CUBLAS_OP_N,
                        BatchedOrmqrOp::T => CUBLAS_OP_T,
                        BatchedOrmqrOp::C => unreachable!("op = C rejected above"),
                    };
                    let status = unsafe {
                        $gemm_strided(
                            h,
                            trans_t,
                            CUBLAS_OP_N,
                            nb, n, nb,
                            &one as *const $T,
                            t_block_ptr as *const $T, nb, t_slot_stride,
                            w_typed as *const $T, nb, w_slot_stride,
                            &zero as *const $T,
                            w2_typed, nb, w2_slot_stride,
                            b,
                        )
                    };
                    if status != 0 {
                        return Err(Error::CutlassInternal(-status));
                    }

                    // --------- GEMM 3: C := C - V · W2    (M × N)
                    //   op(A) = V   → transa = N,  A = V [M, nb],   lda = M
                    //   op(B) = W2  → transb = N,  B = W2 [nb, N],  ldb = nb
                    //   C_out = C [M, N],  ldc = M    (α = -1, β = 1)
                    let status = unsafe {
                        $gemm_strided(
                            h,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            m, n, nb,
                            &neg_one as *const $T,
                            v_typed, m, v_slot_stride,
                            w2_typed as *const $T, nb, w2_slot_stride,
                            &one as *const $T,
                            c_ptr, m, c_slot_stride,
                            b,
                        )
                    };
                    if status != 0 {
                        return Err(Error::CutlassInternal(-status));
                    }
                }

                Ok(())
            }
        }
    };
}

impl_batched_ormqr_wy_run!(
    f32,
    baracuda_kernels_batched_ormqr_wy_build_t_f32_run,
    baracuda_kernels_batched_ormqr_wy_extract_v_f32_run,
    cublasSgemmStridedBatched
);
impl_batched_ormqr_wy_run!(
    f64,
    baracuda_kernels_batched_ormqr_wy_build_t_f64_run,
    baracuda_kernels_batched_ormqr_wy_extract_v_f64_run,
    cublasDgemmStridedBatched
);

impl<T: Element> Drop for BatchedOrmqrWyPlan<T> {
    fn drop(&mut self) {
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cublasDestroy_v2(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
    }
}
