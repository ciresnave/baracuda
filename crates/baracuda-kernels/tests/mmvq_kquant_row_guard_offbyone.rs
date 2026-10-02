//! #140: the K-quant MMVQ kernels (`mmvq_q*_K_tmpl` in
//! `crates/baracuda-kernels-sys/kernels/gguf/mmvq.cu`) computed
//! `row = blockIdx.x*blockDim.y + threadIdx.y` and guarded with
//! `if (row > nrows) return;` -- for a `row` that is 0-indexed over a
//! valid range of `0..nrows-1`, that lets `row == nrows` through: one row
//! past the end of the `dst` output buffer, an out-of-bounds write.
//!
//! # Why this is a host-side arithmetic test, not a GPU repro
//!
//! The launcher sizes every one of these kernels' grids as
//! `block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y)` blocks with
//! `blockDim.y = GGML_CUDA_MMV_Y`, and `GGML_CUDA_MMV_Y` is a single
//! shared `constexpr int = 1` (see `kernels/gguf/mmvq.cu`). Under that
//! constant, `row = blockIdx.x * 1 + threadIdx.y(=0) = blockIdx.x`, and
//! `blockIdx.x` ranges over exactly `[0, nrows)` (`ceil_div(nrows, 1)
//! == nrows` blocks) -- so `row` can never actually reach `nrows` in a
//! real launch today. The bug is real but **currently unreachable**; it
//! would only misbehave if `GGML_CUDA_MMV_Y` is ever raised above 1 for
//! occupancy reasons (the K-quant kernels would then process more than
//! one row per block via `threadIdx.y`, and the off-by-one would let the
//! last block's final row land one past `nrows`).
//!
//! Forcing an actual device OOB repro would require recompiling the
//! kernel with a non-default `GGML_CUDA_MMV_Y`, which isn't exposed as a
//! build-time parameter -- not worth the churn for a latent path. This
//! test instead encodes the exact row/grid arithmetic from
//! `kernels/gguf/mmvq.cu`'s K-quant launchers (`block_num_y =
//! ceil_div_host(nrows, GGML_CUDA_MMV_Y)`, `row = blockIdx.x * blockDim.y
//! + threadIdx.y`) as plain host-side integer math, and proves: (a) under
//! today's `GGML_CUDA_MMV_Y = 1`, the old and new guards are
//! observationally identical (confirming the "currently latent" claim is
//! not just asserted but demonstrated), and (b) under a hypothetical
//! `GGML_CUDA_MMV_Y > 1`, the old guard (`row > nrows`) lets `row ==
//! nrows` through for the grid's last block while the new guard (`row >=
//! nrows`) correctly rejects every `row` outside `[0, nrows)`.

fn ceil_div_host(p: i64, q: i64) -> i64 {
    (p + q - 1) / q
}

/// Every `row` a real launch produces, for a given `(nrows, mmv_y)`, via
/// the same grid/block formula the kernel launchers use.
fn launched_rows(nrows: i64, mmv_y: i64) -> Vec<i64> {
    let block_num_y = ceil_div_host(nrows, mmv_y);
    let mut rows = Vec::new();
    for block_idx_x in 0..block_num_y {
        for thread_idx_y in 0..mmv_y {
            rows.push(block_idx_x * mmv_y + thread_idx_y);
        }
    }
    rows
}

fn old_guard_rejects(row: i64, nrows: i64) -> bool {
    row > nrows
}

fn new_guard_rejects(row: i64, nrows: i64) -> bool {
    row >= nrows
}

/// Today's shipped `GGML_CUDA_MMV_Y` (see `kernels/gguf/mmvq.cu`). Every
/// `row` a real launch produces is `< nrows` under this value -- the
/// bug's trigger condition (`row == nrows`) is unreachable, so neither
/// guard observably differs for any launch that can actually happen.
#[test]
fn ggml_cuda_mmv_y_one_makes_the_bug_unreachable() {
    let mmv_y = 1;
    for nrows in [1_i64, 2, 7, 64, 1000] {
        for &row in &launched_rows(nrows, mmv_y) {
            assert!(
                row < nrows,
                "nrows={nrows}, mmv_y={mmv_y}: launched row {row} reached or exceeded nrows -- \
                 the 'currently unreachable' claim in the #140 fix is false"
            );
            // Neither guard ever fires for a real launch under mmv_y=1 --
            // both accept every row a real launch actually produces.
            assert!(!old_guard_rejects(row, nrows));
            assert!(!new_guard_rejects(row, nrows));
        }
    }
}

/// Positive control: the bug is real, not hypothetical. At a
/// `GGML_CUDA_MMV_Y` above 1 (not what's shipped today, but a stand-in
/// for "if this constant is ever raised"), the grid's last block pads
/// out to a multiple of `mmv_y`, so `threadIdx.y` can land a `row`
/// exactly at `nrows` -- which the OLD guard let through and the NEW
/// guard correctly declines.
#[test]
fn old_guard_lets_row_equal_nrows_through_when_grid_is_padded() {
    // nrows=5, mmv_y=2: block_num_y = ceil_div(5,2) = 3 blocks x 2 rows/block
    // = rows [0,1, 2,3, 4,5]. row=5 == nrows=5 is the one-past-the-end row
    // the old `row > nrows` guard would NOT reject (5 > 5 is false), but a
    // valid output buffer only has indices [0,4].
    let nrows = 5_i64;
    let mmv_y = 2_i64;
    let rows = launched_rows(nrows, mmv_y);
    assert!(
        rows.contains(&nrows),
        "test setup: expected a padded grid to actually produce row == nrows (got {rows:?})"
    );

    assert!(
        !old_guard_rejects(nrows, nrows),
        "sanity: this test exists to demonstrate the old guard's failure to reject row==nrows"
    );
    assert!(
        new_guard_rejects(nrows, nrows),
        "the new guard (row >= nrows) must reject row == nrows"
    );

    // Full sweep: every row the old guard would have let through the new
    // guard rejects, and every row both guards accept is a genuinely
    // in-bounds row (< nrows).
    for &row in &rows {
        let old_rejects = old_guard_rejects(row, nrows);
        let new_rejects = new_guard_rejects(row, nrows);
        if row < nrows {
            assert!(!new_rejects, "row {row} < nrows {nrows} must be accepted");
        } else {
            assert!(
                new_rejects,
                "row {row} >= nrows {nrows} must be rejected by the fixed guard"
            );
            if row == nrows {
                assert!(
                    !old_rejects,
                    "row {row} == nrows {nrows} is exactly the case the OLD guard mishandled"
                );
            }
        }
    }
}
