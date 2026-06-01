# Session prompt — Build `baracuda_smem_tile.cuh` helper

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Multiple parallel kernel-helper sessions running — read
[`docs/internals/kernel-helpers.md`](../internals/kernel-helpers.md) FIRST.

## Context

Matmul-style kernels (GEMM, conv im2col, attention QK^T) stage 2D
tiles in SMEM with bank-conflict-avoiding padding: a tile shaped
`[BLOCK_M, BLOCK_K]` of `T` is allocated as `T smem[BLOCK_M][BLOCK_K + 1]`
in SMEM — the `+1` padding prevents 32-thread-warp accesses to the
same SMEM bank when threads stride through the inner dimension.

This pattern is everywhere in baracuda's CUTLASS-vendored kernels, in
the bespoke flash attention, in the strided im2col code, etc. A
typed helper makes the pattern explicit + reduces bugs (forgetting the
`+1` padding is a classic CUDA bank-conflict footgun).

## File layout

Create `crates/baracuda-kernels-sys/kernels/include/baracuda_smem_tile.cuh`.

## Conventions

Same as Phase 65a — include guard `BARACUDA_SMEM_TILE_CUH`, namespace,
file-top docstring.

## Scope

```cuda
namespace baracuda { namespace tile {

// Bank-conflict-padded 2D SMEM tile.
// PAD = 1 by default; can be overridden via template arg for unusual
// element sizes (e.g. f64 tiles may want PAD = 2 if Stride causes
// 8-byte / 32-bank alignment quirks).
template <typename T, int ROWS, int COLS, int PAD = 1>
struct SmemTile2D {
    T data[ROWS][COLS + PAD];

    __device__ __forceinline__ T&       operator()(int r, int c)       { return data[r][c]; }
    __device__ __forceinline__ const T& operator()(int r, int c) const { return data[r][c]; }

    static constexpr __device__ __host__ int rows() { return ROWS; }
    static constexpr __device__ __host__ int cols() { return COLS; }
    static constexpr __device__ __host__ size_t bytes() {
        return ROWS * (COLS + PAD) * sizeof(T);
    }
};

// Cooperative tile load — `BLOCK_THREADS` threads in a block load a
// `ROWS x COLS` tile from a row-major source matrix.
template <int BLOCK_THREADS, typename Tile, typename SrcT>
__device__ __forceinline__ void tile_load_row_major(
    Tile& tile,
    const SrcT* __restrict__ src,
    int src_stride_row);

// Cooperative tile store — symmetric writeback.
template <int BLOCK_THREADS, typename Tile, typename DstT>
__device__ __forceinline__ void tile_store_row_major(
    DstT* __restrict__ dst,
    const Tile& tile,
    int dst_stride_row);

// Transposed load — read from column-major source into row-major tile
// (used for K matrix in `C = A * B^T` patterns).
template <int BLOCK_THREADS, typename Tile, typename SrcT>
__device__ __forceinline__ void tile_load_col_major(
    Tile& tile,
    const SrcT* __restrict__ src,
    int src_stride_col);

} }  // namespace baracuda::tile
```

## Implementation guidance

- Use `__syncthreads()` at the end of `tile_load_*` so the tile is visible to all threads before the compute phase.
- Don't sync at the end of `tile_store_*` — let the caller decide if a sync is needed (often the next iteration's tile load doesn't depend on this store).
- Vectorize loads when possible: f32 → load via `float2` or `float4` when `COLS % 4 == 0` and `src_stride_row % 4 == 0`. Use template specialization or `if constexpr` to dispatch. Out-of-scope for the first ship: vectorized half-precision (f16/bf16) loads via `__half2` / `__nv_bfloat162` — those are tricky enough to warrant a separate session.

## Coordination with CUB

CUB has `cub::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD>` which does
similar work but is geared toward 1D rows. For 2D tiles with padded
SMEM the baracuda helper covers a different use case. Don't try to
replace CUB; complement it.

## Deliverables

1. The new `.cuh` (~200 LOC).
2. Update the index doc.
3. Commit message lists candidate kernels to migrate next (GEMM,
   bespoke flash attention, im2col, attention QK^T, etc.).

## Tests

Optional standalone kernel test that builds a small tile, loads from
device memory, compares against expected. Not required for the first
ship.

## Out of scope

- Vectorized half-precision loads (f16x2 / bf16x2). Separate session
  for `baracuda_vec_load.cuh`.
- Cross-warp tile-shuffling helpers (`tile_transpose_in_smem`,
  `tile_broadcast_col`). Add later if multiple kernels need them.
- Migration of existing kernels to use this helper. Future audit phase.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase67e-smem-tile`
- Read first: `docs/internals/kernel-helpers.md`
- No version bump, no publish.
- Commit on branch + push + update index + stop.

## Stop conditions

- If baracuda's existing GEMM / flash-attention kernels use a
  fundamentally different padding strategy (e.g. swizzled SMEM layout
  via XOR addressing): note the difference, ship the helper anyway
  (it's still useful for simpler kernels), but document the limitation.
