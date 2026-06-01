# Session prompt — Build `baracuda_block_atomic.cuh` helper

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Multiple parallel kernel-helper sessions running — read
[`docs/internals/kernel-helpers.md`](../internals/kernel-helpers.md) FIRST.

## Context

Several baracuda kernels write to a single output buffer from multiple
blocks (cross-block reduction). Each block accumulates a partial
result, then atomically merges into the shared output. The pattern is
duplicated across:

- `embedding_bag` BW (atomicAdd over grad output)
- `segment_*` BW (atomicAdd into grad output, with bf16/f16 atomicCAS for half-precision)
- `index_add` BW
- `scatter` reductions
- `unsorted_segment_prod` (atomicCAS-retry multiplication)
- BatchNorm BW (channel-wise atomicAdd)

The half-precision case is non-trivial: `atomicAdd` natively supports
f32 + f64 + i32 + i64 + u32 + u64 + (on sm_70+) f16; but bf16 needs an
`atomicCAS`-loop wrapper. baracuda has this wrapper somewhere
(probably in `baracuda_indexing.cuh` or similar — Phase 11.3
introduced it). Find it and lift it.

## File layout

Create
`crates/baracuda-kernels-sys/kernels/include/baracuda_block_atomic.cuh`.

## Conventions

Same as Phase 65a — include guard, namespace, `__device__
__forceinline__` everywhere, file-top docstring.

## Scope

**A unified atomic-add API across all numeric dtypes**:

```cuda
namespace baracuda { namespace atomic {

template <typename T>
__device__ __forceinline__ void add(T* addr, T value);

// Specializations: f32, f64, i32, i64, u32, u64 → native atomicAdd
//                  __half       → atomicAdd on sm_70+, atomicCAS-loop fallback
//                  __nv_bfloat16 → atomicCAS-loop (no native atomic on Ada/Ampere)

template <typename T>
__device__ __forceinline__ void max(T* addr, T value);

template <typename T>
__device__ __forceinline__ void min(T* addr, T value);

// atomic::mul is rare but needed for segment_prod; CAS-loop form.
template <typename T>
__device__ __forceinline__ void mul(T* addr, T value);

} }  // namespace baracuda::atomic
```

## Half-precision atomic-CAS pattern

Find baracuda's existing bf16/f16 atomicAdd wrapper (grep for
`atomicCAS` in the `kernels/include/` directory) and lift it. The
pattern is roughly:

```cuda
__device__ __forceinline__ void add(__nv_bfloat16* addr, __nv_bfloat16 value) {
    uintptr_t aligned = ((uintptr_t)addr) & ~0x3;
    bool hi = (((uintptr_t)addr) & 0x2) != 0;
    unsigned int* word_addr = (unsigned int*)aligned;
    unsigned int old = *word_addr, assumed;
    do {
        assumed = old;
        unsigned int updated = update_half_in_word(assumed, value, hi);
        old = atomicCAS(word_addr, assumed, updated);
    } while (assumed != old);
}
```

Make sure the half-word alignment handling is correct for both lower
and upper 16 bits of each 32-bit word.

## Coordination with `baracuda_dtype_promote.cuh`

This helper may benefit from `baracuda_dtype_promote.cuh`'s
`load_as_f32` / `store_from_f32` for the CAS-loop body (load f16 to
f32, add, store back as f16). If that header is already shipped: use
it. If not: write self-contained inline conversions; future merge can
factor them out.

## Deliverables

1. The new `.cuh` (~150-200 LOC).
2. Update the index doc.
3. Commit message references which existing kernel files had local
   atomic wrappers that should be migrated next (as a follow-up).

## Tests

Pure templates. Verification happens when a kernel migrates to use it.

You CAN add a small standalone kernel-test harness if you want — write
a `.cu` file in `kernels/test/` that exercises each atomic against a
known input, then a Rust `#[ignore]` test that runs it. That's
optional polish, not required.

## Out of scope

- Don't add `atomic::sub` — callers use `add` with negated value.
- Don't add `atomic::div` — no meaningful CAS-loop semantics for it.
- Don't migrate existing kernels to use this helper in this session.
  Migration is a separate phase (future audit + 1-kernel-per-commit).
- Don't add lock-free queue / list helpers. Pure scalar atomics only.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase67c-block-atomic`
- Read first: `docs/internals/kernel-helpers.md`
- No version bump, no publish.
- Commit on branch + push + update index doc + stop.

## Stop conditions

- If you can't find baracuda's existing bf16 atomicCAS wrapper after
  searching: stop, ask Eric — maybe it lives in a different place than
  expected.
- If the existing wrapper's alignment handling looks wrong: stop, ask
  before refactoring.
