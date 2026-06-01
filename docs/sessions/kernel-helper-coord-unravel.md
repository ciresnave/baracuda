# Session prompt — Build `baracuda_coord_unravel.cuh` helper

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Multiple parallel kernel-helper sessions running — read
[`docs/internals/kernel-helpers.md`](../internals/kernel-helpers.md) FIRST.

## Context

Every strided baracuda kernel duplicates the same coordinate-unraveling
loop: given a linear thread index `i`, decompose it into a multi-coord
across the tensor's shape, dot-product with per-axis input + output
strides to derive byte offsets. Example from
[`baracuda_elementwise.cuh`](../../crates/baracuda-kernels-sys/kernels/include/baracuda_elementwise.cuh)
(`unary_pointwise_strided_kernel` around line 4379):

```cuda
int64_t linear = i;
int64_t off_x = 0, off_y = 0;
for (int d = rank - 1; d >= 0; --d) {
    int32_t s = shape.v[d];
    int64_t c = (s == 0) ? 0 : (linear % (int64_t)s);
    if (s != 0) linear /= (int64_t)s;
    off_x += c * stride_x.v[d];
    off_y += c * stride_y.v[d];
}
```

This identical pattern appears in flip, roll, permute, affine_strided,
where_strided, ternary_clamp_strided, the rms_norm helpers, the
indexing kernels, etc. Lifting it into a helper would reduce
duplication AND make `__restrict__` / `__forceinline__` discipline
consistent.

## File layout

Create
`crates/baracuda-kernels-sys/kernels/include/baracuda_coord_unravel.cuh`.

## Conventions

Same as Phase 65a helpers (see `baracuda_smem_row_stager.cuh` as
reference for style). Include guard `BARACUDA_COORD_UNRAVEL_CUH`,
`namespace baracuda { ... }`, file-top docstring, `__device__
__forceinline__` everywhere.

## Scope

**Define `DimsI32` / `DimsI64` once in this header** if not already in
a shared location (currently duplicated across 8+ headers — see Phase
62 audit). Mark them `inline constexpr int MAX_RANK = 8;` + small POD
structs. Then add the canonical unravel functions:

```cuda
// Unravel `linear` into multi-coord, return packed offset into ONE
// stride array.
__device__ __forceinline__ int64_t unravel_offset_1(
    int64_t linear,
    int32_t rank,
    const DimsI32& shape,
    const DimsI64& stride);

// Same, for TWO independent stride arrays (input + output).
// Returns offsets via output parameters.
__device__ __forceinline__ void unravel_offsets_2(
    int64_t linear,
    int32_t rank,
    const DimsI32& shape,
    const DimsI64& stride_a,
    const DimsI64& stride_b,
    int64_t& off_a,
    int64_t& off_b);

// And THREE stride arrays (for ternary / where kernels).
__device__ __forceinline__ void unravel_offsets_3(
    int64_t linear,
    int32_t rank,
    const DimsI32& shape,
    const DimsI64& stride_a,
    const DimsI64& stride_b,
    const DimsI64& stride_c,
    int64_t& off_a,
    int64_t& off_b,
    int64_t& off_c);
```

Maybe a 4-stride variant for ternary ops with output. Decide based on
how many call sites currently exist.

## Important — DimsI32/DimsI64 deduplication risk

`DimsI32` / `DimsI64` are currently defined in 8 different headers
(verified by Phase 62 audit: `baracuda_indexing.cuh`,
`baracuda_elementwise.cuh`, `baracuda_reduce_to.cuh`,
`baracuda_fill.cuh`, `baracuda_triu_tril.cuh`, `baracuda_affine.cuh`,
`baracuda_write_slice.cuh`, `baracuda_contiguize.cuh`).

**Do NOT move the definitions yet.** Putting `DimsI32` / `DimsI64` in
your new header would cause redefinition errors in any `.cu` file that
includes BOTH this header AND one of the existing 8. Instead:

- Define `baracuda::coord::DimsI32` (under sub-namespace) in your
  header — strictly scoped to your helper.
- Have your unravel functions take the existing in-scope structs as
  templates: `template <typename Shape, typename Stride> ... unravel_offset_1(linear, rank, shape, stride)`. C++ deduces the type at call site, so callers can pass either `baracuda::elementwise::DimsI32` or `baracuda::affine::DimsI32` — they're structurally identical PODs.

This avoids the ODR violation while still consolidating the unravel logic.

## Deliverables

1. The new `.cuh` header (~120 LOC).
2. Update the index doc, move row from "planned" to "existing".
3. Commit message explains the design decision around DimsI32/DimsI64.

## Tests

Pure templates — no standalone test. Verification when first kernel
migrates to use them.

## Out of scope

- Don't modify the 8 existing headers' `DimsI32` / `DimsI64`
  definitions. Migration is future work, one kernel at a time, after
  the helper exists.
- Don't add 4+ stride variants unless you find ≥3 call sites in
  existing kernels that need them.
- Don't add broadcast handling here — the stride-0-for-broadcast convention is already baked into the existing usage; your helper just transparently honors it.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase67b-coord-unravel`
- Read first: `docs/internals/kernel-helpers.md`
- No version bump, no publish.
- Commit on branch + push + update index doc + stop.

## Stop conditions

- If the existing duplicates of `DimsI32` / `DimsI64` use INCOMPATIBLE
  definitions (different MAX_RANK, different field layout): stop,
  report, ask Eric.
- If you discover a separate "shared coord" header already exists
  (someone got here first): stop, report.
