# In-place op coverage — comprehensive matrix

This guide is the single source of truth for **which baracuda kernels
can safely be dispatched with same-pointer aliasing** (the standard
"in-place" pattern: output buffer equals one of the input buffers).

Phases 61/62/64 documented the stable public contract on the
relevant trailblazers; downstream callers (e.g. Fuel's in-place op
executor) can rely on it without coordinating with baracuda for each
new in-place op family.

## TL;DR table

| Family | In-place? | How |
|---|---|---|
| Unary contig (~30 ops × 4-8 dtypes) | ✅ Safe | `x_ptr == y_ptr` (Phase 61) |
| Unary strided | ✅ Safe (conditional) | `x_ptr == y_ptr` AND `stride_x == stride_y` (Phase 62) |
| Binary contig (add/sub/mul/div/min/max/...) | ✅ Safe | `a_ptr == y_ptr` or `b_ptr == y_ptr` (Phase 61) |
| Binary strided | ✅ Safe (conditional) | aliased input's stride must equal `stride_y` (Phase 62) |
| Parameterized-unary (powi/threshold/elu/prelu/lerp) | ✅ Safe | same as unary trailblazer (Phase 61) |
| Ternary clamp contig + strided | ✅ Safe | any input == `y` (Phase 61); strided requires equal strides (Phase 62) |
| Affine | ✅ Safe (forward + dedicated `_inplace_` symbols) | Phase 61 + 62 |
| **Cast** | ✅ Safe **(only when same byte width)** | Phase 64 |
| **Where** | ✅ Safe | `a == y` or `b == y` (Phase 64) |
| **Triu / Tril** | ✅ Safe | `input == output` (Phase 64) |
| **Activation BW** (gelu/silu/relu/tanh/sigmoid/... `_backward_*`) | ✅ Safe | `dx == saved` or `dx == dy` (Phase 64) |
| **Fill** | ✅ Trivial | write-only, no input to alias (Phase 64) |
| **Flip** | ❌ NOT safe | thread `i` reads at source coord, writes at flipped dest coord — two threads touch each cell (Phase 64 warning) |
| **Roll** | ❌ NOT safe | cyclic shift — same race as flip (Phase 64 warning) |
| **Permute** | ❌ NOT safe | axis permutation — same race (Phase 64 warning) |
| **RoPE** | ❌ NOT safe | pair rotation — both threads in a pair read both elements (Phase 64 warning) |
| Reductions, Softmax/LayerNorm/RMSNorm/BatchNorm, Attention, GEMM, Conv, Pool, Linalg, Sort/Topk, Scans, FFT, Losses, Quantization, Embedding | ❌ Fundamentally not in-place safe | multi-pass / shape-changing / cross-cell deps |

## Why some kernels look like they should be in-place but aren't

A common-looking elementwise kernel body of the form `y[i] = f(x[i])`
**looks** in-place-safe at a glance, but the underlying question is:
**does each thread read from and write to the SAME memory address?**

For Cast, Where, Triu/Tril, Activation BW, and Fill: yes, each thread's
read offset equals its write offset (they're indexed by the same `i`).
Same-pointer is safe.

For Flip, Roll, Permute, RoPE: each thread reads from one coordinate
(a flipped / shifted / permuted / paired source) and writes to a
different coordinate. Two different threads touch the same physical
memory cell — one reads it as a source, another writes it as a
destination. Without synchronization between those two threads (which
the kernels don't provide), the read-then-write ordering is
unpredictable, leading to silent data corruption.

```text
Flip example (shape [4]):
  Thread 0:  reads x[0]  →  writes y[3]   (same buffer: byte 0 → byte 12 if f32)
  Thread 3:  reads x[3]  →  writes y[0]   (same buffer: byte 12 → byte 0)

If Thread 3 writes byte 0 BEFORE Thread 0 reads byte 0, Thread 0
reads the new value (= original x[3]) instead of original x[0].
Result: corrupted output.
```

If a caller needs an in-place flip / roll / permute / RoPE, the
correct pattern is to materialize the output into a fresh buffer and
copy back, OR write a bespoke in-place algorithm with explicit
paired-swap synchronization (which baracuda does not provide today).

## Cast aliasing — byte-width condition

Cast is special: `y[i] = static_cast<TOut>(x[i])`. With `x_ptr == y_ptr`:

- If `sizeof(TIn) == sizeof(TOut)`: byte offset of `x[i]` equals byte
  offset of `y[i]`. Each thread reads then writes the same address.
  **Safe.** Examples: f32↔i32, f32↔u32, i32↔u32, f16↔bf16, f64↔i64,
  u8↔i8.
- If `sizeof(TIn) != sizeof(TOut)`: byte offsets diverge. Thread `i`
  reads at `i * sizeof(TIn)` but writes at `i * sizeof(TOut)` — and
  some thread `j ≠ i` reads from the byte range thread `i` writes to.
  **NOT safe.** Examples: f32→f64, f16→f32, f64→f32, f32→f16, any
  fp↔int width-mismatch.

The kernel does no validation. Caller must verify.

## Activation BW aliasing — saved-x vs saved-y

The activation BW family is `dx[i] = f(saved[i]) * dy[i]` where
`saved` is either the saved input (saved-x family: ReLU, GELU, SiLU,
ELU, HardSwish, HardSigmoid, Mish, LeakyReLU, Erf, Erfc) or the saved
output (saved-y family: Sigmoid, Tanh).

Same-pointer aliasing with EITHER `saved` or `dy` is safe:

- `dx_ptr == saved_ptr`: thread `i` reads `saved[i]` and `dy[i]`,
  writes `dx[i]`. The `saved[i]` read happens before the `dx[i]`
  write (within thread `i`); other threads don't touch this cell.
  **Safe.**
- `dx_ptr == dy_ptr`: symmetric. **Safe.**
- `saved_ptr == dy_ptr`: both inputs, both read. No write conflict.
  **Trivially safe.**
- All three aliased (`dx == saved == dy`): each thread reads twice
  from the same address then writes back. **Safe** (read-before-write
  ordering preserved per thread).

## Reference: the Phase 61/62 stride-equality contract still applies

For the strided variants of unary/binary/ternary, aliasing requires
that the aliased input's stride array equals `stride_y`
element-for-element. Use [`baracuda_kernels_types::strides_equal`](../../crates/baracuda-kernels-types/src/tensor.rs)
to check. See [`fa2-saved-tensor-contract.md`](fa2-saved-tensor-contract.md)
for a different but related "contract specified at one place, callers
validate" pattern.

## What's NOT covered (and likely never will be)

These op families fundamentally CANNOT be done in-place at the kernel
level, by structure:

- **Reductions** (`reduce_sum`, `reduce_max`, etc.) — output shape ≠
  input shape.
- **Norms** (Softmax/LogSoftmax/LayerNorm/RMSNorm/BatchNorm/GroupNorm/
  InstanceNorm) — multi-pass: must read ALL of input before writing
  any output cell.
- **Attention** (FlashAttn / SDPA / Ring / paged) — multi-pass with
  softmax + value-aggregation.
- **GEMM / matmul / convolution / pooling** — output shape differs;
  output cells depend on many input cells.
- **Linalg** (Cholesky, LU, QR, SVD, etc.) — complex factorization;
  some have in-place library variants (cuSOLVER) but baracuda's safe
  wrapper layer doesn't expose them.
- **Sort / Topk / Argsort / Unique** — multi-pass.
- **Scans** (cumsum, cumprod, cummax, cummin) — sequential
  dependencies.
- **FFT** — frequency-domain transform.
- **Losses** (involve reductions).
- **Quantization** (per-group/-channel/-token/-tensor) — read-then-
  derive-scale-then-write requires a multi-pass structure or a
  separately-prepared scale tensor.
- **Embedding / EmbeddingBag** — gather with aggregation.

Don't ask for in-place versions of any of the above — the answer is
fundamental, not policy.

## See also

- [`crates/baracuda-kernels-sys/src/lib.rs`](../../crates/baracuda-kernels-sys/src/lib.rs)
  — search for "Aliasing" in the docstrings; each family's contract
  is documented at the trailblazer.
- [`crates/baracuda-kernels-types/src/tensor.rs`](../../crates/baracuda-kernels-types/src/tensor.rs)
  — `strides_equal` helper for the strided contract.
- Phase 61 / 62 / 64 in `ROADMAP.md` — historical decisions.
