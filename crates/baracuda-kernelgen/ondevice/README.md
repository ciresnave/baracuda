# On-device validation harnesses

Manual `nvcc` harnesses (not wired into `cargo test`) that launch the **generated**
`.cu` kernels on the GPU and diff against a host/CPU reference — the checks that
are catchable only on device. The `#include`d kernel names track the catalog cells
in `bin/kernelgen.rs`; update both together.

**Run (Windows):** from a Visual Studio dev shell so `nvcc` finds `cl.exe`
(`Enter-VsDevShell`), or an x64 Native Tools prompt. General shape:

```sh
cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>   # generate the catalog .cu
cp crates/baracuda-kernelgen/ondevice/<harness>.cu <outdir>/  # place harness beside them
nvcc -O3 -arch=sm_89 <outdir>/<harness>.cu -o <outdir>/<harness> && <outdir>/<harness>
```

---

## `reduce_validate.cu` — general reduction path (item 03)

Launches the general-path reduction kernels (`_reduce_{tag}_ax{hex}[_kd]`) with
small hand-checkable shapes vs a CPU reference. Validates:

- the emitter↔host **ABI** — `shape[]` / `s0[]` / `so[]` indexing and `n_out`;
- the **keepdim ⇒ `so` by input axis** vs **collapse ⇒ `so` by kept position** split;
- **NaN propagation** in the `Max` `has`-flag fold (torch.amax semantics);
- multi-axis, middle-axis (two kept axes), and reduce-all (kept empty);
- **integer accumulation** — i32 last-axis Sum/Max fold in a `long long` accumulator
  (exact, no float rounding), including negatives.

Expected: `ALL PASSED` (bit-exact, `maxerr 0`; NaN propagated).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **all 9 cases PASS**
(incl. the InnerContig block-per-row last-axis path and the i32 exact-int Sum/Max cases).

### Benchmark — `reduce_bench.cu`

Compares the fast-path (last-axis) vs general (outer-axis) reduction on a large
`[8192,8192]` f32 tensor against a copy-bandwidth reference (reductions are
memory-bound, so GB/s vs. the copy peak is the figure of merit).

**RTX 4070 Laptop (sm_89):**

| kernel | ms | GB/s |
| --- | --- | --- |
| copy (bandwidth ref, read+write) | 2.74 | 195.8 |
| reduce **last** axis (block-per-row) | **1.18** | **227.4** |
| reduce **axis 0** (general/outer, 1 thread/col) | 2.27 | 118.3 |

The block-per-row rewrite gave a **4.4× win** on the last axis (was 5.15 ms /
52.2 GB/s with the old one-thread-per-row *sequential*, uncoalesced fold); it now
reads at ~227 GB/s — above the copy's read+write ceiling because a reduction is
read-only, i.e. memory-optimal. The outer-axis follow-up is now the **split-K
variant** (see `splitk_validate.cu` below) — regime-dependent, shipped as a
bench-gated schedule variant beside the baseline.

---

## `splitk_validate.cu` — split-K outer-axis reduction VARIANT (phase 2)

The first bench-gated schedule variant (ship-top-K policy — see
`docs/planning/foundational/11-variant-generators-backlog.md`): a two-kernel
split-K (`_splitk_partial` → workspace → `_splitk_combine`) beside the
single-pass baseline for the outer-axis Sum/Mean cell. Deterministic for a fixed
`chunk_rows`, no atomics — but a **different association** than the baseline
(`VariantFidelity::ReassociatedDeterministic`), so it is selectable only through
its honest contract, never silently.

Checks: baseline + split-K (ragged chunks) vs a CPU f64 oracle; degenerate
`n_chunks=1` **memcmp-identical** to the baseline (same association); run-to-run
determinism (memcmp).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 — **all 3 cases PASS** (oracle
relerr 0.0), and the sweep (fixed 0.27 GB read, two stable runs):

| cols | baseline (1 thread/col) | split-K | speedup |
| --- | --- | --- | --- |
| 256 | 12.4 GB/s (starved) | 229.9 GB/s | **18.5×** |
| 1024 | 53.0 GB/s | 241.6 GB/s | **4.6×** |
| 4096 | 204.1 GB/s | 243.8 GB/s | 1.19× |
| 16384+ | ~248 GB/s | ~244 GB/s | 0.98× |

**Regime-dependent — the variant thesis in one table.** The `StructureKey`
deliberately carries no literal extents, so all these shapes are ONE cell: the
within-cell winner depends on a runtime extent (`cols` vs GPU width). That makes
this a **launch-config-class decision for the runtime selector** (Fuel, per
call), exactly why the ship-top-K policy ships both kernels: winner-only would
bake in an 18× loss or a 2% regression depending on the bench shape. (The old
"118 GB/s" figure from the item-03 session also did not reproduce at cols=8192
— laptop clock variance; the starved regime is the real, stable gap.)

---

## `dag_validate.cu` — shared-interior DAG emitter (item 02)

Launches the diamond kernels — `out = g / (g + 1)` with `g = a * b`, the shared
product hoisted to one `tmp` — vs a host oracle. Validates the one thing catchable
only on device: that the DAG rewrite (emit a shared value once, reference it twice)
is a **no-op on the computed values**, and that the hoisted-`tmp` source compiles.
Two cells exercise both hoist paths:

- `baracuda_gen_diamond_f32_scalar` — `float tmp0 = (in0[i]*in1[i]); out[i] = (tmp0 / (tmp0 + 1.0));`
- `baracuda_gen_diamond_f32_co_v4` — per-lane scoped block `{ float tmp0 = (v0.x*v1.x); vo.x = (tmp0 / (tmp0 + 1.0)); }` (no cross-lane name collision).

Expected: `ALL PASSED` (`maxerr 0` — bit-exact; dedup changes text, not values).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **both cases PASS**
(scalar and per-lane vectorized hoist, bit-exact; PTX `.entry` present under a
headerless `-ptx` compile). The **fused-reduction epilogue** dedup (Softmax's
shared `exp(x-max)`), the DAG-based contract flops count, and the `region_to_op`
seam hash-cons are the item-02 follow-up (see `docs/planning/foundational/`).

---

## `packed_validate.cu` — packed f16/bf16 pair path (item 09 Stage 1)

Runs each **packed** kernel (`_co_v8`: half2/bf162 pairs, 128-bit accesses) and
its **scalar sibling** (the oracle, `_scalar` via a 2-byte-aligned cell) over a
corpus where input 0 sweeps **every 16-bit pattern** — all NaN payloads, ±Inf,
±0, every subnormal, max-finite — and requires the raw u16 outputs to be
memcmp-identical. Cases: `add` (Tier A native pair ops) and `relu_add` (Tier A
add + Tier B pair-scalarized relu), f16 + bf16.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 — **all 10 cases bit-identical (add, relu_add, neg, abs, sqr × f16, bf16 — incl. every NaN payload through the Tier-A intrinsics)**
over the full sweep; compute-sanitizer `initcheck` + `memcheck` **0 errors**.

### Bench — `packed_bench.cu` (honest finding)

f16 `add`, scalar vs packed, per-kernel launcher-realistic grids:

| n (halves) | regime | packed vs scalar |
| --- | --- | --- |
| 64k | launch-bound | parity (noise) |
| 1M–4M | **L2 / instruction-bound** | **+3–8% (consistent)** |
| 16M+ | DRAM-bound | parity |

**The item-09 brief's premise ("largest unclaimed memory-bound win") is
empirically wrong on sm_89**: the coalescer merges a warp's adjacent 2-byte lane
accesses into optimal transactions, so the *scalar* f16 kernel already runs at
the DRAM ceiling (~203 GB/s) for large coalesced elementwise. The packed path is
a **modest pure win**: bit-identical always, +3–8% where instruction issue is
the limiter, never a consistent regression, and fewer issue slots burned per
element (headroom for fused compute-heavy bodies). The deferred packed stages
(Tier-A transcendentals, packed reductions) should be built as **measured
variants** gated by the item-07 bench harness, not assumed wins.

---

## `contract_validate.cu` — skinny contraction go/no-go vs cuBLAS (item 10)

The generated `_contract_tll` cell ([M≤8,K]·[K,N], f32) vs a sampled CPU f64
oracle and `cublasSgemm` (row-major via the C^T = B^T·A^T mapping), then the
long-tail bench at M ∈ {1, 8}, K = N = 4096. Needs `-lcublas`.

**Last run (RTX 4070/sm_89, CUDA 13.3): correctness EXACT (0.0 rel err vs both
oracle and cuBLAS) — but the v1 skinny SIMT schedule is a perf NO-GO: ~62 GB/s
vs cuBLAS ~245 (0.25×).** Diagnosis: one thread per column = 4096 threads in 16
blocks — the SAME occupancy starvation the outer-axis reduction baseline showed
(12–53 GB/s starved), with a sequential K load-use chain on top; cuBLAS's M=1
path split-Ks internally. The proven in-repo fix is the split-K schedule (16×
on the reduction analogue) as a bench-gated **variant** of this cell — queued
as the node's first variant. Fourth instance of measure-don't-assume: this time
the gate protected us from shipping our own thesis kernel as the default.

**Split-K rematch (same harness, variant pair added):** all 6 correctness cases
PASS (splitk exact vs cuBLAS; degenerate `n_chunks=1` memcmp-identical to base
at both shapes), and the bench:

| M | base | split-K | cuBLAS | splitk vs cuBLAS |
| --- | --- | --- | --- | --- |
| 1 | 63 GB/s | **233 GB/s** | 245 GB/s | 0.95× |
| 8 | 74 GB/s | **217 GB/s** | 234 GB/s | 0.93× |

The variant closed the 4× gap to within ~5–7% of cuBLAS on the vendor's own
plain decode cell — inside/near the `MIN_FLIP_MARGIN` noise floor, so the
honest per-cell verdict is "vendor keeps the plain cell." The generated node's
winning ground, per the §1 long-tail thesis, is the **fused-epilogue** cell
(matmul+bias/act in ONE launch, epilogue folded into `_splitk_combine`) that
the vendor serves only as a two-kernel round trip — the next rematch.

**Fused-epilogue rematch (matmul_relu):** correctness exact (0.0 vs the vendor
round-trip), but **fusion did NOT win — 0.94–0.96×**. Structural finding: at
Tiny-M the output is tiny (16 KB at M=1), so the vendor's separate relu pass
costs ~2 µs and our ~5–7% GEMM gap eats it; epilogue fusion pays only when the
epilogue's traffic is large relative to the GEMM, which at Tiny-M it never is.
The contraction long tail lives elsewhere: **dequant-fused matmul** (int4/nf4
weights dequantized in-kernel — the real quantized-decode traffic),
irregular-K, and batched-many-tiny. Sixth measure-don't-assume instance.

---

## `audit_reduce_softmax.cu` — generated-vs-bespoke audit, round 1

Generated cells vs the hand-written `baracuda-kernels-sys` kernels, called
through their own `extern "C" _run` launchers (their path selection — what
dispatch actually calls). Compile with
`-std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" -I <kernels-sys>/kernels/include`.

**Round-1 results (RTX 4070/sm_89, all correctness PASS):**

| matchup | generated | bespoke | verdict |
| --- | --- | --- | --- |
| mean last-axis [8192²] | 248 GB/s | 234 GB/s | **GEN 1.06×** |
| sum axis-0 [65536×1024] | 242 GB/s (splitk) | 171 GB/s (legacy) | **GEN 1.41×** |
| softmax [4096²] | 235 GB/s (recompute) | 229 GB/s (smem) | **GEN 1.03×** |
| softmax [2048×16384] | 200 GB/s | **0.2 GB/s** (global fallback) | **GEN 884×** |

Notes: (1) the bespoke softmax fast path IS the smem row-cache — independently
confirming the earlier gate finding that recompute ≥ smem-cache on this card;
above 47 KB rows it collapses to an O(numel·extent) fallback the generated
kernel simply doesn't have. (2) **Extract-the-delta, first application — from a
LOSING kernel:** the bespoke legacy reducer (171 GB/s) beats our general-path
BASE (55 GB/s) at identical parallelism because it passes shape/strides **by
value in kernel params** (`DimsI32/I64` → constant bank) while ours re-reads
`shape[]/s0[]/so[]` from **global pointers every loop iteration**. Our split-K
still wins the cell, but by-value dims params are a legitimate technique to
extract into the general strided/reduction emitters (queued in the backlog).

---

## `int_validate.cu` — int ops (increment 0c)

Launches the increment-0c integer kernels against the **bespoke**
`binary_bitwise_*_int.cu` / `binary_logical_*_bool.cu` kernels (bit-exact
diff, included by absolute path like `audit_reduce_softmax.cu` does) and CPU
references: two's-complement models on the defined subset, exhaustive 256×256
for every 8-bit case, and the documented promote-then-truncate model for the
8-bit shifts.

**Regeneration:** these cells are **not yet emitted by the `bin/kernelgen.rs`
catalog** (the exception to the header note above). Generate them with the
library into `<outdir>`, then copy this harness beside them as usual:

```rust
use baracuda_kernelgen::ir::BinaryOp;
use baracuda_kernelgen::{generate, input, Cuda, OpDef};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};
use ElementKind::{I32, S8, U8};

let out = std::env::args().nth(1).expect("outdir");
// Scalar binary cell (align 4 keeps the vector classifier off; int dtypes
// take the scalar path regardless — pinned in the unit suite).
let key = |dt: ElementKind| {
    let a = OperandDesc::new(1, &[1 << 16], &[1], dt, 4);
    structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
};
let emit = |name: &str, dt: ElementKind, body| {
    let k = generate(&OpDef::elementwise(name, 2, &[dt], body), &key(dt), &Cuda);
    std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
};
for (n, b) in [("band", BinaryOp::BitAnd), ("bor", BinaryOp::BitOr),
               ("bxor", BinaryOp::BitXor), ("shl", BinaryOp::Shl), ("shr", BinaryOp::Shr)] {
    emit(n, I32, input(0).binary(b, input(1)));
}
for (n, b) in [("land", BinaryOp::LogicalAnd), ("lor", BinaryOp::LogicalOr),
               ("lxor", BinaryOp::LogicalXor)] {
    emit(n, U8, input(0).binary(b, input(1)));
}
for dt in [U8, S8] {
    emit("addw", dt, input(0) + input(1));
    emit("mulw", dt, input(0) * input(1));
}
emit("shl", U8, input(0).binary(BinaryOp::Shl, input(1)));
emit("shr", U8, input(0).binary(BinaryOp::Shr, input(1)));
emit("shr", S8, input(0).binary(BinaryOp::Shr, input(1)));
```

**Last run:** RTX 4070 Laptop (sm_89), 2026-07-03 — **ALL PASSED**:

- **i32 bitwise/shift** (`band`/`bor`/`bxor`/`shl`/`shr`): generated vs
  bespoke **bit-exact** over the edge cross + 65,536 randoms per op,
  **including the out-of-range shift amounts b = 0/31/32/33/-1/64/-32**.
  Observed (gen == bespoke, architecture-inherited): `1<<31 = -2³¹`,
  `1<<32 = 0`, `1<<33 = 0`, `1<<-1 = 0`, `1>>32 = 0`, `1>>-1 = 0`. The CPU
  two's-complement reference additionally matches on the defined subset
  (and/or/xor everywhere; shifts at b ∈ [0,31]).
- **u8 logical** (`land`/`lor`/`lxor`): exhaustive 65,536 pairs, generated vs
  bespoke bit-exact AND vs the CPU `(a != 0) OP (b != 0)` reference —
  including the normalization probe `2 && 4 == 1` (never the bitwise
  `2 & 4 == 0`).
- **u8/i8 wrapping add/mul** (`addw`/`mulw`): exhaustive 65,536 pairs vs a CPU
  wrapping reference (no bespoke elementwise int add/mul exists — CPU is the
  oracle).
- **8-bit shifts** (`shl`/`shr` u8, `shr` i8): match the documented promotion
  model (promote to int, C shift, store-truncate mod 2⁸) for b ∈ [0,31];
  i8 `shr` is ARITHMETIC (sign-replicating) — `-128 >> 7 == -1`.

## `coord_validate.cu` — `Coord` leaf (increment 0d)

Validates `ScalarExpr::Coord(axis)` (the output coordinate along `axis`, as a
float). Three bodies: a triu **mask-multiply** `x * (coord(1) >= coord(0) + k)`
(k = 0/-1/2, f32 + f64) diffed against the **bespoke** `triu` kernel
(`baracuda_triu_tril.cuh`, included by absolute path); a pure `iota` `coord(1)`;
and an alibi-slope `(coord(1) - coord(0)) * p0` (launch param) — the last two vs
a CPU reference. The generated kernels route to the STRIDED schedule (the Coord
body forces it) even on contiguous cells.

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.
Generate them with the library into `<outdir>`, then copy this harness beside
them:

```rust
use baracuda_kernelgen::ir::BinaryOp;
use baracuda_kernelgen::{coord, generate, input, konst, param, Cuda, OpDef};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

let out = std::env::args().nth(1).expect("outdir");
let write = |k: baracuda_kernelgen::GeneratedKernel| {
    std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
};
let key_1in = |dt: ElementKind| {  // one input + output
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
    structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89)
};
let key_0in = |dt: ElementKind| {  // zero inputs (pure coord) + output
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
    structure_key(OpCategory::UnaryElementwise, &[a], ArchSku::Sm89)
};
let triu = |name: &str, dt: ElementKind, k: f64| OpDef::elementwise(
    name, 1, &[dt],
    input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0) + konst(k)));
write(generate(&triu("triu_mask", ElementKind::F32, 0.0), &key_1in(ElementKind::F32), &Cuda));
write(generate(&triu("triu_mask_km1", ElementKind::F32, -1.0), &key_1in(ElementKind::F32), &Cuda));
write(generate(&triu("triu_mask_k2", ElementKind::F32, 2.0), &key_1in(ElementKind::F32), &Cuda));
write(generate(&triu("triu_mask", ElementKind::F64, 0.0), &key_1in(ElementKind::F64), &Cuda));
write(generate(&OpDef::elementwise("iota1", 0, &[ElementKind::F32], coord(1)),
               &key_0in(ElementKind::F32), &Cuda));
write(generate(&OpDef::elementwise("alibi", 0, &[ElementKind::F32], (coord(1) - coord(0)) * param(0)),
               &key_0in(ElementKind::F32), &Cuda));
```

Compile like `audit_reduce_softmax.cu` (the bespoke header needs the
preprocessor flags): `nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler
"/Zc:preprocessor /std:c++17" -I <kernels/include> coord_validate.cu`.

**Last run:** RTX 4070 Laptop (sm_89), 2026-07-04 — **ALL PASSED**:

- **`Coord` is bit-exact.** `iota` (`out = coord(1)`, column index to 4095, a
  coordinate axis > 2¹¹) and `alibi` (`(coord(1)-coord(0))*slope`) match the
  CPU definition **bit-for-bit** across every shape.
- **triu mask-multiply is VALUE-exact to bespoke, not bit-exact — and the gap
  is *precisely* the sign of zero.** The generated body is a mask-MULTIPLY
  (`in * (cond ? 1 : 0)`); bespoke `triu` is a SELECT (`cond ? in : 0`). On a
  masked-out **negative** entry the multiply yields `negative * 0.0f = -0.0`
  while the select stores `+0.0`. Across all f32/f64 shapes (incl. non-square
  37×53, degenerate 1×1, and coordinate axes > 2¹¹: 5000×33 / 33×5000) the
  generated output is `==`-equal to both bespoke and the mathematical
  definition, and **every** bit-difference was verified to be exactly that
  `-0.0`-on-masked-negative case (e.g. 84,489 of them at 5000×33, all
  accounted). A bit-identical `triu` needs a `Where`/select op (a future
  increment); the mask-multiply idiom is value-correct modulo signed zero.
  **Route implication for the eventual triu audit:** value-equal with `-0` on
  masked negatives — a consumer needing exact `+0` requires the select op.

## `reduction_upgrades_validate.cu` — reduction upgrades (increment 0e)

Validates the three 0e reduction additions against a CPU reference and, where a
bespoke sibling exists, against the hand-written `baracuda-kernels-sys` reduce
kernels (called through their `extern "C" _run` launchers, keepdim ABI):

1. **`ReduceOp::Prod`** — f32 (`reduce_prod_fp.cu`) and i32 (`reduce_prod_int.cu`,
   the widened `long long` accumulator + wrap-on-store). Bespoke siblings.
2. **Fused post-expression** — `norm2 = Sqrt(Sum(Sqr(x)))` (the `Sqr` pre-body
   folds, `Sqrt` post applies to the fold result via `red0`). Bespoke sibling
   `reduce_norm2_fp.cu`.
3. **Hetero output dtype** — `any` (`Sum(x≠0)` with a `Cmp*` post → `u8`) and
   `count` (`Sum(x≠0)` with the identity post → `i64`). No bespoke `OpKind`
   (Fuel has no Prod/Any/All/CountNonzero reduce dispatch — CPU is the oracle).

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.
Generate them into `<outdir>`, then copy the harness beside them:

```rust
use baracuda_kernelgen::ir::BinaryOp;
use baracuda_kernelgen::{generate, input, konst, reduced, Cuda, OpDef, ReduceOp, UnaryOp};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

let out = std::env::args().nth(1).expect("outdir");
let write = |k: baracuda_kernelgen::GeneratedKernel| {
    std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
};
// last-axis reduce cell: [256,128] f32 input, [256] output of `out_dt`.
let key = |out_dt: ElementKind| {
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let o = OperandDesc::new(1, &[256], &[1], out_dt, 256);
    structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89)
};
let key_uniform = |dt: ElementKind| {
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
    let o = OperandDesc::new(1, &[256], &[1], dt, 256);
    structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89)
};
// (1) Prod fp + int.
write(generate(&OpDef::reduction("prod", 1, &[ElementKind::F32], input(0), ReduceOp::Prod), &key_uniform(ElementKind::F32), &Cuda));
write(generate(&OpDef::reduction("prod", 1, &[ElementKind::I32], input(0), ReduceOp::Prod), &key_uniform(ElementKind::I32), &Cuda));
// (2) norm2 = Sqrt(Sum(Sqr(x))).
write(generate(&OpDef::reduction_post("norm2", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Sqr), ReduceOp::Sum, reduced(0).sqrt()), &key_uniform(ElementKind::F32), &Cuda));
// (3) hetero-out: any -> u8 (Cmp* post), count -> i64 (identity post).
let mut anyv = OpDef::reduction_post("anyv", 1, &[ElementKind::F32], input(0).binary(BinaryOp::CmpNe, konst(0.0)), ReduceOp::Sum, reduced(0).binary(BinaryOp::CmpGt, konst(0.0)));
anyv.out_dtype = Some(ElementKind::U8);
write(generate(&anyv, &key(ElementKind::U8), &Cuda));
let mut countv = OpDef::reduction("countv", 1, &[ElementKind::F32], input(0).binary(BinaryOp::CmpNe, konst(0.0)), ReduceOp::Sum);
countv.out_dtype = Some(ElementKind::I64);
write(generate(&countv, &key(ElementKind::I64), &Cuda));
```

Compile like `audit_reduce_softmax.cu` (the bespoke reduce headers want c++17):
`nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" -I <kernels/include> reduction_upgrades_validate.cu`.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **ALL 11 cases PASS**:

| case | vs CPU | vs bespoke |
| --- | --- | --- |
| `prod_f32` | relerr 4.3e-07 | relerr **0.0** (bit-exact) |
| `prod_i32` (wrap) | **bit-exact** (i64→i32) | **bit-exact** |
| `norm2_f32` | relerr 3.9e-08 | relerr **0.0** (bit-exact) |
| `any_u8` | **bit-exact** | — (no sibling) |
| `count_i64` | **bit-exact** | — (no sibling) |

Notes: (1) `prod_i32` exercises the i32 wrap-on-store from the widened `long long`
accumulator (3²⁰ ≈ 3.5e9 fits i64, wraps i32) — bit-exact to both the CPU
`(i32)(i64 product)` model and the bespoke i64-accumulator kernel (integer product
is exactly associative mod 2⁶⁴, so the block-tree and the bespoke sequential fold
agree bit-for-bit). (2) `prod_f32` / `norm2_f32` came out **bit-identical** to the
bespoke sibling on this corpus (relerr 0.0), and both are correctly-rounded-close
(< 1e-6) to the f64 oracle. (3) the hetero-out `any`/`count` have no bespoke
`OpKind` (see `fuel-cuda-backend/src/baracuda/reduce.rs`), so CPU is the oracle;
both bit-exact — `any` via the Cmp* post's exact 0/1 → u8, `count` via the float
accumulator → i64 store (exact while count ≤ 2²⁴).

## `multi_output_validate.cu` — multi-output elementwise (increment 1)

Launches the **generated** MULTI_OUTPUT kernels — one kernel writing N outputs
from a shared body-DAG, with cross-body CSE (the shared `dy` load / an interior
product emitted once, then N stores) — vs an f64 CPU oracle **per output**, on a
contiguous and a strided cell, plus a generated-vs-bespoke audit (the sibling
`binary_mul_backward_fp.cu` / `binary_div_backward_fp.cu` functor math, inlined).

Validates:

- **`mul_backward`** (3 in → 2 out: `da=dy·b`, `db=dy·a`) — the shared `dy` load
  hoists to one `tmp0` referenced by both stores; both outputs oracle-exact.
- **`div_backward`** (`da=dy/b`, `db=−dy·a/b²`) — the `dy/b` interior is shared
  (body 0's root AND body 1's interior), computed once; both outputs within a few
  f32 ULP of the oracle.
- **`fma_backward`** (3 outputs, one a plain **copy** of `dy` reusing the hoisted
  load) — all three exact.
- **strided cell** — col-major inputs, row-major outputs: both stores land at
  their own unraveled offsets (`oo0`/`oo1`).
- **determinism** — two runs of the multi-store are bit-identical.

**Run (from a VS dev shell):**

```sh
nvcc -O3 -arch=sm_89 multi_output_validate.cu -o multi_output_validate && ./multi_output_validate
compute-sanitizer --tool memcheck  ./multi_output_validate
compute-sanitizer --tool racecheck ./multi_output_validate
```

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **ALL 15 checks PASS**;
`compute-sanitizer memcheck` = **0 errors**, `racecheck` = **0 hazards** (the
multi-store to distinct buffers is race-free and in-bounds).

| case | vs f64 oracle | vs bespoke |
| --- | --- | --- |
| `mul_bw` da / db (contig) | **maxrel 0.0** (exact) | **bit-identical** (single multiply) |
| `mul_bw` da / db (strided) | **maxrel 0.0** | — |
| `mul_bw` determinism (2 runs) | — | **bit-identical** |
| `div_bw` da (contig) | maxrel 4.5e-08 | **bit-identical** (`dy/b` same formula) |
| `div_bw` db (contig) | maxrel 1.1e-07 | oracle-close, **not** bit-equal (see note) |
| `div_bw` da / db (bespoke) | maxrel ≤ 4.8e-08 | — |
| `fma_bw` da / db / dc-copy | **maxrel 0.0** (exact) | — (no 3-out sibling) |

Notes: (1) `mul_backward` is a single multiply per output, so the generated dual
store is **bit-identical** to the bespoke `MulBackwardFunctor` — a tie at the
memory wall on the contig fast path, exactly the audit prediction; the generator
additionally serves the **strided** cell (bespoke is contig-only). (2) `div_backward`
`db` differs by rounding: the generator shares the `dy/b` interior (`db =
−((dy/b)·a/b)`), the bespoke recomputes (`db = −(dy·a)/(b·b)`) — the interior-share
is the whole point (fewer ops/loads), and both land within ~1e-7 of the f64 oracle.
(3) no elementwise multi-output backward has a Fuel `OpKind` (Fuel splits
multi-output backward into per-output kinds, e.g. `FlashAttnBackwardQ/K/V`), so
these ship as generated AOT kernels with **no FKC contract** (honest miss — the
`return.outputs`/§5.5-bundle envelope needs a forest-pattern identity Baracuda
cannot yet advertise); the kernels generate and run correctly, proven here.

## `rowreduce_bw_validate.cu` — compound-backward RowReduce (increment 2)

The increment-2 proof vehicles: a fused RowReduce with a **second row-streamed
input** (softmax bw reads `y` AND `dy`) and **per-row saved-stat scalars** hoisted
once per row (layer_norm bw dx reads `mean`/`rstd` as `in_i[row]`). One block per
row, block-parallel tree reduce; the generated kernels are diffed against an f64
CPU oracle and the **bespoke** `softmax_backward_fp` / `layer_norm_backward_fp`
launchers (`baracuda-kernels-sys`, the path dispatch calls).

- **softmax bw**: `dx[j] = y[j]·(dy[j] - Σ_l y[l]·dy[l])` — `y`, `dy` both
  RowStreamed (`in_i[base+j]`); the row-dot is one block reduce (bespoke recomputes
  it per thread). Bespoke launcher `launch_softmax_backward_fp(dy, y, dx, …)` takes
  the **saved forward output `y`** (not recomputed).
- **layer_norm bw dx**: `x_hat=(x-mean)·rstd; dx = rstd·(dy - mean(dy) -
  x_hat·mean(dy·x_hat))` — `x`, `dy` RowStreamed; `mean`, `rstd` per-row scalars
  (stride `[1,0]`, hoisted). Bespoke `launch_layer_norm_backward_fp(dy, x, gamma=null,
  mean, inv_std, dx, …)` takes **mean + `inv_std` (= rstd)** indexed `[row]` with
  `stride_save=[1,0]` — the identical saved-stats convention (gamma=null ⇒ the
  dx-only path matching the generated epilogue).

**Run (from a VS dev shell):**

```sh
nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
     -I <kernels/include> rowreduce_bw_validate.cu -o rowreduce_bw_validate
./rowreduce_bw_validate                                  # correctness (5 shapes) + bench
compute-sanitizer --tool memcheck  ./rowreduce_bw_validate san   # generated kernels, small shapes
compute-sanitizer --tool racecheck ./rowreduce_bw_validate san
```

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **all 10 correctness
cases PASS**; `compute-sanitizer memcheck` = **0 errors**, `racecheck` = **0
hazards** (the block-reduce smem, the per-row-scalar hoist, and the dual
row-streamed loads are race-free and in-bounds).

Correctness (scale-relative L-inf vs f64 oracle; abs-diff vs the bespoke sibling —
per-element relative error is meaningless for backward grads that cancel to ≈0):

| shape | softmax_bw oracle / vs bespoke | layer_norm_bw oracle / vs bespoke |
| --- | --- | --- |
| 256×128 | 7.8e-08 / 7.5e-09 | 8.8e-08 / 3.6e-07 |
| 1000×777 (non-square) | 8.6e-08 / 1.9e-09 | 1.0e-07 / 3.6e-07 |
| 64×16384 (64 KB row, smemrow regime) | 9.0e-08 / 5.8e-11 | 1.1e-07 / 3.6e-07 |
| 4096×1024 (catalog cell) | 1.0e-07 / 1.9e-09 | 1.0e-07 / 4.8e-07 |
| 131072×32 (many rows, tiny k) | 1.1e-07 / 6.0e-08 | 1.0e-07 / 6.0e-07 |

The generated kernels match the shipped bespoke to **f32 precision** (abs-diff
≤ 6e-7 across every shape) and the f64 oracle to the same (worst-element error
≤ 1.1e-7 of the tensor's peak magnitude — clean f32-accumulation level even at
k = 16384, thanks to the block tree reduce).

**Extract-the-delta — the generator WINS decisively (not a tie).** The bespoke
backwards are one-thread-per-cell with an inner O(extent) recompute of the row
statistic (`Σ y·dy` / `sum_dxh`+`sum_dxhxh`) — O(numel·extent) total, and there is
**no smem/block-cooperative BW fast path** (only the *forward* softmax/layernorm
have one). The generated fused RowReduce does one block-parallel tree reduce per
row, so it is memory-bound where the bespoke is compute-bound:

| bench cell | gen GB/s | bespoke ms | gen speedup |
| --- | --- | --- | --- |
| softmax_bw 8192×2048 | 240 | 79.7 | **95×** |
| layer_norm_bw 8192×2048 | 170 | 1061 | **893×** |
| softmax_bw 2048×16384 | 140 | 1210 | **421×** |
| layer_norm_bw 2048×16384 (64 KB row) | 141 | 17 018 | **5976×** |

The gap widens with `k` (the recompute is quadratic in the reduced extent), so the
wide-row cell — exactly the smemrow-variant regime — is the generator's largest
win. The technique to extract for a bespoke follow-up is the one this generator
already embodies: **replace the per-thread row-statistic recompute with a
block-cooperative tree reduce** (the same lesson as the reduction/softmax-fwd
rewrites). No cliff, no loss to record.

**Fuel contract (honest miss, confirmed):** Fuel's JIT/FKC vocabulary (`OpTag`,
`fuel-kernel-seam-types`) is forward/functional only — it has **no `*Backward`
tag**. Autograd emits softmax/layernorm backward as atomic `Op::Fused(…_BACKWARD)`
nodes and `op_to_tag(Op::Fused) → None`, so they never enter a JIT region; the
registry backward matchers are stubbed (`canonical_pattern → None`). On the
Baracuda side `derive_pattern` rejects the RowReduce region (`NotElementwise`). So
these fused backwards emit **no contract** and stay AOT-only — the same honest miss
as the reduction family and the multi-output elementwise increment, no new panic
path.

## `view_validate.cu` — layout/shape views (item 01)

Validates `OpDef::views` — a fused op reading an INPUT through a layout change in
ONE pass, skipping a materialized `contiguize`/transpose copy (the §1
memory-optimal win). Two bodies, both routing to the STRIDED schedule (a viewed
read is non-contiguous — `build_plan` forces it, never vectorized/packed):

- **`relu_t`** — `out[i,j] = relu(x[j,i])`, input 0 read through `View::Permute{[1,0]}`.
  `x` is the PRODUCER buffer, physically `[N,M]` row-major contiguous; the emitter
  folds the transpose into address math as `o0 = c0*s0_1 + c1*s0_0` (**swapped
  strides** — iteration axis `d` reads producer stride `perm[d]`), the output
  offset unchanged. Diffed BIT-EXACT vs a CPU double reference AND vs the
  **bespoke** materialize-then-op path = `baracuda::contiguize(x^T)`
  (`baracuda_contiguize.cuh`, `launch_contiguize<4>`) THEN a contiguous relu — two
  kernels + a materialized transpose buffer + an extra DRAM round-trip.
- **`addb_t`** — `out[i,j] = x[j,i] + b[j]`, in0 transposed (`Permute`), in1 a
  per-column `[N]` bias broadcast over the row axis (`Identity` view; the key
  carries stride-0 on axis 0, so `o1 = c1*s1_1` drops the row term). Diffed
  bit-exact vs a CPU reference — the transpose remap composed with a key broadcast.

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.
Generate them into `<outdir>`, then copy the harness beside them:

```rust
use baracuda_kernelgen::ir::View;
use baracuda_kernelgen::{generate, input, Cuda, OpDef};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

let out = std::env::args().nth(1).expect("outdir");
let write = |k: baracuda_kernelgen::GeneratedKernel| {
    std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
};
// relu(x^T): x producer [N,M] dense (Permute operand 0 must have empty bcast).
let x = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
let relu_t = OpDef::elementwise("relu_t", 1, &[ElementKind::F32], input(0).relu())
    .with_views(vec![View::Permute { perm: vec![1, 0] }]);
write(generate(&relu_t, &structure_key(OpCategory::UnaryElementwise, &[x, o], ArchSku::Sm89), &Cuda));
// out[i,j] = x[j,i] + b[j]: in0 transposed, in1 a per-column bias broadcast (key
// bcast axis 0), Identity view.
let b = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::F32, 256);
let addb_t = OpDef::elementwise("addb_t", 2, &[ElementKind::F32], input(0) + input(1))
    .with_views(vec![View::Permute { perm: vec![1, 0] }, View::Identity]);
write(generate(&addb_t, &structure_key(OpCategory::BinaryElementwise, &[x, b, o], ArchSku::Sm89), &Cuda));
```

Compile (the bespoke `contiguize` header wants the MSVC conforming preprocessor):
`nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" -I <kernels/include> view_validate.cu`.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **ALL PASSED**,
`compute-sanitizer --tool memcheck` **0 errors** (the transposed read is strided —
no OOB). The generator WINS every shape: **one fused pass, no materialized copy**.

| cell | shape | gen==bespoke | gen==ref | gen ms | bespoke (contig+relu) ms | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| relu_t square | 512×512 | yes | yes | 0.014 | 0.057 | **4.01×** |
| relu_t wide | 384×1024 | yes | yes | 0.020 | 0.083 | **4.11×** |
| relu_t tall | 1024×384 | yes | yes | 0.021 | 0.081 | **3.93×** |
| relu_t row | 1×4096 | yes | yes | 0.007 | 0.181 | **27.0×** |
| relu_t large | 4096×4096 | yes | yes | 1.254 | 4.503 | **3.59×** |
| addb_t square | 512×512 | — | yes | — | — | (bit-exact) |
| addb_t wide | 384×1024 | — | yes | — | — | (bit-exact) |
| addb_t large | 4096×4096 | — | yes | — | — | (bit-exact) |

The win is structural: the bespoke path materializes `x^T` to DRAM then re-reads it
(2× the tensor traffic + a second launch); the generated kernel reads `x`
transposed in place and writes once. The degenerate `1×4096` row is a 27× blowout
because the bespoke contiguize + relu is dominated by launch/round-trip overhead
there. Bit-exact throughout — a transposed read is pure index arithmetic (no math
reordering), so there is no precision delta to record.

**Fuel contract (honest miss, confirmed against Fuel's sources):** a viewed op
emits **no contract**. The kernel computes `body(transpose(input))`, but Baracuda's
emitted pattern grammar (`pattern::PatternNode` = `Op` + `Bind`, no layout node, no
attrs channel) can only describe reading `Input(i)` at the iteration coordinate —
`derive_pattern` walks `op.body` alone and would silently drop the transpose.
Fuel's own grammar CAN express it (`fuel-kernel-seam-types` `PatternNode::Op { op:
OpTag::Permute, attrs: OpAttrs { perm } }` with a `perm` guard — the fkc §4.3 rule
for a load-bearing-attribute layout op, explicitly NOT `see_through`, whose skip is
a no-op stub in `fuel-graph jit.rs` today anyway), but Baracuda has no matching
`OpTag`/attrs vocabulary to author that guard, and the concrete-region direction
rejects layout re-emit outright (`fuel-graph runtime_fused.rs`: Transpose/Permute/
Reshape are `UnRepresentable`). So `contract()`/`derive_pattern` miss honestly
(typed `PatternError::ViewUnsupported`; the kernel still AOT-generates and runs) —
the Coord/multi-output precedent. A same-rank `Reshape`/`Identity` view is NOT
address-affecting and still advertises normally.

**rope — DEFERRED to the gather increment (#4).** The bespoke rope
(`baracuda_attention.cuh`) rotates pairs `(2i, 2i+1)`:
`y[2i] = x[2i]·cos θ − x[2i+1]·sin θ`, `y[2i+1] = x[2i+1]·cos θ + x[2i]·sin θ`. This
is NOT pure-stride-expressible, on three independent counts, each landing squarely
in #4: (1) **pair-partner cross-read** — each output reads BOTH lanes of its pair,
so the "odd" stream is the "even" stream at a **+1 element base offset**, and the
item-01 boundary is explicit that there is no `base_offset` field (a slice offset
is a runtime launch arg / gather, not a stride view); (2) **interleaved output** —
`y[2i]` and `y[2i+1]` scatter back into ONE buffer at stride 2, which the
MULTI_OUTPUT emitter (N distinct contiguous buffers) does not express; (3)
**θ = pos·base^(−2·pair/D)** needs a transcendental of a *feature* `Coord`
(`powf`, outside the item-0d `(float)c{d}` vocabulary), and the production path
(`rope_apply`) instead reads a precomputed cos/sin cache indexed by (position,
pair) — a GATHER, the definitional #4 case. Not forced.

## `gather_validate.cu` — GATHER (increment 4)

Validates `OpDef::read_index` (the `ReadIndex::Indexed` role) — the first
DATA-DEPENDENT access pattern: the gathered-axis coordinate `c{axis}` is replaced
by a value loaded from an integer index tensor. One strided emitter mechanism
covers the whole bespoke gather surface, distinguished only by the index
operand's broadcast mask + the OOB policy:

- **`gather`** — `out[r,c] = src[index[r,c], c]` (axis 0, **full-shape** i32/i64
  index). OOB policy `Skip` — the OOB output cell is left UNWRITTEN (bespoke
  `gather` `continue;`). Diffed BIT-EXACT vs `baracuda::indexing::launch_gather`
  (`baracuda_indexing.cuh`) AND a CPU reference.
- **`isel`** — `out[r,c] = src[idx[r], c]` (axis 0, **1-D** index broadcast over
  axis 1 ⇒ `gidx_off = c0*s1_0`, the bespoke 1-D `index_select` lookup). `Skip`.
  Diffed vs `launch_index_select`.
- **`emb`** — `out[n,d] = weight[ids[n], d]` (axis 0, 1-D ids). OOB policy
  `ZeroFill` — the OOB / negative row is zeroed (bespoke `embedding`). Diffed vs
  `baracuda::embedding::launch_embedding` (`baracuda_embedding.cuh`, `padding_idx`
  disabled). (The bespoke `padding_idx` — zero the row where `ids[n]==padding_idx`
  — is a per-op runtime scalar predicate deferred in v1; the harness disables it so
  only the OOB path is exercised.)
- **`gclamp`** — `out[r,c] = src[clamp(index[r,c],0,V-1), c]` (`Clamp`, a
  generator-only policy no bespoke op has). Diffed vs a CPU clamp reference.

The emitter emits the offset `o0 = (gidx_clamped)*s0_0 + c1*s0_1` — the runtime
index value replaces the loop coordinate on the gathered axis, matching bespoke's
`src_off = idx_val*stride_src[0] + coord[1]*stride_src[1]` exactly. **The load
address is always CLAMPED in-bounds**, so an OOB gather never issues an
out-of-range read; the OOB policy shapes only the WRITE (Skip predicates the
store, ZeroFill selects the fill). Negative indices are OOB (no PyTorch from-end
wrap) — bespoke parity, confirmed per kernel.

**OOB PROBES are the point.** Every run feeds negative + out-of-range indices and
requires the generated kernel to match the bespoke policy EXACTLY; the index dtype
rides the `entry_point` symbol (`gather_f32_i32` vs `gather_f32_i64`), never the
structure-key token.

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.
Generate them into `<outdir>`, then copy the harness beside them:

```rust
use baracuda_kernelgen::ir::OobPolicy;
use baracuda_kernelgen::{generate, Cuda, OpDef};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

let out = std::env::args().nth(1).expect("outdir");
let write = |k: baracuda_kernelgen::GeneratedKernel| {
    std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
};
let data = OperandDesc::new(2, &[128, 64], &[64, 1], ElementKind::F32, 256);
let outp = OperandDesc::new(2, &[128, 64], &[64, 1], ElementKind::F32, 256);
// gather: FULL-shape index (dense on every axis), i32 + i64.
let idx32 = OperandDesc::new(2, &[128, 64], &[64, 1], ElementKind::I32, 256);
let idx64 = OperandDesc::new(2, &[128, 64], &[64, 1], ElementKind::I64, 256);
let gk32 = structure_key(OpCategory::BinaryElementwise, &[data, idx32, outp], ArchSku::Sm89);
let gk64 = structure_key(OpCategory::BinaryElementwise, &[data, idx64, outp], ArchSku::Sm89);
write(generate(&OpDef::gather("gather", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::I32), &gk32, &Cuda));
write(generate(&OpDef::gather("gather", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::I64), &gk64, &Cuda));
write(generate(&OpDef::gather("gclamp", &[ElementKind::F32], 0, OobPolicy::Clamp, ElementKind::I32), &gk32, &Cuda));
// isel / emb: 1-D index broadcast over axis 1 (stride 0).
let idx1d = OperandDesc::new(2, &[128, 64], &[1, 0], ElementKind::I32, 256);
let k1d = structure_key(OpCategory::BinaryElementwise, &[data, idx1d, outp], ArchSku::Sm89);
write(generate(&OpDef::index_select("isel", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::I32), &k1d, &Cuda));
write(generate(&OpDef::embedding("emb", &[ElementKind::F32], ElementKind::I32), &k1d, &Cuda));
```

Compile (the bespoke headers want the MSVC conforming preprocessor):
`nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" -I <kernels/include> gather_validate.cu`.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **ALL PASSED**,
`compute-sanitizer --tool memcheck` **0 errors** under EVERY OOB policy (Skip /
ZeroFill / Clamp, with negative + out-of-range indices — the load-bearing check:
the address-clamp keeps every OOB gather read in-bounds).

| cell | shape | policy | gen==bespoke | gen==ref | gen ms | bespoke ms | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather tiny | 6×4 (V=6) | Skip | yes | yes | 0.0063 | 0.0074 | 1.17× |
| gather mid | 512×64 (V=128) | Skip | yes | yes | 0.0069 | 0.0192 | **2.78×** |
| gather large | 2048×128 (V=1000) | Skip | yes | yes | 0.0168 | 0.1220 | **7.28×** |
| gather V=1 | 64×8 (V=1) | Skip | yes | yes | 0.0068 | 0.0064 | 0.94× |
| gather(i64) mid | 512×64 (V=128) | Skip | yes | yes | — | — | (bit-exact) |
| index_select tiny | 10×4 (V=6) | Skip | yes | yes | — | — | (bit-exact) |
| index_select mid | 1024×96 (V=200) | Skip | yes | yes | — | — | (bit-exact) |
| embedding tiny | 10×4 (V=6) | ZeroFill | yes | yes | — | — | (bit-exact) |
| embedding mid | 2048×128 (V=512) | ZeroFill | yes | yes | — | — | (bit-exact) |
| gather-clamp tiny | 6×4 (V=6) | Clamp | (no bespoke) | yes | — | — | (bit-exact) |
| gather-clamp mid | 1024×64 (V=256) | Clamp | (no bespoke) | yes | — | — | (bit-exact) |

Bit-exact vs bespoke on every cell (a gather is pure address arithmetic — no math
reorder, so no precision delta), and the OOB probes confirm the Skip/ZeroFill
policies match bespoke to the byte. Perf was expected to TIE at the memory wall
(both are plain gathers) — instead the generator **wins 2.8×–7.3× on the mid/large
cells** and ties at tiny/degenerate: the bespoke gather unravels the linear index
into a `coord[MAX_RANK]` array and re-reads dims/strides from `DimsI32`/`DimsI64`
structs per element, while the generated kernel carries dims BY VALUE as flattened
scalars (extraction #1) with the rank fully unrolled — the same lesson as the
audit's general-path win, now on the gather path.

**Fuel contract (honest miss, AOT-only, confirmed against Fuel's sources):** a
gathered op emits **no contract** (`PatternError::GatherUnsupported`). Two
independent blockers: (1) the index operand's dtype is **unkeyable** — Baracuda's
`StructureKey` has no per-operand dtype FIELD (a single operand-0 dtype, "v1
assumes a uniform operand dtype"), so the token does not name the index operand
as i32 vs i64. (The dtype's byte size leaks *incidentally* into that operand's
`vec_width` — a full-shape i32 index vectorizes wider than an i64 one, so the
`gk32`/`gk64` tokens above actually differ there — but that side-channel is
unreliable: it collapses to equal for the 1-D index of `index_select`/`embedding`
where both are `Scalar`. So the token neither reliably distinguishes nor is meant
to distinguish index dtype.) Fuel's gather admissibility is instead an explicit
per-operand dtype TUPLE — key `[T, U32, T]` (`fuel-dispatch fkc/cpu_link.rs`
fixes `indices` as a U32 slot, `out: passthrough(source)`). A contract keyed on
`T` alone would let Fuel bind an i32-index kernel to an i64/U32 call — no keyed
field guards it. (2) The `Op`+`Bind` pattern grammar cannot carry the
gather `axis`/OOB semantics; Fuel names `OpTag::Gather`/`IndexSelect` but their
identity rides `OpAttrs.axis` + a `fdx.gather.kind` enum Baracuda has no vocabulary
for. So the kernels ship **AOT-only** — the Contraction-node precedent — until the
per-operand-dtype key extension lands (a `STRUCTURE_KEY_VERSION` bump = a Fuel
propose-first).

**rope — RE-EVALUATED with gather in hand, STILL DEFERRED (partial closure).**
Gather closes ONE of rope's three blockers: the precomputed cos/sin cache read
indexed by (position, pair) IS now an `Indexed` read (the production `rope_apply`
path). But the other two remain out of #4's reach: (a) the interleaved output —
`y[2i]`, `y[2i+1]` scatter into ONE buffer at stride 2 — needs SCATTER (#5, a
`ScatterIndexed` output role), which the read-side gather does not provide; and
(b) the pair-partner cross-read `(2i ↔ 2i+1)` is a +1 *element* base-offset slice,
not an index-tensor gather (there is still no `base_offset` operand field). So rope
stays deferred to #5 (scatter) + the base-offset/slice operand work; gather alone
does not close it. Not forced.
