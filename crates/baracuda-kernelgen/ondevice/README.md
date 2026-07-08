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
rides the `entry_point` symbol (`gather_f32_i32` vs `gather_f32_i64` vs
`gather_f32_u32`), never the structure-key token.

- **`gatheru`** — the Model-A `u32`-index gather (the FUEL-FACING variant; Fuel keys
  the gather index operand as a fixed U32 slot `[T, U32, T]`, so `u32` is the variant
  that carries the keyed contract). NO bespoke sibling (bespoke `launch_gather` is
  i32/i64-templated), so it is diffed vs a CPU ref AND cross-checked against the i32
  kernel on the same non-negative index values (must be bit-identical there).

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
// u32-index gather (the Fuel-facing Model-A variant; FULL-shape U32 index).
let idxu = OperandDesc::new(2, &[128, 64], &[64, 1], ElementKind::U32, 256);
let gku = structure_key(OpCategory::BinaryElementwise, &[data, idxu, outp], ArchSku::Sm89);
write(generate(&OpDef::gather("gather", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::U32), &gku, &Cuda));
// isel / emb: 1-D index broadcast over axis 1 (stride 0).
let idx1d = OperandDesc::new(2, &[128, 64], &[1, 0], ElementKind::I32, 256);
let k1d = structure_key(OpCategory::BinaryElementwise, &[data, idx1d, outp], ArchSku::Sm89);
write(generate(&OpDef::index_select("isel", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::I32), &k1d, &Cuda));
write(generate(&OpDef::embedding("emb", &[ElementKind::F32], ElementKind::I32), &k1d, &Cuda));
```

Compile (the bespoke headers want the MSVC conforming preprocessor):
`nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" -I <kernels/include> gather_validate.cu`.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **ALL PASSED**
(incl. the new `gather(u32)` cases: `u32==i32 yes` bit-exact cross-check + `u32==ref
yes` across tiny/mid/large/V=1, Skip OOB matched), `compute-sanitizer --tool
memcheck` **ERROR SUMMARY: 0 errors** under EVERY OOB policy (Skip / ZeroFill /
Clamp, with negative + out-of-range indices, i32/i64/u32 index dtypes — the
load-bearing check: the address-clamp keeps every OOB gather read in-bounds; for
the u32 path the `< 0` branch is statically dead, the `>= gext` bound still fires).

| cell | shape | policy | gen==bespoke | gen==ref | gen ms | bespoke ms | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gather tiny | 6×4 (V=6) | Skip | yes | yes | 0.0063 | 0.0074 | 1.17× |
| gather mid | 512×64 (V=128) | Skip | yes | yes | 0.0069 | 0.0192 | **2.78×** |
| gather large | 2048×128 (V=1000) | Skip | yes | yes | 0.0168 | 0.1220 | **7.28×** |
| gather V=1 | 64×8 (V=1) | Skip | yes | yes | 0.0068 | 0.0064 | 0.94× |
| gather(i64) mid | 512×64 (V=128) | Skip | yes | yes | — | — | (bit-exact) |
| gather(u32) tiny/mid/large/V=1 | 6×4 … 2048×128 | Skip | u32==i32 yes | u32==ref yes | — | — | (bit-exact; Fuel-facing) |
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

## `scatter_validate.cu` — SCATTER + ATOMIC_HISTOGRAM (increment 5)

Validates `OpDef::write_index` (the `WriteIndex::ScatterIndexed` role) — the
write-side mirror of #4's gather: the OUTPUT store's scattered-axis coordinate
`c{axis}` is replaced by a value from an integer index tensor, and the store
becomes a `WriteCombine` op. **DETERMINISM is the core discipline** (see the
module header): duplicate/OOB indices write the same cell from many threads, so
the combine's algebra decides the op's determinism class:

- **scatter (Assign, unique idx)** — `out[idx[r,c], c] = upd[r,c]`. Unique targets
  ⇒ race-free ⇒ deterministic. Diffed BIT-EXACT vs `launch_scatter` + a CPU ref.
- **integer scatter_add (i32)** — `out[idx[r,c], c] += upd[r,c]`. Integer atomicAdd
  is exact + associative ⇒ order-**independent** ⇒ deterministic. BIT-EXACT vs
  `launch_scatter_add` + CPU ref. **Ships unconditionally** (the safe primary).
- **bincount (i32)** — `counts[x[i]] += 1` (the ATOMIC_HISTOGRAM representative).
  Integer counts ⇒ deterministic. BIT-EXACT vs `launch_bincount` + CPU ref.
- **FP scatter_add** — the determinism SPLIT. Float atomicAdd is
  **run-to-run non-deterministic** (order-varying + non-associative). So the BASE
  lower() is the deterministic **gather-sum** (`_scatter_gathersum_`: one thread
  per output cell scans the update domain, sums matching values in a fixed order,
  NO atomics) — BIT-EXACT vs a CPU ref summed in the same order. The ATOMIC
  scatter (`_strided_`) is the gated `Nondeterministic` VARIANT — value-correct
  within an FP tolerance vs the f64 ref, never bit-exact, never silently selected.

**OOB PROBES** feed negative + out-of-range indices; the generated kernel clamps
the write address in-bounds and skips the OOB target (bespoke `continue;`) — so
`memcheck` is the load-bearing check (0 errors = no OOB scatter), and `racecheck`
must be 0 hazards on the deterministic kernels (a real race there is a correctness
bug) while treating the atomic variant's atomicAdd as legitimate.

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.
Generate them into `<outdir>`, then copy the harness beside them:

```rust
use baracuda_kernelgen::{generate, generate_variants, Cuda, OpDef};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

let out = std::env::args().nth(1).expect("outdir");
let write = |k: &baracuda_kernelgen::GeneratedKernel| {
    std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
};
let (f32, i32) = (ElementKind::F32, ElementKind::I32);
// rank-2 value cells: updates[128,64], index[128,64] full-shape, dst[*,64]; axis 0.
let upd = OperandDesc::new(2, &[128, 64], &[64, 1], f32, 256);
let idx = OperandDesc::new(2, &[128, 64], &[64, 1], i32, 256);
let dst = OperandDesc::new(2, &[128, 64], &[64, 1], f32, 256);
let key = structure_key(OpCategory::BinaryElementwise, &[upd, idx, dst], ArchSku::Sm89);
write(&generate(&OpDef::scatter("scatter", &[f32], 0, i32), &key, &Cuda));
// FP scatter_add: base (gather-sum) + the Nondeterministic atomic variant.
for v in generate_variants(&OpDef::scatter_add("scatter_add", &[f32], 0, i32), &key, &Cuda) {
    write(&v.kernels[0]);
}
// integer scatter_add (deterministic base).
let iupd = OperandDesc::new(2, &[128, 64], &[64, 1], i32, 256);
let ikey = structure_key(OpCategory::BinaryElementwise, &[iupd, iupd, iupd], ArchSku::Sm89);
write(&generate(&OpDef::scatter_add("scatter_add", &[i32], 0, i32), &ikey, &Cuda));
// bincount (rank-1 counts).
let x = OperandDesc::new(1, &[4096], &[1], i32, 256);
let cnt = OperandDesc::new(1, &[64], &[1], i32, 256);
let bkey = structure_key(OpCategory::UnaryElementwise, &[x, cnt], ArchSku::Sm89);
write(&generate(&OpDef::bincount("bincount", i32), &bkey, &Cuda));
```

Compile (the bespoke headers want the MSVC conforming preprocessor):
`nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" -I <kernels/include> scatter_validate.cu`.
Sanitize with `compute-sanitizer --tool {memcheck,racecheck} ./scatter_validate san`.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **ALL PASSED** (16
correctness cases); `compute-sanitizer memcheck` = **0 errors**, `racecheck` = **0
hazards** (the load-bearing checks — no OOB scatter under the negative/out-of-range
probes; the deterministic kernels are race-free and racecheck treats the atomic
variant's atomicAdd as legitimate).

| class | case | vs CPU ref | vs bespoke | determinism |
| --- | --- | --- | --- | --- |
| scatter Assign (unique idx) | 100×32 M=128, 64×64 M=100 | **bit-exact** | **bit-exact** | deterministic |
| integer scatter_add | 128×64 M=64, 256×16 M=40 | **bit-exact** | **bit-exact** | deterministic (int add assoc.) |
| bincount | N=4096 B=64, N=65536 B=256 | **bit-exact** | **bit-exact** | deterministic (int counts) |
| FP scatter_add **gather-sum BASE** | 128×32 M=48, 64×8 M=16 | **bit-exact** (j-order) | — | **deterministic default** |
| FP scatter_add gather-sum vs f64 | — | rel ≤ 1.8e-6 | — | — |
| FP scatter_add **atomic VARIANT** | — | rel ≤ 3.7e-6 (f64) | within FP tol | **Nondeterministic** |

The FP-atomic variant's **run-to-run non-determinism is DEMONSTRATED**: two launches
of the identical config differed in 56/1536 (and 12/128) output cells, maxabs
1.5e-5 — genuinely order-nondeterministic (not merely reassociated), confirming it
must ship as the gated `VariantFidelity::Nondeterministic` variant while the
gather-sum base is the reproducible default. bincount / integer scatter_add /
Assign are all bit-exact to both the bespoke sibling and the CPU reference (integer
add is exactly associative, so the atomic interleave doesn't affect the result).

**Fuel contract (honest miss, AOT-only, confirmed against Fuel's sources):** a
scattered op emits **no contract** (`PatternError::ScatterUnsupported`), for the
gather reasons plus the determinism discipline: (1) the index dtype is **unkeyable**
in Baracuda's single-operand-0-dtype token, while Fuel keys `scatter_add`/`index_add`
as `[T, U32, T, T]` (`fuel-dispatch fkc/cpu_link.rs`: `base`, fixed U32 `indices`,
`src`, `passthrough(base)` out) — a token keyed on `T` alone could bind the wrong
index dtype. (2) The `Op`+`Bind` pattern grammar can't carry the scatter
`axis`/OOB/combine; Fuel names `OpKind::ScatterAdd`/`IndexAdd` (`fuel-ir dispatch.rs`)
but has **no bare `Scatter`, no `Bincount`/`Histogram` op-kind, and no scatter-reduce
mode enum** — so `scatter`/`bincount`/AtomicMax-Min are a *double* honest miss
(net-new vocabulary). (3) **Determinism** — an FP-atomic scatter's honest contract
must set `determinism: nondeterministic` (`fuel-dispatch fkc/schema.rs`, one of
`bitwise` | `same_hardware_bitwise` | `nondeterministic`), which by Fuel's precision
coherence rule (`fkc/validate.rs` Rule 9) ALSO obligates
`precision.bit_stable_on_same_hardware: false` + `audited: true` — a coupled block
Baracuda does not yet author. The determinism flip is spelled + ready
(`VariantFidelity::determinism_str` → `nondeterministic`) for when the
per-operand-dtype key extension lands (a `STRUCTURE_KEY_VERSION` bump = a Fuel
propose-first, the same gate as gather).

**rope — RE-EVALUATED with scatter in hand, 2 of 3 blockers now CLOSED (STILL
DEFERRED on the third).** #4 closed the cos/sin cache read (an `Indexed` gather);
**#5 closes the interleaved output** — `y[2i]`, `y[2i+1]` writing into ONE buffer at
stride 2 is now expressible as a `ScatterIndexed` write (a stride-2 destination
address is exactly the write-side substitution this increment adds). The **one
remaining blocker** is the pair-partner cross-read `(2i ↔ 2i+1)`: each output reads
BOTH lanes of its pair, i.e. the "odd" stream is the "even" stream at a **+1
*element* base offset** — a slice, needing a `base_offset` operand field that no
increment provides yet (it is a runtime launch-arg slice, not a stride view or an
index gather). So rope is now 2/3 closed; the last third is a small base-offset/slice
operand (NOT another access-pattern increment). Not forced.

## `scan_validate.cu` — prefix scan (increment 6)

Validates `Access::Scan` — a cumsum/cumprod/cummax/cummin along the innermost
(contiguous) axis. Two forms ship:

- **serial-fold BASE** (`_scan_{sum,prod,max,min}[_rev][_excl]`) —
  `VariantFidelity::BitIdentical`, thread 0 walks the axis in order (the honest
  deterministic bit-reference). Diffed **bit-exact** (memcmp / NaN-aware) vs a CPU
  **float-serial** oracle scanned in the MATCHING direction.
- **block-scan VARIANT** (`..._blockscan`) — a Kogge-Stone warp scan
  (`__shfl_up_sync`) + cross-warp exclusive-offset carry, re-emitted **inline**
  (`smem_scan` does not exist in kernelgen; the source is headerless), chunked so a
  `k > blockDim` row threads its running carry across tiles. **FP `Sum`/`Prod` only**
  (Max/Min + integer ride the base). `ReassociatedDeterministic` — the warp tree
  reassociates, so it is selectable only through an honest contract, never silently,
  and there is **no bit-identical degenerate config** (unlike split-K, even a single
  blockDim-wide chunk reassociates — the degenerate config is within-ULP of the base).

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.
Regenerate the `.cu` sources with the library dump tool, then copy the harness
beside them:

```sh
SCAN_OUT=<outdir> cargo test -p baracuda-kernelgen dump_scan_sources -- --ignored --nocapture
cp crates/baracuda-kernelgen/ondevice/scan_validate.cu <outdir>/
nvcc -O3 -arch=sm_89 <outdir>/scan_validate.cu -o <outdir>/scan_validate && <outdir>/scan_validate
```

The dump tool (`cuda::scan_tests::dump_scan_sources`) writes the 16 f32 base cells
(4 combines × incl/excl × fwd/rev), the f64 base, and the 4 block-scan variants
(Sum/Prod × incl/excl) — the exact `#include` set the harness names.

**Sanitizers** (small shapes via the `san` argv — the block-scan smem carry +
`__syncthreads` make racecheck/synccheck/initcheck load-bearing, not just memcheck):

```sh
compute-sanitizer --tool memcheck  ./scan_validate san
compute-sanitizer --tool racecheck ./scan_validate san
compute-sanitizer --tool synccheck ./scan_validate san
compute-sanitizer --tool initcheck ./scan_validate san
```

**Extract-the-delta audit vs the bespoke naive scan** (`scan_cumsum_fp.cu`, called
through its `baracuda_kernels_scan_cumsum_f32_run` launcher) — the header form:

```sh
nvcc -O3 -arch=sm_89 -std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" \
     -DWITH_BESPOKE -I <kernels-sys>/kernels/elementwise -I <kernels-sys>/kernels/include \
     scan_validate.cu -o scan_validate_bespoke && ./scan_validate_bespoke
```

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **RESULT: ALL PASSED**
(52 checks pure; +2 the bespoke audit); `compute-sanitizer` **memcheck / racecheck /
synccheck / initcheck all 0 errors / 0 hazards** on the `san` shapes (the block-scan
smem carry + the two `__syncthreads` per chunk are race/sync/init-clean).

- **serial BASE bit-exact** — all 4 combines × incl/excl × fwd/rev, memcmp-exact vs
  the float oracle on both `8×13` and `64×200`; f64 base memcmp-exact vs a double
  oracle.
- **FP boundary semantics** — signed-zero (an all-`−0.0` row sums to `+0`, bits
  `0x00000000` — the `0.0f`-seeded fold); NaN propagation (Max/Min NaN-propagating
  via `v != v`; Sum/Prod poison downstream, upstream unaffected — all 4 combines);
  empty row (`k=0` leaves the output untouched); single-element (incl=in,
  excl=identity).
- **block-scan VARIANT** — run-to-run **memcmp-identical** (deterministic) on every
  **device-launched** cell (the four **forward** f32 Sum/Prod × incl/excl blockscans);
  **within-ULP** of an f64 oracle to a depth-aware `max(1e-5, 5e-7·√k)` bound
  (a 16384-deep f32 sum legitimately carries ~1.4e-5 — the reassociated tree, being
  deterministic, only sizes the *value* tolerance for deep rows); degenerate
  single-chunk within-ULP of the serial base (relerr ≤ 3.9e-6). Multi-warp `64×16384`
  (64 KB row — crosses warps AND chunks) is the device-only carry-propagation check.
  The f16/bf16/f64 **forward** blockscans share this exact warp/carry algorithm (the
  acc-width + `__half2float`/`__float2half` convert is the same primitive the
  device-validated f64 base and the packed-f16 elementwise path already exercise), so
  they are covered by equivalence, not a separate launch.

> **Post-review (2026-07-04) — value-preserving emitter fixes, prior run still holds.**
> Three adversarial-review fixes landed after this run; none change any output, so the
> ALL-PASSED result above stands: (1) the Max/Min identity is now emitted header-light
> via `__int_as_float(0xff800000u)` / `__longlong_as_double(...)` instead of the
> `<cmath>` `INFINITY` macro (same ±inf bit pattern; **nvcc-verified to compile clean to
> an sm_89 cubin headerless** — matches the reduce path's no-`INFINITY` discipline); (2)
> the warp-exclusive `wexc` shuffle is now emitted only on the exclusive path (dead in
> inclusive kernels — no output change); (3) the **reverse** block-scan variant is
> **declined to the serial base in v1** (`scan_blockscan_variant` returns `None` for
> `reverse`): it is traced-correct but was never device-launched, and an AOT scan is an
> honest miss (no Fuel contract), so the validator is the ONLY gate — a reassociated
> path ships only once device-validated. Reverse scans run the BitIdentical base (17×
> bespoke). **Follow-up:** device-validate + re-enable reverse block-scan; add explicit
> f16/bf16 block launches.

**Extract-the-delta — the generator WINS decisively (and ties on correctness).** The
bespoke scan family (`scan_axis_kernel`) is **one-thread-per-cell**, each thread
re-scanning its own O(extent) prefix → **O(numel·extent)** total. The generated
serial base is memcmp-**exact** to it (same naive math order — forward and reverse
inclusive) but one-thread-**per-row** (O(numel)); the block-scan variant is
cooperative. On `4096×4096` f32 cumsum (0.13 GB read+write):

| kernel | technique | ms | GB/s (read+write) | vs bespoke |
| --- | --- | --- | --- | --- |
| gen block-scan variant | Kogge-Stone warp + carry | **0.590** | **227.4** | **43×** |
| gen serial base | one thread per row | 1.461 | 91.9 | **17×** |
| bespoke (naive) | one thread per cell | 25.325 | 5.3 | 1× |

The block-scan variant reads at **227 GB/s — the copy-bandwidth ceiling** (memory
optimal, same ~227 the reduce/rowreduce rewrites hit), 43× the bespoke; even the
deliberately-unparallelized base is 17× the bespoke because it drops the quadratic
per-cell rescan. No losing cell to record; the winning technique the generator
already embodies is the cooperative scan (the same lesson as the reduction/softmax
rewrites). **De-scoped from v1** (queued as follow-ups): the block-scan `Max`/`Min`
variant (a `(value, has)`-flag warp scan — exactly associative, would be
`BitIdentical`), integer block-scan, the **reverse** block-scan (traced-correct but
declined pending device validation — reverse rides the base), and the non-inner scan
axis.

**Fuel contract (honest miss, AOT-only):** a scan emits **no contract** — neither
`contract.rs` nor `pattern.rs` has any Scan/Cumsum/Prefix vocabulary and Fuel
exposes no Scan/Cumsum `OpTag`, so `derive_pattern` rejects it as `NotElementwise`
before any body walk and `contract()` returns `None` (the Reduction/RowReduce/
Contraction precedent — pinned by `contract::tests::scan_is_an_honest_miss_no_contract`).
Scan is a **stronger miss** than those: before this increment it could not even be
represented; after it, the AOT kernel generates and runs (proven here) but still
crosses no Fuel wire. Keying stays **additive** (`baracuda-kernels-types` UNTOUCHED —
the `_blockscan` entry_point disambiguates the variant on the wire, never the token).

## `window_validate.cu` — sliding-window pooling (increment 7)

Validates `Access::Window` — a sliding-window reduction (the POOLING family:
`max_pool` / `min_pool` / `sum_pool` / `avg_pool`) along the innermost (contiguous)
axis, with `size` / `stride` / `dilation` / `pad_lo` / `pad_hi` and a
`count_include_pad` divisor policy (Mean only). One form ships:

- **serial-fold BASE** (`_window_{max,min,sum,mean}[_cip]`) —
  `VariantFidelity::BitIdentical`, **one thread per OUTPUT element** (grid-stride)
  walking the local window at input tap `p = o*stride − pad_lo + kk*dilation` for
  `kk in 0..size`. Each output is an **independent fixed-order fold** (no cross-output
  dependence, unlike the scan prefix), so it is naturally parallel AND
  bit-reproducible. Padding taps are **skipped** for Max/Min (padding never wins) and
  **contribute the additive identity** for Sum; `avg_pool` divides by `size`
  (`count_include_pad`) or the valid-tap count. NaN propagates through Max/Min via the
  `v != v` probe. No variant (a pool has no reassociation to offer — it is already
  memory-optimal, see the perf row).

The window geometry is baked into each kernel as **compile-time literals**, and the
kernel name encodes op/dtype/combine but NOT geometry, so the dump tool gives each
`(combine, geometry)` cell a geometry-encoding `op_name` (`mx_a`, `av_c`, …) to keep
entry symbols distinct.

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.

```sh
WINDOW_OUT=<outdir> cargo test -p baracuda-kernelgen dump_window_sources -- --ignored --nocapture
cp crates/baracuda-kernelgen/ondevice/window_validate.cu <outdir>/
nvcc -O3 -arch=sm_89 <outdir>/window_validate.cu -o <outdir>/window_validate && <outdir>/window_validate
```

The dump tool (`cuda::window_tests::dump_window_sources`) writes the 8 f32 cells
(max/min/sum + avg exclude/include, across a stride/dilation/pad spread) and 2 f64
cells (avg + max, dilated + padded) — the exact `#include` set the harness names.

**Sanitizers** (small shapes via the `san` argv):

```sh
compute-sanitizer --tool memcheck  ./window_validate san
compute-sanitizer --tool racecheck ./window_validate san
compute-sanitizer --tool synccheck ./window_validate san
compute-sanitizer --tool initcheck ./window_validate san
```

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **RESULT: ALL PASSED**
(12 checks); `compute-sanitizer` **memcheck / racecheck / synccheck / initcheck all 0
errors**.

Every cell below is **device-launched** (not equivalence-covered) unless noted:

- **max/min/sum-pool bit-exact** — memcmp-exact vs a CPU float oracle mirroring the
  tap order, across **stride 1 and >1**, **dilation 1 and >1**, and **pad both ends**
  (`s2 d1 p0`, `s1 d1 p1`, `s2 d2 p2`). The `s2 d2 p2` geometry's edge windows
  **overhang both ends** (span 5, pad 2) — the padding-skip path, device-exercised.
- **avg_pool** — both divisor policies device-launched: **exclude-pad** (divide by the
  valid-tap count, `cnt>0`-guarded) and **include-pad** (`_cip`, divide by the `size`
  literal), across stride 1 and the dilated `s2 d2 p2`. Diffed **within 1 ULP** for the
  FP divide — and observed **bit-exact (0 ULP)** in this run (the CPU oracle divides in
  the same float order).
- **f64 oracle-exact** — avg + max at `s2 d2 p2` memcmp-exact vs a double oracle
  (0 ULP). The f16/bf16 pools use the same **float accumulator + fold as the
  device-validated f32 window cells** (only the loads/stores differ, through the
  `__half2float`/`__float2half` convert primitive the packed-f16 elementwise path
  already exercises), so they are **covered by equivalence, not a separate launch**.
- **NaN propagation** — a planted NaN inside a `max_pool` window makes every covering
  output NaN (`v != v`), the others unaffected; device-launched, and the full output
  is additionally memcmp-exact vs the NaN-aware oracle.

**Extract-the-delta — vs the memory-bandwidth ceiling (NOT a math-matched bespoke).**
The bespoke fixed-window `MaxPool1d`/`AvgPool1d` (`crates/baracuda-kernels/src/pool/
{max,avg}_pool1d.rs`) ride **cuDNN's Nd-pooling descriptor** — an **opaque library**
path with **no exposed `_run` launcher and no fixed-window `.cu`** whose math order we
could match bit-for-bit. The bespoke pool `.cu` kernels that DO expose `_run`
launchers — `adaptive_{avg,max}_pool*`, `fractional_max_pool*`, `lp_pool1d` — are all
DIFFERENT reductions/windowing schemes (non-uniform windows, or an Lp p-norm), none a
math-matched sibling for fixed-window max/avg/sum pooling. So there is nothing to
memcmp against; pooling is memory-bound, so the figure of merit is GB/s vs the copy
ceiling:

| kernel | technique | ms | GB/s (read+write) | vs copy ceiling |
| --- | --- | --- | --- | --- |
| gen `max_pool` 2× | one thread per output | 1.75 | **230.6** | **102%** |
| `cudaMemcpy` D2D | copy | — | 225.4 | 100% |

`max_pool` 2×-downsample on `8192×8192` f32 reads at **230.6 GB/s — the
copy-bandwidth ceiling** (memory optimal; the ~102%-of-copy figure reflects the pool
writing only half the input on a 2× downsample). No losing cell to record; the
generator saturates bandwidth, which the opaque cuDNN path cannot beat on a
memory-bound op. (A math-matched bespoke comparison is impossible until a bespoke
fixed-window pool `.cu` with a `_run` launcher exists.)

**Fuel contract (honest miss, AOT-only):** a window emits **no contract** — neither
`contract.rs` nor `pattern.rs` has any Pool/Window vocabulary and Fuel exposes no
Pool/Window `OpKind` (the pool family rides bespoke cuDNN, opaque), so
`derive_pattern` rejects it as `NotElementwise` before any body walk and `contract()`
returns `None` (the Reduction/Scan/Contraction precedent — pinned by
`contract::tests::window_is_an_honest_miss_no_contract`). Keying stays **additive**
(`baracuda-kernels-types` UNTOUCHED — the window params ride the `OpDef` + the
`_window_<combine>[_cip]` entry_point, never the token).

**De-scoped from v1** (queued follow-ups, documented in `Access::Window`): im2col
(dimension EXPANSION, not reduction), causal_conv1d (needs a weight operand → windowed
contraction), interpolate/bilinear (Coord-computed weights, 2-D window), N-D /
multi-axis windows, overlap-backward (rides atomics / gather-sum), and the non-inner
window axis. The `in_len → out_len` window arithmetic is a **runtime-launch-arg caller
precondition** (the structure key abstracts numeric extents away — the same trust
level as RowReduce's `k`/`n_out`); the output operand's LAYOUT (forward-dense
contiguous, downsampled) IS keyed and gate-checked.

## `sort_validate.cu` — row sort / argsort (increment 8, SORT_PERM)

Validates `Access::RowSort` — a row sort / argsort along the innermost (contiguous)
axis (the `sort` / `argsort` / `msort` family). Each output row is a permutation of
its input row under a **total order on `(key, original-index)` pairs** (index
tie-break ⇒ every pair is distinct ⇒ a **unique** sorted sequence, so the result is
deterministic, algorithm-independent, and **stable** = ascending original index within
equal keys). Two single-output ops share the variant: `row_sort` (values output,
dtype-preserving) and `row_argsort` (`I32` index output). Two forms ship:

- **rank-sort BASE** (`_rowsort_{asc,desc}_stable[_idx]`) —
  `VariantFidelity::BitIdentical`, **one thread per OUTPUT element** (grid-stride):
  each thread scans its row computing the element's RANK under the total order
  (O(k²), no smem, no `__syncthreads`, **any k**), then writes it to `out[base+rank]`.
  Stable by construction; the correctness reference.
- **bitonic pair-sort VARIANT** (`…_bitonic`) — also `VariantFidelity::BitIdentical`
  (a pair sort is a pure permutation — no FP arithmetic, so **NO** FP-only gate,
  unlike the scan blockscan; int sorts ride the same network). One block per row, the
  whole `next_pow2(k)`-padded row staged in dynamic smem as `(key, index)` pairs,
  sorted by a bitonic network; the values writeback gathers **raw** input bits through
  the final permutation. **Launch contract (`launch_note`, k ≤ 1024):** blockDim a
  multiple of 32, ≤ 1024; dynamic smem = `next_pow2(k) × (sizeof(acc)+4)` bytes.

**NaN convention (PINNED, PyTorch):** NaN compares **greater than every non-NaN** —
ascending ⇒ NaN block **last**, descending ⇒ NaN block **first**. NaN-vs-NaN and
`-0.0`-vs-`+0.0` are key-ties resolved by index. The values output is a **raw-bit
permutation** (it gathers original storage bytes), so NaN payloads and `-0.0` signs
are preserved exactly (`memcmp`-checkable). The bitonic **pad sentinel** is the
MAXIMUM of the pair order — a **quiet NaN** for ascending FP (`+inf` would be wrong: a
real NaN sorts *after* it under NaN-greatest), `-inf` for descending FP, and the type
extreme for integers — all emitted **header-light** (`__int_as_float(0x7fc00000u)` /
`__longlong_as_double(…)`, never the `INFINITY` macro).

**Regeneration:** these cells are **not** in the `bin/kernelgen.rs` catalog.

```sh
SORT_OUT=<outdir> cargo test -p baracuda-kernelgen dump_sort_sources -- --ignored --nocapture
cp crates/baracuda-kernelgen/ondevice/sort_validate.cu <outdir>/
nvcc -O3 -arch=sm_89 <outdir>/sort_validate.cu -o <outdir>/sort_validate && <outdir>/sort_validate
```

The dump tool (`cuda::sort_tests::dump_sort_sources`) writes 34 cells: `{f32,f64,i32}`
× `{asc,desc}` × `{sort,argsort}` × `{base,bitonic}`, plus i64 asc sort+argsort and
f16/bf16/f32s asc values-sort (base+bitonic) — the exact `#include` set the harness
names.

**Sanitizers** (small shapes via the `san` argv — race/sync are **load-bearing** for
the smem bitonic swaps + per-phase barriers; initcheck covers the global in/out
buffers — it does **not** see dynamic-shared-memory reads, so the smem pad cells are
protected by the memcmp oracle + the pad-tie invariant cells below, not by initcheck):

```sh
compute-sanitizer --tool memcheck  ./sort_validate san
compute-sanitizer --tool racecheck ./sort_validate san
compute-sanitizer --tool synccheck ./sort_validate san
compute-sanitizer --tool initcheck ./sort_validate san
```

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **RESULT: ALL PASSED**
(212 device-launched checks + 1 host-side contract assertion, the bitonic k > 1024
refusal); `compute-sanitizer` **memcheck / racecheck / synccheck / initcheck all 0
errors** (racecheck: 0 hazards).

**Post-review emitter fix (re-validated on device):** the review caught the base rank
sort truncating its tie index — which is also the **load address** — to `int` while
claiming "any k" (an OOB read past 2³¹; a rank collision past 2³²). The tie index is
now `long long` end-to-end (`pair_lt(acc, long long, acc, long long)`; the bitonic's
`int sidx[]` widens losslessly), so the **values-sort base is genuinely any-k**;
**argsort** keeps the inherent `k ≤ 2³¹−1` precondition (its I32 index output cannot
represent more — documented on the constructor). The results above were re-run against
the fixed emitter output.

**Mutation-check record (each broken → its test failed → restored → green):** stable
panic drop → `unstable_rejected`; body-`Input(0)` gate drop → `composed_body` +
`param_body`; desc comparator flip → desc golden; tie-break `ia<ib`→`ia>ib` →
**61 on-device memcmp fails**; per-phase `__syncthreads` delete → **54 on-device
fails and 56 racecheck hazards**; asc pad qNaN→+inf → **4 asc-NaN bitonic fails**;
(post-review) output-layout assert weakened to `!flipped`-only →
`non_contig_output_rejected` + `broadcast_output_rejected`; `n_operands` assert
neutralized → `wrong_operand_count_rejected`.

Every cell below is **device-launched** (memcmp-exact vs a CPU oracle that implements
`pair_lt` EXACTLY — order-adjusted keys, NaN-greatest, ascending-index tie-break, then
a raw-byte value gather) unless explicitly labeled *equivalence-covered*:

- **random rows, k = 1000** (non-pow2, near the 1024 cap) — f32 / f64 / i32 × asc+desc
  × sort+argsort × base+bitonic: **memcmp-oracle**, **base ≡ bitonic** (the D4 unique-
  permutation claim), **argsort ∘ gather ≡ values-sort** (mutual consistency), and
  **run-to-run determinism** (memcmp of two launches) — all device-launched.
- **edge k = 1, 5, 33, and exactly 1024** (pad-heavy small non-pow2; and pow2 == k with
  zero pads, one full block) — f32 asc+desc, base+bitonic; plus **already-sorted +
  reverse-sorted** rows (the bitonic worst paths), k = 512.
- **stability witness** (tie-heavy keys from `{0,1,2}`, k = 257) — the argsort indices
  are **strictly ascending within every equal-key run**, both directions, f32 + i32
  (device-checked against the input-order rule).
- **NaN-planted** (multiple NaNs, distinct payloads, interior positions) — asc ⇒ NaN
  block last, desc ⇒ NaN block first; **payload bits preserved** (raw memcmp vs the
  oracle); pins the qNaN-pad-vs-real-NaN tie invariant. f32 + f64, base+bitonic.
- **mixed −0.0/+0.0** — a key tie broken by index; **sign bits preserved** (memcmp).
  f32 asc+desc, base+bitonic.
- **extreme-value rows** containing real `INT_MAX`/`INT_MIN` (i32 asc) and real
  `−inf`/`+inf` (f32 desc) — pins the **pad-tie invariant** (a real element equal to
  the pad key has index < k < every pad index ⇒ sorts before all pads ⇒ the k real
  elements occupy `[0,k)`), device-exercised through the bitonic path.
- **k = 1500 (> 1024)** — the **base ONLY** is device-launched (proves the any-k rank
  sort); the harness **refuses** the bitonic variant for k > 1024 — a host-side
  **contract-reject** mirroring the `launch_note` (labeled, not a silent launch).
- **dtypes** — i64 asc sort+argsort device-launched. **f16 / bf16 / f32-strict** asc
  values-sort (base+bitonic) device-launched and oracle-checked (the f16/bf16 key uses
  the `__half2float`/`__bfloat162float` convert primitive; f32-strict uses the `double`
  accumulator). Their **descending and argsort variants are *equivalence-covered***:
  they share the identical acc/convert primitive and the order/argsort logic already
  device-validated on f32/f64/i32 — not folded into the pass count. **i64 desc
  (sort+argsort) is likewise equivalence-covered**: the desc comparator inversion is
  device-validated on i32, the 64-bit key staging on i64 asc, and the `INT64_MIN`/`MAX`
  extreme literals are the shared type-extreme path — also not in the pass count.

**Extract-the-delta — vs the bespoke stable `msort` (`baracuda_sort.cuh`, STABLE=1).**
`msort` is the one bespoke sort whose semantics are math-matchable (its `descending`
flag maps to our `Desc`). Built with `-DWITH_BESPOKE` (`#include`s the bespoke `sort.cu`);
on **NaN-free** inputs, k = 1000:

| comparison | result |
| --- | --- |
| **values** vs msort (ties included, asc+desc f32; asc i32) | **bit-exact** ✓ |
| **indices** vs msort on **distinct-key** rows (asc+desc f32, asc i32) | **bit-exact** ✓ |
| **indices** vs msort on **tie** rows (f32 asc+desc) | **differ — 64/64 rows** (delta) |

**Headline finding (a semantics delta, not a bug):** our **values** output matches the
bespoke `msort` bit-for-bit (ties included, both directions), and our **indices**
match on distinct-key rows — but on **tie** rows the two argsort permutations differ on
**every** row. Our `(key, original-index)` pair-sort produces the **input-order-stable**
permutation (verified bit-exact, for every cell above, against a CPU `std::sort` over
`(key, original-index)` pairs under the `pair_lt` total order — equivalent to a
`stable_sort` because the index tie-break makes every pair distinct); the bespoke
bitonic STABLE tie-break
(`cmp_swap_needed`: `ascending_block ? a_idx<b_idx : a_idx>b_idx`, `baracuda_sort.cuh`
:173/180) is stable-by-value but is **not input-order-preserving** for the index
output. There is exactly one input-order-stable permutation, so this is a real
convention difference — **ours is the stable one** (matching `torch.sort(stable=True)`).
NaN rows are a further documented delta: bespoke treats NaN as an equality tie
(network-position-dependent, NOT PyTorch NaN-last), while ours pins NaN-last(asc)/
NaN-first(desc) — device-confirmed on a planted-NaN row.

**Performance (f32, 4096 × 1024, sm_89):**

| kernel | technique | ms | Gelem/s | GB/s | vs |
| --- | --- | --- | --- | --- | --- |
| gen `rowsort` base | one thread per output, O(k²) rank | 14.56 | 0.29 | 2.3 | — |
| **gen `rowsort` bitonic** | one block per row, smem bitonic | **1.42** | **2.96** | **23.7** | **10.3× base** |
| bespoke `msort` | block-bitonic (STABLE) | 2.25 | 1.87 | 14.9 | — |

The generated **bitonic variant WINS**: **1.59× faster than the bespoke `msort`**
(same algorithm class, one block per row) and **10.3× faster than our own O(k²) rank
base** (the base is the correctness reference / the any-k long-row path, never the
perf claim). No losing cell to record.

**Fuel contract (honest miss, AOT-only):** a sort emits **no contract** — neither
`contract.rs` nor `pattern.rs` has any Sort/ArgSort vocabulary and Fuel exposes no
Sort/ArgSort `OpTag` (sort already rides the bespoke kernels
`crates/baracuda-kernels/src/sort/*`, so kernelgen SORT_PERM is AOT-only by
construction — a *stronger* miss than scan/window, like pooling↔cuDNN). `derive_pattern`
rejects it as `NotElementwise` before any body walk and `contract()` returns `None`
(pinned by `contract::tests::{sort,argsort}_is_an_honest_miss_no_contract`). Keying
stays **additive** (`baracuda-kernels-types` UNTOUCHED — the order/stable/argsort/variant
tokens ride the `OpDef` + the `_rowsort_{asc,desc}_stable[_idx][_bitonic]` entry_point,
never the structure-key token).

**De-scoped from v1** (queued follow-ups, documented in `Access::RowSort`): `topk`
(needs a k-select parameter + truncated writeback; bespoke covers it), single-kernel
`(values, indices)` dual output (blocked on hetero multi-out — a permutation is not a
`ScalarExpr` body), `argsort` I64 indices (bespoke pegs i32), a >1024 fast path
(multi-block merge / CUB — CUB needs headers + a workspace, violating the header-light
+ fixed-ABI envelope; the rank base is the honest any-k answer), `stable=false` (the
pair-sort makes stability free — an unstable variant would be dead keying), non-inner /
multi-axis sort, `sparsemax` fusion, and S8/U8 dtypes. The bitonic `k ≤ 1024` bound is
a **runtime-launch precondition** (the structure key abstracts numeric extents away —
the same trust level as smemrow/blockscan), harness-enforced + on-device-validated,
with NO emitted guard beyond `k == 0`.
