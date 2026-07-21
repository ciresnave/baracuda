# KISS-Ops Recipe Grammar — spec for the kiss-ref recipe-evaluation API

**Version:** recipe-grammar **v1** (the "recipe-grammar version" leg of the kiss-ref consumption triple: `(KISS-Ops spec version, recipe-grammar version, kiss-ref semver)`).
**Purpose:** the stable input contract for kiss-ref's `eval_recipe(dag, inputs) -> (outputs, per_node_det)`. Baracuda AND Fuel both emit this canonical form, so a single evaluator makes kiss-ref the shared semantics reference for both producers.
**Source of truth:** Baracuda's emitter `crates/baracuda-kernelgen/src/recipe.rs` (`semantics_dag`) + the flat-DAG structure `pattern.rs` (`PatternNode`). This doc is the consolidated, evaluator-facing extract.

> **Two layers, don't conflate.** kiss-ref evaluates the **logical flat-DAG** (a decoded Rust structure: nodes + edges + logical attrs). The **wire encoding** of that DAG (§6.4-0009/0010 + §6.19 OpAttrs, the bytes) is being pinned by the 3B/Appendix F work (mlgheozs) and is referenced in §7 — the evaluator does not parse wire bytes; a decoder produces the DAG it walks.

---

## 1. Canonical flat-DAG structure (§6.4-0009)

A recipe is a DAG of nodes; each node is one of:
- **`Op { name, op_attrs }`** — a KISS-Ops operation (§3 vocabulary) with logical attributes.
- **`Bind(input_index)`** — a leaf binding the fused op's `input[input_index]` (Baracuda text spelling: `in{i}`).

**Edges.** Each node carries an ordered list of **child edges** (operand node-indices). **Operand order is significant and part of identity** (positional, §6.4-0009). The canonical order is **§6.4-0010 byte-lex** over each operand's canonical subtree serialization, and **commutativity is canonicalization-only** (§6.2-0005: `add(a,b) == add(b,a)` → one canonical form; positional ops preserve order). *(Baracuda's internal matching-canonicalization is separate and NOT used for signature/wire order — see the 3B thread.)*

**`Reduced(i)` is NOT a leaf — it is a child edge to the i-th fold node.** Inside a reduction/contraction/scan/RowReduce epilogue, `Reduced(i)` references the i-th structural fold node (the K-sum for matmul, the reduce/scan fold, or RowReduce stage `i`). A bare elementwise body has no fold nodes.

---

## 2. Op vocabulary

Notation per op: **`name{attrs}` — edges — DetClass**. DetClass is per-node from `(op, attrs)`; enums in §3.

### Scalar / elementwise atoms
| Node | Attrs | Edges | DetClass |
|---|---|---|---|
| `Bind(i)` | input_index | 0 | (leaf) |
| `const` | `bits` (raw value bit-pattern) | 0 | exact-byte |
| `add` / `sub` / `mul` / `div` | — | 2 | exact-byte (IEEE) |
| `<unary>` | — | 1 | exact-byte OR ULP (per atom, §3) |
| `<binary>` | — | 2 | exact-byte OR ULP (per atom, §3) |
| `select` | — | 3 (cond, a, b) | exact-byte (raw-bit-preserving) |
| `iota` | `axis` | 0 (source) | exact-byte |
| `runtime_scalar` | `slot` (param index) | 0 (source) | exact-byte |

`<unary>`/`<binary>` names are the KISS-Ops atom set (Baracuda `unary_kiss_name` / `binary_kiss_name`): exact-byte atoms (neg, abs, floor, ceil, round_even, bitwise, rem_floor, max_prop/min_prop, copysign, nextafter, comparisons, complex algebraic) vs **ULP** transcendentals (exp, log, sin, cos, sqrt, erf, atan, atan2, lgamma — §6.8).

### Structural / fold nodes
| Node | Attrs | Edges | DetClass |
|---|---|---|---|
| `matmul` | `roles` = `<lhs>.<rhs>` over `{b,m,n,k}` (Batch/FreeM/FreeN/ContractedK) | 2 (lhs, rhs) | instance: exact-byte if a fixed schedule is pinned, else order-invariant/nondeterministic (default: tolerance) |
| `reduce` | `monoid`, `axes`, `keepdim` | 1 (pre) | **instance:** `max`/`min` → exact-byte; float `sum`/`prod` → order-invariant/nondeterministic |
| `prefix_scan` | `monoid`, `axis`, `exclusive` | 1 (pre) | same instance rule as `reduce` |
| `flip` | `axis` | 1 | exact-byte |
| `reduced_count` | `axes` | 0 (shape-derived leaf) | exact-byte |
| `gather` | `axis`, `oob`, `index_dtype` | 2 (data, index) | exact-byte |
| `scatter` | `axis`, `combine`, `oob`, `index_dtype` | 2 (value, index) | **instance:** float `atomic-add` → order-invariant/nondeterministic |

**`Mean` is not a monoid** — it composes as `div( reduce{monoid=sum, axes, keepdim}(pre), reduced_count{axes} )`. **Integer `Mean` is not expressible** (rounds; no single-dtype cell).

---

## 3. Attribute enumerations

- `monoid` ∈ `{sum, prod, max, min}` (Mean is composed, not a monoid).
- `axes` = `last` (empty-mask trailing-axis default, resolved against the interface rank) **or** `0x<hex>` (raw axis bitmask). *(Reader-side also accepts §6.7-0005 `rall`/`rlast`; emit is `last`/`0x<hex>`.)*
- `keepdim` ∈ `{kd, nokd}`.
- `exclusive` ∈ `{excl, incl}`.
- `oob` ∈ `{skip, clamp, zero_fill}`.
- `combine` ∈ `{atomic-add}` (Baracuda floor; assign/atomic-max/atomic-min are honest misses today).
- `index_dtype` ∈ `{u32, i32, i64}`.
- `roles` chars ∈ `{b, m, n, k}` (rank-2 `mk.kn`, batched `bmk.bkn`).
- `DetClass` = `{ exact-byte, ulp(bound), order-invariant/nondeterministic }` — returned **per node** from `(op, attrs)` (a flat per-op-name table is WRONG for the monoid-parameterized ops).

---

## 4. Fold-node / epilogue semantics

- **Elementwise:** body over `Bind`/`const`/source-op leaves; no fold node.
- **Reduction / Contraction / Scan:** one fold node + an epilogue over `Reduced(0)`. Contraction's fold is `matmul`; a fused bias/activation is ordinary elementwise nodes over `Reduced(0)` (bias rides `Bind(2)`).
- **RowReduce:** staged fold nodes; stage `i` → `Reduced(i)`; a later stage's `pre` may read earlier `Reduced(j<i)`; the epilogue reads `Reduced(0..n)` **and** the full-width row-streamed `Bind`s (RmsNorm/LayerNorm shape).

---

## 5. Honest-miss boundary (evaluator returns "unexpressible", never a guess)

Not in v1 grammar → the emitter yields no recipe; the evaluator should reject rather than fabricate: **pooling/window, sort, im2col**; **integer `Mean`**; a **fused gather body** (v1 = identity read `data[index]` only); **scatter combine ∉ {atomic-add}**; an **index_dtype ∉ {u32,i32,i64}**. Dtype-breadth (**FP8 e4m3/e5m2, bool, complex c32/c64, batched matmul**) is the kiss-ref coverage roadmap — gate per-op via the ledger (§6).

---

## 6. Evaluator contract (the API to build against)

```
eval_recipe(dag: FlatDag, inputs: &[Tensor<T>]) -> Result<(Vec<Tensor<T>>, Vec<DetClass>), Error>
```
- Walk the canonical DAG; dispatch each `Op{name, attrs}` to its kernel (attrs → monoid/axis/oob/combine/roles/…).
- `Bind(i)` → `inputs[i]`.
- `Reduced(i)` → the output of the i-th fold node (staged).
- Return the output tensor(s) **+ the per-node `DetClass`** (from `(op, attrs)`) for the consumer's comparator selection.
- `dtype = T` (the tensor element type). **Float lane first** (f16/bf16/f32/f64); other lanes gated on coverage.
- `Error` on an unexpressible node (§5) or an op/dtype not `support(op, dtype) == Done`.

**Precision:** kiss-ref is hardware-precision (`evaluation_precision: compute-dtype`, libm) → a same-precision differential: **byte-exact on exact-byte nodes, ULP-band on transcendentals** (libm ≠ CUDA intrinsics), composing with Baracuda's device-diff comparator.

---

## 7. Versioning & wire alignment

- **Triple:** `(KISS-Ops spec version, recipe-grammar version [this doc], kiss-ref semver)`. Baracuda pins a kiss-ref commit; each op's migration gates on the coverage ledger `support(op, dtype) == Done`.
- **Wire (3B/Appendix F, mlgheozs):** the DAG's canonical serialization is §6.4-0009/0010 + §6.19 OpAttrs — `Bind` = `0x00` + `input_index` u32-LE; §6.4-0010 byte-lex canonical child order; commutativity canonicalization-only; per-node OpAttrs = §6.19 positional blob (u32-LE outer byte-length). The **shape-oracle §6.20 DimExpr** grammar (`SameAs`/`Extent`/`Const`/`Param`/arith, u16-LE child lengths, `0xFF` last-sentinel) is a **separate** grammar for *shape rules*, not value semantics — do not conflate it with this recipe grammar.
