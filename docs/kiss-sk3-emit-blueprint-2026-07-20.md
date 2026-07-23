# sk3 emit — Baracuda implementation blueprint

**Status: APPLIED (2026-07-22, branch `feat/sk3-codec-bump`).** sk3 was formally adopted 2026-07-21 (RFC `C:\Projects\KISS\rfcs\sk3-gemm-precision-coordinates.md`, all sign-offs in, §6.17 pin folded); Eric greenlit the Baracuda bump ("Now, on a branch"). The plan below executed as written, with two implementation resolutions the code surfaced (recorded in §Resolutions at the bottom): (1) the `F32Strict` fold lives at the TOKEN codec only — the `StructureKey` STRUCT keeps `F32Strict` as the in-process MathPrecision carrier (`plan.dtype` reads `key.dtype`, so a derivation-side fold would have silently downgraded the strict-reduce kernels); (2) the kernelgen skinny-contraction kernel's `F32Strict` accumulator changed `double`→`float` to honor the §4.2 `accumulation_type`↔`<acc>` pin and the F32Strict element contract ("full IEEE 754 binary32 multiply-add throughout"). Golden regen with KISS + Fuel still pending (coordinates when the KISS spec clauses complete).

---

## Grammar (folded, signed off)

```
sk2 gem:            c<m><n><k>/<kdiv>[/b<class>]
sk3 gem non-batched: c<m><n><k>/<kdiv>/<wdt>/<acc>/<out>/<mp>
sk3 gem batched:     c<m><n><k>/<kdiv>/b<class>/<wdt>/<acc>/<out>/<mp>
```
- `<batch>` = `b<class>`, **conditionally present** (iff batched) — Baracuda already emits this (`structure_key.rs` to_token ~L847–849, from_token ~L917–927).
- `<wdt>` = operand-1 (weight) dtype token; `<acc>` = accumulator/compute dtype; `<out>` = output dtype — all closed-set dtype tokens.
- `<mp>` = math-precision, `{st}` bit-stable / `{rm}` reduced-mantissa (never begins with `b`, so no collision with `b<class>`). Derived from `MathPrecision` (§6.17): `F32`/`F32Strict` → `st`; `Tf32` → `rm`.
- **FP8 variant-explicit spellings:** `Fp8E4M3` → `e4m3fn` (OCP, SATFINITE, no-inf, max 448) — a rename from bare `e4m3`; `e4m3fnuz` (AMD) reserved, unused. `Fp8E5M2` → **`e5m2` (bare, UNCHANGED)** — it is already the variant-explicit IEEE-style spelling (inf/NaN, max 57344); `e5m2fnuz` (AMD finite) reserved, unused. So only `e4m3→e4m3fn` is a rename; `e5m2` stays. (Matches KISS reference codec, cross-checked with g6uuwo0p on `feat/sk3-implementation`.)
- Non-gem cells: unchanged except the global `sk2`→`sk3` version prefix.

---

## Code changes

### 1. `crates/baracuda-kernel-vocab/src/structure_key.rs`

**(a) Version.** `STRUCTURE_KEY_VERSION: u16 = 2` → `3` (L52). Update the doc comment (L47–51) to describe the sk3 gem-precision coordinate add. Update the golden token asserts in-file (the `sk2|…` string literals at ~L626, L1411, L1426, L1440, L1497, L1699, L1705, and the `STRUCTURE_KEY_VERSION == 2` assert at ~L1490).

**(b) `ContractionKey`** (currently `{ m, n, k, k_div, batch }`, ~L199–213): add
```rust
wdt: ElementKind,   // operand-1 (weight) dtype
acc: ElementKind,   // accumulator / compute dtype
out: ElementKind,   // output dtype
mp:  MathPrecision, // st (bit-stable) / rm (reduced-mantissa)  — or a small MathPrecisionKey enum
```
(Consider a 2-value `MpCode { St, Rm }` rather than the full `MathPrecision` enum, since the key only needs the mantissa/determinism axis; map `MathPrecision → MpCode` at derive time.)

**(c) `to_token`** (the `if let Some(c) = self.contraction` block, ~L837–850): after the existing `c<mnk>/<kdiv>` and the optional `/b<class>`, append `/<wdt>/<acc>/<out>/<mp>`:
```rust
token.push_str(&format!("/{}/{}/{}/{}",
    dtype_code(c.wdt), dtype_code(c.acc), dtype_code(c.out), mp_code(c.mp)));
```
Order: batch (if present) BEFORE the precision group, matching the grammar. Add `const fn mp_code(MpCode) -> &'static str { St => "st", Rm => "rm" }`.

**(d) `from_token`** (the `parts.get(9)` contraction block, ~L908–947): after parsing `classes`, `div`, optional `b<class>`, parse the four trailing precision components in order `<wdt>/<acc>/<out>/<mp>` via `dtype_from_code` and `mp_from_code` (reject unknown). Note: the current parser treats the batch as the last optional component — reorder so batch is parsed BEFORE the precision group, and the precision group is REQUIRED for a gem cell (an sk3 gem token without it is malformed → decline).

**(e) `derive_contraction`** (~L491+): populate the four new fields from the GEMM operands + plan:
- `wdt` = operands[1].dtype; `out` = output operand dtype.
- `acc` = the accumulator dtype from `PrecisionGuarantee.accumulator` (int8/int4/bin→s32, fp8/f16/bf16/f32→f32, f64→f64 — `baracuda-cutlass/src/types.rs:354-362`, already computed by `plan.rs:68-85`).
- `mp` = map the cell's `MathPrecision` → `MpCode` (`F32`/`F32Strict`→`St`, `Tf32`→`Rm`).

**(f) Retire the `f32s` dtype token (D4).** In `dtype_code`/`dtype_from_code` remove the `f32s` spelling. The `F32Strict` *Rust type stays* (drives kernel selection); its identity-key spelling moves to `<mp>=st` on an `f32`-primary gem cell. Non-gem cells that previously used `f32s`: confirm none exist (the reconciliation notes `f32s` is today the gem key's only SIMT-vs-TF32 discriminator, so this retirement is atomic with `<mp>` existing — do NOT retire before (c)/(e) land).

### 2. `crates/baracuda-kernelgen/src/contract.rs`

**Add `accumulation_type` to the emitted Guarantees/precision block** (~L936–946, alongside `bit_stable_on_same_hardware` / `max_ulp` / `audited` / `notes`): emit `accumulation_type: <dtype>` from `PrecisionGuarantee.accumulator`. **Consistency invariant (KISS-Contract §6.8, per the RFC):** the contract's `accumulation_type` and the key's `<acc>` coordinate MUST denote the same dtype using the same closed dtype-token spelling (same value, different serialization surfaces — not identical wire bytes).

---

## Golden / test impact

- **Every gem-cell token golden changes** (sk3 prefix + the four new coordinates). **Every non-gem token golden changes only the `sk3` prefix.** This includes the KISS-repo `relu_add` condition-1 golden (a `bin` cell) — a `sk2`→`sk3` re-prefix.
- **Coordinate a golden regen with KISS + Fuel**, same process as the sk2/#60 regen: Baracuda emits sk3 from the branch → KISS regens its fixtures/goldens via a KISS-side issue (change-via-issue) → Fuel's independent deriver rebuilds against sk3. Re-run the recorded head-to-head at sk3.
- In-crate: update the ~8 `sk2|…` literals + the version assert (listed in 1a), add gem-cell sk3 token tests (a non-batched and a batched gem cell showing the precision group; a `st` vs `rm` pair proving SIMT-f32 and TF32 now hold distinct tokens — the collision the RFC fixes).

---

## Sequencing (the pin)

1. Land 1(b)→(e) + 2 **together** as one `sk2→sk3` bump (the D1+D4+D5 coordinate set), with 1(f) `f32s` retirement atomic with `<mp>` existing. Never partial.
2. Do it on a branch off `feat/kiss-convergence`; hold until sk3 is formally adopted (Eric's stamp).
3. On adoption: apply, coordinate the KISS/Fuel golden regen, re-run the recorded byte-match at sk3.
4. Estimated scope: ~1 field-set on `ContractionKey`, ~4 codec touch-points, ~1 contract field, ~8 in-crate golden updates + new gem tests, + the cross-repo regen. Mechanical once the grammar is frozen (it is).

---

## Resolutions recorded at application time (2026-07-22)

Two seams the blueprint's step 1(e)/(f) left implicit, resolved during the bump:

1. **The `F32Strict` fold is TOKEN-side only.** First attempt folded `F32Strict→F32` in
   `structure_key()` derivation — which broke kernelgen immediately
   (`reduction_f32strict_folds_in_double`): `build_plan` reads the cell dtype from
   `key.dtype` (`plan.rs` — `OpDef.dtypes` is the accepted-dtype *list*, not the cell's
   dtype), so a derivation-side fold silently downgrades every strict kernel. Final shape:
   the STRUCT keeps `F32Strict` (in-process MathPrecision carrier); `dtype_code` spells it
   `f32` (the closed set has no `f32s`, §6.1-0005); on a gem cell the strictness re-surfaces
   as `<mp>=st`; on a non-gem cell it is out-of-band per RFC §4.1.5 (§6.6-0018 —
   token-collides with the plain-f32 cell BY SPEC, disambiguated provider-side). Parsing an
   `F32Strict`-emitted token yields the canonical `F32` twin, which re-emits the identical
   token (token-level round-trip holds; struct-level round-trip is defined over canonical keys).
2. **kernelgen's skinny contraction accumulator for `F32Strict`: `double`→`float`.** The
   emitter (`cuda.rs emit_contraction` + `contraction_splitk_variant`) accumulated
   F32Strict in `double` — contradicting the F32Strict element contract ("full IEEE 754
   binary32 multiply-add throughout", `element.rs`), the canonical `<acc>` lattice
   (F32Strict→f32, matching CUTLASS `GemmSku::precision_guarantee`), the KISS reference
   tokens (`…/f32/f32/f32/st`), and the §4.2 `accumulation_type`↔`<acc>` pin. Numerics
   change: strict-f32 contraction results are now a true binary32 chain (last-bit different
   from the old double-accumulated values, and now genuinely bit-stable-per-contract). The
   REDUCTION-family double-accumulate convention (`F64|F32Strict→double` across
   reduce/scan/window/rowreduce/sort) is UNTOUCHED — those cells carry no key `<acc>`
   coordinate; their contracts now declare `accumulation_type: f64` honestly.

Also landed with the bump (blueprint §2 refined): `accumulation_type` emits for **gem cells
from the key's `<acc>`** (the §4.2 pin holds by construction; test
`contract_accumulation_type_matches_key_acc`) and for **reduction-bearing cells from the
generated fold's accumulator rule** (double for f64/f32-strict, native for integers, float
otherwise); pure elementwise cells omit the field (no fold, no claim). Spelling goes through
the new `baracuda_kernel_vocab::dtype_token` — the one closed-set spelling surface.
