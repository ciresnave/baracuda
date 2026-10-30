# API freeze — public-surface inventory, classification, policy (slice 1)

Backs ROADMAP "Pre-1.0 must-haves → API freeze + 1.0 stability review". Measured at `origin/main`
`8786bf0f` (alpha.96). **Docs only: no code was changed and nothing here cuts `0.1.0-beta.0`.**
This supersedes the line-grep first pass (`../api-freeze-pub-surface-audit.md`, `fc1a97de`), whose
numbers were a ranking and are replaced by the census below.

Three deliverables, all in this directory:

| file | what |
|---|---|
| `per-crate-summary.tsv` | per-crate counts by class, all 74 library crates (incl. the two C-ABI and the vendor-mirror crates) |
| `items.tsv` | every item **not** in `stable-c-abi` / `vendor-mirror`, one row each (9 996 rows): crate, file, line, kind, name, reach, class, non_exhaustive |
| `crates-scanned.tsv` | which manifests were scanned, whether each has a lib root, `publish = false`, and how many `src/` files the module walk did not reach |
| `tool/` | the enumerator + classifier source (`*.txt` so repo rustfmt/feature-coverage ignore them); method below |

## 1. Method

- **Enumeration is by `git ls-files`** (2 347 tracked files, 1 218 `.rs`), never a disk walk.
- A `syn` (v2, `full`) walker parses each library root named by its manifest (`[lib] path` or
  `src/lib.rs`), resolves `mod x;` through `x.rs` / `x/mod.rs` / `#[path]`, skips `#[cfg(test)]` and
  `mod tests`, and records every `pub` item: fn, struct **and pub fields**, enum **and variants**,
  union, trait **and members**, type, const, static, `pub use`, `extern` import fn/static/type,
  inherent `pub` methods / assoc consts / assoc types, `#[macro_export]`, and `pub mod`.
- Per item it records `reach` (**`public`** = every enclosing module is `pub` up to the lib root;
  **`private-mod`** = a `pub` item inside a non-`pub` module), `#[doc(hidden)]`, `#[deprecated]`,
  `#[non_exhaustive]`, and whether it is FFI.
- **Limits, stated:** items emitted by macro invocations are not seen (only literal `macro_rules`
  exports are counted); trait-impl items are excluded (they are not new surface); `#[cfg(feature)]`
  gating is **not** applied, so a feature-gated item counts once whichever feature enables it;
  `private-mod` + a matching `pub use` is treated as exported by name only (heuristic).
- **Controls.** (a) All 74 manifests with a `[lib]` root were walked, **0** `UNRESOLVED mod` /
  `PARSE-ERR` / `NOT-TRACKED` lines on stderr. (b) The 3 crates with an unreached `src/` file are
  explained: `baracuda-cuda-emit` → `src/bin/kernelgen.rs` (a binary), `baracuda-kernels-bench` →
  `src/bin/baracuda-bench.rs` (a binary), `baracuda-cuda-sys` → `src/bindings/generated.rs`
  (1 851 `pub` lines by grep) — **that file is an `xtask` output (`xtask/src/main.rs:298`) that no
  `mod` declares, so it is not compiled and is not surface**; the earlier line-grep counted it, which is
  why `baracuda-cuda-sys` read 3 270 then and reads 1 742 now. (c) The classifier's plumbing rule
  (§3) matches 30 trait members that are *already* `doc(hidden)`, so the rule finds the pattern it
  is looking for.
- Reproduce: build `tool/` (rename `Cargo.toml.txt`, `main.rs.txt` into `src/`), `pubinv <repo> <out>`,
  then `python -I classify.py <out>/items.tsv <out>`.

## 2. The surface

**20 076 `pub` items across 74 library crates** (3 further manifests have no lib root: `baracuda-examples`,
`kiss-ref-diff`, `xtask` — see `crates-scanned.tsv`).
19 693 are `public`, 383 are `pub` inside a private module.

| class | items | meaning (rule, mechanical) |
|---|---:|---|
| `stable` | 9 905 | public, not hidden, not deprecated; **the freeze is a promise on these** |
| `stable-c-abi` | 5 809 | `extern` symbols/types in `baracuda-kernels-sys` and `baracuda-cutlass-kernels-sys` — **our** C ABI |
| `vendor-mirror` | 4 271 | every other `*-sys` crate: raw mirror of a vendor header (CUDA, cuDNN, cuBLAS…); stability follows the vendor |
| `doc-hidden` | 62 | already `#[doc(hidden)]` |
| `sealed-trait-pattern` | 21 | `pub trait Sealed*` in a private module — the idiomatic sealed-trait; intentional |
| `candidate-for-removal` | 8 | `#[deprecated]` (list below) |
| `candidate-doc-hidden` | 0 | rule below; **nothing to propose** — FFI plumbing is already hidden |
| `narrow-visibility-candidate` | 0 | pub-in-private-module and not re-exported; the 20 hits were all `Sealed` traits |

Concentration (`per-crate-summary.tsv`): `baracuda-kernels` 6 185 (30.8%), `baracuda-kernels-sys`
5 798 (28.9%; 5 673 extern fns), `baracuda-cuda-sys` 1 742, `baracuda-driver` 626,
`baracuda-runtime` 431, `baracuda-cudnn-sys` 421, `baracuda-cudnn` 241. **Ours to decide** is
`stable` (9 905) + `stable-c-abi` (5 809) = 15 714; the 4 271 vendor-mirror items are decided by NVIDIA.

### Candidates for removal (all 8, each already `#[deprecated]` with a replacement)

| item | since | replacement (from the attribute) |
|---|---|---|
| `Adaptive{Avg,Max}Pool{1,2,3}dPlan::derived_kernel_stride` (6) | alpha.33 | none: "no single (kernel, stride) pair applies" — remove outright |
| `baracuda_nccl::Communicator::nranks` | — | `world_size()` |
| `baracuda_npp::adds_32f_in_place` | 0.2.0 | `baracuda_npp::signal::add_32f_in_place` |

Removing any of them is a breaking change under §4; do it in the beta.0 window, not before.

### Findings the numbers expose

1. **`doc(hidden)` is used on 62 items (0.3%)**, so "what is hidden today" is "almost nothing": the
   freeze is almost entirely about deciding what to *start* promising. The 6 185-item
   `baracuda-kernels` surface is dominated by per-op `Plan` / `Descriptor` / `Args` types and their
   `pub` fields (3 660 pub fields workspace-wide); **pub fields on descriptor structs are the
   single largest unfrozen promise** (adding a field breaks struct-literal callers; Phase 32
   already retrofitted `#[non_exhaustive]` + builders onto 18 descriptors, and only 51 pub items
   workspace-wide carry `#[non_exhaustive]` today — see recommendation R2).
2. **`baracuda-kernels-types` re-exports Unpopped by glob** (`lib.rs:40`
   `pub use unpopped_vocab::*;`). Every item `unpopped-vocab` adds or removes therefore changes
   baracuda's public surface without a baracuda diff. See R3.
3. **Our C ABI is 5 809 items with no ABI version symbol** (`git grep -i abi_version` over
   `baracuda-kernels-sys` finds none) and 5 `#[repr(C)]` types. Compatibility is entirely
   carried by the lockstep crate version. **Policy ruled 2026-10-14 (board 198): see §4.2 and R4.**

## 3. Classification rules (applied by `tool/classify.py`, in this order)

1. `#[deprecated]` → `candidate-for-removal`.
2. `#[doc(hidden)]` → `doc-hidden`.
3. `pub trait Sealed*` in a private module → `sealed-trait-pattern`.
4. other `pub` item in a private module with no matching `pub use` → `narrow-visibility-candidate`.
5. `trait-member` whose name starts with `_` or ends `_raw` / `_ffi` → `candidate-doc-hidden`
   (this is the FFI-plumbing shape already hidden on 30 members in `baracuda-cublas`,
   `-cusolver`, `-megatron`, `-optim`, `-nccl`, `-cuvs`).
6. in `baracuda-kernels-sys` / `baracuda-cutlass-kernels-sys` and FFI → `stable-c-abi`.
7. in any other `*-sys` crate → `vendor-mirror`.
8. else `stable`.

**What this is not:** a per-item judgement that each of the 9 905 `stable` items *deserves* to be
stable. That needs per-family owner review of the kernels crate; this slice only
makes the denominators and the mechanical exits explicit. The `as_raw` / `from_raw` accessors
(135 / 14 methods) were deliberately **not** treated as plumbing: they are the documented escape hatch
to the vendor handle.

## 4. Breaking-change policy — PROPOSED, not adopted

Facts fixed by CireSnave's standing rules (`C:\Projects\CLAUDE.md` §9) are inputs, not choices: a
pushed change changes the version; a breaking change moves the major; **pre-1.0 the major is the
second number** (0.5.x → 0.6.0, never 1.0.0); one version for every crate in the workspace except
crates that track Unpopped (`baracuda-cuda-emit`, `-cuda-parse`); the PM allocates the number at
gate time. Today the workspace is `0.0.1-alpha.N`, where N is the only moving counter and the
README says "API stability is **not** promised before beta.0".

### 4.1 Rust API (all crates except the C ABI)

**Breaking:** removing or renaming a `pub` item or feature; changing a signature, generic bound
(tightening), associated type, `const` value, or enum discriminant; adding a **required** trait
method or a variant/field to a type that is neither `#[non_exhaustive]` nor sealed; making a
`pub` field private; changing default features; raising the MSRV (permitted by CireSnave 2026-10-08,
but noted as Breaking on pre-release lines per `CLAUDE.md` §9); a dependency bump that changes a
re-exported type (see R3).

**Not breaking:** adding an item, inherent method, optional (defaulted) trait method, or feature;
adding a variant / field to a `#[non_exhaustive]` type; loosening a bound; changing `doc(hidden)`
items; documentation; internal kernel changes that stay within the op's documented tolerance.

### 4.2 C ABI (`baracuda-kernels-sys`, `baracuda-cutlass-kernels-sys`)

Symbols are `baracuda_kernels_<family>_<dtype>[_<variant>]_{run,can_implement,workspace_size}`
(e.g. `baracuda_kernels_gemm_s8_rrr_sm80_run`), as declared in `baracuda-kernels-sys/src/lib.rs`.

**Breaking:** removing or renaming a symbol; changing parameter count, order or type; changing
`#[repr(C)]` size, alignment or field order; changing a status-code or enum value; changing
a **documented contract** — in-place/aliasing safety (`docs/guides/inplace-op-coverage.md`),
stride/lda meaning, workspace-size formula, null-pointer behaviour; and a `*_can_implement` that
turns an *accepted* configuration into a *declined* one.

**Not breaking:** adding a symbol; a `*_can_implement` that turns a declined configuration into an
accepted one; making an op faster; tightening a documented tolerance.

**Policy (ruled by CireSnave 2026-10-14, board 198; verbatim):** *"The C ABI should be stable across a
single major version after a 1.x release."* Read as: **from 1.0 onward the C ABI does not break within
a major version; breaking it requires the next major.** Before 1.0 the usual pre-1.0 rule applies
(§4.6: the second number is the major), so a C-ABI break pre-1.0 is a second-number bump, as today.
The wording says "after a 1.x release"; I read it as binding from `1.0.0` itself, because a floor of
`1.1` would leave the whole `1.0.x` line unprotected. If that is not meant, it is a one-line change.

**What this implies for the 5 809 `stable-c-abi` items** (count as measured in §1; vendor mirrors
are unaffected, §4.4):

1. **The breaking list above becomes a freeze list from 1.0.** Every item under *Breaking* is
   forbidden inside a major, including a `*_can_implement` that turns an accepted configuration into
   a declined one and any documented-contract change (aliasing, stride/lda, workspace formula,
   null-pointer behaviour). Within a major the set is **additive-only**.
2. **A signature change is a new symbol, never an edit** (the mechanism below). The old symbol stays
   exported, with its behaviour, for the whole major.
3. **Symbol versioning is by name.** The existing `_v2` / descriptive-suffix scheme is the symbol
   version; the lockstep crate version is not an ABI promise, because crates move on every pushed
   change (CLAUDE.md §9) and the C ABI must not.
4. **Deprecation path.** A symbol is marked deprecated (doc comment on the `extern` declaration and a
   `CHANGELOG.md` entry; the doc comment's first deprecation line is exactly `/// Deprecated since <version>: <replacement or reason>`,
   which `scripts/check-abi-manifest.py` records in the fifth column of `abi/abi-manifest.tsv` (deprecating is not an ABI break); Rust-side wrappers get `#[deprecated(since, note)]`), kept through the rest of
   that major, and removed only in the next major. §4.5's "one breaking version" is the *minimum*;
   from 1.0 it is *the remainder of the major*.
5. **`baracuda_kernels_abi_version()` becomes the loader's check.** It returns the C-ABI major, not
   the workspace epoch, so a loader refuses a `.dll`/`.so` from another major instead of crashing.
   Adding it is additive; it does not exist today (`git grep -i abi_version` finds none).
6. **Before 1.0, the 5 809 items are not frozen**, but anything that will be public at 1.0 should
   already follow items 2–4 so that 1.0 does not need a flag day.

**Mechanism (recommendation R4, now adopted):** a signature change is a *new symbol* (`…_v2` or a descriptive
suffix) plus deprecation of the old one — never an in-place edit. History shows in-place edits
(Phase 31 changed 8 ELU signatures; Phase 21 changed f32/f64 interpolate signatures) were absorbed
as "alpha breaking"; that is acceptable pre-beta and should stop at beta.0. A `baracuda_kernels_abi_version()`
symbol (returns the workspace's breaking-epoch integer) would let a loader refuse a mismatched
`.dll`/`.so` instead of crashing; it is additive.

### 4.3 Numerical behaviour

An op's accuracy contract is `PrecisionGuarantee` (Unpopped's type) plus the tolerance recorded
in its tests. A change that makes results leave that tolerance is breaking; a change
within it is not, even if the bits differ. Bit-exactness is promised only where the op's doc says
so (e.g. the dense-GEMM f32 path is IEEE binary32, not TF32).

### 4.4 Vendor mirrors (`*-sys` other than ours)

They track the vendor header at the supported minimum toolkit. Adding symbols for a newer toolkit
is non-breaking; removing a symbol is breaking only if the minimum supported toolkit still has it.

### 4.5 Deprecation

Deprecate with `#[deprecated(since, note)]` naming the replacement and keep it through at least
one *breaking* version (second-number bump) before removal. The 8 existing deprecations (§2) are
the first removal batch.

### 4.6 What each version bump means (pre-1.0)

| change | bump |
|---|---|
| any non-breaking pushed change (incl. docs/CI) | next `0.0.1-alpha.N` (lockstep), PM-allocated |
| a breaking change per §4.1–4.3 | second number: `0.0.x` → `0.1.0`-class bump, PM-allocated; for `-cuda-emit`/`-cuda-parse` it follows Unpopped's `0.M` |
| cut of the stability review | `0.1.0-beta.0` — **explicitly not done here** |

## 5. `T: Element` vs `T: DeviceRepr + Copy` — recommendation: **keep the split, document it**

Measured at `8786bf0f` (`git grep`, `crates/**/*.rs`, lines — not a census of generics):
`T: Element` 920 lines / 227 files; `T: …DeviceRepr…` 194 lines / 34 files; `T: KernelDtype` 2
lines / 1 file. In `unpopped-vocab 0.16.0` (`src/element.rs`): `pub trait KernelDtype: DeviceRepr +
kerneldtype_sealed::Sealed + Copy + 'static { const KIND: ElementKind; }` (L131) and
`pub trait Element: KernelDtype + sealed::Sealed { type Scalar: ScalarType; }` (L255).

- `Element` is a **sealed subtrait** of the `DeviceRepr + Copy + 'static` layer, not a rival
  bound. It adds the α/β `Scalar` projection that only arithmetic ops need.
- Compute plans (GEMM, conv, norms, …) use `T: Element` because they need `Scalar`; carriers and
  layout/pack plans (`TensorRef`, `ContiguizePlan`, `CastSubBytePlan`, GGUF/MMVQ batched) use
  `DeviceRepr + Copy + 'static` so packed / sub-byte types can ride them without an arithmetic
  `Scalar`. A single umbrella would force every sub-byte type to invent an α/β type it has no use for.
- **Unifying is not a baracuda change**: both traits and their sealing live in Unpopped.
  Recommendation: **document as intentional** (R1), and record `Element` / `KernelDtype` /
  `ElementKind` as an Unpopped-owned stability dependency of every `Plan<T>` signature.

## 6. Recommendations (each needs a decision; none is applied)

- **R1** Document the layering in `baracuda-kernels-types` rustdoc and `ARCHITECTURE.md` (docs only).
- **R2** Before beta.0, classify descriptor structs: `#[non_exhaustive]` + builder for any struct
  whose fields may grow (the 3 660 pub fields are the exposure); this is a breaking pass, so it
  should be the *last* breaking batch before the freeze.
- **R3** Replace `pub use unpopped_vocab::*;` with an explicit re-export list (or stop
  re-exporting and have consumers depend on `unpopped-vocab` directly), so Unpopped additions stop
  moving baracuda's surface. **Cross-lane:** consumers (Fuel) may rely on the glob — ask before. **Status: DONE** (landed at 0.1.0-alpha.11; `crates/baracuda-kernels-types/src/lib.rs:48-54` now re-exports an explicit list, re-checked 2026-10-29).
- **R4** **RULED 2026-10-14 (board 198):** adopt §4.2 for the C ABI; from 1.0 it does not break within a
  major (verbatim ruling and implications in §4.2). Built at alpha.164: `baracuda_kernels_abi_version()` as an independent ABI integer (not the crate
  version), the checked-in `abi/abi-manifest.tsv` and the CI guard `scripts/check-abi-manifest.py`. Still to build:
  the C-ABI deprecation note convention.
- **R5** Remove the 8 deprecated items in the first breaking bump after adoption. **Status: DONE** (landed at 0.1.0-alpha.11; 0 `#[deprecated]` attributes remain under `crates/`, re-checked 2026-10-29).
- **R6** Per-family review of the 9 905 `stable` items by plan family before beta.0; this inventory
  is the denominator, not the verdict.

## 7. Slice 2 — family denominators and the R2 worklist (measured at `92ba7311`, alpha.98)

Re-run of the §1 census at `92ba7311`: identical totals (20 076 items; docs-only commits since
`8786bf0f`). Two new mechanical outputs, regenerated by `tool/slice2.py.txt`:

- **`kernels-families.tsv`** — `baracuda-kernels` public items per top-level module family
  (23 families, 6 180 public items; kinds, `doc(hidden)`, deprecated, `#[non_exhaustive]` structs).
  Largest: `loss` 726, `attention` 716, `quantize` 625, `pool` 390, `conv` 371, `segment` 368,
  `elementwise` 353, `shape_layout` 324, `linalg` 295. This is the R6 review denominator: one owner
  review pass per family, not per item.
- **`r2-worklist.tsv`** — every public struct that has at least one `pub` field and is **not**
  `#[non_exhaustive]`: **792 structs / 3 328 pub fields** workspace-wide, of 1 320 public structs
  (818 have pub fields; only **26** carry `#[non_exhaustive]`). The `repr` column flags
  `repr(C)`/`transparent`/`packed` (168 structs) — those must stay literal-constructible.

Reading the worklist (kernels crate: 577 of the 792):

| shape | structs | pub fields | R2 verdict |
|---|---:|---:|---|
| `*Descriptor` (problem-shape description; fields grow as ops gain options) | 257 | 1 179 | **the R2 target**: `#[non_exhaustive]` + `::new()` + `with_*` builders, as Phase 32 did for 18 |
| `*Args` (borrowed device-buffer bundles passed once per call) | 307 | 1 182 | **leave literal** — adding a buffer is already breaking for every caller; `non_exhaustive` would make them unconstructible without a builder for no gain |
| other (`Block*` GGUF block layouts, `GptqWeights`, `MarlinWeights`) | 13 | 42 | leave: byte layouts / on-disk formats |

Outside the kernels crate: 151 are in `*-sys` crates (68 in `baracuda-cuda-sys`; vendor mirrors, not
ours to attribute) and 64 are in other wrapper crates (`baracuda-cutlass`, `-runtime`, `-kernels-bench`, …).
In the kernels crate 11 of the 577 are `repr`-flagged.

**Control:** `Conv2dDescriptor` (retrofitted in Phase 32) is absent from the worklist and present
among the 26 `non_exhaustive` structs. **Limit:** "has a pub field" is read from the pub field rows,
not from whether the struct is ever built by literal downstream; whether Fuel struct-literals any
`*Descriptor` is unmeasured here (R2 is a breaking pass and needs that check first).

### 7.1 Consumer exposure of the R2 target (measured 2026-10-14)

`git grep -w -F -f <the 257 kernels *Descriptor names> origin/main -- '*.rs'` over Fuel at
`e6cbf782` matches **0 files**; the same `-f` path with `ElementKind` matches 9 files (positive
control). Fuel's use of `baracuda_kernels{,_sys,_types}` is `ElementKind`, `OperandDesc`, `ArchSku`
and a few contiguize FFI symbols — no `Plan` / `Descriptor` / `Args`. So the R2 pass has no Fuel
struct-literal exposure at that ref. **Not measured:** lightbulb and other in-portfolio consumers,
crates.io dependents.

### 7.2 Keeping the census current (process)

Re-run `tool/` at each release PR gate and diff `per-crate-summary.tsv` against the previous run: a
new `pub` item, a new un-`non_exhaustive` descriptor, or a changed `doc(hidden)` count is then a
visible freeze delta instead of an accident. The tool is read-only and takes ~1 s over 1 218 files;
it was deliberately not a CI job until the policy (R4) was adopted, because a gate with no adopted
rule only produces noise. R4 is adopted (§4.2), so a C-ABI diff gate is now a candidate follow-up; it is
not built here. **Update 2026-10-29: it has since been built** as the "C-ABI manifest guard" CI step (`.github/workflows/ci.yml`, `scripts/check-abi-manifest.py --base`).

**Gate re-run at `272d9d18` (alpha.131):** totals identical (20 076 items), but one delta surfaced and was a bug, not a
decision: `baracuda_kernels::ROPE_DEFAULT_BASE` had become `doc(hidden)` because #185 left an orphaned `#[doc(hidden)]`
(for the removed `map_status_pub`) directly above it. Fixed at alpha.132; with the fix the census matches the committed
TSVs on every column except line numbers (compared with `cut -f1,4-8 | sort`). The committed TSVs are not rewritten for
line-number drift alone.

### 7.3 R2 decision: which descriptors actually grew (measured 2026-10-23, `origin/main` 405bf7fc-era)

**Ruled (PM, 2026-10-23):** R2(b). Only descriptors whose pub fields have *actually grown* become
`#[non_exhaustive]` + `new()` + `with_*`; the rest are **declared frozen-as-is, in writing, here**. The
edit is breaking, so it ships in the single beta.0 window with R3 and R5, not before.

**Method** (`tool/r2growth.py.txt`, output `r2-growth.tsv`): for each of the 257 kernels `*Descriptor`
structs in `r2-worklist.tsv`, walk the history of the file that holds it (oldest first, `git log --reverse`),
read the struct's `pub` field names at every revision that touches the file, and count revisions where a name
appeared that the previous revision lacked.

**Result: 3 of 257 grew (1.2%)**, each by one field:

| struct | field added | fields now |
|---|---|---:|
| `BatchPagedPrefillDescriptor` | `enable_kv_split` | 7 |
| `FlashDecodingDescriptor` | `num_kv_heads` | 7 |
| `GgufMmvqDescriptor` | `w_start_byte_offset` | 4 |

Those three are the beta.0 R2 retrofit list (**done at 0.1.0-alpha.11**; R3 and R5 landed with it). The other **254 are frozen as-is**: their field sets have not changed
over the measured history, and they stay literal-constructible (this avoids a breaking pass over 453 test files
and 182 source/bench/example files, about 2 017 `XDescriptor {` sites, for a surface Fuel uses 0 of, section 7.1).
If one of the 254 gains a field before 1.0 it joins the retrofit list; after 1.0 it is a breaking change.

**Controls:** the three positives are real additions (cross-checked: `git log -L :Name:file` shows 2 to 3 commits
touching each, and a `+ pub` field line in the diff); `AffineDescriptor`, `SdpaDescriptor` and `UnaryDescriptor`
read 0 growth and `git log -L` shows only their creation plus one non-field edit (checked for `AffineDescriptor`).
**Limits:** the history is 800 squashed commits starting at `e36466e6` ("Initial commit: baracuda v0.1.0"), so
anything that grew before that is invisible; a struct moved between files reads as "first seen"; one squash commit
that adds several fields counts once; a renamed field reads as one addition. "Has not grown" is a statement about
this history, not a prediction.

## 8. What this slice does not decide

Per-item stable/hidden/removal verdicts for the 9 905 `stable` items; whether beta.0 happens at all.
(The C-ABI policy was decided 2026-10-14, §4.2.) Those need CireSnave via the PM.
