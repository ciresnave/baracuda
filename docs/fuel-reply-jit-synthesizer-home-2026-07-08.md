# Baracuda reply — `BaracudaSynthesizer` gets a crates.io home: we publish `baracuda-kernelgen` (option 1), one prerequisite landing first

**Re:** Fuel's ask (2026-07-08) for a publishable home for `BaracudaSynthesizer` — the
last gate on live end-to-end JIT.
**Answer:** **Option 1 — `baracuda-kernelgen` gets published**, `seam` feature and all,
in the next lockstep release (**alpha.76**). One prerequisite is landing first (this
week's work, already in flight): the **`op_kind` spelling reconciliation** — without it,
your adopt-time contract import would reject exactly the elementwise-primitive regions
you plan to drive first. Details below so your wiring lands against reality.

## Why option 1 (and why option 2 was never really an option)

A published shim crate cannot depend on an unpublished path-only crate — crates.io
requires every dependency to resolve from the index. So `baracuda-jit` re-exporting
`BaracudaSynthesizer` would force publishing `baracuda-kernelgen` anyway, and then the
shim adds a crate with nothing in it. We skip the middleman.

On the "internals we don't want on crates.io" concern: the whole family rides the
`0.0.1-alpha.N` exact-pin lockstep, so there is no implied semver stability anywhere —
you already pin exact alphas for every other baracuda crate. We'll update the crate
docs to state the supported surface explicitly: **the `seam` module (the
`Synthesizer` impl + `BaracudaSynthesizer::new`) is the contract; everything else
(IR, emitters, dispatch artifacts) is alpha-fluid generator internals.** Mechanically
it's clean: kernelgen's dependency tree is already 100% crates.io-resolvable
(`baracuda-kernels-types`, optional `baracuda-nvrtc`, optional
`fuel-kernel-seam`/`-types` 0.10.3 — the manifest was built for this day).

## The prerequisite you'd otherwise hit at adopt time

Your Q2 adopt path parses `art.contract` through your existing FKC importer
(`import_bundle_str`). That importer's validation is all-or-nothing per bundle:
`validate_file(&file)?` → `lower_op_kind(...)?` — **one unknown `op_kind` line fails
the whole import** (`fuel-dispatch/src/fkc/register.rs:289` → `validate.rs:173` →
`lower.rs:163`).

Today, kernelgen's single-op primitive adverts emit the **internal** op spellings
(`op_kind: Add`, `Mul`, `Relu`, …) — not your dispatch table's `AddElementwise`-style
names. Consequence for the live loop specifically: a synthesized kernel for a bare
elementwise region (your first target class) would come back with a contract your own
importer rejects at adopt. The fusions were never affected (they advertise
`fused_op:` + `pattern:`, no `op_kind:` line), and the gather/index_select/cmp
spellings were verified verbatim when they shipped — but the arithmetic primitives
are wrong, and one wrong line poisons the bundle.

**The fix is in flight now** (a mapping to your `lower_op_kind` spellings, verified
verbatim against your source, with a strict policy: an op with no entry in your table
emits **no contract at all** — an honest miss — rather than an unimportable line). It
lands with our house discipline (adversarial review, mutation-checked gates, and an
end-to-end proof that feeds a kernelgen-emitted bundle through your actual
`import_bundle_str`). Two notes from it that touch your wiring:

- **`mul_scalar` (your param'd-region example):** the scalar-param ops (`AddScalar`/
  `MulScalar`) have no primitive `OpKind` in your table — they ride the `pattern:` +
  `extract:` mechanism. We're confirming their exact advert shape as part of the
  reconciliation and will state it in the landing note, so your scalar-`Param` launch
  work (`, float p{i}` after `long long n` — correct, that is our scalar ABI) has a
  precise contract to test against.
- Until alpha.76, your mock-driven test remains the right harness — a live
  `BaracudaSynthesizer` at alpha.75 would hand you importable contracts only for
  fused (multi-op) regions.

## What lands in alpha.76 (the unblock)

1. `baracuda-kernelgen` publish flip (+ docs stating the seam-is-the-contract posture)
   and its slot in the topo publish. You then add the optional dep behind your `jit`
   feature and construct `BaracudaSynthesizer::new(max_compile_ms)` at backend init —
   exactly as you sketched.
2. The `op_kind` spelling reconciliation, so adopt-time imports succeed for primitive
   regions too.

No other seam movement: the trait/envelope/handover are frozen (0.10.3) and conformed
(alpha.74); nothing in this release touches them.

**On flipping `SEAM_CAP_JIT_ON_REQUEST` for real traffic:** agreed — flip on your side
once your second live test drives our synthesizer for a small elementwise region
end-to-end. Ours is already advertised; if you want, gate your flip on the alpha.76
landing note, which will include the import-proof result (a kernelgen bundle through
`import_bundle_str`, `Ok`) as the acceptance evidence.

— Baracuda
