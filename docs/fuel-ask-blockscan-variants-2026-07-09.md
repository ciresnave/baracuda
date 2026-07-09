# Baracuda ask — block-scan variants shipped (reverse, Max/Min, integer; FP Max/Min + integer now BitIdentical device-proven); STILL an AOT-only honest miss, nothing to wire (2026-07-09)

**No action needed — informational.** This is a propose-first heads-up in the
landing-doc "radar item" class. It records that the cooperative block-scan now
covers the three sub-cases the SCAN increment deferred, and — importantly — that
**scan remains a contract-less honest miss with nothing for Fuel to bind.** There
is no ask here; it exists so the eventual "should scan cross the wire?" conversation
starts from an accurate picture.

## What shipped (Baracuda, `feat/kernel-specialization`)

The ONE cooperative Kogge-Stone block-scan variant (`scan_blockscan_variant` +
`emit_scan_impl(block=true)`) extends beyond forward-FP-Sum/Prod to the three
increment-6 deferrals:

- **reverse** — `j = k-1-p` remaps the reverse j-scan to a forward p-space scan
  (same reassociation as forward for FP Sum/Prod).
- **Max/Min** — a NaN- and signed-zero-correct combiner
  `comb_max(a,b) = (b != b || b > a) ? b : a` (min flips to `<`; integer drops the
  dead NaN test), LEFT = earlier prefix, order-critical because Max/Min is
  non-commutative. Proven associative + identity-clean (independent design panel:
  monoid-homomorphism proof, 0/2744 exhaustive triple failures, 20000+ regroup
  matches) and **device-validated `BitIdentical`** — whole-buffer `memcmp==0` vs the
  serial base plus hand-derived raw-bit probe rows (P1 signed-zero tie, NaN absorb,
  later-NaN-wins, exclusive-out[0]=ident, MIN mirror) on sm_89.
- **integer** {`i32`,`i64`} — native wrapping acc; Sum/Prod modular-associative,
  Max/Min select — all `BitIdentical`, `memcmp==0` device-proven.

Fidelity by op: **FP Sum/Prod stay `ReassociatedDeterministic`** (within-ULP,
`same_hardware_bitwise`); **FP Max/Min + ALL integer scans are `BitIdentical`**
(`bitwise`). The **forward-FP-Sum/Prod emission is byte-for-byte unchanged** (the
shipped path; verified by before/after source `diff`). **S8/U8 decline to the serial
base by design** — `__shfl_up_sync` has no 8-bit overload and promoting to an int acc
would break the base's native 8-bit modular wrap (a different domain, not
bit-identical); an explicit FP+{i32,i64} allowlist, NOT `is_int_dtype`. IEEE-strict
build required for the FP Max/Min bit-identity (no `-ffast-math`; the forge
nvrtc/nvcc path already is) — the same requirement Sum/Prod already relies on.

Device proof: `crates/baracuda-kernelgen/ondevice/README.md`, `scan_validate.cu`
section — **RESULT: ALL PASSED**, 216 BitIdentical base-vs-blockscan `memcmp==0`
checks + 19 combiner probe checks + 18 reassociated within-ULP/determinism checks;
`compute-sanitizer` memcheck/racecheck/synccheck/initcheck all 0.

## Why there is nothing to wire (honest miss — unchanged, confirmed)

Scan is a **contract-less honest miss**, and this increment does not change that:

- **No `OpKind` / no contract.** Neither `contract.rs` nor `pattern.rs` has any
  Scan/Cumsum/Prefix vocabulary, and Fuel exposes no Scan/Cumsum `OpTag`, so
  `derive_pattern` rejects the region as `NotElementwise` before any body walk and
  `contract()` returns `None` (the Reduction/RowReduce/Contraction precedent, pinned
  by `contract::tests::scan_is_an_honest_miss_no_contract`). The block-scan variant
  is a **schedule** of that same contract-less cell — a faster kernel for the same
  miss, not a new advert surface.
- **Keying stays additive.** `baracuda-kernels-types` is UNTOUCHED and there is **no
  `STRUCTURE_KEY_VERSION` bump** — the `_blockscan` `entry_point` disambiguates the
  variant on the wire (never the structure-key token), exactly as the forward variant
  already did. Nothing new keys.

So: the AOT kernels generate, run, and are device-proven, but cross **no** Fuel wire.

## The ask (when/if scan should ever cross the wire)

Only relevant if you decide prefix-scan is worth a JIT/FKC surface at all. That would
need, on your side, a Scan/Cumsum `OpTag` (+ the reverse/exclusive/op/dtype attrs on
the attrs channel — the same attrs-bridge gap the Iota/Triu adverts are blocked on),
and on ours a `pattern.rs`/`contract.rs` Scan node. None of that is built or implied
by this increment — recorded here so that, if you ever want it, the conversation
starts from "the kernels already exist and are bit-characterized" rather than from
zero. Until then, nothing to do.
