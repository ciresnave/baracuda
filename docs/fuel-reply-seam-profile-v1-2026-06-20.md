# Baracuda review — Kernel-Seam Interop Contract, Profile v1 (2026-06-20)

Reply to Fuel's *Kernel-Seam Interop Contract — Profile v1* (DRAFT for
ratification, branch `feat/kernel-contracts-dlpack`) and its cover note.
Companion to [`fuel-reply-telemetry-fdx-fkc-2026-06-19.md`](fuel-reply-telemetry-fdx-fkc-2026-06-19.md)
and [`fuel-reply-fkc-patterns-2026-06-19.md`](fuel-reply-fkc-patterns-2026-06-19.md).

TL;DR — **strong yes to the shape**: a single living contract that pins
who-speaks-what-at-which-version, role-disjoint subsets, a bundled Profile, and a
frozen handshake is exactly the right artifact, and the §1/§2/§4 structure is
sound. We're ready to ratify Profile v1 once **two conditions** are met, and with
one **drift correction** (we're further along than §7.1 assumes):

- **Blocking (§A1): Profile v1 pins fusion-patterns rev 4, but we reviewed rev 2
  and raised two must-fix spec bugs + a blocking question (E1 commutativity).**
  We can't ratify a profile pinning a spec version we haven't seen, which had
  known must-fixes at the version we did review — and our generator emits to that
  grammar, so we must re-verify against rev 4.
- **Pin before freeze (§A2): the `baracuda_seam_hello()` C-ABI.** The `SeamHello`
  envelope is "frozen forever," but its `profiles` array is variable-length — so
  the C-ABI can't return it by value. Nail the concrete calling convention now.
- **Endorse**, capability-gated, no v1 burden: the §3 handshake and the §5
  JIT-on-request layer (it's the telemetry-driven-fusion loop we already designed
  together).

---

## A. Conditions before we ratify

### A1. Fusion-patterns rev 4 vs. our rev-2 review (blocking)

Profile v1 §2 pins **FKC fusion patterns rev 4**. Our review
([`fuel-reply-fkc-patterns-2026-06-19.md`](fuel-reply-fkc-patterns-2026-06-19.md))
was of **rev 2**, and it raised items that must be resolved in rev 4 before we
ratify — *and* our `derive_pattern` generator emits patterns to that grammar, so
"rev 4 is fine" isn't enough; we have to re-verify our output conforms to rev 4.
Specifically, please confirm rev 4 resolved:

- **A1-a (was must-fix A1):** §8.2's RMSNorm guard `self.axis == input(0).rank - 1`
  used subtraction the §5 guard grammar doesn't define ("no general arithmetic").
  Confirm the fix (we suggested `self.axis == -1` / a `last_axis` sentinel).
  *Blocks every norm pattern.*
- **A1-b (was must-fix A2):** §8.1's FusedLinear bias guard `operand(0)` was carried
  by a `bind` leaf, which §5 says has no operands. Confirm the fix
  (`input(1).dim[-1]`).
- **A1-c (was BLOCKING E1): commutative-operand canonicalization.** This is the one
  that gates *us* as an auto-generator. Our `derive_pattern` emits **one** operand
  ordering (the body's natural order). If Fuel does not canonicalize commutative
  operands before matching, our patterns miss every graph written the other way
  (the 2ᵏ problem). Please confirm rev 4 states normatively that Fuel canonicalizes
  commutative operands before matching, *and* the canonical order, so our emitter
  and your matcher agree by construction.

Send us rev 4 (and the rev-3 delta) and we'll re-run our conformance check against
it — fast, since the generator is built (§D).

### A2. The `baracuda_seam_hello()` C-ABI (pin before the envelope freezes)

We endorse the §3.1 envelope, but flag a concrete ABI detail that must be settled
*before* ratification precisely because the envelope is "frozen forever":
`SeamHello` contains a **variable-length** `profiles: u16[profiles_len]`, so
`baracuda_seam_hello() -> SeamHello` **cannot return it by value across the C
ABI.** We need the concrete realization pinned — e.g. a caller-provided buffer:

```c
// fill *out (fixed header) + write up to cap profile ints into profiles_buf;
// return the count actually available (caller re-calls if cap was too small).
uint16_t baracuda_seam_hello(SeamHello* out, uint16_t* profiles_buf, uint16_t cap);
```

or a fixed `profiles[MAX]` cap in the struct. Whichever — let's lock the exact
byte layout + calling convention into the FDX-style size-asserted header, since a
"frozen-forever" envelope with an unpinned C realization is the worst thing to get
wrong. Trivial to agree; just not implicit.

---

## B. Endorsements

### §3 — the connect-time handshake: yes

The frozen-envelope, highest-mutually-supported, hard-fail-on-disjoint design is
right, and designing it in from v1 even though v1 negotiation is trivial is the
correct discipline — it's the same shared-canonicalization-from-day-one stance we
already took with `structure_key`. Reusing the FDX `BackendProbe` tokens as the
low bits of `capabilities` is elegant (no new concept). On our side
`baracuda_seam_hello()` + `seam_profiles: [1]` in each FKC bundle's front-matter
is a small, clean ask — our generator already emits the front-matter, so
`seam_profiles` is a one-line add. (Modulo A2.)

### §5 — JIT-on-request: yes (it's the loop we already designed)

This is the same telemetry-driven-fusion loop we worked out in the FDX/FKC reply —
miss signal → request → synthesize → cost-gated adoption as a multi-sibling
alternative. The **Fuel-strategist / Baracuda-synthesizer** split is the right
constitution: Fuel chooses the region (= the fusion decision) and decides
adoption; we synthesize within it. Three points:

- **One alignment to state explicitly so it doesn't bite later:** our planned
  **algebraic optimizer (e-graphs)** operates as the *synthesizer's hardware
  knowledge applied within the Fuel-chosen region* — it rewrites/extracts the best
  kernel for the `JitRequest.region` we're handed. It is **not** a DAG-scanner and
  does **no** opportunity-finding across Fuel's graph. So it's fully compatible
  with "no backend-side opportunity-finding"; we just want that reading on record,
  because our optimizer *is* real fusion machinery — pointed only inward, at the
  region you choose.
- **The JitResponse reuses our generator.** `(kernel + full FKC contract +
  recipe[pattern: + decompose])` is exactly what our generator produces — the
  pattern emitter is built (§D), the full-contract emitter is the remaining piece.
  So the JIT endpoint is a thin wrapper over the generator, not new machinery.
- **Capability-gating is correct for us.** We'll ship FDX+FKC at ratification and
  advertise `SeamCapJitOnRequest` later, when both sides have built the
  base-emission seam. A Profile-v1 Baracuda without the JIT bit is fully
  conformant — agreed and appreciated.

### Tier-2 declarative registration: fine

Fuel-internal plumbing; it doesn't change what we author. Our generated contracts
already carry the recipe (`pattern:` now; `decompose` to follow). Implementing
your declarative-pattern engine (the `PatternKind::Declarative` stub) is the
prerequisite on your side — noted.

---

## C. Profile v1 bundle (§2) + capability bits we'd advertise

- **FDX v1, FKC v1, telemetry v1** — **confirmed**; they match our 2026-06-19
  acceptances (the honesty invariant, the `ImplId` 5-field separable tuple,
  `StructureKey` computed by us and called by you via the minimal
  `FdxOperandDesc` projection, miss = generic-match, `I4/U4/B1`, `F32Strict` as a
  precision mode, FKC as a generated projection).
- **Fusion patterns rev 4** — **pending §A1**.
- **JIT v1** — design-acknowledged; implement later behind the capability bit.
- **Capability bits at first connect:** FDX+FKC core now. We'd advertise the FDX
  `BackendProbe` tokens matching our shipped kernel families — `DlpackExtV1`
  (sub-byte `S4/U4/Bin`), `DlpackExtMx` (FP8 microscaling), `DlpackExtGgml`,
  `DlpackExtAffine`, `DlpackExtSymbolic` (attention KV), `DlpackExtGather` (paged
  KV) — i.e. most of them, since the broader Baracuda library already has those
  kernel families. `SeamCapJitOnRequest`: **off at first**, on later. (We'll send
  the exact token set with the round-trip confirmation.)

---

## D. Drift correction — §7.1 understates where we are

§7.1's "Open on Baracuda's side: `structure_key` + the FKC/`link_registry`
generator are not yet shipped (the elementwise pilot)" is **stale**. Since
2026-06-19 we've built and GPU-validated, on branch `feat/kernel-specialization`
(PR #2):

- **`structure_key`** — shipped in `baracuda-kernels-types` (the join token, with
  the lossless string codec; the `FdxOperandDesc` minimal-projection input is
  designed).
- **`baracuda-kernelgen`** — the generator: an op IR (`Const`/`Unary`/`Param` +
  arithmetic), three schedules (vectorized `float4`/`double2`, scalar, strided
  with broadcast-hoisting), dtype coverage (f32/f16/bf16/f64), and
  **`derive_pattern`** emitting FKC `pattern:` blocks — including, as of today,
  **`AddScalar`/`MulScalar` with `extract:`** for scalar-param fusions. Every
  kernel is nvcc-compiled + run on sm_89; the go/no-go is **2.03×**; the emitted
  patterns were adversarially verified conformant against the FKC §3 grammar.

So the generator core that §7.1 lists as unshipped is substantially built and
exercised. What genuinely **remains** for a Profile-v1-conforming *publish*:

1. The **full FKC contract emitter** — we emit `pattern:` today; `accept`/
   `return`/`op_params`/`cost`/`precision`/`determinism` (from our
   `KernelSku`/`PrecisionGuarantee`/OP-MATRIX) are next.
2. The **`link_registry`** (symbol → `KernelRef`) generated alongside.
3. **`baracuda_seam_hello()`** + `seam_profiles` front-matter (§A2).
4. The **`structure_key` public callable** packaged for your trampoline (the
   `FdxOperandDesc` projection + the `From` adapters).

All four are bounded and on the same generator pipeline — none is a research risk.

---

## E. Process

We're a **yes** to ratifying Profile v1 once §A1 (rev 4 confirms our must-fixes +
E1) and §A2 (the `seam_hello` ABI) are settled — both are quick. Send us rev 4 and
let's pin the `SeamHello` C realization; we'll return the round-trip confirmation
([`baracuda-seam-v1-roundtrip.md`](../outreach/baracuda-seam-v1-roundtrip.md)) with
the exact capability-token set and a re-verified-against-rev-4 generator. Your
parallel vertical slice (§8.2) is the right de-risking — implementation-true
ratification beats paper, and our generator is far enough along to make our half
implementation-true too.
