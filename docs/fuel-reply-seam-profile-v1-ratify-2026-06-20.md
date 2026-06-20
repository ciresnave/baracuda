# Baracuda — ratifying Profile v1 (2026-06-20)

Reply to Fuel's *FKC fusion-patterns rev 2 → rev 4 delta* and the updated
*Kernel-Seam Interop Contract — Profile v1*. Follows our conditions reply
([`fuel-reply-seam-profile-v1-2026-06-20.md`](fuel-reply-seam-profile-v1-2026-06-20.md)).

TL;DR — **both our conditions are resolved; Baracuda ratifies Profile v1.** The
delta confirms all three rev-2 blocking items landed in rev 3 and carry unchanged
into rev 4 (the grammar our generator targets is rev-3-stable), and the updated
§3.1 pins the `seam_hello` ABI to a fixed-size POD + out-param. The §5.1 e-graph
clarification and the §7.1 drift correction are exactly right. One bounded
conformance action remains on our generator (the commutative-operand canonical
sort) plus the publish work — neither blocks ratification.

---

## A. Both conditions — resolved

### A1 — fusion-patterns rev 4

- **A1-a** (`self.axis == input(0).rank - 1` arithmetic) → `self.axis == -1`
  (normalized negative-from-end). **Resolved.**
- **A1-b** (`operand(0)` on a `bind` leaf) → `rank == 1 and dim[0] == input(1).dim[-1]`.
  **Resolved.**
- **A1-c / E1** (commutativity — the one that gated us) → **§3a.2a, normative**:
  Fuel canonicalizes commutative-op operands before matching, patterns match
  against the canonical order. **Resolved exactly as our condition stated** ("must
  be stated normatively"). We emit one ordering; you canonicalize the graph;
  positional matching holds — no 2ᵏ blow-up.

The delta's headline — rev 4 changed *nothing* in the pattern/guard/extract grammar
or §3a.2a, only the adaptive-fusion prose — means our `derive_pattern` (which
conforms to the rev-3 grammar) conforms to rev 4. We'll re-run our conformance
check against the full rev-4 text to confirm (fast).

### A2 — the `baracuda_seam_hello()` ABI

The updated §3.1 resolves it precisely: a fixed **56-byte `SeamHello` POD**
(`SEAM_MAX_PROFILES = 16` fixed array + `profiles_len`, no variable-length member),
the out-param entry point `int baracuda_seam_hello(SeamHello* out)`, and frozen
offsets cross-checked by size/`offset_of!` asserts the way FDX does its `#[repr(C)]`
structs. No by-value variable-length return; clean to implement. The 16-profile cap
with `SEAM_ENVELOPE_VERSION` as the one escape hatch is the right call (profiles
retire by floor advance, so the live set stays tiny).

## B. The two things you added for us — exactly right

- **§5.1 clarification (e-graph, on the record).** "The synthesizer **MAY** use
  arbitrarily powerful fusion machinery — including an e-graph / equality-saturation
  optimizer — to produce the best kernel for the region Fuel handed it … pointed
  only inward." That is precisely the alignment we wanted: our optimizer is real
  fusion machinery, aimed *inward* at a Fuel-chosen region, never scanning the graph
  to *pick* regions. Confirmed, and we're glad it's on the record.
- **§7.1 drift correction.** Accurate now — `structure_key` + `baracuda-kernelgen`
  built and GPU-validated on PR #2, with the corrected "what remains."

## C. Our one conformance action (generator — not a ratification blocker)

§3a.2a imposes exactly one concrete thing on us: **emit commutative-op operands
(`Add`/`Mul`/`Maximum`/`Minimum`) in your canonical order.** Our `derive_pattern`
currently emits the op author's *natural* order, so `Add(MatMul, bias)` and
`Add(bias, MatMul)` would serialize differently — we need both to serialize to the
canonical order your matcher expects.

To implement the matching sort we need **§3a.2a's exact sort-key definition**. The
delta names it ("the same canonicalization `structure_key` uses") but doesn't quote
the key, and the full `fkc-fusion-patterns.md` rev 4 wasn't in what reached us — so,
per your own §8 circulation rule (bundle the full pinned annexes), **please include
the full rev-4 fusion-patterns spec** and we'll add the canonical-operand sort to
`derive_pattern` and re-verify. Bounded generator change.

## D. Profile v1 bundle (§2) + capability bits — confirmed

- **FDX v1 / FKC v1 / fusion-patterns rev 4 / telemetry v1 / JIT v1**: confirmed
  (FDX/FKC/telemetry per our 2026-06-19 acceptances; fusion-patterns now per §A1;
  JIT design-acknowledged, implemented later behind the capability bit).
- **Capability bits at first connect:** FDX+FKC core now; we'd advertise the FDX
  `BackendProbe` tokens matching our shipped kernel families — `DlpackExtV1`
  (sub-byte `S4/U4/Bin`), `DlpackExtMx` (FP8 microscaling), `DlpackExtGgml`,
  `DlpackExtAffine` (NF4/QLoRA), `DlpackExtSymbolic` (attention KV), `DlpackExtGather`
  (paged KV). `SeamCapJitOnRequest`: **off now, on later**. (Exact final token set
  in the round-trip confirmation.)

## E. Publish plan (the lockstep bump)

Per §7.1's corrected "what remains," our conforming crate version needs, all on the
one generator pipeline:

1. the **full FKC contract emitter** — `accept`/`return`/`op_params`/`cost`/
   `precision`/`determinism` from `KernelSku`/`PrecisionGuarantee`/OP-MATRIX (we emit
   `pattern:` today, incl. `AddScalar`/`MulScalar` `extract:`);
2. the **`link_registry`** (symbol → `KernelRef`);
3. **`baracuda_seam_hello()`** + `seam_profiles: [1]` front-matter (§A2);
4. the **`structure_key` callable** packaged for your trampoline (`FdxOperandDesc`
   projection + `From` adapters);
5. the **commutative-operand canonical sort** (§C).

None is a research risk.

## F. Net

**Baracuda ratifies Profile v1.** Send the full rev-4 `fkc-fusion-patterns.md` so we
pin the commutative sort key, and we proceed to the conforming publish for the
lockstep crate bump. Stamp it — *Profile v1, ratified*.
