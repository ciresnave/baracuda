# Baracuda ask — bundle-schema reconciliation for live JIT adopt (1 question), the Relu NaN convention (1 proposal), and a Max/Min doc FYI

**Context:** the op_kind spelling reconciliation landed (every primitive advert now
emits your `lower_op_kind` spellings, proven through your real `import_bundle_str` —
all 28 mapped spellings import Ok and resolve; details ride the alpha.76 landing
note). Running your importer against a full raw Baracuda bundle surfaced the
remaining schema divergences, which we're fixing Baracuda-side for alpha.76. One
question needs your answer to finish the JIT wiring; one NaN-semantics decision is
yours to make; one FYI about your own docs.

## 1. QUESTION — who wraps `art.contract` for the adopt-time import?

Your adopt path parses `art.contract` through the existing FKC importer. Your parser
collects a ```fkc block only under a `## ` heading, inside a file with front matter
(`provider:`/`backend:`/`link_registry:`), and — important hazard we verified —
**silently drops headingless blocks**: a file whose sections are all headingless
imports `Ok` with **zero** registrable sections, so a mis-framed adopt would look
like success and adopt nothing.

Today `BaracudaSynthesizer` retains the **bare per-kernel contract block** (exactly
what `contract()` emits — no front matter, no heading). Two options; we're fine with
either, but the seam needs ONE answer:

- **(a) Fuel wraps:** your adopt path wraps `art.contract` in its own front matter +
  a `## <entry_point>` heading before importing. Zero Baracuda change; you control
  the provider identity of adopted kernels (arguably where it belongs — the adopt
  context knows the provider/link registry).
- **(b) Baracuda ships a full mini-bundle:** `synthesize` stores front matter +
  heading + block in `art.contract` so it is `import_bundle_str`-ready verbatim.
  One-line-ish change on our side; your adopt path stays a straight pipe.

Tell us (a) or (b) and we ship accordingly in alpha.76. **Either way, we recommend a
Fuel-side guard**: treat an imported provider with zero registrable sections as an
error at adopt time (it is always a framing bug, never a valid adopt) — that closes
the silent-no-op hazard for every future producer.

## 2. Baracuda-side fixes landing in alpha.76 (FYI — no action, correct us if wrong)

Verified against your parser/lowerer/corpus (`elementwise-binary.fkc.md` as ground
truth), the emitter will change to:

| today (Baracuda emits) | alpha.76 (matching your schema) |
|---|---|
| `backend: cuda` | `backend: Cuda` (your `lower_backend` casing) |
| headingless ```fkc blocks | `## <section>` heading per kernel in bundle assembly |
| `layout: contiguous` (string) | `layout: { contiguous: required, strided: rejected, ... }` (your `LayoutSpec` inline map, driven from our structure-key facts) |
| anonymous `inputs:` entries | `- name: in0` / `in1` … (names, so rules can reference them) |
| `dtype_rule: same_as_input(0)` | `dtype_rule: passthrough(in0)` (your `parse_dtype_rule`; the old spelling parsed as `DtypeRule::Other`, which **silently omits the output dtype from the binding key** — worth knowing it fails soft) |

## 3. PROPOSAL — the Relu NaN convention (currently blocks our Relu advert)

Our op_kind sweep deliberately **withholds** the Relu primitive advert (honest miss)
because the semantics diverge and an adopted JIT relu would silently change results:

- **Our synthesized relu NaN-PROPAGATES** — `x < 0 ? 0 : x` (torch.relu semantics:
  `torch.relu(nan) = nan`).
- **Your ReluElementwise slot NaN-SCRUBS** in all three of your authorities: the CPU
  reference core (`x.max(0.0)` — Rust max returns the non-NaN operand), your FKC doc
  ("NaN-as-missing (f32::max)"), and the incumbent CUDA binding (our bespoke
  `unary_relu_fp.cu`, `fmaxf(x, 0)` — also scrubbing).

So your CPU and CUDA agree with each other but **both disagree with torch.relu**.
Proposal: **reconcile Fuel-side to NaN-propagating** (torch parity), i.e. the CPU
core becomes `if x.is_nan() { x } else { x.max(0.0) }`-equivalent and the CUDA
binding routes to a propagating kernel (we can ship one bespoke, or you adopt the
generated form). If instead you deliberately want NaN-as-missing relu, tell us and
we'll emit an fmaxf-form kernel for the ReluElementwise advert — either way, once
the convention is pinned we lift the withhold. Until then Relu simply doesn't
advertise (everything else does).

## 4. FYI — your FKC doc misdescribes your own CUDA Maximum/Minimum

While auditing the same class: your `elementwise-binary.fkc.md` claims NaN-as-missing
for Maximum/Minimum **on all backends**, but your own incumbent CUDA binding for
`OpKind::MaximumElementwise`/`MinimumElementwise` routes to our
`binary_maximum_fp.cu`, which is explicitly **NaN-PROPAGATING** (torch.maximum
parity; it reserves `fmaxf` for the separate Fmax op). So the doc is wrong about
your CUDA path, and your CPU (`a.max(b)`, scrubbing) genuinely diverges from your
CUDA on NaN inputs — a pre-existing Fuel-internal CPU↔CUDA inconsistency,
independent of anything we ship. Our Maximum/Minimum adverts stay mapped (they are
behaviorally identical to your CUDA incumbent). Flagging so you can fix doc or CPU,
whichever you intend.

— Baracuda
