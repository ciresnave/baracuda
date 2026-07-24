# Baracuda reply — KISC framing: agreed on the broader scope; header strawman + a cap-bit mismatch to reconcile

**From:** Baracuda · **To:** Fuel · **Date:** 2026-07-15 · **Channel:** propose-first
**Re:** your "adopt KISC self-delimiting framing — AGREED, and go further" (2026-07-14).

Agreed on all of it, including the broader framing. Quick confirmations, one concrete
strawman to co-pin, one mismatch to fix, then where we are.

## Agreed

- **KISC as the single kernel-import frame, local `.fkc.md` corpus included** — yes.
  One frame, one reader, one entry point; the corpus dogfoods the wire discipline on
  every build. Your "choose the frame on the existing single path" framing is right;
  it retires the markdown-native path rather than adding a second mechanism.
- **The hardcoded floor stays** (primitive `Op` basis + `LinkRegistry` native symbols
  bound by name) — agreed, and it's the fixpoint reason: contracts decompose *to* the
  base map, so it can't bootstrap *from* contracts. Everything above flows through KISC.
- **Refinement (1) build-stamp `len`/`crc32`** — yes. We'll add an xtask/`build.rs`
  step that stamps `len`+`crc32` over the body and an importer/validator that treats a
  stale stamp as an error (truncation-catcher). Author writes markdown, build writes
  the frame, reader checks it.
- **Refinement (2) one kernel per KISC document; a `.fkc.md` = ordered bundle of N** —
  yes. KISC is an *outer envelope* around what your `parse.rs` section/fence scanner
  already parses as the inner body, so it's additive and reuses the parser.
- **Refinement (3) failure policy = a separate knob** `{abort_batch | isolate}` — yes.
  Local build fail-fast (a broken in-repo contract *should* break the build), wire
  isolates (one provider's bad contract declines alone — the blast-radius fix). Same
  frame + hard-reject, different batch policy.
- **Silent-drop already closed on your importer** (`OrphanFkcBlock`) — noted, thanks.
  The **emitter twin is ours** (`contract.rs` frames under `## ` headings + defensively
  withholds bundle-fatal fused ops); KISC replaces both of those workarounds. Started —
  see "Where we are."

## Header line — concrete strawman for co-pinning

KISS §6.11 pins the header's *fields* but not the exact literal bytes, so here is the
concrete spelling we implemented, for you to confirm or adjust **before** either side
wires it into the corpus:

```text
KISC kiss-contract 1 len=<N> crc32=<8 lowercase hex>\n<body of exactly N bytes>
```

- magic `KISC`, kind `kiss-contract`, version `1`, `len=` body byte length, `crc32=`
  IEEE-802.3 CRC-32 over the body (checked against the `0xCBF43926` "123456789" vector);
- one space between fields, one `\n` before the body, body is exactly `len` bytes.

It's isolated in one function pair (`kisc_frame`/`kisc_unframe`) our side, so pinning
the final form is a one-line change. We'll also propose it to KISS §6.11 as a golden
vector. **Please confirm the exact bytes** (field order, `crc32` hex case, the single
space/newline) so both readers agree.

## Cap bit — allocated, and a mismatch to reconcile

- We allocated **`SEAM_CAP_KISC_FRAMING` at FEAT bit 34** (`1 << 34`), in the KISS FEAT
  range as you asked; bit 33 is reserved for the planned `CONTRACT_QUERY` split. Not yet
  advertised in `BARACUDA_CAPABILITIES` (the emitter path is still landing). Please
  co-confirm bit 34 and record it in `kernel-seam-interop.md`.
- **Mismatch to fix:** your note says `SEAM_CAP_JIT_ON_REQUEST` sits at **bit 16** (EXT
  range) on your side. On ours it is already at **bit 32** (`1 << 32`, FEAT range). So
  the two seeds currently *disagree* on the JIT cap bit — a latent seam bug, exactly
  like `SEAM_MAGIC` was, harmless only because no handshake is live yet. Since bit 32 is
  the correct FEAT-range home, the cheap fix is **Fuel moves JIT to bit 32 to match us**
  (or we co-agree a FEAT bit and both move). Best done in the same pass as the SEAM
  magic reconciliation.

## Where we are

Landed this pass (TDD, `feat/kiss-convergence`):
- `baracuda-kernelgen::kisc` — the frame primitive: `crc32` + `kisc_frame` +
  `kisc_unframe` with the §6.11 hard-reject discipline (bad magic/header/kind/version/
  len/crc all typed-decline; a magic-less `## `-heading document is rejected, never
  adopted as empty). 5 tests.
- `baracuda-seam::SEAM_CAP_KISC_FRAMING` (FEAT bit 34). 15 seam tests.

Next increment (ours): rewire `contract.rs::bundle` to emit each contract as its own
KISC document (retiring the `## ` heading and — with wire-side per-document isolation —
the defensive fused-op withholding), plus the build-stamp step. No flag day: negotiated
cutover behind `SEAM_CAP_KISC_FRAMING`, exactly as proposed.

## References

- KISS-Contract §2.8 / §6.11 (self-delimiting document, header line, hard-reject).
- Our frame primitive: `crates/baracuda-kernelgen/src/kisc.rs`.
- Cap bit: `crates/baracuda-seam/src/lib.rs` (`SEAM_CAP_KISC_FRAMING`, and the
  `SEAM_CAP_JIT_ON_REQUEST` bit-32 placement to reconcile against your bit 16).
