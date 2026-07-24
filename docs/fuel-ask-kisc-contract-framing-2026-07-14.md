# Fuel ask — adopt KISC self-delimiting framing for the kernel contract

**From:** Baracuda · **To:** Fuel · **Date:** 2026-07-14 · **Channel:** propose-first
**Companion:** KISS-Contract §2.8 / §6.11 (self-delimiting document framing); ThinkersJournal/KISS issue tracker.

This is a request to change **one thing on the shared seam**: how a kernel contract
document is *framed* (its outer envelope), not what a contract *says*. The seven
section blocks, the `ImplId` tuple, the `accept`/`return`/`op_params`/`caps`/`cost`/
`precision`/`determinism` fields — all unchanged. Only the bytes that wrap them, and
the reader's failure discipline, change. Nothing about your `Synthesizer` trait, the
`JitRequest`/`JitResponse` envelope, or `take_kernel` moves.

## TL;DR

1. Our current contract framing (a markdown `##` heading + a ```` ```fkc ```` fence,
   [`contract.rs:82-136`](../crates/baracuda-kernelgen/src/contract.rs)) has **two
   real bugs**: it **silently drops a headingless block** (adopts it as an empty /
   no-op contract), and one bad `op_kind` is **bundle-fatal**
   ([`contract.rs:884-892`](../crates/baracuda-kernelgen/src/contract.rs)) — a single
   malformed kernel poisons the whole bundle.
2. KISS-Contract already defines the fix as its transport: a **self-delimiting
   structured/text document** framed by a pinned `KISC` magic header line + version +
   `len=<N>` + `crc32=<…>`, with a **hard-reject** discipline — a magic-less,
   headingless, or crc-mismatched document is a *typed decline*, never silently
   adopted (KISS-Contract §2.8, §6.11, clauses 6.1-0001/-0002).
3. We propose Baracuda emits, and Fuel's FKC importer reads, this KISC framing on the
   shared seam. It is a **strictly louder** reader — every contract you accept today
   still parses; the only new behavior is that malformed/empty documents that today
   pass as "success" now fail loudly at the point they can be fixed.

## The problem, concretely

Our emitter frames each kernel contract as a markdown heading plus a fenced block. Two
failure modes fall out of that choice:

- **Silent-drop.** A block that arrives without its `##` heading is currently adopted
  as an *empty* contract rather than rejected. An empty contract "succeeds" — the
  consumer binds nothing and discovers the miss later, far from the cause. This is the
  single worst class of seam bug: a malformed input that *looks like success*.
- **Bundle-fatal blast radius.** Because the bundle is one concatenated text, a single
  unimportable `op_kind` aborts the whole bundle
  ([`contract.rs:884-892`](../crates/baracuda-kernelgen/src/contract.rs)) instead of
  declining just the one offending contract and keeping its siblings.

Both are framing problems, not schema problems — the section contents are fine.

## The proposal — KISC framing (KISS-Contract §2.8 / §6.11)

A contract becomes a **self-delimiting structured/text document** (still text, still
greppable — *not* a binary envelope), with:

- a **header line** beginning with the 4-byte magic `KISC`, then the `kiss-contract`
  kind, the document **version**, an inner-body byte length `len=<N>`, and a
  `crc32=<…>` over that body;
- the **seven section blocks** under pinned heading lines (unchanged content);
- a **hard-reject** reader: a document that does not begin with `KISC`, whose header
  line is absent/malformed, whose version is unknown, or whose `crc32` mismatches is a
  **typed decline** — the reader never repairs it and **never imports a headingless or
  magic-less block as an empty contract** (§6.1-0002).

On the contract-query / provision seam the whole document travels as an **opaque,
length-delimited payload** (a `u32` byte-length + that many bytes) — Announce/Synth
never parse the inner framing, so this change is invisible above the transport.

## What changes on each side

- **Baracuda (us):** replace the `##`+fence emission in `contract.rs` with the KISC
  header line + `len`/`crc32`; make per-contract emission independent so a withheld or
  malformed contract is one typed decline, not a bundle abort. (This also unblocks the
  per-document **sibling isolation** we've asked KISS to make normative.)
- **Fuel (you):** your FKC importer reads the `KISC` header line, verifies `len`/
  `crc32`, hard-rejects a magic-less/headingless/mismatched document with a typed
  decline, and imports each document independently. A contract you accept today still
  imports byte-for-byte identically once past the header.

## Compatibility & rollout

- **No schema change.** Every field and section is identical; only the wrapper and the
  reader's failure mode change.
- **Cutover, not a flag day.** Suggest we gate on a seam capability bit (KISC-framing
  supported) during the transition, so a mixed fleet negotiates the framing the way
  `SeamHello` negotiates everything else. Once both sides advertise it, the old
  `##`+fence path retires.
- **Determinism.** The `crc32` is over the inner body only; identical contract bodies
  produce identical frames, so byte-for-byte reproducibility of emitted contracts is
  preserved (and now checkable).

## What we need from you

1. A read of KISS-Contract §2.8 / §6.11 and a thumbs-up (or pushback) on adopting KISC
   framing on the shared seam.
2. Agreement on the **capability bit** for negotiated cutover, so neither side breaks
   during rollout.
3. Confirmation that your FKC importer can move to **per-document hard-reject +
   isolation** (one bad contract declines alone) — this is the half that kills the
   bundle-fatal blast radius.

Nothing here is blocking on your side today; it needs no code from you this week. The
point is to align before either emitter or importer touches the shared framing.

## References

- KISS-Contract §2.8 (transport — self-delimiting document that fails loudly), §6.11
  (header line + section framing), clauses 6.1-0001 / 6.1-0002 (hard-reject; no
  headingless/magic-less adoption).
- Current framing + bugs: [`contract.rs:82-136`](../crates/baracuda-kernelgen/src/contract.rs)
  (silent-drop), [`contract.rs:884-892`](../crates/baracuda-kernelgen/src/contract.rs)
  (bundle-fatal).
- Companion KISS issues (steward tracker): sibling-document isolation is part of the
  cross-cutting robustness issue; the typed-decline code currency is the decline-code
  work already landing in `baracuda-kernelgen` (`JitError::decline_code` → KISS
  `CANNOT_PROVISION`).
