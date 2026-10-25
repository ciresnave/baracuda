# FDX-successor — Baracuda's co-design position (prep)

**Status:** prep for the queued three-way (KISS + Fuel + Baracuda) FDX-successor design thread (RECONCILIATION **D7**). Not yet sent — Baracuda's opening position, ready when the thread opens.

## Frame

D7 blesses a **neutralized** shared quant/MX sidecar to replace each corner's private FDX mirror — *not* a verbatim lift of Fuel's current struct. Baracuda today carries a private projection of FDX at the seam and wants to drop it for the shared one. The guardrail (Eric): the successor must stay project-/language-agnostic so any ML ecosystem can adopt it.

## The invariant Baracuda holds (C-3)

**Quant/MX facts stay OUT of the identity key, in the sidecar.** The `structure_key` stays a finite, published closed dtype set (finiteness = publishability); the sidecar is the open overlay for quant granularity/scale/MX block structure. This is the pattern that lets any ecosystem add a quant scheme without a key-schema break — it serves KISS neutrality and Baracuda equally. MX is modeled at the *quant* layer, never as a dtype token.

## What Baracuda needs represented (from its live model)

Baracuda's private FDX projection (`baracuda-kernel-vocab/src/structure_key.rs`), which the successor must express:
- **`QuantFamily`** — `{ Ggml, Mx, AffineInt, AffineFloat, AffineBlock }`. Covers the shipped GEMM families: GGUF `mmvq`, NF4/QLoRA, Marlin/AWQ int4, FP8, SmoothQuant, OCP microscaling (MX, per-block F8E8M0 scale).
- **`ScalePlacement`** — `{ Inline, SeparateBuffer, BroadcastPerAxis }`.
- **`QuantFacts`** — the per-operand quant descriptor (`OperandDesc.quant`).
- **Seam advertisement** — `SEAM_CAP_DLPACK_EXT_{V1,MX,GGML,AFFINE,SYMBOLIC,GATHER}` (an FDX overlay bit-set, never the DLPack ABI, never the key).

## Alignment with Fuel's substance (jvwnb5ut)

Fuel brings: overlay-at-the-seam pattern; **model-B** (quant scale = a *sibling operand*, never embedded in the tensor); the self-describing **DType / SType / Encoding** split. Baracuda concurs with model-B (scale-as-sibling is cleaner than inline and matches `ScalePlacement::SeparateBuffer`), and the DType/SType/Encoding split maps onto Baracuda's `QuantFamily` (encoding) + `ScalePlacement` (scale layout) + logical dtype.

## Baracuda's proposed neutral shape

`{ logical_dtype, quant_family (encoding), scale: {placement, granularity, dtype}, block_structure (for MX: block size + scale dtype) }` — a neutral superset of Baracuda's `QuantFamily`/`ScalePlacement` and Fuel's DType/SType/Encoding + model-B. DLPack codes are a *spelling guide*, never the normative owner. Bless the neutralized successor; Baracuda then drops its private mirror for the shared one.

### Gaps raised by Fuel (2026-10-25), accepted into the proposal

Fuel concurred and listed four things the neutral shape must cover before its mirror can go. Baracuda accepts all four; none is contradicted by Baracuda's model (`QuantFamily` / `ScalePlacement` / `QuantFacts` no longer exist in this repo: `baracuda-kernel-vocab` left at alpha.79, so there is no in-tree type to compare against):

1. `scale.placement` keeps `Inline` (GGUF Q4_0 / Q8_0 carry an f16 scale in each block); scale-as-sibling operand is the default, not the only case.
2. `scale` is a recursive descriptor (bnb NF4 double-quant: a per-block scale that is itself quantized, with an offset).
3. An optional `zero` operand beside `scale`, with the same placement / granularity vocabulary (AWQ / GPTQ int4, `AffineInt`). The op vocabulary already carries `zero_point` (per-tensor, per-channel, per-token, per-group).
4. `block_structure` records packing: nibble order and the packing axis (AWQ interleave, bnb nibble order), else two incompatible layouts share one descriptor. The Unpopped `S4` / `U4` element type already fixes low nibble = even index.


## Non-goals

Not in the identity key (C-3). Not a verbatim Fuel-struct lift. Not the DLPack ABI structs (those stay cuVS-FFI-only).
