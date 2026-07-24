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

## Non-goals

Not in the identity key (C-3). Not a verbatim Fuel-struct lift. Not the DLPack ABI structs (those stay cuVS-FFI-only).
