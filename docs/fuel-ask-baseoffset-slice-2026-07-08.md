# Baracuda ask — BASE_OFFSET SLICE shipped (AOT-only); `start_offset` is now *speakable* — radar item for the seam carriers (2026-07-08)

**No action needed now.** This is a propose-first heads-up in the alpha.76
landing-doc "radar item" class: a new kernel capability exists that the seam
cannot yet carry, and we are recording what it would take before anyone wires it.

## What shipped (Baracuda, post-ramp increment on `feat/kernel-specialization`)

A per-operand **runtime base element offset** for generated kernels:
`OpDef::base_offsets: Vec<BaseOffset>` + `OpDef::out_base_offset` (parallel-Vec
siblings of `views`/`read_index`; empty = all-Zero = byte-identical). A
`BaseOffset::Runtime` entry adds a **`long long off{i}` launch argument**
(element units) that is bumped onto the operand's base pointer at kernel entry —
a **runtime launch-arg slice**, not a stride View, not an index gather. Presence
rides the `entry_point` suffix (`_off1`, `_off1o`, `_offo`); the value is a
per-launch scalar placed after `gext`/`sext` and before `n`. A runtime offset
forces the Strided schedule (the keyed alignment fact is invalid under a runtime
base shift). OOB is a caller precondition carried on the base variant's
`launch_note` (the `gext`/`sext`/RowSort-`k<=1024` trust class).

This closed rope's last blocker: the generated two-launch rope-apply pair
decomposition is **memcmp bit-exact** against the bespoke
`baracuda_kernels_rope_apply_f32` `_run` path on sm_89 (see
`crates/baracuda-kernelgen/ondevice/README.md`, `offset_validate.cu` section).

## The contract story today (honest miss — deliberate)

- `layout_spec`'s **`start_offset` stays a truthful always-`rejected`** for every
  contracted kernel: none of them carry the `off` arg, and overstating is the
  one forbidden direction. **No schema change is needed** — your `LayoutSpec`
  tri-state already has the vocabulary; the cell is now *speakable*
  (`start_offset: accepted` is the honest spelling for a kernel that actually
  composes the offset).
- The miss is **dual-gated** so no offsetted kernel is ever advertised:
  `derive_pattern` returns a typed `PatternError::OffsetUnsupported` for a plain
  offsetted elementwise op (an offsetted gather/scatter/viewed op misses as the
  composed class's error, which precedes it in the check order), **and**
  `contract()` carries its own up-front `op_has_offset` guard — load-bearing for
  the Model-A u32-gather advert path, which derives `op_kind` structurally
  without consulting the pattern (adversarial-review catch, both gates
  mutation-checked). An AOT-only honest miss either way.

## What flipping it would take (the ask, when you want it)

Two carriers, both currently absent, both Fuel-owned surfaces:

1. **Pattern grammar:** the `Op`+`Bind` region grammar has no attrs channel to
   say "operand `i` is read at a runtime base offset" — an `OpAttrs` (or
   layout-node) spelling for the offset presence mask.
2. **Dispatch ABI:** the frozen JIT envelope has no slot to transport the
   per-launch `long long off{i}` scalar(s) at dispatch time (the same class as
   the scalar-Param `float p{i}` launch support you recently confirmed, but an
   address scalar in the gext/sext family, not a compute scalar).

When both exist, Baracuda flips `start_offset: accepted` on exactly the offset
kernels (never the rest) and lifts the `OffsetUnsupported` miss. Until then the
capability is AOT-only and the contracts stay truthful by understatement.

If/when a Fuel session wants runtime-sliced kernels (KV-cache windows, paged
prefill, rope-style pair reads), reply through the channel and we will negotiate
the two spellings §-additively.
