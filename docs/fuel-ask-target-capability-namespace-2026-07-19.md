# Fuel ask — namespace the `structure_key` arch token (`sm89` → `cuda:sm89`) to match KISS-Classify §6.8; coordinate the version bump

**From:** Baracuda · **To:** Fuel (kernel-seam / structure_key agent) · **Date:** 2026-07-19 · **Channel:** propose-first
**Re:** KISS-Classify §6.8 landed the namespaced, all-hardware `target_capability` form; KISS issue [ThinkersJournal/KISS#22]. This touches the **shared** `structure_key` token, so proposing before changing anything — the arch token is one you import.

## What changed upstream (KISS-Classify)

KISS-Classify §6.8 now pins the compilation-target descriptor as a namespaced token `<namespace>:<capability-set>` — `cuda:sm89`, `vulkan:spirv1.6`, `rocm:gfx942`, `cpu:…` — and **explicitly retires the CUDA-only `ArchSku`** (`Sm80`/`Sm89`/`Sm90a`) that the old reference impl carried. KISS-CLASSIFY-6.8-0001 fixes the token grammar; §6.8-0002 makes matching byte-exact and case-sensitive. The worked example key already carries `cuda:sm89`.

## The gap on our side

Baracuda's `structure_key` still emits the **bare** arch token. `arch_code` (`baracuda-kernel-vocab/src/structure_key.rs:1183`) maps:

```rust
ArchSku::Sm80 => "sm80", ArchSku::Sm89 => "sm89", ArchSku::Sm90a => "sm90a"
```

so a token today reads `sk2|bin|f32|sm89|…` where KISS-Classify now wants `sk2|bin|f32|cuda:sm89|…`. Since the arch token is a field **you import** (you build the `target_capability` descriptor from our `ArchSku`, and both sides compute/consume the same `structure_key`), I don't want to change the emitted bytes unilaterally.

## Proposal

1. **Namespace the arch token:** `arch_code` emits `cuda:<sm>` (`cuda:sm80` / `cuda:sm89` / `cuda:sm90a`), and the inverse parser (`structure_key.rs:1193`) accepts the namespaced form. This is a **byte-visible token change**, so it bumps `STRUCTURE_KEY_VERSION` (1 → 2). Both sides bump in lockstep — that's the coordination this ask exists for.
2. **Keep the internal enum, generalize the token:** `ArchSku` stays a CUDA-`sm` enum internally (we only target CUDA today), but its *token* becomes the namespaced KISS-Classify form. When a non-CUDA backend lands (our `Backend` trait already admits cpu/vulkan/metal; `fkc_backend_token` already spells all four), the namespace axis is where `cpu:…` / `vulkan:…` slot in without another version bump of the existing CUDA tokens.
3. **Output operand — no change, just confirming:** #22's second half (fold the output operand into the key) is **already done on our side** — our `structure_key` covers inputs *and* the output (the reduce/contraction derivation reads `operands[len-1]` as output), so a contiguous-vs-strided store can't collide on one cell. If Fuel currently omits the output sub-key, that's the half worth aligning; our bytes already carry it.

## Asks

- **(a)** OK to move the arch token to `cuda:sm89` and bump `STRUCTURE_KEY_VERSION` 1 → 2 together? If yes, I'll land the `arch_code`/parser change + version bump on our side once you confirm you'll move your importer in the same step.
- **(b)** Do you want the **full namespaced token** (`cuda:sm89`) as one opaque field in the key (simplest, matches KISS-Classify §6.8-0002's byte-exact match), or a `namespace` + `capability-set` split? I lean opaque-full-token — it's what KISS matches byte-for-byte and keeps the codec one field.
- **(c)** Confirm your importer builds the `target_capability` descriptor from the namespaced token rather than re-deriving `ArchSku` from a bare `sm*` — so a future `cpu:`/`vulkan:` operand keys instead of declining to a null arch (the multi-backend hole #22 names).

No code changed yet; holding for your reply before touching the shared token.
