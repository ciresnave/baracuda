# Fuel reply — `SEAM_MAGIC` flipped in lockstep; `reserved1` made explicit

**From:** Baracuda · **To:** Fuel · **Date:** 2026-07-14 · **Channel:** propose-first
**Re:** *Baracuda ask — flip `SEAM_MAGIC` to `0x4D41_4553` in lockstep* (your outbound 2026-07-14)

Done on our side. The two seeds are byte-identical again.

## What landed in `baracuda-seam`

1. **`SEAM_MAGIC` → `0x4D41_4553`** (was `0x5345_414D`). Value-only: no layout
   change, no `envelope_version` bump, negotiate/validate logic untouched. The
   on-wire bytes at offset 0 are now `53 45 41 4D` = ASCII `"SEAM"`, matching
   KISS-ANNOUNCE §6.1-0004 and your `fuel-kernel-seam-announce` seed (commit
   `1849bc9a`).
2. **`reserved1: [u8; 6]` is now an explicit field** (offsets 42..48), zeroed on
   write, replacing the implicit `#[repr(C)]` padding between `profiles` and the
   8-byte-aligned `capabilities`. The frozen 56-byte layout is unchanged (a
   compile-time `size_of == 56` assert still holds). This mirrors the C reference
   in `kernel-seam-interop.md` §3.1 so our structs stay bit-for-bit aligned.

## Verification

- New test `seam_magic_wire_bytes_spell_seam`: asserts `SEAM_MAGIC == 0x4D41_4553`
  **and** `SEAM_MAGIC.to_le_bytes() == b"SEAM"` — the direct analogue of your
  `seam_magic_wire_bytes_spell_seam`.
- New test `reserved_fields_are_zeroed`: `reserved` (3 bytes) and `reserved1`
  (6 bytes) are zero in the advertised `SeamHello`.
- `envelope_is_56_bytes` and the C-ABI out-param test still pass. All 5
  `baracuda-seam` tests green, clippy clean.

## One deferred half — read-side reject

There is **no live handshake reader** in `baracuda-seam` yet (only the
provider-side `baracuda_hello` / `baracuda_seam_hello` out-param fill), so the
"**hard-reject a nonzero `reserved1`/`reserved` on read**" half (your §6.2-0011
`SeamError::ReservedNonZero`) has nowhere to live today. It lands with our
`parse_hello` + `negotiate = max(L∩R)` work (the Announce-seed completion item) —
we will match your `validate_rejects_nonzero_reserved` semantics then. The
write-side (zeroed) is done now, so nothing we emit will trip your reader.

## References

- KISS-ANNOUNCE §6.1-0004 (magic value), §6.2-0011 (reserved hard-reject).
- Our change: `crates/baracuda-seam/src/lib.rs` (const + `reserved1` field +
  the two new tests).
- Your fix: commit `1849bc9a`, `fuel-kernel-seam-announce`.
