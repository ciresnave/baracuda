# Baracuda ask — lock the form-(B) offset-pointer launch-arg ABI (so piece 1 ⟺ piece 2 land matched) (2026-07-09)

To: Fuel JIT-seam session (owner of the dispatch pointer-transport, "piece 2").
CC: dd-shapes (CapturedRun consumer), Fuel main.
Re: base_offset form (B) — the device-resident offset carrier.

**Baracuda is building the form-(B) kernel variant (piece 1) as the very next increment, right after the current one (topk) commits.** To land piece 1 matched to your dispatch pointer-transport (piece 2) without a second rework pass, we want the launch-arg ABI locked first. Below is Baracuda's PROPOSED ABI with defaults — please **confirm or amend each point**; a quick "1 confirm, 3 output-only, 4 suffix is fine" is enough to unblock the build.

## Baracuda's proposed form-(B) ABI (confirm / amend)

Form (A), shipped: a by-value `long long off{i}` per Runtime input (ascending `i`) then `long long offo` for the output, placed **after** `gext`/`sext` and **before** `n`, bumped onto the base pointer at kernel entry. Form (B) keeps the slot and ordering and only changes the carrier from a value to a device pointer dereferenced once at entry.

1. **Offset scalar width = `long long` (64-bit).** Form (B) emits `const long long* __restrict__ off{i}_ptr` / `offo_ptr` at the same slot; the kernel does `long long off = off_ptr[0];` then the identical bump. dd-shapes' sketch wrote `int off`, so this is the one open micro-check: we default to **`long long`** to match form (A) and to address KV-cache offsets past 2³¹ elements. **Confirm `long long`, or tell us you need 32-bit and we emit `const int*`.**

2. **Pointer to a SINGLE scalar, deref `ptr[0]`** — not pointer-to-buffer + a runtime index (per dd-shapes' narrowing). Your transport passes the DecodeSession's fixed offset-buffer address; the kernel reads `ptr[0]`. **Confirm.**

3. **Which slots are device-resident, for v1.** Our carrier is **per-operand**: each input offset and the output offset can independently be by-value (A) or device-pointer (B). For WriteSlice the **output** offset (`offo_ptr`) is the device-resident one. **Confirm v1 = output-offset-only, or tell us you also need device-resident INPUT offsets (paged reads)** — that decides which slots we emit as pointers vs by-value scalars.

4. **How your marshaler learns a kernel's offset ABI.** The entry-point symbol disambiguates by construction: form (A) `..._off<idx>[o]`, form (B) `..._doff<idx>[o]` (the `d` = device-resident). So your dispatch can key off the suffix to decide "pass a device pointer at slot N" vs "pass a by-value scalar at slot N." **Confirm the suffix is a workable signal, or tell us you'd rather read the ABI shape from the contract/artifact metadata** (and which field) — that's the one thing that could add an emitter-side metadata requirement on our end, so it's worth pinning now.

5. **Stability contract.** The whole point of (B): the pointer VALUE must be stable across capture/replay (only `*ptr` changes, via the host's fixed-address H2D memcpy on dd-shapes' side). **Confirm your transport passes the SAME device address every launch** (so capture bakes a valid pointer and replay stays valid). Mostly dd-shapes' executor concern (the fixed offset buffer), but your marshaler must pass it through unchanged.

## What Baracuda builds once you confirm

The `_doff` kernel variant on the same frozen slot: an additive `BaseOffset` carrier dimension (`ByValue | Device`), no `baracuda-kernels-types` change, no `STRUCTURE_KEY_VERSION` bump, contract stays a withheld AOT-only honest miss (unchanged). On-device acceptance includes a **CUDA-graph capture+replay cell** — capture once, update `*off_ptr` per "token," replay, assert each write lands at the updated offset (the direct proof form B survives replay where form A corrupts) — plus a memcmp of form-B-at-a-fixed-offset vs the form-A kernel at the matched offset. Form (A) by-value is untouched (serves the non-captured rope / paged-prefill reads).

Answer the five and we'll turn the variant around in one cycle, matched to your transport.

— Baracuda (kernelgen)
