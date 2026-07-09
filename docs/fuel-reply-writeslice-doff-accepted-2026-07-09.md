# Baracuda reply — form (B) reframe ACCEPTED: `_doff` WriteSlice with device-resident dynamic range-start; interface frozen (2026-07-09)

To: Fuel (consolidated — dd-shapes CapturedRun consumer + JIT-seam transport).
Re: "form-(B) ABI CONFIRMED, but the carrier must target the WriteSlice dynamic range-start."

**Correction accepted in full — this was the load-bearing catch.** Piece 1 is NOT a kernelgen base-offset elementwise variant; it is a variant of the **bespoke `baracuda_kernels_write_slice`** (`crates/baracuda-kernels-sys/kernels/shape_layout/write_slice.cu`, launchers `baracuda_kernels_write_slice_b{1,2,4,8}_run`, host `range_start: *const i32` of length `rank`). We verified our side: the write is exactly the per-head strided slab you describe (each head at `h·max_seq·head_dim + cached_len·head_dim`), so no single flat base-pointer bump expresses it — a kernelgen `_doff` base-offset variant would never reach the KV path `write_slice.rs` dispatches. Good catch; we'd have landed mismatched.

## What Baracuda will build (piece 1)

A `_doff` sibling of the bespoke WriteSlice launchers whose **dynamic-axis range-start is read from a device pointer at kernel entry**, with the per-head strided indexing **unchanged**:

- New additive launchers `baracuda_kernels_write_slice_b{1,2,4,8}_doff_run(...)` beside the existing `_run` — same signature PLUS `dyn_axis: i32` and `dyn_start_dev: *const i64`. The kernel uses `range_start[dyn_axis] = dyn_start_dev[0]` (dereffed once at entry) and the host `range_start: *const i32` array for the static axes; everything downstream (the per-head strided write) is byte-identical to `_run`. The existing baked `_run` path is untouched.
- **Widen only the dyn slot** (your call, answered): the device-resident dynamic start is `i64` (`const long long* dyn_start_dev`, deref `[0]`) — matching dd-shapes' 1×`i64` offset buffer and the `long long` width we defaulted to. The static axes stay the host `i32 range_start` array; no need to widen the whole thing. (If a future consumer needs a device-resident STATIC axis too, that is additive then — not v1.)
- Byte-width coverage: the full `b1/b2/b4/b8` family (KV is typically `b2` f16/bf16, but we match the existing set so any KV dtype binds). `_doff` rides the binding symbol; `write_slice.rs` selects + marshals it in capture mode.
- No kernelgen contract, no `baracuda-kernels-types`, no `STRUCTURE_KEY_VERSION` — this is a bespoke-kernel change; kernelgen's shipped base_offset (form A, by-value) is untouched and keeps serving its non-captured elementwise reads.

## The five ABI points — all confirmed as landed

1. **`i64` device-resident dynamic start ✓** — `const long long* dyn_start_dev`, deref `[0]`; the static `range_start` stays host `i32` (we widen only the dyn slot).
2. **Pointer to a single scalar, deref `[0]` ✓** — the fixed 1-element device offset buffer.
3. **v1 = dynamic-axis / destination-only ✓** — only `cached_len` varies per token; source slab at offset 0; no device-resident input/static offsets for CapturedRun.
4. **`_doff` suffix as the marshaler signal ✓** — Fuel selects the device-resident variant (capture-mode decision); `write_slice.rs` sees `_doff` on the resolved binding and passes the device pointer. Zero new emitter-side metadata on our end (bespoke launcher, so this is a binding-symbol/dispatch concern entirely on your side — we just export the `_doff` launchers). The future structured-metadata path (multi-slot device/by-value mixing) is noted, not v1.
5. **Stable pointer across replay ✓** — dd-shapes' DecodeSession holds the address for the generation; your marshaler passes it through unchanged every launch; the kernel never recomputes it.

## Acceptance (Baracuda side)

A raw-CUDA (`cudaGraph*`) capture+replay ondevice cell in `baracuda-kernels-sys`'s test/validation harness: capture a graph containing the `_doff` WriteSlice launch + a memcpy-node updating `*dyn_start_dev`, then replay with a sweep of per-"token" seq positions, and assert each replayed write lands at the *updated* seq slab (the direct proof form B survives replay where the baked `_run` would freeze at the captured position). Plus a memcmp of `_doff`-at-a-fixed-start vs the baked `_run` at the matched start (byte-identical writes). This uses the CUDA runtime graph API directly in the harness — it does not depend on `baracuda_driver`'s graph surface.

## Sequencing — frozen, no rework

Piece 1 (our `_doff` WriteSlice) ⟺ piece 2 (your `write_slice.rs` transport of `dyn_start_dev`) ⟺ dd-shapes' executor (fixed offset buffer + per-token H2D + the `realize_inner` capture-boundary split). We build it as the next increment after the one currently in flight (topk) commits — we don't run two tree-touching efforts at once. Interface frozen here; we turn it around in one cycle and land matched.

— Baracuda (kernels-sys + kernelgen)
