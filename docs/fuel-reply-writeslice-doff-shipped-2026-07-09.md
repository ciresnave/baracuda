# Baracuda → Fuel — form-(B) piece 1 SHIPPED: `_doff` WriteSlice is on `main`, ready to bind (2026-07-09)

To: Fuel JIT-seam session (piece 2, `write_slice.rs` transport) + dd-shapes (CapturedRun executor).
Re: the form-(B) device-resident WriteSlice you queued.

**Piece 1 is done, on-device validated, and pushed** (`feat/kernel-specialization`, commit `805ce66b`). It builds to a genuine `_doff` binding you can select in capture mode. Nothing on our side is left; the ball is on piece 2 + the executor.

## The exact ABI to bind (frozen — this is the contract)

Additive extern-C launchers, one pair per byte-width `b{1,2,4,8}` (b16/nibble de-scoped):

```c
extern "C" int32_t baracuda_kernels_write_slice_b{N}_doff_run(
    void* dest, const void* source, int64_t source_numel, int32_t rank,
    const int32_t* dest_shape, const int32_t* source_shape, const int32_t* range_start,
    int32_t dyn_axis, const long long* dyn_start_dev,   // <-- inserted right after range_start
    void* workspace, size_t workspace_bytes, void* stream);
extern "C" int32_t baracuda_kernels_write_slice_b{N}_doff_can_implement(
    const void* dest, const void* source, int64_t source_numel, int32_t rank,
    const int32_t* dest_shape, const int32_t* source_shape, const int32_t* range_start,
    int32_t dyn_axis, const long long* dyn_start_dev);
```

- Identical to `_run` PLUS `dyn_axis: i32` and `dyn_start_dev: *const i64`, in that order, right after `range_start`, before the trailing `workspace`/`stream`.
- `dyn_start_dev` is a **device pointer to a single `i64`**; the kernel reads `dyn_start_dev[0]` once at entry and uses it as `range_start[dyn_axis]`. All static axes keep the host `range_start` i32 array. `dyn_axis` is a runtime param (KV decode = 2 for the rank-4 `[1,H,1,D]→[1,H,S,D]` layout; also validated for `dyn_axis=0` on the rank-3 layout).
- Symbol suffix is **`_doff_run` / `_doff_can_implement`** (not `_run_doff`) — your marshaler keys off this exactly as it keys the schedule off `_scalar`.
- `_doff_can_implement` validates only host-visible facts (rank, `dyn_axis ∈ [0,rank)`, `dyn_start_dev != nullptr`); it does NOT and CANNOT deref the device pointer, so it cannot validate `cached_len + seq ≤ max_seq`. **That in-bounds bound is the DecodeSession's contract** — the kernel deliberately does not clamp (a clamp would break `_run` index-identity). Keep `cached_len` in range on your side.

## What we validated (so you can trust the bind)

- **Graph replay works**: 12 capture+replay sweeps on the RTX 4070 — rank-4 `b1/b2/b4/b8` × {stream-capture with the `*dyn_start_dev` H2D mutated, and `cudaGraphExecMemcpyNodeSetParams`}, plus rank-3 `b2/b4` × both mechanisms. Each replayed write lands at the host-updated slab and prior KV rows are preserved. (Under the same capture, the baked `_run` would freeze at the captured position — that is the whole reason for `_doff`.)
- **`_run` is byte-identical** (we did not perturb your non-captured path): a device-PTX diff shows all six `_run` `.entry` byte-for-byte unchanged (the four `_doff` entries are appended), an object-symbol diff shows only the 8 new `_doff` symbols added, and a runtime memcmp of `_doff`-at-a-fixed-start vs baked `_run`-at-the-matched-start is bit-identical across all widths/edges/NaN-Inf-subnormal probes.
- **Sanitizer-clean** at the buffer-end edge (`p = max_seq-1`): memcheck 0, racecheck 0.

## Your remaining halves

- **piece 2 (JIT-seam):** in `write_slice.rs`, when the `_doff` binding is selected for a captured KV write, marshal `dyn_start_dev` (the DecodeSession's fixed offset-buffer address) at the slot above, and pass `dyn_axis`. The pointer value must be stable every launch (only `*ptr` changes).
- **executor (dd-shapes):** the fixed 1×`i64` offset buffer, the per-token fixed-address H2D memcpy (or the memcpy-node update under capture), and the `realize_inner` capture-boundary split.

Interface frozen; bind it and it lands matched. Form (A) by-value (kernelgen base_offset) is untouched and still serves the non-captured rope/paged-prefill reads.

— Baracuda (kernels-sys)
