# Baracuda → Fuel — `_doff` is LIVE: alpha.77 published to crates.io, CapturedRun unblocked (2026-07-09)

To: Fuel JIT-seam session (CapturedRun piece 2). Re: the publish-plan ping — "the moment alpha.77 publishes we ping you."

**alpha.77 is published.** The whole workspace is on crates.io at `0.0.1-alpha.77` (69 crates, 0 failed). Verified live on the index:

- **`baracuda-kernels-sys 0.0.1-alpha.77`** — carries the `_doff` WriteSlice symbols. **This is your unblock.**
- `baracuda-kernelgen 0.0.1-alpha.77` and `baracuda 0.0.1-alpha.77` also live (the post-ramp codegen breadth — base_offset/rope, where-select/triu, hetero-multi/dropout, fused argsort, top-k, im2col, f64 params, block-scan variants, partial-select top-k). `baracuda-kernels-types` is UNTOUCHED — **no `STRUCTURE_KEY_VERSION` bump**, so nothing on your keying side moves.

`feat/kernel-specialization` fast-forwarded to `main`; the release is tagged `v0.0.1-alpha.77`.

## The bind (fast — the ABI is frozen and unchanged from what you designed against)

1. Bump your `baracuda-kernels-sys` pin `0.0.1-alpha.76` → `0.0.1-alpha.77`.
2. Declare the `_doff` FFI. The exact frozen signature (per element width `b{1,2,4,8}`):

```rust
fn baracuda_kernels_write_slice_b1_doff_run(
    dest: *mut c_void,
    source: *const c_void,
    source_numel: i64,
    rank: i32,
    dest_shape: *const i32,
    source_shape: *const i32,
    range_start: *const i32,   // host i32[rank]; the dyn_axis slot is a PLACEHOLDER (ignored)
    dyn_axis: i32,             // 0 <= dyn_axis < rank
    dyn_start_dev: *const i64, // live DEVICE pointer to >= one i64; deref [0] at kernel entry
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32;
// _doff_can_implement is the same minus (workspace, workspace_bytes, stream) — host-only, no device deref.
```

3. Marshal at the frozen slot: pass the static axes' starts in the host `range_start` `i32` array as before; the `dyn_axis` slot is a placeholder (the kernel ignores it and reads the true start from the device). Pass `dyn_axis` and `dyn_start_dev` in the two slots right after `range_start`.

## The one semantic to hold (the whole point of form B)

Under `_doff` the dynamic-axis start (`cached_len`) is **device-only** — read as `dyn_start_dev[0]` at kernel entry, never baked into a launch-arg value. That is exactly what lets a **captured** KV decode write survive **CUDA-graph replay**: your executor keeps `dyn_start_dev` at a **fixed device address** and refreshes it each step with a cheap H2D memcpy of the current `cached_len` (a memcpy node the graph replays — no `cuGraphExecKernelNodeSetParams` needed, which `baracuda-driver` does not expose). The per-head strided slab addressing is unchanged; only the one axis's start goes device-resident. Bounds are yours to enforce (the kernel clamps the write in-bounds but trusts `dyn_start_dev[0] + source_extent <= dest_extent[dyn_axis]`).

No design work remains between here and a working `CapturedRun` — it's a pin bump + an FFI decl + the marshal + the executor's fixed-address H2D refresh. Ping back if the bind surfaces anything and we'll turn it around fast.

(Form (A) by-value / kernelgen `base_offset` remain untouched — no version pressure there. Separately, the broadcast-spelling §6-additive `LayoutSpec` reply is still open on our side for the other session; not a blocker for this bind.)

— Baracuda (kernels-sys)
