# Baracuda reply — stream-ordered allocation/free for async dispatch (Step E A3)

**To:** Fuel dispatch-core, Step E **Phase A3** (CUDA async dispatch).
**Re:** `baracuda-ask-stream-ordered-free` (Q1–Q4 + the possible additive ask).
**Status:** **shipped** in `0.0.1-alpha.72` — the additive change you flagged is built,
GPU-tested on sm_89, adversarially reviewed, and published. A3 is now the small, pure-Fuel
path (Q1/Q2 = yes).
**TL;DR:** you read the alloc half correctly. The *free* half was synchronous — so we made
`*_async`-allocated buffers free stream-ordered on `Drop`, and added `zeros_async` so your
**output** buffers get the same treatment as workspaces. No behavior change for sync callers.
**One precondition to honor:** the implicit `Drop` free is ordered only against the buffer's
**origin stream** — see *Safety precondition* below. For A3 (one stream per device) that's
automatic; it only bites if you ever consume a buffer on a *different* stream than it was
allocated on.

> **Version note.** alpha.71 first shipped this; a post-ship adversarial review found a
> defensive-path `Drop` hazard (a failed async-free call could fall back to a *synchronous*
> free of still-in-use pool memory) and an over-broad "safe by construction" claim. Both are
> fixed in **alpha.72** — pull alpha.72, not alpha.71. Details under *Hardening*.

---

## Short answers

| # | Question | Answer (as of alpha.72) |
|---|----------|-------------------------|
| **Q1** | Does a `new_async` buffer free via `cudaFreeAsync` on `Drop`, or sync `cuMemFree`? Does it retain its stream? | **alpha.70: synchronous.** `Drop` always called `cuMemFree`; the buffer did **not** retain its stream, so it *couldn't* enqueue an async free. **alpha.72: stream-ordered.** `new_async` retains its origin stream and `Drop` enqueues `cuMemFreeAsync` on it. |
| **Q2** | Coverage — do `alloc_zeros` (outputs) / plain `new` also go stream-ordered, or only `new_async`? | **Sync paths stay sync** (`new` / `zeros` / `from_slice` → `cuMemAlloc` + sync `cuMemFree`). For **outputs**, use the new **`zeros_async(ctx, len, stream)`** — async alloc + async zero + async free. So both workspaces *and* outputs can be stream-ordered now. |
| **Q3** | Is the stream-ordered pool enabled? Release-threshold knob? | **Yes — `new_async` uses the device's *current* stream-ordered mem-pool** (the default pool unless you've called `set_current_pool`), no setup. Freed blocks feed reuse, not the OS. The default **release threshold is 0 (aggressive release)**; for a hot realize loop you'll likely want to raise it — `baracuda_driver::mempool` exposes it (below). |
| **Q4** | Safe to have hundreds of `new_async` allocs/frees outstanding on one stream across a realize? | **Yes — that's exactly the pooled-allocator workload** (PyTorch caching-allocator shape). The pool reuses freed blocks in stream order, so peak ≈ live working set, not sum-of-allocs. The one knob that matters is the release threshold (Q3). |

---

## Q1 — Free semantics (the detail you asked to confirm)

In alpha.70, `baracuda_driver::DeviceBuffer` stored `{ ptr, len, context }` — **no stream** — and
`Drop` unconditionally called `cuMemFree`, whether the buffer came from `new` or `new_async`.
`new_async` differed from `new` *only* in the alloc call (`cuMemAllocAsync` vs `cuMemAlloc`). So
your read was right: alloc was stream-ordered, free was not. Dropping a `new_async` buffer while a
kernel was still pending on the stream would have synchronously freed it → use-after-free, exactly
the blocker you described.

**alpha.72 fixes this at the source.** The buffer now carries `stream: Option<Stream>`:

```rust
pub struct DeviceBuffer<T: DeviceRepr> {
    ptr: CUdeviceptr,
    len: usize,
    context: Context,
    stream: Option<Stream>,   // Some(_) for *_async constructors; None for sync
    _marker: PhantomData<T>,
}

impl<T: DeviceRepr> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.0 == 0 { return; }
        let Ok(d) = driver() else { return };
        if let Some(stream) = &self.stream {
            if let Ok(cu) = d.cu_mem_free_async() {
                // Stream-ordered free; ordered after work already on `stream`.
                // On enqueue error we LEAK rather than sync-free a pool pointer
                // that may still be in use (context reclaims it at teardown).
                let _ = check(unsafe { cu(self.ptr, stream.as_raw()) });
                return;
            }
            // cuMemFreeAsync symbol missing (pre-11.2): fall through defensively.
        }
        // Synchronous free: sync-allocated (None) buffers + the pre-11.2 fallback.
        if let Ok(cu) = d.cu_mem_free() { let _ = unsafe { cu(self.ptr) }; }
    }
}
```

Retaining the stream is **cheap**: `baracuda` `Stream` is `#[derive(Clone)]` over an
`Arc<StreamInner>`, so `stream.clone()` is one atomic refcount bump — no FFI, no new stream. The
struct grows by one `Option<Stream>` word; sync buffers carry `None` and pay nothing at free time.

The runtime-API `baracuda_runtime::DeviceBuffer` got the identical treatment (`cudaMallocAsync` /
`cudaFreeAsync`), in case any path uses it — but note your `fuel-cuda-backend` uses the **driver**
API (`baracuda_driver::DeviceBuffer`, `device.rs:5`), which is the one that matters for A3.

## Q2 — Coverage: workspaces *and* outputs

Your two consumers map cleanly:

- **Workspaces / scratch** — already on `new_async` (`device.rs:163`). **No change needed**; they now
  free stream-ordered for free.
- **Output buffers** — today `alloc_zeros` → `DeviceBuffer::zeros(ctx, len)`, which is fully
  synchronous (`cuMemAlloc` + `cuMemsetD8`, sync `cuMemFree` on drop). Switch these to the new
  **`zeros_async(ctx, len, stream)`** and outputs become stream-ordered end-to-end — alloc, zero,
  *and* free all on the stream. That's what removes the need for executor-side `force_synchronize`
  lifetime guards on data buffers.

`zeros_async` is just `new_async` + `zero_async` (a `cuMemsetD8Async` on the same stream), so the
zero is ordered before any consumer reads the buffer, and the buffer retains the stream like
`new_async`.

```rust
// Fuel device.rs, suggested:
pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> Result<DeviceBuffer<T>> {
    DeviceBuffer::zeros_async(&self.context, len, &self.stream).w()   // was: ::zeros(&self.context, len)
}
```

Sync constructors (`new`, `zeros`, `from_slice`) are deliberately left **fully synchronous** — no
behavior change for any existing caller, same contract discipline as elsewhere. The async path is
opt-in by which constructor you call.

## Safety precondition — single origin stream (please read before flipping A3 on)

The implicit `Drop` free enqueues `cuMemFreeAsync` on the buffer's **origin stream** (the stream you
passed to `new_async` / `zeros_async`). CUDA orders that free after everything already submitted **to
that same stream** — which is the whole guarantee. The precondition:

- **Use a buffer only on its origin stream** (allocate on `S`, launch all producers/consumers on `S`,
  drop) → the free is ordered after all of them. **Safe, no extra care.** This is exactly A3's shape:
  one stream per device, everything submitted to it.
- **If you ever consume a buffer on a *different* stream `T`** than it was allocated on (raw pointer
  handed to a kernel launched on `T`, or a cross-stream dependency), the `Drop` free on `S` is **not**
  automatically ordered after `T`'s work. The pool could reclaim/reissue the block while `T`'s kernel
  is still reading it. To stay safe in that case: record an event on `T` and `S.wait_event(it)` before
  the buffer drops, or free explicitly via `free_async(self, stream)` on a stream ordered after `T`.

We rejected the "safe by construction, full stop" phrasing from the first draft because it omitted
this — it's the standard CUDA stream-ordered-allocator contract, not a baracuda quirk, but worth
stating so a future multi-stream scheduler doesn't silently reintroduce the UAF. The rustdoc on
`new_async`/`zeros_async` now carries the same precondition.

## Q3 — Mem pool + release threshold

`new_async` / `zeros_async` allocate from the device's **current** stream-ordered pool that
`cuMemAllocAsync` uses — which is the **default pool** unless you've switched it via
`set_current_pool`. Already enabled, nothing to turn on. Freed blocks return to that pool and are
reused by subsequent same-stream allocations, so a producer→consumer chain recycles one workspace
footprint instead of issuing a real `cuMemAlloc` per op.

The default **release threshold is 0** (the pool trims aggressively back toward the OS between
syncs). For a tight realize loop that repeatedly allocs/frees, you'll usually want a non-zero
keep-threshold so the pool *holds* blocks for reuse. `baracuda_driver::mempool` exposes the knobs:

```rust
use baracuda_driver::mempool;
let pool = mempool::default_pool(&ctx, &device)?;   // tune the pool new_async actually uses;
                                                    // if you've called set_current_pool, tune THAT one
pool.set_release_threshold(u64::MAX)?;   // keep everything until an explicit trim
// pool.trim_to(bytes)?  // reclaim down to `bytes` at a quiescent point (e.g. realize-end)
// pool.used_bytes()? / pool.reserved_bytes()?  // observability
```

Recommendation: set a generous release threshold once at device init, and (optionally) `trim_to`
at realize-end where you already sync for the D2H read — that gives reuse during the realize and
bounded residency between them. Pool config is **process/device-global**, so set it once, not per-op.
(One precision: `new_async` uses the device's *current* pool; the default pool and the current pool
are the same object until someone calls `set_current_pool`, so tune whichever one is current.)

## Q4 — Concurrency at realize scale

Hundreds of outstanding async alloc/free on one stream is the intended use — it's the same pattern
the PyTorch caching allocator runs at much larger scale. Because frees are stream-ordered and feed
the pool, **peak VRAM tracks the live working set**, not the cumulative alloc count, and there's no
per-op real allocation after the pool warms. The only caveat is the Q3 threshold: with the default
0, the pool can hand memory back to the OS at sync points and you'd re-grow it next realize — set
the threshold to avoid that churn. No fragmentation gotcha specific to baracuda beyond what the
driver's pool already manages.

---

## What this means for A3

Your "if Q1/Q2 yes" path is now the live one — **A3 is small and pure-Fuel**:

1. `Workspace` / scratch: already `new_async` — nothing to change; they free stream-ordered now.
2. Output buffers: `alloc_zeros` → `zeros_async` (one-line change above).
3. Drop the per-op `device.synchronize()`; keep the D2H sync in `to_cpu_bytes` and at realize-end.
4. (Recommended) raise the pool release threshold once at device init; optional `trim_to` at
   realize-end.
5. Keep each buffer's use on one stream (the *Safety precondition* above) — already true for A3's
   one-stream-per-device dispatch.

No Fuel-side retention pool, no executor `force_synchronize` guards. The heavier fallback you
scoped is **not needed**.

## API surface (alpha.72)

```rust
// baracuda_driver::DeviceBuffer<T>  (and the mirror in baracuda_runtime)
fn new(ctx, len)                  -> sync alloc,  sync  Drop   (unchanged)
fn zeros(ctx, len)                -> sync alloc+zero, sync Drop (unchanged)
fn from_slice(ctx, &[T])          -> sync                       (unchanged)
fn new_async(ctx, len, stream)    -> async alloc, ASYNC Drop   (retains stream)        ← workspaces
fn zeros_async(ctx, len, stream)  -> async alloc+zero, ASYNC Drop                       ← NEW; outputs
fn free_async(self, stream)       -> explicit async free at a chosen point (unchanged)
```

- `new_async` is **source-compatible** — same signature; the `Drop` behavior changed (sync → async,
  stream-ordered free) and the buffer now retains its stream. You can build against it with no
  call-site edit and immediately get stream-ordered free for workspaces.
- `zeros_async` is the one new call site (outputs).
- `free_async(self, stream)` still exists if you ever want an explicit free point; with the new
  `Drop` you mostly won't need it (it's only required when you want the free *before* scope exit, or
  on a stream other than the origin — see the precondition).

## Hardening (what changed alpha.71 → alpha.72)

A post-ship adversarial review (4 independent lenses + skeptic verification) of the alpha.71 change
surfaced two things, both fixed in alpha.72:

1. **`Drop` fallback (should-fix, code).** alpha.71's `Drop` fell through to a **synchronous**
   `cuMemFree`/`cudaFree` whenever the async-free *call* returned an error — not only when the symbol
   was missing. For a pool allocation with pending stream work, that immediate sync free is the exact
   use-after-free the feature was built to prevent. alpha.72 restricts the sync fallback to the
   genuine pre-11.2 case (the async-free *symbol* is absent); on an async-free *call* error it **leaks**
   the block (the owning context reclaims it at teardown) rather than risk a sync free of in-use
   memory. The happy path is unchanged (the async free succeeds and returns); the error path is
   unreachable on a healthy stream.
2. **Over-broad doc claim (should-fix, docs).** The "safe by construction" wording didn't state the
   single-origin-stream precondition. Now documented (see *Safety precondition*) in the rustdoc and
   here.

The review also confirmed the design's soundness on the points we most wanted checked: no double-free
(the consuming `free_async` nulls the pointer so `Drop` no-ops), `Send` impl still holds, and the
retained `Arc<Stream>` keeps the stream handle (and its context) alive through the free, so a
cross-thread drop is sound.

## Verification

GPU-gated tests, RTX 4070 / sm_89, driver 610.47, CUDA 11.2+ — all green (alpha.71 + re-run on
alpha.72 after the hardening):

- **`async_drop_is_stream_ordered`** (driver) — the decisive one: allocate two inputs via
  `new_async`, launch `vector_add`, then **drop the inputs before any synchronize**. The async free
  is ordered after the kernel, so the 16K-element output is bit-exact. (A non-ordered free would
  corrupt the pending read — it doesn't.)
- **`zeros_async_is_zeroed`** (driver) — `zeros_async` buffer reads back all-zero.
- **`zeros_async_and_implicit_stream_drop`** (runtime) — `zeros_async` zeroing + a `new_async` buffer
  dropped with **no explicit free**, reclaimed via `cudaFreeAsync` on the retained stream; stream
  stays valid afterward.
- Existing `async_alloc_roundtrip` (explicit `free_async`) still green.

One honesty note: compute-sanitizer (memcheck/racecheck) was **not** run — it isn't installed on the
local box. The ordered-free correctness is exercised functionally by the pending-read test above; if
you want a sanitizer pass on your side before flipping A3 on, the test is
`cargo test -p baracuda-driver --test graph_smoke -- --ignored async_drop_is_stream_ordered`.

## Pointers

- Impl: `crates/baracuda-driver/src/memory.rs` (`DeviceBuffer::{new_async, zeros_async}`, `Drop`),
  mirror in `crates/baracuda-runtime/src/memory.rs`. Pool knobs:
  `crates/baracuda-driver/src/mempool.rs`.
- Tests: `crates/baracuda-driver/tests/graph_smoke.rs`,
  `crates/baracuda-runtime/tests/runtime_extras_smoke.rs`.
- CHANGELOG: `0.0.1-alpha.71` (feature) + `0.0.1-alpha.72` (hardening).
- Pull `baracuda-driver = "0.0.1-alpha.72"` (workspace bumped lockstep; **use alpha.72, not alpha.71**).
