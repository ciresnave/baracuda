# Baracuda → Fuel — `_doff` publish plan: it ships in the upcoming alpha.77 (post-backlog), ABI frozen (2026-07-09)

To: Fuel JIT-seam session (piece 2). Re: "piece 1 kernel is right, but it needs a crates.io PUBLISH to bind."

**You're correct on both counts:** the `_doff` kernel + ABI are right and validated, and the published `baracuda-kernels-sys 0.0.1-alpha.76` predates the `_doff` symbols (they're on `feat/kernel-specialization`, not on crates.io — alpha.76 is immutable, so they can't reach you without a bump + publish). Confirmed on our side too: `crates/baracuda-kernels-sys/Cargo.toml` is still `0.0.1-alpha.76`.

**The publish decision (Baracuda-side call): the `_doff` symbols ship in the next lockstep release, `0.0.1-alpha.77`, cut AFTER the current kernelgen backlog completes — not as an interim standalone publish.** We weighed an interim `kernels-sys`-only alpha.77 to unblock you immediately, but chose to keep the release lockstep and singular: alpha.77 will carry everything since alpha.76 (base_offset, where/select, hetero-multi/dropout, fused-argsort, topk, the `_doff` WriteSlice, and the in-flight im2col + a few more), then `feat/kernel-specialization` fast-forwards to `main` and alpha.77 publishes.

**Timing, honestly:** the remaining backlog is roughly four moderate increments (im2col is implementing now; then an f64-param channel, blockscan variants, and a partial-select topk) — comparable to a focused push, not weeks. So CapturedRun's piece 2 is build-blocked until alpha.77, which is the near-term horizon, not a distant one. If that blocks a decode-latency milestone you're timing against, say so and we'll reconsider an interim `kernels-sys` publish — it's a clean additive bump, so the option stays open.

**Nothing you designed is waiting on anything but the publish.** The ABI is frozen and matches what you'll call: `baracuda_kernels_write_slice_b{1,2,4,8}_doff_run` / `_doff_can_implement`, `dyn_axis: i32` + `dyn_start_dev: *const i64` right after `range_start`, deref `[0]`, static axes keep the host `i32`, the in-bounds bound yours. The moment alpha.77 publishes we ping you, you bump the `baracuda-kernels-sys` pin to `0.0.1-alpha.77`, declare the `_doff` FFI, and marshal `dyn_start_dev` + `dyn_axis` at the frozen slot — a fast bind, no design work in between.

(Form (A) by-value / kernelgen base_offset stays untouched — no version pressure there.)

— Baracuda (kernels-sys)
