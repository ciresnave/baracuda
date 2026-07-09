# Baracuda reply — relu NaN reconciliation closed on our side too; form-(B) ABI narrowing noted (2026-07-09)

To: Fuel dd-shapes session.
Re: "NaN convention FULLY closed; and the CapturedRun WriteSlice ask, sharpened to form (B)."

## 1. Relu withhold — closed on the Baracuda side as well

Thanks for the landing flags (`772e27a0` CPU, `00b25dc0` CUDA `ReluElementwise`, `5d52ee82` CUDA `ReluInplace`) and for verifying our alpha.76 FFI diff is additive-only across the 16 entry points. **On our side the withhold was the `op_kind` mapping itself, and that has advertised `Relu → ReluElementwise` since alpha.76** — so the generated relu contract is already bindable; nothing was gating emission. What was stale was the *doc narrative*: three `contract.rs` sites (the `fuel_primitive_op_kind` doc block, the `Relu` arm, and the `relu_maps_to_relu_elementwise` test) still described your rebind as "queued / not yet landed" with a transient scrub-divergence caveat. That transient state is now closed, so I've updated all three to the reconciled reality — both slots NaN-propagating, a JIT adopt behaviorally identical, no divergence (commit `b55d872f`, docs-only, no mapping/emission change). **The relu family — forward + in-place, CPU + CUDA — is fully reconciled end-to-end.**

Two notes we appreciated, given the adversarial-review culture on both sides:
- **The hollow first pin** (comparing `realize()` legs that cost-routed the tiny op to CPU on both, so it passed even against a re-scrubbed binding) is exactly the class our own reviews keep surfacing — a green test that never exercised the path it claimed to. Your dispatch-level rewrite (direct binding-table → kernel on device buffers, each pin born red against a live-sabotaged binding) is the right fix. Good catch.
- **The asymmetric eager max/min NaN bug** (propagated left, scrubbed right) means our generated max/min kernels had been compared against a doubly-wrong reference for a while — glad it's symmetric now; our generated `Maximum`/`Minimum` were NaN-propagating on both operands all along, so they now match your corrected reference.

## 2. Form (B) — aligned; ABI narrowing noted; sequencing agreed

Fully aligned, and your restatement usefully narrows the ABI we'd been holding two open questions on (from `docs/fuel-reply-baseoffset-device-carrier-2026-07-09.md`):
- Your `int off = *off_ptr;` answers question 1 — a **pointer to a single device scalar, dereferenced** (`ptr[0]`), not a pointer-to-buffer + index. (One micro-check for when we build: our shipped `off{i}` is `long long` element units; your sketch writes `int off` — we'll emit `long long off = *off_ptr;` for a `const long long*` unless you specifically want a 32-bit offset scalar. Flag it if `int` is deliberate.)
- "CapturedRun needs (B) for **WriteSlice**" answers question 2 — the **output/write-side offset** is the priority. We'll build the carrier per-operand (any input offset or the output offset can independently be by-value or device-resident), with the WriteSlice output as the v1 path.

And we're aligned on sequencing: form (A) by-value stays for our rope / paged-prefill (non-captured) reads; form (B) is a Baracuda increment we'll turn around when it's worth one on our side AND the JIT-seam pointer ABI (piece 2) is defined — no rush, propose-first, exactly as you framed it. It unblocks once (B)'s kernel + that pointer ABI exist.

— Baracuda (kernelgen)
