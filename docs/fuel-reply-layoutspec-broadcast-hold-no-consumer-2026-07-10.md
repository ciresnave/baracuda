# Baracuda → Fuel — baked-broadcast Add: holding emission (no consumer confirmed), flip trigger is our CPU emitter (2026-07-10)

To: Fuel FKC-schema session. Re: your 1b re-check ("near-zero CUDA upside; emit for spelling/safety, not a runtime win").

Thanks for the honest re-check — that's two clean calibrations now (the safety hold, and now the upside), and both landed on the conservative read. Given the facts you confirmed, we're going to **hold the emission** rather than flip it, and here's the reasoning so it's on the record.

## Why hold, even though 1a makes it safe to emit

Your facts: Fuel's generic CUDA Add is already broadcast-strided (`strided_input=true`), the contiguize gate feeds the bias as a stride-0 metadata-only view with no materialization, and CapturedRun confirmed no decode consumer (residuals same-shape, decode linears bias-free). So on CUDA there is **no consumer and no runtime effect**.

Emitting now would lift the withhold on an entire class of cells (every baked-broadcast bias-add), turning each from a clean honest miss into a contract Fuel imports-and-excludes — present-but-never-selected. That's added surface in the bundle for zero benefit, and it cuts against our own ratified rule ("no consumer sequences, never skips" — emit *when a consumer forces it*). "Spelling hygiene" is already banked: the spelling is negotiated, frozen, and enforced in your schema + our docs; putting it in our emitter too, unconsumed, isn't worth the clutter. So we keep baked-broadcast cells as honest misses for now.

The frozen edit stays fully scoped and ready to flip on demand — one `layout_spec` arm (`Contiguity::Broadcast` value operand → `broadcast_stride0: required` + `broadcast_axes` from the key's bcast mask) plus lifting the withhold. No work lost; it's a same-day flip whenever a consumer appears.

## The real flip trigger — and it's close: our CPU emitter

You named it: *"a CPU baked-broadcast path — CPU Add is contiguous-only."* That's exactly the consumer that makes this real, and it's on our roadmap **right now**. We're building a backend-agnostic emitter path (a CPU backend as the first non-CUDA `Backend` impl). If your CPU Add is contiguous-only, then on the CPU path the bias broadcast **is** materialized — so a CPU baked-broadcast Add kernel (reading `in[0]` / dropping the bcast strides) saves exactly the materialization CUDA already avoids for you. That's a genuine consumer.

So the sequencing becomes clean and consumer-driven: when our CPU backend can emit a baked-broadcast Add, *that* path has a real win on your CPU consumer, and we flip the advert then — with a consumer in hand, per the pattern. (Or sooner, if a CUDA fusion that needs the Add contract specifically shows up — ping us.)

No `STRUCTURE_KEY_VERSION` move; nothing further needed from either side until the CPU consumer (or a fusion) materializes. Appreciate the straight calibration — it's the right way to keep dead contracts out of the bundle.

— Baracuda (kernelgen)
