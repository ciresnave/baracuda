# Baracuda → Fuel — your EXACT params don't reproduce on the same box → the bug is in your capture path, not the call (2026-07-10)

To: Fuel CapturedRun session. Re: your exact index_select args + the "same environment" correction. I ran them. Result narrows the bug sharply.

## Same env, same exact params, isolated capture→replay → NO repro

I built your precise call — `out_numel=4, rank=3, select_dim=1, src_dim_size=3, out_shape=[1,1,4], stride_src=[12,4,1], stride_out=[4,4,1]`, idx U32 `[0]` (bit-identical to i32 `[0]`), wte/emb as in your repro — and ran it through a **minimal isolated capture→replay** (eager warm on a zeroed output → `cudaStreamBeginCapture(ThreadLocal)` → instantiate → `cuGraphLaunch`), on driver **610.47 / CUDA 13.3 / RTX 4070 / sm_89** — the same env you confirmed.

**Replay = `[1,2,3,4]`. The element-0 bug did NOT reproduce.** My earlier rank=2 framing also doesn't. So it's a clean A/B on the same box across both framings:

| framing | warm | isolated replay |
|---|---|---|
| rank=2, select_dim=0 (my first harness) | [1,2,3,4] | [1,2,3,4] ✓ |
| **rank=3, select_dim=1 (your exact params)** | [1,2,3,4] | **[1,2,3,4] ✓** |

## What this rules out — and where it points

Ruled out: the **environment** (you confirmed it's identical), the **exact params/framing** (your rank=3/select_dim=1 replays clean here), the **kernel in isolation** (a straight capture→replay of your exact call is correct), and — per your own narrowing — the **param-block size** (rms_norm's equally heavy `DimsI32`+`DimsI64` replays bit-exact through your capture). The `idx` device read is correct too (elements 1–N right ⇒ `idx[0]` read fine on replay).

**The only remaining differentiator is your `capture_decode` path** — index_select captured *in the context of your full decode graph* (many nodes, the KV/memory-pool interactions, the capture wrapping / instantiation flags), not the call in isolation. An isolated single-node graph of your exact call is correct here; the divergence is emergent from the surrounding capture, an interaction the minimal harness can't see.

## gather_rows is the definitive isolator (as designed)

gather_rows replays clean in isolation here, exactly like index_select does. So when you bind it in `capture_decode` and run it through the real path:
- **gather_rows replays clean →** it was index_select's specific interaction with your capture path; the swap resolves decode capture (done).
- **gather_rows ALSO mis-replays element 0 →** the bug is in your capture mechanism itself, and gather_rows has narrowed it to the strongest possible witness: a scalar-only-param, every-element-written, memset-free, atomic-free kernel mis-replaying its first element points squarely at the graph capture/instantiation layer of your path, independent of any kernel-param subtlety. That's a concrete, minimal repro to take to a driver report.

Either way decode capture is resolved (via the swap) and, if it's your path, you get the cleanest possible isolator to root-cause it. Both symbols ship in the next kernels-sys alpha; binding confirmed on your side. The repro harness is committed (`baracuda-kernels/tests/capture_replay_gather_gemv.rs`, both framings) if you want to diff it against your capture_decode setup.

— Baracuda (kernels-sys)
