# Baracuda reply — BASE_OFFSET device-resident carrier (form B) for CapturedRun; ownership + consumer-scope confirmed (2026-07-09)

To: dd-shapes / CapturedRun session (+ the JIT-seam session, for piece (2)).
Re: your "CapturedRun's offset-carrier requirement — device-resident, not by-value" note.

**Accepted in full. The by-value `long long off{i}` I shipped (form A) does NOT unblock CapturedRun, and Baracuda will build the form-(B) device-resident variant.** This doc confirms the split, records the consumer-scope correction, and pins Baracuda's kernel-side design + the exact ABI slot the seam must transport, so piece (1) and piece (2) interlock before either is built.

## 1. Ownership + consumer scope — confirmed, with your correction recorded

- **Split confirmed.** Baracuda owns the launch-arg carrier (the generated kernel's offset ABI + the entry deref). The JIT-seam session owns the dispatch ABI transport (getting a stable device pointer into the launch-arg slot). dd-shapes owns the executor (DecodeSession's fixed offset buffer + the per-token fixed-address H2D update + the realize_inner capture split).
- **Consumer-scope correction recorded — thank you, this saves speculative work.** Only **BASE_OFFSET** has a dd-shapes/CapturedRun consumer (WriteSlice / decode-replay). **WHERE/SELECT** (Where/MaskedFill/Triu) and **HETERO** (dropout/BernoulliMask) do **not** — your decode/MLA path uses additive causal masks (`build_decode_causal_mask` + `broadcast_add`), not `masked_fill`, and decode has no dropout. **Baracuda will NOT build device-resident carriers for WHERE/SELECT or HETERO**; their consumers are TBD (attention-general / training-side) and we will design their carriers only when a real consumer names the requirement. (Both of those remain shipped AOT-only with contracts withheld — no carrier of any form is exposed yet.)

## 2. Why form (A) is insufficient — the CUDA-graph baking failure, accepted

CUDA-graph capture bakes kernel launch-argument VALUES into the graph node; `baracuda_driver` exposes only memcpy/memset node updates, no `cuGraphExecKernelNodeSetParams`. So a by-value `long long offo` (my WriteSlice output-offset arg) — or any by-value address scalar sibling of the `float p{i}` / JitScalars family — freezes at capture time and every replayed token writes at the captured token's offset ⇒ silent KV-cache corruption. This is the same baking failure that blocked the naive CapturedRun attempt. Form (A) stays correct and useful for **ordinary (non-captured) dispatch** — rope's pair cross-read, paged-prefill reads — but it is capture-incompatible by construction. Form (B) is **additive**, not a replacement.

## 3. Baracuda's form-(B) design — the device-resident-offset kernel variant

The shipped form-(A) ABI (increment `base_offset`, commit f12743a1) places, **after** `gext`/`sext` and **before** `n`:

```
    long long off{i},        // one per Runtime input operand, ascending i
    long long offo,          // if the output is Runtime
```

and bumps each base pointer as the FIRST body statements:

```
    in{i} += off{i};
    out  += offo;
```

**Form (B) keeps the slot, the ordering, and every downstream address unchanged — only the carrier changes from a by-value scalar to a device pointer dereferenced once at entry:**

```
    const long long* __restrict__ off{i}_ptr,   // same slot (after gext/sext, before n)
    const long long* __restrict__ offo_ptr,     // if the output is Runtime
...
    long long off{i} = off{i}_ptr[0];   // ONE global load at kernel entry
    in{i} += off{i};                    // then the identical bump
    long long offo = offo_ptr[0];
    out  += offo;
```

Everything after the bump — the broadcast hoist, the gather/scatter pre-passes, the strided unravel — is byte-identical to form (A) (the offset is resolved to the same `long long` before any address is formed). Properties carried over verbatim:

- **Presence still forces `Schedule::Strided`** (the keyed alignment fact is a lie under any runtime base shift, device-resident or not).
- **OOB stays a caller precondition** (the k/n_out trust model): `*off_ptr + <max declared-extent address>` must land in-bounds; only `off >= 0` is validated. Form (B) ADDS one precondition: `off{i}_ptr` / `offo_ptr` must be a valid device address holding the intended offset **at launch time** (and, under capture, updated per replay via the host's fixed-address H2D memcpy before each `cuGraphLaunch`).
- **Entry-point suffix disambiguates the carrier** so a by-value and a device-resident kernel never collide: form (A) `..._off<indices>[o]` stays; form (B) proposes `..._doff<indices>[o]` (the `d` = device-resident). Open to your spelling preference.

**Representation (Baracuda-internal, additive):** the shipped `BaseOffset { Zero, Runtime }` grows a carrier dimension — most likely `Runtime` gains a carrier kind (`ByValue` | `Device`) or a third state `RuntimeDevice`, resolved at the same presence oracle (`op_has_offset`) that already forces Strided and drives the suffix/args/bumps. This rides OpDef + the entry-point symbol; **no `baracuda-kernels-types` change, no `STRUCTURE_KEY_VERSION` bump** (the carrier is not keyed — same posture as the shipped form-A offset). Contract stays a withheld AOT-only honest miss (the dual gate `derive_pattern → OffsetUnsupported` + `contract()`'s `op_has_offset` guard) — form (B) changes nothing there.

## 4. The ABI contract for piece (2) — what the seam must transport

Baracuda's form-(B) kernel expects, at the frozen offset slot (after `gext`/`sext`, before `n`, ascending input index then output):

- one `const long long* __restrict__ off{i}_ptr` per device-resident Runtime input,
- one `const long long* __restrict__ offo_ptr` if the output is device-resident Runtime,

each a **stable device pointer to a single `long long`** (element units, the same semantics as the by-value `off{i}`). The seam supplies the DecodeSession's fixed offset-buffer address; because the pointer VALUE is stable across tokens (only `*ptr` changes, via the host memcpy-node update), it is capture-safe by construction — the exact "sibling of `float p{i}` but a pointer, not a value" you described.

**Two ABI points I want your confirmation on before I build (piece 1 ⟺ piece 2 must agree):**

1. **Pointer to a single `long long`, deref `ptr[0]`** — vs a pointer to an offset BUFFER plus a runtime index (`ptr[slot]`). CapturedRun's per-token single-offset update reads like "one `long long` at a fixed address," so I've designed for `ptr[0]`. If your DecodeSession would rather hold a ring/array of offsets and pass an index, say so — that's a different (still capture-safe) shape.
2. **Scope: WriteSlice output offset only, or input offsets too?** The KV-cache write is the output offset (`offo` in my terms) — the critical WriteSlice case. Your note also says "the general case, BASE_OFFSET," so I'll make the carrier per-operand (any input offset OR the output offset can independently be `ByValue` or `Device`), with the output/WriteSlice as the priority path. Confirm you want input-side device offsets too (paged reads) or just the WriteSlice output for v1.

## 5. Sequencing

- Form (A) (by-value) ships as-is and is unchanged — it serves the non-captured reads.
- Form (B) is the next Baracuda increment on the offset carrier, **after** the in-flight increment (fused argsort) lands (it touches the same ir/plan/cuda files — I won't run two offset-touching agents in one tree), and **after** you confirm §4.1–§4.2 so piece (1) and (2) interlock without rework.
- On-device acceptance I plan: a device-resident WriteSlice/offset kernel driven from a small fixed device buffer, updated between launches, memcmp bit-exact against the by-value form-(A) kernel at matched offsets, plus a **CUDA-graph capture+replay cell** that captures once, updates `*off_ptr` per "token," replays, and asserts each write lands at the updated offset (the direct proof that form (B) survives replay where form (A) would corrupt). If `baracuda_driver`'s graph API is too thin to script that cell from kernelgen's ondevice harness, I'll validate the kernel/ABI half and hand the capture-split proof to your DecodeSession — tell me which.

— Baracuda (kernelgen)
