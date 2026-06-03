# baracuda-flashinfer

Safe, typed Rust wrappers for **NVIDIA FlashInfer**'s inference-serving
kernels — the vLLM-style serving surface for the baracuda CUDA stack.

Wraps [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)
(Apache-2.0). The vendored kernels live under
`crates/baracuda-kernels-sys/vendor/flashinfer/`; this crate is the
typed safe facade.

## What it covers

| Family | Plan types | Use case |
| --- | --- | --- |
| **Paged-KV attention** | `BatchPagedDecodePlan`, `BatchPagedPrefillPlan`, `BatchRaggedPrefillPlan`, `BatchPagedDecodeFp8Plan` | vLLM-style decode + prefill against a paged KV-cache |
| **KV-cache append** | `PagedKvAppendPlan` | Decode-time write of the freshly-computed K/V into the paged store |
| **Cascade attention** | `CascadeAttentionPlan`, `CascadeMergeStatesPlan` | LSE-aware merge of partial attention states for prefix-cache / shared-prompt reuse |
| **Sort-free sampling** | `TopKTopPSamplingPlan`, `PerRowSamplingPlan`, `SpeculativeSamplingPlan` | Decode-time token sampling without a global vocabulary argsort |
| **Native** | `TokenPenaltyPlan` | Repetition / frequency / presence penalty logit transform (baracuda-native, not feature-gated) |

Dtypes are f16 / bf16 / f32 for the attention plans (head_dim ∈ {64, 128, 256});
FP8 KV cache (e4m3 / e5m2) is handled by `BatchPagedDecodeFp8Plan`.

## When to use this crate

- You're building an LLM serving stack (vLLM-style continuous batching,
  paged-KV memory management, speculative decoding).
- For pre-training and single-prompt inference, prefer the contiguous
  attention plans in [`baracuda-kernels`] (`FlashSdpaPlan`,
  `FlashSdpaSm89Plan`).

## Quick example

```rust,no_run
# #[cfg(feature = "flashinfer")]
# fn demo(stream: &baracuda_driver::Stream) -> baracuda_flashinfer::Result<()> {
use baracuda_flashinfer::prelude::*;
use half::f16;

let desc = BatchPagedDecodeDescriptor {
    batch_size: 8,
    num_qo_heads: 32,
    sm_scale: 1.0 / (128.0_f32).sqrt(),
    paged_kv: PagedKvCacheDescriptor {
        page_size: 16,
        num_total_pages: 1024,
        num_kv_heads: 8, // GQA group size 4
        head_dim: 128,
        element: ElementKind::F16,
    },
};
let plan = BatchPagedDecodePlan::<f16>::select(stream, &desc, PlanPreference::default())?;
let _ws_bytes = plan.workspace_size();
// ... allocate workspace + page table, then plan.run(stream, ws, args)
# Ok(())
# }
```

The same `Descriptor → Plan::select → query_workspace_size → Args →
Plan::run` lifecycle applies to every type in the crate.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `flashinfer` | no | Build the vendored FlashInfer kernels (pulls `baracuda-kernels/flashinfer` + `baracuda-flashinfer-sys/flashinfer`). With the feature off, plans still type-check and `select` / `can_implement` still validate shapes, but `run` returns `Error::Unsupported`. `TokenPenaltyPlan` is a baracuda-native op and runs regardless. |

## Status / scope

- Phase 46 landed the initial paged-KV decode + sampling surface.
- Phase 66 extended to paged prefill, ragged prefill, cascade /
  many-way state merge, FP8 paged decode, per-row sampling, and
  speculative-sampling verification.
- Determinism: sampling is bit-stable across launches given the same
  `(seed_val, offset_val)` and probability tensor. Set
  `deterministic = true` in the descriptor to make the rare
  cumulative-boundary tiebreak reproducible.
- Not wrapped yet: anything FlashInfer ships that isn't in the symbol
  list above. File a bug if you need it.

## Related crates

- [`baracuda-flashinfer-sys`] — raw FFI surface (FlashInfer C-ABI).
- [`baracuda-kernels`] — the unified op facade; plans are also re-exported there under `flashinfer` cargo feature.
- [`baracuda-driver`] — `Stream` / `Context` / `DeviceBuffer`.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-flashinfer-sys`]: https://docs.rs/baracuda-flashinfer-sys
[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-driver`]: https://docs.rs/baracuda-driver
