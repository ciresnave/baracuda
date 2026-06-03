# baracuda-flashinfer-sys

Raw C-ABI FFI re-exports for the vendored **FlashInfer** inference
kernels in the baracuda CUDA stack.

## What it wraps

[`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)
(Apache-2.0). FlashInfer is header-only / template-heavy, so there is no
shared library to dynamically load. Instead, baracuda compiles thin
C-ABI launcher shims around the vendored FlashInfer headers inside
[`baracuda-kernels-sys`]; this crate re-exports those `extern "C"`
symbols under a dedicated crate name so downstream FFI consumers don't
have to pull the whole kernels-sys symbol table into scope.

## When to use this crate

Almost never directly — the safe wrapper [`baracuda-flashinfer`] gives
you typed plans + checked dimensions. Reach for this crate only when
the safe layer hasn't wrapped a symbol you need (file a bug if so), or
when you're bridging the FlashInfer C-ABI into another foreign-language
runtime.

## Symbol families

- `*_paged_decode_*` — batched paged-KV decode
  (`BatchDecodeWithPagedKVCacheDispatched`). f16 / bf16 / f32.
- `*_paged_kv_append_decode_*` — decode-time KV-cache append.
- `*_merge_state_in_place_*` / `*_merge_states_*` — cascade /
  prefix-cache LSE-aware attention-state merge.
- `*_top_k_sampling_*` / `*_top_p_sampling_*` /
  `*_min_p_sampling_*` / `*_top_k_top_p_sampling_*` — sort-free
  sampling from a row-normalized probability tensor.

Each kernel ships a `*_can_implement` companion that returns 0 for
"supported" without executing the launch — used by the safe layer's
plan-selection path.

## Quick example

```rust,no_run
# #[cfg(feature = "flashinfer")]
# unsafe fn raw_can_check() {
use baracuda_flashinfer_sys::baracuda_kernels_flashinfer_paged_decode_can_implement;

let ok = baracuda_kernels_flashinfer_paged_decode_can_implement(
    /* batch_size   */ 8,
    /* num_qo_heads */ 32,
    /* num_kv_heads */ 8,
    /* head_dim     */ 128,
    /* page_size    */ 16,
    /* element      */ 0, // 0=f16, 1=bf16, 2=f32
);
debug_assert_eq!(ok, 0);
# }
```

Real call sites should go through the safe wrapper.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `flashinfer` | no | Transitively enables `baracuda-kernels-sys/flashinfer`, which compiles the vendored launcher `.cu` files via nvcc. With the feature off, this crate is empty. |

## Related crates

- [`baracuda-flashinfer`] — safe, typed API; the documented entry point.
- [`baracuda-kernels-sys`] — owns the vendored FlashInfer sources +
  launcher shims; this crate is a thin re-export facade.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-flashinfer`]: https://docs.rs/baracuda-flashinfer
[`baracuda-kernels-sys`]: https://docs.rs/baracuda-kernels-sys
