// baracuda-kernels Phase 6 Category K — KV-cache append for FP types
// (Milestone 6.5).
//
// Pure copy from new K/V vectors into running K/V caches at per-sample
// offsets supplied by `cache_offsets[b]`. Single launcher per dtype
// fires two device-side copy kernels (K then V) on the same stream.

#include "../include/baracuda_attention.cuh"

BARACUDA_KERNELS_KV_CACHE_APPEND_INSTANTIATE(kv_cache_append_f32, float)
BARACUDA_KERNELS_KV_CACHE_APPEND_INSTANTIATE(kv_cache_append_f16, __half)
BARACUDA_KERNELS_KV_CACHE_APPEND_INSTANTIATE(kv_cache_append_bf16, __nv_bfloat16)
BARACUDA_KERNELS_KV_CACHE_APPEND_INSTANTIATE(kv_cache_append_f64, double)
