// Ring Attention — Phase 56 Tier 1.
//
// Sequence-parallel attention with online-softmax reconstruction across
// rotating K/V chunks. See `kernels/include/baracuda_ring_attention.cuh`
// for the algorithm narrative + tile geometry.
//
// Algorithm credit: Hao Liu, Matei Zaharia, Pieter Abbeel,
// "Ring Attention with Blockwise Transformers for Near-Infinite Context"
// (NeurIPS 2023; arXiv:2310.01889; Apache-2.0 reference at
// https://github.com/lhao499/RingAttention). This file is a clean-room
// hand-port of the CUDA equivalent — no upstream source vendored.

#include "../include/baracuda_ring_attention.cuh"

// Dtype-independent init helper — zero o_acc / set m_acc=-INF / l_acc=0.
extern "C" int32_t baracuda_kernels_ring_attention_init_run(
    void* o_acc, void* m_acc, void* l_acc,
    int64_t o_len, int64_t ml_len,
    void* stream_ptr)
{
    if (o_acc == nullptr || m_acc == nullptr || l_acc == nullptr) return 2;
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return baracuda::ring_attention::launch_ring_init(
        static_cast<float*>(o_acc),
        static_cast<float*>(m_acc),
        static_cast<float*>(l_acc),
        o_len, ml_len, stream);
}

extern "C" int32_t baracuda_kernels_ring_attention_init_can_implement(
    const void* /*o_acc*/, const void* /*m_acc*/, const void* /*l_acc*/,
    int64_t o_len, int64_t ml_len)
{
    if (o_len < 0 || ml_len < 0) return 2;
    return 0;
}

// Workspace bytes for the Ring Attention persistent accumulator state.
extern "C" size_t baracuda_kernels_ring_attention_workspace_bytes(
    int32_t batch, int32_t heads, int32_t q_local, int32_t d)
{
    if (batch <= 0 || heads <= 0 || q_local <= 0 || d <= 0) return 0;
    return baracuda::ring_attention::ring_attention_workspace_bytes_host(
        (int64_t)batch, (int64_t)heads, (int64_t)q_local, (int64_t)d);
}

// Tier 1: f16, bf16. f32 / f64 deferred.
BARACUDA_KERNELS_RING_ATTENTION_INSTANTIATE(f16, __half)
BARACUDA_KERNELS_RING_ATTENTION_INSTANTIATE(bf16, __nv_bfloat16)
