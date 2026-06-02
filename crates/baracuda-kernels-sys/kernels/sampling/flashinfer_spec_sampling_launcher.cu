// baracuda-kernels Phase 66 Tier 2 — FlashInfer speculative-decode
// verification launcher.
//
// Wraps FlashInfer's `ChainSpeculativeSampling` (already vendored in
// `sampling.cuh`): given a draft model's per-step probabilities + the
// tokens it sampled, and the target model's probabilities for the same
// steps, perform the standard speculative-decoding accept/reject
// verification and emit the corrected token sequence.
//
// Layout:
//   - draft_probs       : [batch, num_spec, vocab] f32
//   - draft_token_ids   : [batch, num_spec]        i32
//   - target_probs      : [batch, num_spec + 1, vocab] f32
//   - output_token_ids  : [batch, num_spec + 1]    i32  (written)
//   - output_accepted_token_num     : [batch] i32 (written)
//   - output_emitted_draft_token_num: [batch] i32 (written)
//
// Status codes: 0 ok, 2 invalid_problem, 3 unsupported.

#include <cstdint>
#include <cuda_runtime.h>

#include "../../vendor/flashinfer/include/flashinfer/sampling.cuh"

namespace {
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;

inline int translate(cudaError_t e) {
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}
}  // namespace

extern "C" {

int baracuda_kernels_flashinfer_chain_speculative_sampling_f32_run(
    int32_t batch, int32_t num_speculative_tokens, int32_t vocab,
    int32_t deterministic, uint64_t seed_val, uint64_t offset_val,
    const void* draft_probs, const void* draft_token_ids, const void* target_probs,
    void* output_token_ids, void* output_accepted_token_num,
    void* output_emitted_draft_token_num, void* stream)
{
    if (batch <= 0 || num_speculative_tokens <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!draft_probs || !draft_token_ids || !target_probs || !output_token_ids ||
        !output_accepted_token_num || !output_emitted_draft_token_num)
        return STATUS_INVALID_ARG;
    cudaError_t e = flashinfer::sampling::ChainSpeculativeSampling<float, int32_t>(
        reinterpret_cast<float*>(const_cast<void*>(draft_probs)),
        reinterpret_cast<int32_t*>(const_cast<void*>(draft_token_ids)),
        reinterpret_cast<float*>(const_cast<void*>(target_probs)),
        reinterpret_cast<int32_t*>(output_token_ids),
        reinterpret_cast<int32_t*>(output_accepted_token_num),
        reinterpret_cast<int32_t*>(output_emitted_draft_token_num),
        static_cast<uint32_t>(batch), static_cast<uint32_t>(num_speculative_tokens),
        static_cast<uint32_t>(vocab), deterministic != 0,
        /*seed_arr=*/nullptr, seed_val, /*offset_arr=*/nullptr, offset_val,
        reinterpret_cast<cudaStream_t>(stream));
    return translate(e);
}

int baracuda_kernels_flashinfer_chain_speculative_sampling_f32_can_implement(
    int32_t batch, int32_t num_speculative_tokens, int32_t vocab)
{
    if (batch <= 0 || num_speculative_tokens <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

}  // extern "C"
