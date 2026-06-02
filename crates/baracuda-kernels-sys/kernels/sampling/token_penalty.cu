// baracuda-kernels Phase 66 Tier 2 — bespoke token-penalty logit transform.
//
// FlashInfer does not ship a penalty kernel, so this is a small native
// baracuda elementwise op (NOT behind the `flashinfer` feature). It
// applies the three standard autoregressive sampling penalties to a
// logits tensor in place, given a per-(request, token) occurrence count.
//
//   logits : [batch, vocab] f32, modified in place
//   counts : [batch, vocab] i32, # prior occurrences of each token
//
// For each cell with count c:
//   * repetition penalty (HF convention, multiplicative): if c > 0,
//       logit = logit > 0 ? logit / rep : logit * rep   (rep >= 1 penalizes)
//   * frequency penalty (OpenAI, additive):  logit -= freq * c
//   * presence  penalty (OpenAI, additive):  logit -= pres * (c > 0)
//
// Pass rep=1.0 / freq=0.0 / pres=0.0 to disable a given penalty.
//
// Status codes: 0 ok, 2 invalid_problem.

#include <cstdint>
#include <cuda_runtime.h>

namespace {
constexpr int STATUS_OK          = 0;
constexpr int STATUS_INVALID_ARG = 2;

__global__ void apply_token_penalty_f32_kernel(
    float* __restrict__ logits, const int32_t* __restrict__ counts,
    int64_t n, float rep_penalty, float freq_penalty, float pres_penalty)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        int32_t c = counts[i];
        if (c <= 0) continue;
        float x = logits[i];
        // Repetition (multiplicative).
        if (rep_penalty != 1.0f) {
            x = x > 0.0f ? x / rep_penalty : x * rep_penalty;
        }
        // Frequency + presence (additive).
        x -= freq_penalty * static_cast<float>(c);
        x -= pres_penalty;  // c > 0 guaranteed here
        logits[i] = x;
    }
}
}  // namespace

extern "C" {

int baracuda_kernels_apply_token_penalty_f32_run(
    int32_t batch, int32_t vocab, float rep_penalty, float freq_penalty, float pres_penalty,
    void* logits, const void* counts, void* stream)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    if (!logits || !counts) return STATUS_INVALID_ARG;
    int64_t n = static_cast<int64_t>(batch) * static_cast<int64_t>(vocab);
    cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
    constexpr int kThreads = 256;
    int64_t blocks64 = (n + kThreads - 1) / kThreads;
    if (blocks64 > 65535) blocks64 = 65535;  // grid-stride handles the rest
    dim3 grid(static_cast<unsigned int>(blocks64));
    apply_token_penalty_f32_kernel<<<grid, kThreads, 0, s>>>(
        reinterpret_cast<float*>(logits), reinterpret_cast<const int32_t*>(counts),
        n, rep_penalty, freq_penalty, pres_penalty);
    cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? STATUS_OK : STATUS_INVALID_ARG;
}

int baracuda_kernels_apply_token_penalty_f32_can_implement(int32_t batch, int32_t vocab)
{
    if (batch <= 0 || vocab <= 0) return STATUS_INVALID_ARG;
    return STATUS_OK;
}

}  // extern "C"
