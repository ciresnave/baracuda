// On-device numeric validation of the item-03 general reduction path.
// Launches the *actual generated* kernels with small hand-checkable shapes and
// diffs against a CPU reference. Compile: nvcc -arch=sm_89 reduce_validate.cu
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#include "baracuda_gen_sum_f32_reduce_sum_ax1.cu"    // reduce axis0, rank2, collapse
#include "baracuda_gen_mean_f32_reduce_mean_ax3.cu"  // reduce {0,1}, rank3, collapse
#include "baracuda_gen_sum_f32_reduce_sum_ax1_kd.cu" // reduce axis0, rank2, keepdim
#include "baracuda_gen_amax_f32_reduce_max_ax1.cu"   // max axis0, rank2 (has-flag/NaN)
#include "baracuda_gen_sum_f32_reduce_sum_ax2.cu"    // reduce axis1, rank3 (two kept axes)
#include "baracuda_gen_sum_f32_reduce_sum_ax3.cu"    // reduce {0,1}, rank2 -> scalar

static int fails = 0;

static void check(const char* name, const std::vector<float>& got,
                  const std::vector<float>& want, float tol = 1e-3f) {
    float maxerr = 0.0f;
    int badi = -1;
    for (size_t i = 0; i < got.size(); ++i) {
        float e = fabsf(got[i] - want[i]);
        if (e > maxerr) { maxerr = e; badi = (int)i; }
    }
    if (maxerr > tol) {
        printf("FAIL %-22s maxerr %g at %d (got %g want %g)\n", name, maxerr, badi,
               got[badi], want[badi]);
        fails++;
    } else {
        printf("PASS %-22s maxerr %g\n", name, maxerr);
    }
}

template <typename T>
static T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc((void**)&d, h.size() * sizeof(T));
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}

int main() {
    // ---- 1: outer-axis Sum, [4,8] reduce axis 0 -> [8] (collapse) ----
    {
        const int R = 4, C = 8;
        std::vector<float> in(R * C);
        for (int i = 0; i < R * C; ++i) in[i] = (float)i;
        std::vector<long long> shape = {R, C}, s0 = {C, 1}, so = {1};
        std::vector<float> want(C, 0.0f);
        for (int j = 0; j < C; ++j) for (int i = 0; i < R; ++i) want[j] += in[i * C + j];
        float *d_in = dev(in), *d_out = nullptr; cudaMalloc((void**)&d_out, C * sizeof(float));
        auto *d_sh = dev(shape); auto *d_s0 = dev(s0); auto *d_so = dev(so);
        baracuda_gen_sum_f32_reduce_sum_ax1<<<32, 128>>>(d_in, d_out, d_sh, d_s0, d_so, C);
        cudaDeviceSynchronize();
        std::vector<float> got(C); cudaMemcpy(got.data(), d_out, C * sizeof(float), cudaMemcpyDeviceToHost);
        check("sum_ax0_collapse", got, want);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
    }
    // ---- 2: multi-axis Mean, [2,3,4] reduce {0,1} -> [4] (collapse) ----
    {
        const int A = 2, B = 3, D = 4;
        std::vector<float> in(A * B * D);
        for (int i = 0; i < A * B * D; ++i) in[i] = (float)i;
        std::vector<long long> shape = {A, B, D}, s0 = {B * D, D, 1}, so = {1};
        std::vector<float> want(D, 0.0f);
        for (int k = 0; k < D; ++k) { float s = 0; for (int i = 0; i < A; ++i) for (int j = 0; j < B; ++j) s += in[i * B * D + j * D + k]; want[k] = s / (float)(A * B); }
        float *d_in = dev(in), *d_out = nullptr; cudaMalloc((void**)&d_out, D * sizeof(float));
        auto *d_sh = dev(shape); auto *d_s0 = dev(s0); auto *d_so = dev(so);
        baracuda_gen_mean_f32_reduce_mean_ax3<<<32, 128>>>(d_in, d_out, d_sh, d_s0, d_so, D);
        cudaDeviceSynchronize();
        std::vector<float> got(D); cudaMemcpy(got.data(), d_out, D * sizeof(float), cudaMemcpyDeviceToHost);
        check("mean_ax01_collapse", got, want);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
    }
    // ---- 3: keepdim Sum axis 0, [4,8] -> [1,8] (so indexed by input axis) ----
    {
        const int R = 4, C = 8;
        std::vector<float> in(R * C);
        for (int i = 0; i < R * C; ++i) in[i] = (float)i;
        std::vector<long long> shape = {R, C}, s0 = {C, 1}, so = {C, 1};
        std::vector<float> want(C, 0.0f);
        for (int j = 0; j < C; ++j) for (int i = 0; i < R; ++i) want[j] += in[i * C + j];
        float *d_in = dev(in), *d_out = nullptr; cudaMalloc((void**)&d_out, C * sizeof(float));
        auto *d_sh = dev(shape); auto *d_s0 = dev(s0); auto *d_so = dev(so);
        baracuda_gen_sum_f32_reduce_sum_ax1_kd<<<32, 128>>>(d_in, d_out, d_sh, d_s0, d_so, C);
        cudaDeviceSynchronize();
        std::vector<float> got(C); cudaMemcpy(got.data(), d_out, C * sizeof(float), cudaMemcpyDeviceToHost);
        check("sum_ax0_keepdim", got, want);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
    }
    // ---- 4: Max axis 0, [4,8], with a NaN planted at col 5 (propagation) ----
    {
        const int R = 4, C = 8, nan_col = 5;
        std::vector<float> in(R * C);
        for (int i = 0; i < R * C; ++i) in[i] = (float)i;
        in[2 * C + nan_col] = nanf("");
        std::vector<long long> shape = {R, C}, s0 = {C, 1}, so = {1};
        float *d_in = dev(in), *d_out = nullptr; cudaMalloc((void**)&d_out, C * sizeof(float));
        auto *d_sh = dev(shape); auto *d_s0 = dev(s0); auto *d_so = dev(so);
        baracuda_gen_amax_f32_reduce_max_ax1<<<32, 128>>>(d_in, d_out, d_sh, d_s0, d_so, C);
        cudaDeviceSynchronize();
        std::vector<float> got(C); cudaMemcpy(got.data(), d_out, C * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = std::isnan(got[nan_col]); // NaN must propagate (torch.amax)
        for (int j = 0; j < C && ok; ++j) if (j != nan_col && fabsf(got[j] - (float)((R - 1) * C + j)) > 1e-3f) ok = false;
        printf(ok ? "PASS %-22s (nan@%d propagated)\n" : "FAIL %-22s (nan@%d)\n", "max_ax0_nan", nan_col);
        if (!ok) fails++;
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
    }
    // ---- 5: middle-axis Sum, [2,3,4] reduce axis 1 -> [2,4] (two kept axes 0,2) ----
    {
        const int A = 2, B = 3, D = 4;
        std::vector<float> in(A * B * D);
        for (int i = 0; i < A * B * D; ++i) in[i] = (float)i;
        std::vector<long long> shape = {A, B, D}, s0 = {B * D, D, 1}, so = {D, 1};
        std::vector<float> want(A * D, 0.0f);
        for (int i = 0; i < A; ++i) for (int k = 0; k < D; ++k) { float s = 0; for (int j = 0; j < B; ++j) s += in[i * B * D + j * D + k]; want[i * D + k] = s; }
        float *d_in = dev(in), *d_out = nullptr; cudaMalloc((void**)&d_out, A * D * sizeof(float));
        auto *d_sh = dev(shape); auto *d_s0 = dev(s0); auto *d_so = dev(so);
        baracuda_gen_sum_f32_reduce_sum_ax2<<<32, 128>>>(d_in, d_out, d_sh, d_s0, d_so, A * D);
        cudaDeviceSynchronize();
        std::vector<float> got(A * D); cudaMemcpy(got.data(), d_out, A * D * sizeof(float), cudaMemcpyDeviceToHost);
        check("sum_mid_two_kept", got, want);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
    }
    // ---- 6: reduce-all Sum, [4,8] reduce {0,1} -> scalar [1] (kept empty) ----
    {
        const int R = 4, C = 8;
        std::vector<float> in(R * C);
        for (int i = 0; i < R * C; ++i) in[i] = (float)i;
        std::vector<long long> shape = {R, C}, s0 = {C, 1}, so = {1};
        std::vector<float> want(1, 0.0f);
        for (int i = 0; i < R * C; ++i) want[0] += in[i];
        float *d_in = dev(in), *d_out = nullptr; cudaMalloc((void**)&d_out, sizeof(float));
        auto *d_sh = dev(shape); auto *d_s0 = dev(s0); auto *d_so = dev(so);
        baracuda_gen_sum_f32_reduce_sum_ax3<<<32, 128>>>(d_in, d_out, d_sh, d_s0, d_so, 1);
        cudaDeviceSynchronize();
        std::vector<float> got(1); cudaMemcpy(got.data(), d_out, sizeof(float), cudaMemcpyDeviceToHost);
        check("sum_reduce_all", got, want);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_sh); cudaFree(d_s0); cudaFree(d_so);
    }

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    printf(fails ? "\n%d case(s) FAILED\n" : "\nALL PASSED\n", fails);
    return fails ? 1 : 0;
}
