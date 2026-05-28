// Probe M=2048 N=4096 K=4096 CUDA-L2 3090 kernel on RTX 4070 sm_89.
//
// This includes the upstream kernel file directly via the C preprocessor.
// We rename the kernel/launcher to avoid clashes and skip the torch
// binding code at the tail. To do that we wrap the include in a
// `namespace l2_2048 { ... }` block — but that breaks because the file
// uses raw `template` declarations at global scope. So instead we
// inline the relevant portion via a copy, modified to match the 2048
// shape's tuning parameters.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cublas_v2.h>

// Kernel for 2048_4096_4096: __launch_bounds__ + slightly different
// pipeline (move ismem_read advance into the loop body) + cast to T
// (fp16) before R2S epilogue.
template <typename T, int BM, int BN, int BK, int kStage, typename TiledMMA,
          typename G2SCopyA, typename G2SCopyB, typename SmemLayoutA,
          typename SmemLayoutB, typename SmemLayoutC, typename S2RCopyAtomA,
          typename S2RCopyAtomB, typename R2SCopyAtomC, typename S2GCopyAtomC,
          typename S2GCopyC, const bool BlockSwizzle>
__global__ void __launch_bounds__(128, 2) cuda_l2_3090_2048_kernel(
    T *Aptr, T *Bptr, T *Dptr, int m, int n, int k) {
  using namespace cute;
  extern __shared__ T shm_data[];
  T *Ashm = shm_data;
  T *Bshm = shm_data + cute::cosize(SmemLayoutA{});

  int idx = threadIdx.x;
  int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
  int iy = blockIdx.y;

  if (iy * BM >= m || ix * BN >= n) return;

  Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k),
                         make_stride(k, Int<1>{}));
  Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k),
                         make_stride(k, Int<1>{}));
  Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n),
                         make_stride(n, Int<1>{}));

  Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
  Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
  Tensor gD = local_tile(D, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));

  auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

  TiledMMA tiled_mma;
  auto thr_mma = tiled_mma.get_slice(threadIdx.x);

  auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
  auto tCrD = thr_mma.partition_fragment_C(gD);
  clear(tCrD);

  G2SCopyA g2s_tiled_copy_a;
  auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
  auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
  auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

  G2SCopyB g2s_tiled_copy_b;
  auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
  auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
  auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

  auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
  auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
  auto tAsA = s2r_thr_copy_a.partition_S(sA);
  auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

  auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
  auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
  auto tBsB = s2r_thr_copy_b.partition_S(sB);
  auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

  int itile_to_read = 0;
  int ismem_read = 0;
  int ismem_write = 0;

  int ntile = k / BK;

#pragma unroll
  for (int istage = 0; istage < kStage - 1; ++istage) {
    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
               tAsA_copy(_, _, _, istage));
    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
               tBsB_copy(_, _, _, istage));
    cp_async_fence();
    ++itile_to_read;
    ++ismem_write;
  }

  cp_async_wait<kStage - 2>();
  __syncthreads();

  int ik = 0;
  cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
  cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

#pragma unroll 1
  for (int itile = 0; itile < ntile; ++itile) {
    int nk = size<2>(tCrA);
#pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
      int ik_next = (ik + 1) % nk;
      if (ik == 0) {
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                     tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                     tBsB_copy(_, _, _, ismem_write));
          cp_async_fence();
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
      }
      if (ik == nk - 2) {
        cp_async_wait<kStage - 2>();
        __syncthreads();
      }
      if (ik == nk - 1) {
        ismem_read = (ismem_read + 1) % kStage;
      }
      cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                 tCrA_view(_, _, ik_next));
      cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                 tCrB_view(_, _, ik_next));
      cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
    }
  }

  __syncthreads();
  auto sC = make_tensor(sA(_, _, 0).data(), SmemLayoutC{});

  auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
  auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);

  auto tCrD_half = make_tensor_like<T>(tCrD);
#pragma unroll
  for (int i = 0; i < size(tCrD); ++i) {
    tCrD_half(i) = static_cast<T>(tCrD(i));
  }

  auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD_half);
  auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);

  S2GCopyC s2g_tiled_copy_c;
  auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
  auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
  auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

  auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);
  auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);

  int step = size<3>(tCsC_r2s);
#pragma unroll
  for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(r2s_tiled_copy_c, tCrC_r2sx(_, i + j), tCsC_r2s(_, 0, 0, j));
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < step; ++j) {
      cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    }
    __syncthreads();
  }
}

// 2048-tuned launcher: BM=128, BN=64, BK=32, Stages=4.
template <typename T, const int Stages = 4, const bool BlockSwizzle = true>
void launch_2048(T *a, T *b, T *c, int M, int N, int K, int swizzle_stride) {
  using namespace cute;
  auto BM = Int<128>{};
  auto BN = Int<64>{};
  auto BK = Int<32>{};
  auto KStage = Int<Stages>{};
  auto kSmemLayoutCBatch = Int<2>{};

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<8>{}, Int<BK>{}),
                  make_stride(Int<BK>{}, Int<1>{}))));
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BM>{}, Int<BK>{}, Int<KStage>{})));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtom{}, make_shape(Int<BN>{}, Int<BK>{}, Int<KStage>{})));

  using mma_op = SM80_16x8x16_F32F16F16F32_TN;
  using mma_traits = MMA_Traits<mma_op>;
  using mma_atom = MMA_Atom<mma_traits>;

  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;

  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

  using MMA_EU_RepeatT = decltype(make_layout(make_shape(
      Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
  using G2SCopyA = decltype(make_tiled_copy(
      g2s_copy_atom{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;

  using s2r_copy_op = SM75_U32x4_LDSM_N;
  using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
  using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
  using S2RCopyAtomA = s2r_copy_atom;
  using S2RCopyAtomB = s2r_copy_atom;

  using SmemLayoutAtomC = decltype(composition(
      Swizzle<3, 3, 3>{},
      make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                  make_stride(Int<kMmaPN>{}, Int<1>{}))));
  using SmemLayoutC = decltype(tile_to_shape(
      SmemLayoutAtomC{},
      make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

  using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
  using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
  using S2GCopyC = decltype(make_tiled_copy(
      S2GCopyAtomC{},
      make_layout(make_shape(Int<32>{}, Int<4>{}),
                  make_stride(Int<4>{}, Int<1>{})),
      make_layout(make_shape(Int<1>{}, Int<8>{}))));

  int BX = (N + BN - 1) / BN;
  int BY = (M + BM - 1) / BM;
  int BZ = BlockSwizzle ? (N + (swizzle_stride) - 1) / (swizzle_stride) : 1;
  BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;

  dim3 block(size(MMA{}));
  dim3 grid(BX, BY, BZ);

  static constexpr int shm_size_AB =
      cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
  static constexpr int shm_size_C = cute::cosize(SmemLayoutC{});
  static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);

  cudaFuncSetAttribute(
      cuda_l2_3090_2048_kernel<
          T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA,
          SmemLayoutB, SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC,
          S2GCopyAtomC, S2GCopyC, BlockSwizzle>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kShmSize);

  cuda_l2_3090_2048_kernel<
      T, BM, BN, BK, KStage, MMA, G2SCopyA, G2SCopyB, SmemLayoutA, SmemLayoutB,
      SmemLayoutC, S2RCopyAtomA, S2RCopyAtomB, R2SCopyAtomC, S2GCopyAtomC,
      S2GCopyC, BlockSwizzle><<<grid, block, kShmSize>>>(a, b, c, M, N, K);
}

#define CHECK_CUBLAS(x) do { cublasStatus_t r = (x); if (r != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas err %d at %d\n", r, __LINE__); return 1; } } while (0)

int main() {
  // CUDA-L2 expects the second operand as col-major (the kernel reads
  // it as (n, k) row-major, which is the col-major view of a (k, n) row-
  // major B). For a fair vs-cuBLAS comparison we time the same logical
  // product: D = A · B where A is row-major (M, K) and B is row-major
  // (K, N). We pass the col-major-of-B (which is the (N, K) row-major
  // transpose) to CUDA-L2.

  const int M = 2048, N = 4096, K = 4096;
  std::vector<__half> h_a(M*K), h_b_kn_rowmajor(K*N), h_b_nk_rowmajor(N*K), h_c(M*N, __half(0.0f));
  for (int i = 0; i < M*K; ++i) h_a[i] = __float2half(0.01f);
  for (int i = 0; i < K*N; ++i) h_b_kn_rowmajor[i] = __float2half(0.01f);
  // col-major B = transpose. Same values since all 0.01 → identity.
  for (int i = 0; i < N*K; ++i) h_b_nk_rowmajor[i] = __float2half(0.01f);

  __half *d_a, *d_b_kn, *d_b_nk, *d_c;
  cudaMalloc(&d_a, sizeof(__half)*M*K);
  cudaMalloc(&d_b_kn, sizeof(__half)*K*N);
  cudaMalloc(&d_b_nk, sizeof(__half)*N*K);
  cudaMalloc(&d_c, sizeof(__half)*M*N);
  cudaMemcpy(d_a, h_a.data(), sizeof(__half)*M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_kn, h_b_kn_rowmajor.data(), sizeof(__half)*K*N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_nk, h_b_nk_rowmajor.data(), sizeof(__half)*N*K, cudaMemcpyHostToDevice);

  // 1. CUDA-L2 2048 kernel.
  launch_2048<__half, 4, true>(d_a, d_b_nk, d_c, M, N, K, 2048);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "kernel error: %s\n", cudaGetErrorString(err));
    return 1;
  }

  cudaMemcpy(h_c.data(), d_c, sizeof(__half)*M*N, cudaMemcpyDeviceToHost);
  printf("CUDA-L2 C[0] = %f (expected ~0.4096)\n", __half2float(h_c[0]));

  cudaEvent_t s, e;
  cudaEventCreate(&s);
  cudaEventCreate(&e);
  // Warmup.
  for (int i = 0; i < 10; ++i) {
    launch_2048<__half, 4, true>(d_a, d_b_nk, d_c, M, N, K, 2048);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  const int reps = 50;
  for (int i = 0; i < reps; ++i) {
    launch_2048<__half, 4, true>(d_a, d_b_nk, d_c, M, N, K, 2048);
  }
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms = 0;
  cudaEventElapsedTime(&ms, s, e);
  float cuda_l2_us = ms * 1000.0f / reps;
  printf("CUDA-L2 3090 M2048_N4096_K4096 fp16/fp32acc: %.2f us/iter\n", cuda_l2_us);

  // 2. cuBLAS gemmEx reference (same logical product, col-major mapping).
  cublasHandle_t h;
  cublasCreate(&h);
  float alpha = 1.0f, beta = 0.0f;

  // Same trick as gemm_vs_cublas.rs: compute C^T = B^T · A^T col-major.
  // Pass B (row-major K×N) as op_N col-major (N×K, ldb=N), then A
  // (row-major M×K) as op_N col-major (K×M, lda=K). Output is col-major
  // (N, M) == row-major (M, N).
  for (int i = 0; i < 10; ++i) {
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, d_b_kn, CUDA_R_16F, N,
                 d_a, CUDA_R_16F, K,
                 &beta, d_c, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(s);
  for (int i = 0; i < reps; ++i) {
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, d_b_kn, CUDA_R_16F, N,
                 d_a, CUDA_R_16F, K,
                 &beta, d_c, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  }
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  cudaEventElapsedTime(&ms, s, e);
  float cublas_us = ms * 1000.0f / reps;
  printf("cuBLAS gemmEx M2048_N4096_K4096 fp16/fp32acc: %.2f us/iter\n", cublas_us);

  printf("\n=== ratio (CUDA-L2 / cuBLAS) = %.3f ===\n", cuda_l2_us / cublas_us);
  if (cuda_l2_us < cublas_us) {
    printf("CUDA-L2 WINS (%.1f%% faster)\n", 100.0f * (cublas_us - cuda_l2_us) / cublas_us);
  } else {
    printf("cuBLAS WINS (%.1f%% faster)\n", 100.0f * (cuda_l2_us - cublas_us) / cuda_l2_us);
  }

  cublasDestroy(h);
  cudaFree(d_a); cudaFree(d_b_kn); cudaFree(d_b_nk); cudaFree(d_c);
  return 0;
}
