// Compile-time shared-memory capacity guard for the sm_80 CUTLASS kernels.
//
// WHY THIS EXISTS (issue #87)
//
// `baracuda-cutlass`'s `pick_arch` / `pick_int_arch` route ANY device with
// `major >= 8` to `ArchSku::Sm80`, on the grounds that sm_80 kernels are
// PTX-forward-compatible to anything sm_80+. That is true of the ISA and
// says nothing about CAPACITY -- and shared memory is NOT MONOTONIC in
// architecture:
//
//     sm_80   166912 bytes opt-in      (documented)
//     sm_86   101376 bytes opt-in      <-- SMALLER than sm_80
//     sm_89   101376 bytes opt-in      <-- measured on an RTX 4070
//     sm_90   232448 bytes opt-in      (cutlass/arch/arch.h)
//
// So a kernel built for sm_80 can be selected on sm_89 and fail to launch
// there while running on both the older AND the newer part. CUTLASS opts
// into >48 KB via `cudaFuncSetAttribute` in `gemm/device/gemm.h` and maps
// the failure to `Status::kErrorInternal`, which this crate reports as
// status 5 -- an opaque error carrying no hint that the cause is capacity
// or architecture.
//
// This header makes that constraint fail the BUILD instead.
//
// NOTE: no CI job compiles these .cu files (the gating job is the
// driver-free surface), so this assert fires for anyone building with
// CUDA -- which is exactly the population that would otherwise hit the
// runtime failure.

#pragma once

#include <cuda_runtime.h>

namespace baracuda_cutlass {

/// The SMALLEST opt-in shared-memory capacity among the architectures that
/// `pick_arch` can route an sm_80 kernel to. sm_86 and sm_89 (Ampere
/// consumer / Ada) sit BELOW sm_80; they are the binding constraint.
///
/// ⚠️ THIS IS NOT "sm_89's CAP". It is a MINIMUM OVER THE SUPPORTED SET,
/// and sm_86 shares the value. If this crate ever selects kernels for an
/// architecture with a SMALLER opt-in capacity than 101376 -- i.e. if
/// `pick_arch` / `pick_int_arch` in `baracuda-cutlass/src/plan.rs` widen
/// below sm_86 -- THIS CONSTANT MUST COME DOWN WITH IT.
///
/// That is the one way this guard becomes wrong while still compiling:
/// it would keep passing, on a bound that no longer describes the set of
/// devices the kernels are selected for. Nothing detects that but this
/// comment, so change the two together.
constexpr int kMinSupportedSmemOptinBytes = 101376;

/// True when `GemmT`'s shared storage fits the smallest supported capacity.
/// Evaluates the same expression CUTLASS passes to `cudaFuncSetAttribute`.
template <typename GemmT>
struct SmemFitsSmallestSupportedArch {
  static constexpr bool value =
      sizeof(typename GemmT::GemmKernel::SharedStorage)
          <= kMinSupportedSmemOptinBytes;
};

/// Status for a CUTLASS `initialize` / `run` that did not return kSuccess.
///
/// CUTLASS maps every CUDA runtime failure there to the same
/// `Status::kErrorInternal`, so the bare status 5 does not say which call
/// failed. This returns `1000 + cudaError_t` (the encoding the flash-SDPA
/// launchers use) when the CUDA runtime still has an error pending, and 5
/// when it has none.
///
/// What each result means, read from `cutlass/gemm/device/gemm.h` `run()`:
///   * `1000 + n`: a `cudaFuncSetAttribute` call failed. CUTLASS returns right
///     after it, so the error is still pending here.
///   * `5`: the launch itself failed, or CUTLASS failed internally. CUTLASS reads
///     the launch error with `cudaGetLastError()` before returning, which clears
///     it, so it cannot be recovered here.
/// Reading the last error also clears it, so it cannot leak into the next call
/// on this thread.
inline int cuda_failure_code() {
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? 5 : 1000 + static_cast<int>(err);
}

}  // namespace baracuda_cutlass

/// Assert that one instantiated GEMM type fits every architecture this crate
/// will select it for. Place beside the `using Gemm = ...` in the launch path
/// so it is checked for every instantiation the file actually creates.
#define BARACUDA_ASSERT_SMEM_FITS(GemmT)                                       \
  static_assert(                                                               \
      ::baracuda_cutlass::SmemFitsSmallestSupportedArch<GemmT>::value,         \
      "sm_80 kernels are selected for ANY major>=8 device; sm_89's opt-in "    \
      "shared-memory cap is 101376 bytes and is SMALLER than sm_80's. This "   \
      "tile config exceeds it and would fail to launch on sm_86/sm_89 while "  \
      "running on both sm_80 and sm_90. See issue #87.")
