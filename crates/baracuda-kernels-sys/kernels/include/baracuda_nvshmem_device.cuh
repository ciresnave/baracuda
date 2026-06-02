// baracuda_nvshmem_device.cuh — device-side NVSHMEM glue. Phase 69 (device tier).
//
// Thin `__device__ __forceinline__` wrappers over the NVSHMEM *device* API
// (one-sided put/get, cooperative block/warp transfers, typed single-element
// p/g, memory-ordering, and signal ops) so kernels can issue inter-GPU RDMA
// directly from inside a kernel — the fine-grained complement to the
// collectives in `baracuda-nccl` and the *host*-side wrapper in the
// `baracuda-nvshmem` crate.
//
// ============================================================================
// BUILD REQUIREMENTS — this header is NOT free-standing like the smem_*
// helpers. The NVSHMEM device API lives in the static archive
// `libnvshmem_device.a`, which (per NVIDIA's proprietary SLA) baracuda does
// NOT and cannot bundle under its MIT/Apache license. A consumer who wants
// device-side NVSHMEM must, in their OWN build of the kernel that includes
// this header:
//
//   1. Have NVSHMEM installed and its headers on the include path
//      (`<nvshmem.h>`, `<nvshmemx.h>`).
//   2. Compile the translation unit with relocatable device code
//      (`nvcc -rdc=true`) — NVSHMEM device calls require separate
//      compilation + device linking.
//   3. Define `BARACUDA_ENABLE_NVSHMEM_DEVICE` for the TU.
//   4. Device-link the final binary against `libnvshmem_device.a`
//      (and link `libnvshmem_host.so` on the host side).
//
// When `BARACUDA_ENABLE_NVSHMEM_DEVICE` is NOT defined, this header expands to
// an empty `baracuda::nvshmem` namespace: it is always safe to `#include`
// (no `<nvshmem.h>` dependency, no rebuild trigger), but the wrappers simply
// don't exist — a kernel that tries to use them without enabling the gate
// fails to compile, which is the correct signal.
//
// `pe` arguments are PE (processing-element) ranks in the default
// NVSHMEM_TEAM_WORLD team. `dest` / `src` symmetric pointers must come from
// the symmetric heap (`baracuda::nvshmem` host `SymmetricBuffer<T>` /
// `nvshmem_malloc`); a symmetric address is valid on every PE.

#ifndef BARACUDA_NVSHMEM_DEVICE_CUH
#define BARACUDA_NVSHMEM_DEVICE_CUH

#ifdef BARACUDA_ENABLE_NVSHMEM_DEVICE

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

namespace baracuda {
namespace nvshmem {

// ============================================================================
// PE discovery — device-callable identity queries.
// ============================================================================

__device__ __forceinline__ int my_pe() { return nvshmem_my_pe(); }
__device__ __forceinline__ int n_pes() { return nvshmem_n_pes(); }

__device__ __forceinline__ int team_my_pe(nvshmem_team_t team) {
    return nvshmem_team_my_pe(team);
}
__device__ __forceinline__ int team_n_pes(nvshmem_team_t team) {
    return nvshmem_team_n_pes(team);
}

// ============================================================================
// Bulk one-sided RMA — thread-scoped. `n` is an ELEMENT count; the byte
// size is `n * sizeof(T)`. The blocking forms return once the buffer is
// reusable locally (put) / filled (get); the `_nbi` forms return immediately
// and require a later `quiet()` for completion.
// ============================================================================

template <typename T>
__device__ __forceinline__ void put(T* dest, const T* src, size_t n, int pe) {
    nvshmem_putmem(static_cast<void*>(dest), static_cast<const void*>(src),
                   n * sizeof(T), pe);
}

template <typename T>
__device__ __forceinline__ void get(T* dest, const T* src, size_t n, int pe) {
    nvshmem_getmem(static_cast<void*>(dest), static_cast<const void*>(src),
                   n * sizeof(T), pe);
}

template <typename T>
__device__ __forceinline__ void put_nbi(T* dest, const T* src, size_t n, int pe) {
    nvshmem_putmem_nbi(static_cast<void*>(dest), static_cast<const void*>(src),
                       n * sizeof(T), pe);
}

template <typename T>
__device__ __forceinline__ void get_nbi(T* dest, const T* src, size_t n, int pe) {
    nvshmem_getmem_nbi(static_cast<void*>(dest), static_cast<const void*>(src),
                       n * sizeof(T), pe);
}

// ============================================================================
// Cooperative bulk RMA — block- and warp-scoped. ALL threads in the
// block/warp must call with identical arguments; the transfer is split
// across the participating threads for higher per-call bandwidth. This is
// the workhorse for MoE all-to-all / expert-parallel dispatch.
// ============================================================================

template <typename T>
__device__ __forceinline__ void put_block(T* dest, const T* src, size_t n, int pe) {
    nvshmemx_putmem_block(static_cast<void*>(dest), static_cast<const void*>(src),
                          n * sizeof(T), pe);
}

template <typename T>
__device__ __forceinline__ void get_block(T* dest, const T* src, size_t n, int pe) {
    nvshmemx_getmem_block(static_cast<void*>(dest), static_cast<const void*>(src),
                          n * sizeof(T), pe);
}

template <typename T>
__device__ __forceinline__ void put_warp(T* dest, const T* src, size_t n, int pe) {
    nvshmemx_putmem_warp(static_cast<void*>(dest), static_cast<const void*>(src),
                         n * sizeof(T), pe);
}

template <typename T>
__device__ __forceinline__ void get_warp(T* dest, const T* src, size_t n, int pe) {
    nvshmemx_getmem_warp(static_cast<void*>(dest), static_cast<const void*>(src),
                         n * sizeof(T), pe);
}

// ============================================================================
// Typed single-element put/get — the cheapest possible RDMA (one scalar).
// `p(dest, value, pe)` writes one element to `pe`'s heap; `g(src, pe)`
// reads one element from `pe`'s heap. Overloaded over the natively-typed
// NVSHMEM scalars; for other element types use the byte-based `put`/`get`.
// ============================================================================

__device__ __forceinline__ void p(int* d, int v, int pe) { nvshmem_int_p(d, v, pe); }
__device__ __forceinline__ void p(long long* d, long long v, int pe) { nvshmem_longlong_p(d, v, pe); }
__device__ __forceinline__ void p(unsigned int* d, unsigned int v, int pe) { nvshmem_uint_p(d, v, pe); }
__device__ __forceinline__ void p(unsigned long long* d, unsigned long long v, int pe) { nvshmem_ulonglong_p(d, v, pe); }
__device__ __forceinline__ void p(float* d, float v, int pe) { nvshmem_float_p(d, v, pe); }
__device__ __forceinline__ void p(double* d, double v, int pe) { nvshmem_double_p(d, v, pe); }

__device__ __forceinline__ int g(const int* s, int pe) { return nvshmem_int_g(s, pe); }
__device__ __forceinline__ long long g(const long long* s, int pe) { return nvshmem_longlong_g(s, pe); }
__device__ __forceinline__ unsigned int g(const unsigned int* s, int pe) { return nvshmem_uint_g(s, pe); }
__device__ __forceinline__ unsigned long long g(const unsigned long long* s, int pe) { return nvshmem_ulonglong_g(s, pe); }
__device__ __forceinline__ float g(const float* s, int pe) { return nvshmem_float_g(s, pe); }
__device__ __forceinline__ double g(const double* s, int pe) { return nvshmem_double_g(s, pe); }

// ============================================================================
// Memory ordering / synchronization (device-scoped).
//   fence()       — order outstanding RMA to each PE (no completion wait).
//   quiet()       — block until all RMA issued by this PE has completed
//                   remotely (required to retire `_nbi` ops).
//   barrier_all() — cooperative global barrier + remote completion.
//   sync_all()    — lighter cooperative barrier (PE arrival only).
// barrier_all / sync_all are block-cooperative: all threads must call.
// ============================================================================

__device__ __forceinline__ void fence() { nvshmem_fence(); }
__device__ __forceinline__ void quiet() { nvshmem_quiet(); }
__device__ __forceinline__ void barrier_all() { nvshmem_barrier_all(); }
__device__ __forceinline__ void sync_all() { nvshmem_sync_all(); }

// ============================================================================
// Signal ops — the completion-notification mechanism for one-sided patterns
// (a put delivers data AND atomically updates a remote flag the receiver
// spins on). `sig_op` is an NVSHMEM signal op (NVSHMEM_SIGNAL_SET /
// NVSHMEM_SIGNAL_ADD); `cmp` is an NVSHMEM compare op (NVSHMEM_CMP_EQ /
// _GE / ...). Those macros come from the NVSHMEM headers — pass them through.
// ============================================================================

template <typename T>
__device__ __forceinline__ void put_signal(
    T* dest, const T* src, size_t n,
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)
{
    nvshmemx_putmem_signal(static_cast<void*>(dest), static_cast<const void*>(src),
                           n * sizeof(T), sig_addr, signal, sig_op, pe);
}

// Block-cooperative put-with-signal (all threads in the block call).
template <typename T>
__device__ __forceinline__ void put_signal_block(
    T* dest, const T* src, size_t n,
    uint64_t* sig_addr, uint64_t signal, int sig_op, int pe)
{
    nvshmemx_putmem_signal_block(static_cast<void*>(dest), static_cast<const void*>(src),
                                 n * sizeof(T), sig_addr, signal, sig_op, pe);
}

// Spin until `*sig_addr <cmp> cmp_value`; returns the observed value.
__device__ __forceinline__ uint64_t signal_wait_until(
    uint64_t* sig_addr, int cmp, uint64_t cmp_value)
{
    return nvshmem_signal_wait_until(sig_addr, cmp, cmp_value);
}

// Non-blocking read of a signal word.
__device__ __forceinline__ uint64_t signal_fetch(uint64_t* sig_addr) {
    return nvshmem_signal_fetch(sig_addr);
}

}  // namespace nvshmem
}  // namespace baracuda

#else  // !BARACUDA_ENABLE_NVSHMEM_DEVICE

// Gate off: provide only the (empty) namespace so the header is always safe
// to include without pulling in <nvshmem.h> or triggering a rebuild.
namespace baracuda { namespace nvshmem {} }

#endif  // BARACUDA_ENABLE_NVSHMEM_DEVICE

#endif  // BARACUDA_NVSHMEM_DEVICE_CUH
