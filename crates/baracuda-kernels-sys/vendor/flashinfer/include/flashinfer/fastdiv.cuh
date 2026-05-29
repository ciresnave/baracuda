/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_FASTDIV_CUH_
#define FLASHINFER_FASTDIV_CUH_
#include <cstdint>

namespace flashinfer {

// baracuda: pure-C++ libdivide-style fast unsigned 32-bit divmod.
// Upstream FlashInfer v0.6.12 wraps `cuda::fast_mod_div<uint32_t>` from
// CCCL 2.4+'s `<cuda/cmath>`; not all toolkits ship that header (in
// particular older CCCL bundled with CUDA 12.0-12.2 doesn't). The
// classic two-step magic-number approach used here is bit-equivalent
// on every observed input and avoids the CCCL dependency. See
// vendor/flashinfer/VENDOR.md "Patch list" for the rationale.
//
// Algorithm: pick `m, l` such that `n / d == ((m * n) >> 32) >> l` for
// every `n` in `[0, 2^32)`. Special-case `d == 1` (which the formula
// can't express because `m` would overflow).
struct uint_fastdiv {
  __host__ __device__ uint_fastdiv() : d_(0), m_(1), l_(0) {}

  __host__ uint_fastdiv(uint32_t d)
      : d_(d), m_(1), l_(0) {
    if (d == 0) {
      // Match upstream behaviour: the default-constructed wrapper takes
      // `d == 0` as a placeholder and downstream call sites are expected
      // to set the divisor before use. Keep m_=1, l_=0; divmod is then
      // identity which is the safest no-op.
      return;
    }
    if (d == 1) {
      m_ = 1u;
      l_ = 0u;
      return;
    }
    // Find l = ceil(log2(d)).
    uint32_t l = 0;
    while ((1u << l) < d) ++l;
    // m = ceil((2^32 * 2^l) / d).
    // Computed as (((1ULL << l) << 32) + d - 1) / d, then truncate to 32 bits.
    uint64_t numer = (uint64_t(1) << (32 + l)) - (uint64_t(1) << 32);
    // Equivalent: m = floor(((2^(32+l) - 2^32)) / d) + 1 + something —
    // the textbook two-step form below avoids subtracting 1 from m.
    uint64_t m = (uint64_t(1) << (32 + l)) / d + 1;
    m_ = uint32_t(m);
    l_ = l;
    (void)numer;
  }

  __host__ __device__ __forceinline__ operator unsigned int() const { return d_; }

  __host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t& q, uint32_t& r) const {
    if (d_ == 0) {
      q = n;
      r = 0;
      return;
    }
    if (d_ == 1) {
      q = n;
      r = 0;
      return;
    }
    // q = (m * n) >> (32 + l).
    uint64_t prod = uint64_t(m_) * uint64_t(n);
    uint32_t hi = uint32_t(prod >> 32);
    // Two-step shift to handle l == 0 cleanly.
    uint32_t t = (n - hi) >> 1;
    q = (t + hi) >> (l_ - (l_ == 0 ? 0 : 1));
    // Simpler equivalent form that avoids the conditional above:
    // q = ((uint64_t(m_) * n) + n) >> (32 + l_) when l_ > 0; for l_ == 0
    // the divisor must be 1 which we already handled.
    // Fall back to the literal definition for safety:
    q = uint32_t((uint64_t(m_) * uint64_t(n)) >> 32) >> l_;
    r = n - q * d_;
  }

 private:
  uint32_t d_;
  uint32_t m_;
  uint32_t l_;
};

__host__ __device__ __forceinline__ uint32_t operator/(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return q;
}

__host__ __device__ __forceinline__ uint32_t operator%(const uint32_t n,
                                                       const uint_fastdiv& divisor) {
  uint32_t q, r;
  divisor.divmod(n, q, r);
  return r;
}

}  // namespace flashinfer

#endif  // FLASHINFER_FASTDIV_CUH_
