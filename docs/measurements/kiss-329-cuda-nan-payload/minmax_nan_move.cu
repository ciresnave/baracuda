// KISS #329 / §6.8-0010(a) device measurement: does sm_89 (RTX 4070) preserve a
// NaN's payload through a SELECT-MOVED min/max result, or canonicalize it?
//
// The discriminator (per the KISS architect): the measurement must be a MOVE — a
// §6.13 select-decomposed result whose output is a moved float operand — NOT an
// arithmetic op. An arithmetic op (a+b) canonicalizes a COMPUTED NaN, which is
// §6.8-0010's MAIN rule and already settled; minmax is the MOVED case the clause
// excludes. So we compute max_prop AND min_prop as selects over the FLOAT operands
// and read the results' bits, with a+b alongside as the settled-main-rule contrast.
//
// BOTH arms are measured, not just max: `min` and `max` decompose to DIFFERENT
// selects (`a<=b` vs `a>=b`), and nvcc could fold one into a hardware FMNMX while
// leaving the other as FSEL — a fold hazard that must be checked per arm, not
// assumed symmetric. A `minmax` claim over a `max`-only kernel would be a label
// wider than its evidence.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// max_prop / min_prop via the §6.13 select-decomposition: the output is a MOVED
// float operand (a or b), never an arithmetic combination. Written on floats so we
// measure whether a float-register select preserves the payload, not whether an
// explicit bit-move does (which would be trivially yes and measure the wrong thing).
// Both are NaN-PROPAGATING (return the NaN operand), which is why nvcc may NOT fold
// them into a NaN-suppressing hardware FMNMX at default -O3 — the SASS confirms it.
__device__ __forceinline__ float max_prop_select(float a, float b) {
    if (a != a) return a;        // a is NaN -> move a
    if (b != b) return b;        // b is NaN -> move b
    return (a >= b) ? a : b;     // both finite -> move the larger
}
__device__ __forceinline__ float min_prop_select(float a, float b) {
    if (a != a) return a;        // a is NaN -> move a
    if (b != b) return b;        // b is NaN -> move b
    return (a <= b) ? a : b;     // both finite -> move the smaller
}

__global__ void measure(uint32_t* out, uint32_t abits, uint32_t bbits) {
    float a = __uint_as_float(abits);
    float b = __uint_as_float(bbits);
    out[0] = __float_as_uint(max_prop_select(a, b)); // MAX select-moved (open case)
    out[1] = __float_as_uint(min_prop_select(a, b)); // MIN select-moved (open case)
    out[2] = __float_as_uint(a + b);                 // ARITHMETIC add (settled rule)
    out[3] = __float_as_uint(a);                     // bare float round-trip of a
}

int main() {
    const uint32_t sNaN = 0x7F801234u; // signaling NaN, quiet-bit CLEAR, payload 0x1234
    const uint32_t qNaN = 0x7FC01234u; // quiet NaN, same payload
    const uint32_t one  = 0x3F800000u; // 1.0f

    uint32_t *d = nullptr;
    if (cudaMalloc(&d, 4 * sizeof(uint32_t)) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }

    cudaDeviceProp p;
    // Checked: an unchecked query could print a stale/zeroed device name, mislabeling
    // WHICH device produced the bits — the provenance's whole point.
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) { printf("cudaGetDeviceProperties failed\n"); cudaFree(d); return 1; }
    printf("device: %s  sm_%d%d  CUDA runtime measurement\n", p.name, p.major, p.minor);

    struct { const char* name; uint32_t a; } cases[] = { {"sNaN", sNaN}, {"qNaN", qNaN} };
    for (auto& c : cases) {
        measure<<<1,1>>>(d, c.a, one);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(e)); cudaFree(d); return 1; }
        uint32_t h[4];
        // CHECKED: a failed copy would leave h[] holding stack garbage, which the
        // prints below would report AS the measured payload — a silent wrong value
        // indistinguishable in shape from a real result. Refuse to print on failure.
        cudaError_t ce = cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) { printf("cudaMemcpy failed: %s\n", cudaGetErrorString(ce)); cudaFree(d); return 1; }
        printf("\n== %s a=0x%08X, b=1.0 ==\n", c.name, c.a);
        printf("  select-MOVED max_prop(a,b) = 0x%08X  -> payload %s\n",
               h[0], (h[0] == c.a) ? "PRESERVED (moved case; supports 6.8-0010(a))"
                                   : "CANONICALIZED (moved case DIFFERS from x86 -> 6.8-0010(a) needs amending)");
        printf("  select-MOVED min_prop(a,b) = 0x%08X  -> payload %s\n",
               h[1], (h[1] == c.a) ? "PRESERVED (moved case; supports 6.8-0010(a))"
                                   : "CANONICALIZED (moved case DIFFERS from x86 -> 6.8-0010(a) needs amending)");
        printf("  arithmetic   a + b         = 0x%08X  (settled main-rule contrast)\n", h[2]);
        printf("  bare float round-trip of a = 0x%08X  %s\n", h[3],
               (h[3] == c.a) ? "(a survived a load/store unchanged)" : "(a changed on a bare float round-trip)");
    }
    cudaFree(d);
    return 0;
}
