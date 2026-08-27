// KISS #329 / §6.8-0010(a) device measurement: does sm_89 (RTX 4070) preserve a
// NaN's payload through a SELECT-MOVED minmax result, or canonicalize it?
//
// The discriminator (per the KISS architect): the measurement must be a MOVE — a
// §6.13 select-decomposed result whose output is a moved float operand — NOT an
// arithmetic op. An arithmetic op (a+b) canonicalizes a COMPUTED NaN, which is
// §6.8-0010's MAIN rule and already settled; minmax is the MOVED case the clause
// excludes. So we compute max_prop(a,b) as a select over the FLOAT operands and
// read the result's bits, with a+b alongside as the settled-main-rule contrast.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// max_prop via the §6.13 select-decomposition: the output is a MOVED float operand
// (a or b), never an arithmetic combination. Written on floats so we measure
// whether a float-register select preserves the payload, not whether an explicit
// bit-move does (which would be trivially yes and measure the wrong thing).
__device__ __forceinline__ float max_prop_select(float a, float b) {
    if (a != a) return a;        // a is NaN -> move a
    if (b != b) return b;        // b is NaN -> move b
    return (a >= b) ? a : b;     // both finite -> move the larger
}

__global__ void measure(uint32_t* out, uint32_t abits, uint32_t bbits) {
    float a = __uint_as_float(abits);
    float b = __uint_as_float(bbits);
    out[0] = __float_as_uint(max_prop_select(a, b)); // SELECT-MOVED (the open case)
    out[1] = __float_as_uint(a + b);                 // ARITHMETIC add (settled main rule)
    out[2] = __float_as_uint(a);                     // bare float round-trip of a
}

int main() {
    const uint32_t sNaN = 0x7F801234u; // signaling NaN, quiet-bit CLEAR, payload 0x1234
    const uint32_t qNaN = 0x7FC01234u; // quiet NaN, same payload
    const uint32_t one  = 0x3F800000u; // 1.0f

    uint32_t *d = nullptr;
    if (cudaMalloc(&d, 3 * sizeof(uint32_t)) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device: %s  sm_%d%d  CUDA runtime measurement\n", p.name, p.major, p.minor);

    struct { const char* name; uint32_t a; } cases[] = { {"sNaN", sNaN}, {"qNaN", qNaN} };
    for (auto& c : cases) {
        measure<<<1,1>>>(d, c.a, one);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(e)); return 1; }
        uint32_t h[3];
        cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
        printf("\n== %s a=0x%08X, b=1.0 ==\n", c.name, c.a);
        printf("  select-MOVED max_prop(a,b) = 0x%08X  -> payload %s\n",
               h[0], (h[0] == c.a) ? "PRESERVED (moved case; supports 6.8-0010(a))"
                                   : "CANONICALIZED (moved case DIFFERS from x86 -> 6.8-0010(a) needs amending)");
        printf("  arithmetic   a + b         = 0x%08X  (settled main-rule contrast)\n", h[1]);
        printf("  bare float round-trip of a = 0x%08X  %s\n", h[2],
               (h[2] == c.a) ? "(a survived a load/store unchanged)" : "(a changed on a bare float round-trip)");
    }
    cudaFree(d);
    return 0;
}
