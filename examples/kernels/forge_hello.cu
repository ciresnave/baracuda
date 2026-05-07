// Canonical baracuda-forge demo kernel: element-wise vector addition.
//
// Compiled to PTX by examples/build.rs via baracuda-forge, then loaded
// at runtime by examples/forge_hello.rs via baracuda-driver.

extern "C" __global__ void vector_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
