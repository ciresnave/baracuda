// Source for tests/kernels/vector_add.ptx.
//
// If you change this file, regenerate the .ptx with:
//   nvcc --ptx -arch=sm_50 vector_add.cu -o vector_add.ptx
// or the bundled helper:
//   cargo xtask build-kernels
//
// baracuda does not require nvcc at build time — the committed .ptx is the
// source of truth for the test; this .cu is provided only for reference.

extern "C" __global__ void vector_add(
    const float* a,
    const float* b,
    float* c,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
