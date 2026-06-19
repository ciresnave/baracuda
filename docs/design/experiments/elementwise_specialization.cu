// Go/no-go for AOT kernel specialization (docs/design/kernel-specialization.md §11).
//
// Question: does a structure-specialized elementwise kernel beat the GENERIC
// STRIDED sibling enough to justify the specialization machinery?
//
// Three kernels run the SAME contiguous f32 add `y = a + b` over a 4-D tensor:
//   1. generic_strided   — one thread/elem, runtime-rank coord-unravel + dotted
//                          strides per element (mirrors baracuda's generic
//                          strided sibling: it does NOT know the data is contig).
//   2. specialized_scalar— knows contiguous → linear index, scalar load/store.
//   3. specialized_vec4  — knows contiguous + 16B aligned + ext%4==0 → float4.
//
// (generic → scalar) isolates the unravel-elimination win; (scalar → vec4)
// isolates the vectorization win.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_=(x); if(e_){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1; } } while(0)

__global__ void add_generic_strided(
    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ sa, const int64_t* __restrict__ sb, const int64_t* __restrict__ sy,
    int rank, int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (; i < n; i += step) {
        int64_t lin = i, oa = 0, ob = 0, oy = 0;
        for (int d = rank - 1; d >= 0; --d) {          // runtime rank: no unroll
            int64_t s = shape[d];
            int64_t c = lin % s; lin /= s;             // divmod per dim per elem
            oa += c * sa[d]; ob += c * sb[d]; oy += c * sy[d];
        }
        y[oy] = a[oa] + b[ob];
    }
}

__global__ void add_specialized_scalar(
    const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y, int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (; i < n; i += step) y[i] = a[i] + b[i];
}

__global__ void add_specialized_vec4(
    const float4* __restrict__ a, const float4* __restrict__ b, float4* __restrict__ y, int64_t n4)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (; i < n4; i += step) {
        float4 va = a[i], vb = b[i], vy;
        vy.x = va.x + vb.x; vy.y = va.y + vb.y; vy.z = va.z + vb.z; vy.w = va.w + vb.w;
        y[i] = vy;
    }
}

__global__ void fill(float* p, float v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (; i < n; i += step) p[i] = v;
}

static float time_kernel(void (*launch)(void*), void* ctx, int iters) {
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < 5; ++i) launch(ctx);          // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch(ctx);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
}

struct Ctx {
    const float *a, *b; float* y;
    const int64_t *shape, *sa, *sb, *sy;
    int rank; int64_t n;
    int grid, block;
};
static void launch_generic(void* p){ Ctx*c=(Ctx*)p; add_generic_strided<<<c->grid,c->block>>>(c->a,c->b,c->y,c->shape,c->sa,c->sb,c->sy,c->rank,c->n); }
static void launch_scalar (void* p){ Ctx*c=(Ctx*)p; add_specialized_scalar<<<c->grid,c->block>>>(c->a,c->b,c->y,c->n); }
static void launch_vec4   (void* p){ Ctx*c=(Ctx*)p; add_specialized_vec4<<<c->grid,c->block>>>((const float4*)c->a,(const float4*)c->b,(float4*)c->y,c->n/4); }

int main() {
    const int64_t shape_h[4] = {256, 256, 32, 32};    // 67,108,864 elems (64 Mi)
    const int rank = 4;
    int64_t n = 1; for (int d=0; d<rank; ++d) n *= shape_h[d];
    const int64_t stride_h[4] = {256*32*32, 32*32, 32, 1};  // row-major contiguous

    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (sm_%d%d, %d SMs, %.1f GB)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.totalGlobalMem/1e9);
    printf("Tensor: 4-D [256,256,32,32] = %lld f32 elems (%.0f MiB/buffer), contiguous\n\n",
           (long long)n, n*4/1048576.0);

    float *a,*b,*y; int64_t *shape,*sa,*sb,*sy;
    CHECK(cudaMalloc(&a, n*4)); CHECK(cudaMalloc(&b, n*4)); CHECK(cudaMalloc(&y, n*4));
    CHECK(cudaMalloc(&shape, 8*8)); CHECK(cudaMalloc(&sa, 8*8)); CHECK(cudaMalloc(&sb, 8*8)); CHECK(cudaMalloc(&sy, 8*8));
    CHECK(cudaMemcpy(shape, shape_h, rank*8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sa, stride_h, rank*8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sb, stride_h, rank*8, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(sy, stride_h, rank*8, cudaMemcpyHostToDevice));

    int block = 256, grid = prop.multiProcessorCount * 32;
    fill<<<grid,block>>>(a, 1.0f, n); fill<<<grid,block>>>(b, 2.0f, n);
    CHECK(cudaDeviceSynchronize());

    Ctx ctx{a,b,y,shape,sa,sb,sy,rank,n,grid,block};
    const int iters = 100;
    const double bytes = 3.0 * n * 4;                 // 2 reads + 1 write

    // correctness + timing per kernel
    struct Row { const char* name; void(*fn)(void*); };
    Row rows[3] = {
        {"generic_strided  (unravel)", launch_generic},
        {"specialized_scalar (linear)", launch_scalar},
        {"specialized_vec4 (linear+v4)", launch_vec4},
    };
    float base_ms = 0;
    for (int r = 0; r < 3; ++r) {
        CHECK(cudaMemset(y, 0, n*4));
        rows[r].fn(&ctx); CHECK(cudaDeviceSynchronize());
        // verify y == 3.0
        float* yh = (float*)malloc(n*4);
        CHECK(cudaMemcpy(yh, y, n*4, cudaMemcpyDeviceToHost));
        int bad = 0; for (int64_t i=0;i<n && bad<1;++i) if (yh[i]!=3.0f) bad=1;
        free(yh);
        float ms = time_kernel(rows[r].fn, &ctx, iters);
        if (r == 0) base_ms = ms;
        printf("%-30s %7.3f ms  %7.1f GB/s  %s  speedup x%.2f\n",
               rows[r].name, ms, bytes/(ms*1e6), bad?"FAIL":"ok ", base_ms/ms);
    }
    return 0;
}
