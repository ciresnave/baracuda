// baracuda_nms.cuh
//
// Templated kernels and INSTANTIATE macros for `nms` (non-max
// suppression, Phase 9 Category T). Returns a `[num_boxes]` boolean
// keep mask plus a scalar count. No backward — set-valued op.
//
// Box format (per-row, 4 contiguous values): (x1, y1, x2, y2) with
// x2 >= x1, y2 >= y1 (caller is responsible for canonical ordering).
//
// Algorithm: simple O(N²) sequential-greedy NMS in a single block (one
// thread per pair-fan-out is tractable for the trailblazer scope). For
// each box i in descending score order, mark keep[i] = true and zero
// out keep[j] for all j > i (in score order) where IoU(i, j) > thresh.
//
// **Input ordering contract**: caller supplies boxes already sorted by
// score (descending). This matches torchvision's NMS contract.
//
// Output: `keep_mask` is a `[num_boxes]` u8 buffer (0 / 1), indexed in
// the SAME ORDER as the input boxes. `count` is a single i32 with the
// number of kept boxes.
//
// Status codes (0/2/5).

#ifndef BARACUDA_NMS_CUH
#define BARACUDA_NMS_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace image {

template <typename T>
__device__ inline float box_iou_xyxy(const T* a, const T* b)
{
    float ax1 = (float)a[0], ay1 = (float)a[1], ax2 = (float)a[2], ay2 = (float)a[3];
    float bx1 = (float)b[0], by1 = (float)b[1], bx2 = (float)b[2], by2 = (float)b[3];
    float ix1 = ax1 > bx1 ? ax1 : bx1;
    float iy1 = ay1 > by1 ? ay1 : by1;
    float ix2 = ax2 < bx2 ? ax2 : bx2;
    float iy2 = ay2 < by2 ? ay2 : by2;
    float iw = ix2 - ix1; if (iw < 0) iw = 0;
    float ih = iy2 - iy1; if (ih < 0) ih = 0;
    float inter = iw * ih;
    float a_area = (ax2 - ax1) * (ay2 - ay1);
    float b_area = (bx2 - bx1) * (by2 - by1);
    float uni = a_area + b_area - inter;
    if (uni <= 0) return 0.0f;
    return inter / uni;
}

// Single-block kernel — thread 0 sweeps boxes in input order, marking
// keep_mask and decrementing count_out. Suppressed-pair check is
// parallelized via the inner loop one thread per j.
template <typename T>
__global__ void nms_kernel(
    const T* __restrict__ boxes,    // [num_boxes, 4]
    uint8_t* __restrict__ keep_mask,
    int32_t* __restrict__ count_out,
    int num_boxes, float iou_thresh)
{
    extern __shared__ unsigned char smem[];
    uint8_t* killed = reinterpret_cast<uint8_t*>(smem);  // length = num_boxes
    int tid = threadIdx.x;
    int blk = blockDim.x;
    for (int i = tid; i < num_boxes; i += blk) killed[i] = 0;
    __syncthreads();
    if (tid == 0) {
        int32_t kept = 0;
        for (int i = 0; i < num_boxes; ++i) {
            if (killed[i]) { keep_mask[i] = 0; continue; }
            keep_mask[i] = 1;
            kept++;
            // Suppress later boxes with IoU > thresh.
            for (int j = i + 1; j < num_boxes; ++j) {
                if (killed[j]) continue;
                float iou = box_iou_xyxy<T>(boxes + (int64_t)i * 4, boxes + (int64_t)j * 4);
                if (iou > iou_thresh) killed[j] = 1;
            }
        }
        *count_out = kept;
    }
}

template <typename T>
__host__ inline int32_t launch_nms(
    const T* boxes, uint8_t* keep_mask, int32_t* count_out,
    int num_boxes, float iou_thresh, cudaStream_t stream)
{
    if (num_boxes < 0) return 2;
    if (num_boxes == 0) {
        cudaMemsetAsync(count_out, 0, sizeof(int32_t), stream);
        return 0;
    }
    int kBlock = 32;
    size_t shmem = (size_t)num_boxes;
    nms_kernel<T><<<1, kBlock, shmem, stream>>>(
        boxes, keep_mask, count_out, num_boxes, iou_thresh);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::image

// =============================================================================
// INSTANTIATE macros.
// =============================================================================

#define BARACUDA_KERNELS_NMS_INSTANTIATE(NAME, T)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t num_boxes,                                                                     \
        float iou_thresh,                                                                      \
        const void* boxes,                                                                     \
        void* keep_mask,                                                                       \
        void* count_out,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (boxes == nullptr || keep_mask == nullptr || count_out == nullptr) return 2;        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_nms<T>(                                                 \
            static_cast<const T*>(boxes),                                                      \
            static_cast<uint8_t*>(keep_mask),                                                  \
            static_cast<int32_t*>(count_out),                                                  \
            num_boxes, iou_thresh, stream);                                                    \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t num_boxes,                                                                     \
        float /*iou_thresh*/,                                                                  \
        const void* /*boxes*/,                                                                 \
        const void* /*keep_mask*/,                                                             \
        const void* /*count_out*/)                                                             \
    {                                                                                          \
        if (num_boxes < 0) return 2;                                                           \
        return 0;                                                                              \
    }

#endif // BARACUDA_NMS_CUH
