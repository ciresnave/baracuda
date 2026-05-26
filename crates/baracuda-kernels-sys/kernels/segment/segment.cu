// baracuda-kernels Phase 7 Milestone 7.6 — segment / scatter-reduce
// op family (Category S). FW for sorted + unsorted variants; BW for
// sum + mean (sorted and unsorted share the BW launcher).
//
// Phase 11.5 (Fuel team feedback #7): adds i64 segment-id instantiations
// alongside the original i32 ones. PyTorch defaults to int64 for
// indices/segment ids, so i64 spares callers a cast pass.

#include "../include/baracuda_segment.cuh"

// ---------- Sorted FW ----------
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_sum_f32,  float,  baracuda::segment::SEG_SUM,  int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_sum_f64,  double, baracuda::segment::SEG_SUM,  int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_mean_f32, float,  baracuda::segment::SEG_MEAN, int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_mean_f64, double, baracuda::segment::SEG_MEAN, int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_max_f32,  float,  baracuda::segment::SEG_MAX,  int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_max_f64,  double, baracuda::segment::SEG_MAX,  int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_min_f32,  float,  baracuda::segment::SEG_MIN,  int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_min_f64,  double, baracuda::segment::SEG_MIN,  int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_prod_f32, float,  baracuda::segment::SEG_PROD, int32_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_prod_f64, double, baracuda::segment::SEG_PROD, int32_t)

// i64 segment ids — Phase 11.5.
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_sum_i64idx_f32,  float,  baracuda::segment::SEG_SUM,  int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_sum_i64idx_f64,  double, baracuda::segment::SEG_SUM,  int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_mean_i64idx_f32, float,  baracuda::segment::SEG_MEAN, int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_mean_i64idx_f64, double, baracuda::segment::SEG_MEAN, int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_max_i64idx_f32,  float,  baracuda::segment::SEG_MAX,  int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_max_i64idx_f64,  double, baracuda::segment::SEG_MAX,  int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_min_i64idx_f32,  float,  baracuda::segment::SEG_MIN,  int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_min_i64idx_f64,  double, baracuda::segment::SEG_MIN,  int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_prod_i64idx_f32, float,  baracuda::segment::SEG_PROD, int64_t)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_prod_i64idx_f64, double, baracuda::segment::SEG_PROD, int64_t)

// ---------- Unsorted FW ----------
BARACUDA_KERNELS_UNSORTED_SEGMENT_SUM_INSTANTIATE (unsorted_segment_sum_f32,  float,  int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_SUM_INSTANTIATE (unsorted_segment_sum_f64,  double, int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MEAN_INSTANTIATE(unsorted_segment_mean_f32, float,  int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MEAN_INSTANTIATE(unsorted_segment_mean_f64, double, int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MAX_INSTANTIATE (unsorted_segment_max_f32,  float,  int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MAX_INSTANTIATE (unsorted_segment_max_f64,  double, int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MIN_INSTANTIATE (unsorted_segment_min_f32,  float,  int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MIN_INSTANTIATE (unsorted_segment_min_f64,  double, int32_t)

BARACUDA_KERNELS_UNSORTED_SEGMENT_SUM_INSTANTIATE (unsorted_segment_sum_i64idx_f32,  float,  int64_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_SUM_INSTANTIATE (unsorted_segment_sum_i64idx_f64,  double, int64_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MEAN_INSTANTIATE(unsorted_segment_mean_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MEAN_INSTANTIATE(unsorted_segment_mean_i64idx_f64, double, int64_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MAX_INSTANTIATE (unsorted_segment_max_i64idx_f32,  float,  int64_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MAX_INSTANTIATE (unsorted_segment_max_i64idx_f64,  double, int64_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MIN_INSTANTIATE (unsorted_segment_min_i64idx_f32,  float,  int64_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MIN_INSTANTIATE (unsorted_segment_min_i64idx_f64,  double, int64_t)

// ---------- Sum BW (sorted + unsorted share the same kernel) ----------
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(segment_sum_backward_f32, float,  int32_t)
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(segment_sum_backward_f64, double, int32_t)
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(unsorted_segment_sum_backward_f32, float,  int32_t)
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(unsorted_segment_sum_backward_f64, double, int32_t)

BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(segment_sum_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(segment_sum_backward_i64idx_f64, double, int64_t)
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(unsorted_segment_sum_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(unsorted_segment_sum_backward_i64idx_f64, double, int64_t)

// ---------- Mean BW (sorted + unsorted share the same kernel) ----------
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(segment_mean_backward_f32, float,  int32_t)
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(segment_mean_backward_f64, double, int32_t)
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(unsorted_segment_mean_backward_f32, float,  int32_t)
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(unsorted_segment_mean_backward_f64, double, int32_t)

BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(segment_mean_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(segment_mean_backward_i64idx_f64, double, int64_t)
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(unsorted_segment_mean_backward_i64idx_f32, float,  int64_t)
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(unsorted_segment_mean_backward_i64idx_f64, double, int64_t)

// ---------- Phase 25: Max / Min BW (sorted) — argmax recomputed in BW ----------
BARACUDA_KERNELS_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    segment_max_backward_f32, float,  baracuda::segment::SEG_ARG_MAX, int32_t)
BARACUDA_KERNELS_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    segment_max_backward_f64, double, baracuda::segment::SEG_ARG_MAX, int32_t)
BARACUDA_KERNELS_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    segment_min_backward_f32, float,  baracuda::segment::SEG_ARG_MIN, int32_t)
BARACUDA_KERNELS_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    segment_min_backward_f64, double, baracuda::segment::SEG_ARG_MIN, int32_t)

// ---------- Phase 25: Max / Min BW (unsorted) ----------
BARACUDA_KERNELS_UNSORTED_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    unsorted_segment_max_backward_f32, float,  baracuda::segment::SEG_ARG_MAX, int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    unsorted_segment_max_backward_f64, double, baracuda::segment::SEG_ARG_MAX, int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    unsorted_segment_min_backward_f32, float,  baracuda::segment::SEG_ARG_MIN, int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_ARG_BACKWARD_INSTANTIATE(
    unsorted_segment_min_backward_f64, double, baracuda::segment::SEG_ARG_MIN, int32_t)

// ---------- Phase 25: Prod BW (sorted + unsorted share the same kernel) ----------
BARACUDA_KERNELS_SEGMENT_PROD_BACKWARD_INSTANTIATE(segment_prod_backward_f32, float,  int32_t)
BARACUDA_KERNELS_SEGMENT_PROD_BACKWARD_INSTANTIATE(segment_prod_backward_f64, double, int32_t)
BARACUDA_KERNELS_SEGMENT_PROD_BACKWARD_INSTANTIATE(unsorted_segment_prod_backward_f32, float,  int32_t)
BARACUDA_KERNELS_SEGMENT_PROD_BACKWARD_INSTANTIATE(unsorted_segment_prod_backward_f64, double, int32_t)

// ---------- Phase 25: Unsorted Prod FW (atomicCAS retry loop) ----------
BARACUDA_KERNELS_UNSORTED_SEGMENT_PROD_INSTANTIATE(unsorted_segment_prod_f32, float,  int32_t)
BARACUDA_KERNELS_UNSORTED_SEGMENT_PROD_INSTANTIATE(unsorted_segment_prod_f64, double, int32_t)
