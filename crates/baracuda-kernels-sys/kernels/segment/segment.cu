// baracuda-kernels Phase 7 Milestone 7.6 — segment / scatter-reduce
// op family (Category S). FW for sorted + unsorted variants; BW for
// sum + mean (sorted and unsorted share the BW launcher).

#include "../include/baracuda_segment.cuh"

// ---------- Sorted FW ----------
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_sum_f32,  float,  baracuda::segment::SEG_SUM)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_sum_f64,  double, baracuda::segment::SEG_SUM)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_mean_f32, float,  baracuda::segment::SEG_MEAN)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_mean_f64, double, baracuda::segment::SEG_MEAN)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_max_f32,  float,  baracuda::segment::SEG_MAX)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_max_f64,  double, baracuda::segment::SEG_MAX)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_min_f32,  float,  baracuda::segment::SEG_MIN)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_min_f64,  double, baracuda::segment::SEG_MIN)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_prod_f32, float,  baracuda::segment::SEG_PROD)
BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(
    segment_prod_f64, double, baracuda::segment::SEG_PROD)

// ---------- Unsorted FW ----------
BARACUDA_KERNELS_UNSORTED_SEGMENT_SUM_INSTANTIATE (unsorted_segment_sum_f32,  float )
BARACUDA_KERNELS_UNSORTED_SEGMENT_SUM_INSTANTIATE (unsorted_segment_sum_f64,  double)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MEAN_INSTANTIATE(unsorted_segment_mean_f32, float )
BARACUDA_KERNELS_UNSORTED_SEGMENT_MEAN_INSTANTIATE(unsorted_segment_mean_f64, double)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MAX_INSTANTIATE (unsorted_segment_max_f32,  float )
BARACUDA_KERNELS_UNSORTED_SEGMENT_MAX_INSTANTIATE (unsorted_segment_max_f64,  double)
BARACUDA_KERNELS_UNSORTED_SEGMENT_MIN_INSTANTIATE (unsorted_segment_min_f32,  float )
BARACUDA_KERNELS_UNSORTED_SEGMENT_MIN_INSTANTIATE (unsorted_segment_min_f64,  double)

// ---------- Sum BW (sorted + unsorted share the same kernel) ----------
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(segment_sum_backward_f32, float )
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(segment_sum_backward_f64, double)
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(unsorted_segment_sum_backward_f32, float )
BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(unsorted_segment_sum_backward_f64, double)

// ---------- Mean BW (sorted + unsorted share the same kernel) ----------
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(segment_mean_backward_f32, float )
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(segment_mean_backward_f64, double)
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(unsorted_segment_mean_backward_f32, float )
BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(unsorted_segment_mean_backward_f64, double)
