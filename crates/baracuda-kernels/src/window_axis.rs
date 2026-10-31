//! One spatial axis of a pooled / `im2col` window view
//! (KISS-OPS-6.13-0010 and -0011).
//!
//! `D = dilation * (window - 1) + 1`; the output extent is
//! `floor((n + 2 * pad - D) / stride) + 1`, or `0` when `n + 2 * pad < D` (no window fits).
//! Window `o` has tap `j` at operand position `o * stride - pad + j * dilation`, in bounds
//! exactly when that position lies in `[0, n)`.
//!
//! The plans used to evaluate the extent with Rust's truncating `/`, which rounds a
//! negative numerator toward zero: `n = 3, window = 4, stride = 4` gave `(3 - 4) / 4 + 1 = 1`
//! where no window fits. [`out_extent`] is the floor-correct form. For every shape whose
//! numerator is `>= 0` the two agree (pinned by `agrees_with_truncating_form_when_a_window_fits`).

/// Output extent of one axis; `0` when no window fits or a parameter is not expressible
/// (`window`, `stride` or `dilation` < 1, `pad` < 0, `n` < 0). Computed in `i64`.
pub(crate) fn out_extent(n: i32, pad: i32, window: i32, stride: i32, dilation: i32) -> i32 {
    if n < 0 || pad < 0 || window < 1 || stride < 1 || dilation < 1 {
        return 0;
    }
    let span = i64::from(n) + 2 * i64::from(pad);
    let dilated = i64::from(dilation) * (i64::from(window) - 1) + 1;
    if span < dilated {
        0
    } else {
        i32::try_from((span - dilated) / i64::from(stride) + 1).unwrap_or(i32::MAX)
    }
}

/// KISS-OPS-6.13-0011: whether every output window of the axis has at least one in-bounds
/// tap. Closed form per window (the range of tap indices `j` with
/// `0 <= o*s - p + j*d < n`), so it does not enumerate taps. An axis with no windows
/// (extent 0) has no window lacking a tap and returns `true`; the caller declines the
/// empty output separately.
pub(crate) fn every_window_has_tap(
    n: i32,
    pad: i32,
    window: i32,
    stride: i32,
    dilation: i32,
) -> bool {
    let out = i64::from(out_extent(n, pad, window, stride, dilation));
    let (n, k) = (i64::from(n), i64::from(window));
    let (s, d, p) = (i64::from(stride), i64::from(dilation), i64::from(pad));
    (0..out).all(|o| {
        let base = o * s - p;
        let j_lo = if base >= 0 { 0 } else { (-base + d - 1) / d };
        let j_hi = (n - 1 - base).div_euclid(d).min(k - 1);
        j_lo <= j_hi
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pre-fix formula: Rust's truncating `/` on a possibly negative numerator.
    fn truncating(n: i32, p: i32, k: i32, s: i32) -> i32 {
        (n + 2 * p - k) / s + 1
    }

    /// Independent derivation: enumerate every tap of every window that fits.
    fn brute(n: i32, p: i32, k: i32, s: i32, d: i32) -> (i32, bool) {
        let mut windows = 0;
        let mut all_have_tap = true;
        let mut o = 0i64;
        // A window fits when its last tap (o*s + d*(k-1)) is within the padded span.
        while o * i64::from(s) + i64::from(d) * i64::from(k - 1) < i64::from(n + 2 * p) {
            windows += 1;
            let has_tap = (0..k).any(|j| {
                let pos = o * i64::from(s) - i64::from(p) + i64::from(j) * i64::from(d);
                (0..i64::from(n)).contains(&pos)
            });
            all_have_tap &= has_tap;
            o += 1;
        }
        (windows, all_have_tap)
    }

    #[test]
    fn matches_brute_force_over_a_grid() {
        let mut checked = 0;
        let mut lacking = 0;
        for n in 0..10 {
            for k in 1..6 {
                for s in 1..5 {
                    for d in 1..4 {
                        for p in 0..5 {
                            let (w, ok) = brute(n, p, k, s, d);
                            assert_eq!(
                                out_extent(n, p, k, s, d),
                                w,
                                "extent n={n} k={k} s={s} d={d} p={p}"
                            );
                            assert_eq!(
                                every_window_has_tap(n, p, k, s, d),
                                ok,
                                "tap n={n} k={k} s={s} d={d} p={p}"
                            );
                            checked += 1;
                            lacking += usize::from(!ok);
                        }
                    }
                }
            }
        }
        // The grid must exercise both outcomes, or the equality above proves little.
        assert_eq!(checked, 10 * 5 * 4 * 3 * 5);
        assert!(
            lacking > 0,
            "no grid point lacks a tap: the overlap check is untested"
        );
    }

    #[test]
    fn kiss_counterexample_is_declined() {
        // KISS-OPS-6.13-0011: n=1, k=2, d=2, s=1, p=1 has one window, taps at -1 and 1.
        assert_eq!(out_extent(1, 1, 2, 1, 2), 1);
        assert!(!every_window_has_tap(1, 1, 2, 1, 2));
        // The attribute-only bound 2*p <= D holds (2 <= 3), which is why it is not enough.
        let (k, d, p) = (2, 2, 1);
        assert!(2 * p <= d * (k - 1) + 1);
        // Control: the same attributes over a longer axis do have a tap in every window.
        assert!(every_window_has_tap(4, 1, 2, 1, 2));
    }

    #[test]
    fn no_window_fits_is_extent_zero_not_one() {
        // Truncating division gave (3 - 4) / 4 + 1 = 1.
        assert_eq!(
            truncating(3, 0, 4, 4),
            1,
            "control: the old form really returns 1"
        );
        assert_eq!(out_extent(3, 0, 4, 4, 1), 0);
    }

    #[test]
    fn agrees_with_truncating_form_when_a_window_fits() {
        for n in 1..40 {
            for k in 1..8 {
                for s in 1..6 {
                    for p in 0..4 {
                        if n + 2 * p >= k {
                            assert_eq!(out_extent(n, p, k, s, 1), truncating(n, p, k, s));
                        }
                    }
                }
            }
        }
    }
}
