//! Host-side edge-case tests for `baracuda-types`. Exercises numeric
//! conversions, version parsing boundaries, DeviceRepr size invariants,
//! and Complex-number algebra.

use baracuda_types::{
    supports, BFloat16, Complex32, Complex64, CudaVersion, DeviceRepr, Feature, Half,
};

// ---- Numeric conversions ------------------------------------------------

#[test]
fn half_round_trips_exact_values() {
    let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
    for &v in &values {
        let h = Half::from_f32(v);
        assert_eq!(h.to_f32(), v, "exact-representable round-trip failed for {v}");
    }
}

#[test]
fn half_handles_subnormals_and_infinity() {
    let inf = Half::from_f32(f32::INFINITY);
    assert!(inf.to_f32().is_infinite());
    assert!(inf.to_f32() > 0.0);

    let neg_inf = Half::from_f32(f32::NEG_INFINITY);
    assert!(neg_inf.to_f32().is_infinite() && neg_inf.to_f32() < 0.0);

    let nan = Half::from_f32(f32::NAN);
    assert!(nan.to_f32().is_nan());
}

#[test]
fn half_saturates_on_overflow() {
    let big = Half::from_f32(1e10); // outside half's ±65504 range
    // Implementation-defined: either infinity or the max representable.
    let f = big.to_f32();
    assert!(f.is_infinite() || f >= 60000.0, "unexpected overflow handling: {f}");
}

#[test]
fn bfloat16_round_trips_top_byte() {
    // BF16 keeps the top 16 bits of an f32; the exponent range matches f32.
    let values = [0.0f32, 1.0, -1.0, 1e30, -1e-30, 3.14];
    for &v in &values {
        let b = BFloat16::from_f32(v);
        let back = b.to_f32();
        // We lose precision but not range.
        let rel = if v == 0.0 {
            (back - v).abs()
        } else {
            ((back - v) / v).abs()
        };
        assert!(rel < 1e-2, "bf16 round-trip relative error too large: {rel} for {v}");
    }
}

// ---- Complex-number algebra --------------------------------------------

#[test]
fn complex_constants() {
    assert_eq!(Complex32::ZERO, Complex32::new(0.0, 0.0));
    assert_eq!(Complex32::ONE, Complex32::new(1.0, 0.0));
    assert_eq!(Complex32::I, Complex32::new(0.0, 1.0));
    assert_eq!(Complex64::ZERO, Complex64::new(0.0, 0.0));
    assert_eq!(Complex64::I, Complex64::new(0.0, 1.0));
}

#[test]
fn complex_conjugate_identity() {
    let z = Complex32::new(3.0, 4.0);
    let cz = z.conj();
    assert_eq!(cz, Complex32::new(3.0, -4.0));
    // |z|² = z * conj(z) = re*re + im*im
    assert!((z.norm_sqr() - 25.0).abs() < 1e-6);
}

#[test]
fn complex_layout_matches_cucomplex() {
    // baracuda's Complex32/64 must be #[repr(C)] {re, im} so they transmute
    // to NVIDIA's cuComplex / cuDoubleComplex without casts.
    assert_eq!(core::mem::size_of::<Complex32>(), 8);
    assert_eq!(core::mem::size_of::<Complex64>(), 16);
    assert_eq!(core::mem::align_of::<Complex32>(), core::mem::align_of::<f32>());
    assert_eq!(core::mem::align_of::<Complex64>(), core::mem::align_of::<f64>());
}

// ---- Version parsing boundaries ----------------------------------------

#[test]
fn cuda_version_boundaries() {
    let v = CudaVersion::from_raw(12060);
    assert_eq!(v.major(), 12);
    assert_eq!(v.minor(), 6);

    // Major of 13, minor of 0 => 13000.
    let v13 = CudaVersion::from_major_minor(13, 0);
    assert_eq!(v13.raw(), 13000);

    // Lowest supported: floor.
    assert!(CudaVersion::FLOOR.at_least(11, 4));
    assert!(!CudaVersion::FLOOR.at_least(11, 5));
}

#[test]
fn cuda_version_display() {
    let v = CudaVersion::CUDA_12_6;
    assert_eq!(format!("{v}"), "CUDA 12.6");
}

#[test]
fn feature_required_versions_are_monotonic() {
    // Sanity: TensorMapObjects (11.8) must come after VMM (10.2).
    assert!(
        Feature::TensorMapObjects.required_version()
            > Feature::VirtualMemoryManagement.required_version()
    );
    // GraphSwitchNodes (12.8) > GraphConditionalNodes (12.3).
    assert!(
        Feature::GraphSwitchNodes.required_version()
            > Feature::GraphConditionalNodes.required_version()
    );
}

#[test]
fn supports_at_exact_required_version() {
    let cv = Feature::GreenContexts.required_version();
    assert!(supports(cv, Feature::GreenContexts));
    // One minor step back must refuse.
    let before = CudaVersion::from_major_minor(cv.major(), cv.minor().saturating_sub(1));
    if before != cv {
        assert!(!supports(before, Feature::GreenContexts));
    }
}

// ---- DeviceRepr invariants ---------------------------------------------

#[test]
fn primitives_are_devicerepr() {
    fn assert_repr<T: DeviceRepr>() {}
    assert_repr::<u8>();
    assert_repr::<i8>();
    assert_repr::<u16>();
    assert_repr::<i16>();
    assert_repr::<u32>();
    assert_repr::<i32>();
    assert_repr::<u64>();
    assert_repr::<i64>();
    assert_repr::<f32>();
    assert_repr::<f64>();
    assert_repr::<Complex32>();
    assert_repr::<Complex64>();
    assert_repr::<Half>();
    assert_repr::<BFloat16>();
}

#[test]
fn tuples_and_arrays_are_devicerepr() {
    fn assert_repr<T: DeviceRepr>() {}
    assert_repr::<(f32, f32)>();
    assert_repr::<(i32, i32, i32)>();
    assert_repr::<(f32, f64, i32, u32)>();
    assert_repr::<[f32; 4]>();
    assert_repr::<[i64; 8]>();
}
