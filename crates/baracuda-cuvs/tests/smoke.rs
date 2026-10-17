//! Smoke tests for the cuVS safe wrapper.
//!
//! These require BOTH a CUDA GPU **and** a working NVIDIA RAPIDS / cuVS
//! install (`libcuvs.so` on the loader path), neither of which is present on
//! the Windows dev machine, so they are gated behind the `device-tests` feature. Run them on a
//! Linux box with cuVS via:
//!
//! ```text
//! cargo test -p baracuda-cuvs --features cuvs --features device-tests
//! ```
//!
//! Install cuVS with e.g. `conda install -c rapidsai -c conda-forge -c nvidia cuvs`
//! or `pip install cuvs-cu12`.

#![cfg(feature = "cuvs")]

use baracuda_cuvs::Resources;
#[cfg(feature = "device-tests")]
use baracuda_cuvs::{BruteForce, IvfFlat, IvfFlatBuildParams, IvfFlatSearchParams, Metric};
#[cfg(feature = "device-tests")]
use baracuda_driver::{Context, Device, DeviceBuffer, require_optional};

#[cfg(feature = "device-tests")]
const N_ROWS: usize = 100;
#[cfg(feature = "device-tests")]
const DIM: usize = 128;
#[cfg(feature = "device-tests")]
const K: usize = 5;
#[cfg(feature = "device-tests")]
const N_QUERIES: usize = 5;

#[cfg(feature = "device-tests")]
/// Deterministic pseudo-random fixture (xorshift) so tests are reproducible
/// without a `rand` dependency.
fn fixture(n: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // Map to [-1, 1).
            ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[cfg(feature = "device-tests")]
fn ctx() -> Context {
    Context::new(&Device::get(0).expect("device 0")).expect("context")
}

/// The loader degrades gracefully when cuVS is absent: `Resources::new()`
/// either succeeds (cuVS present) or returns a loader error — it must never
/// panic. Always runs (under the `cuvs` feature).
#[test]
fn resources_new_does_not_panic() {
    match Resources::new() {
        Ok(_res) => { /* cuVS present on this host */ }
        Err(e) => {
            // Expected on hosts without RAPIDS.
            let msg = format!("{e}");
            assert!(!msg.is_empty());
        }
    }
}

#[test]
#[cfg(feature = "device-tests")]
fn brute_force_finds_self() {
    let ctx = ctx();
    let res = require_optional!(Resources::new(), "NVIDIA RAPIDS / cuVS runtime");

    let host = fixture(N_ROWS * DIM);
    let dataset = DeviceBuffer::from_slice(&ctx, &host).expect("upload dataset");

    let index = BruteForce::<f32>::build(&res, &dataset, N_ROWS, DIM, Metric::L2Expanded)
        .expect("build brute-force index");

    // Query with the first N_QUERIES dataset rows; each row's exact nearest
    // neighbour is itself (distance ~0).
    let queries = DeviceBuffer::from_slice(&ctx, &host[..N_QUERIES * DIM]).expect("upload queries");
    let (neighbors, distances) = index
        .search(&res, &queries, N_QUERIES, K)
        .expect("brute-force search");

    let mut n_host = vec![0i64; N_QUERIES * K];
    let mut d_host = vec![0f32; N_QUERIES * K];
    neighbors.copy_to_host(&mut n_host).unwrap();
    distances.copy_to_host(&mut d_host).unwrap();

    for q in 0..N_QUERIES {
        assert_eq!(
            n_host[q * K],
            q as i64,
            "query {q}: nearest neighbour should be itself, got {}",
            n_host[q * K]
        );
        assert!(
            d_host[q * K].abs() < 1e-3,
            "query {q}: self-distance should be ~0, got {}",
            d_host[q * K]
        );
        for &idx in &n_host[q * K..(q + 1) * K] {
            assert!(
                (0..N_ROWS as i64).contains(&idx),
                "neighbour index out of range: {idx}"
            );
        }
    }
}

#[test]
#[cfg(feature = "device-tests")]
fn ivf_flat_build_and_search() {
    let ctx = ctx();
    let res = require_optional!(Resources::new(), "NVIDIA RAPIDS / cuVS runtime");

    let host = fixture(N_ROWS * DIM);
    let dataset = DeviceBuffer::from_slice(&ctx, &host).expect("upload dataset");

    // n_lists must be well below N_ROWS for k-means to train on 100 vectors.
    let params = IvfFlatBuildParams {
        metric: Metric::L2Expanded,
        n_lists: 10,
        ..Default::default()
    };
    let index =
        IvfFlat::<f32>::build(&res, &dataset, N_ROWS, DIM, params).expect("build IVF-Flat index");
    assert_eq!(index.dim(), DIM);

    let queries = DeviceBuffer::from_slice(&ctx, &host[..N_QUERIES * DIM]).expect("upload queries");
    let search = IvfFlatSearchParams { n_probes: 10 };
    let (neighbors, distances) = index
        .search(&res, &queries, N_QUERIES, K, search)
        .expect("IVF-Flat search");

    let mut n_host = vec![0i64; N_QUERIES * K];
    let mut d_host = vec![0f32; N_QUERIES * K];
    neighbors.copy_to_host(&mut n_host).unwrap();
    distances.copy_to_host(&mut d_host).unwrap();

    // IVF-Flat is approximate, but with n_probes == n_lists it degenerates to
    // exact, so the self-match should still be the top hit.
    for q in 0..N_QUERIES {
        assert_eq!(
            n_host[q * K],
            q as i64,
            "query {q}: expected self as top hit"
        );
        for &idx in &n_host[q * K..(q + 1) * K] {
            assert!(
                (0..N_ROWS as i64).contains(&idx),
                "neighbour index out of range: {idx}"
            );
        }
        assert!(d_host[q * K].is_finite());
    }
}
