//! Cross-platform bit-exact regression test.
//!
//! The Ozaki path goes through int8 tensor-core matmuls, which are
//! deterministic. With a fixed seed + fixed shape + fixed slice count
//! the output bit pattern should be identical across two runs on the
//! same hardware, and (subject to the Phase 44b Windows port using
//! the typedef-`__uint128_t` path on Linux + the struct emulator on
//! MSVC) **also identical across platforms** when both are built
//! against the same RTX 4070.
//!
//! This test does NOT pin specific byte patterns yet — we don't have
//! a Windows + Linux side-by-side run captured. What it DOES do:
//!
//!   - Run the same fixed-seed problem twice on whatever hardware
//!     the test is on, and assert the two runs are bit-identical.
//!   - Print the first 8 cells' bit patterns to stderr so a future
//!     Windows run can be compared against the prior Linux output
//!     by reading the test log.
//!
//! When we have a Windows + Linux side-by-side test infra, replace
//! the print with `assert_eq!(got, &EXPECTED_BYTES[..])`.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_ozimmu::{Handle, Op, OzakiSlices};

const M: usize = 256;
const SEED: u64 = 0xC0FFEE_DEAD_BEEFu64;

fn fixed_seed_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    // Same xorshift PRNG as accuracy_smoke; inlined to keep this
    // test self-contained.
    let mut x = if seed == 0 { 0xdead_beef } else { seed };
    let mut out = vec![0.0f64; rows * cols];
    for cell in &mut out {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *cell = ((x >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
    }
    out
}

fn run_once(seed: u64) -> Vec<f64> {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let h = Handle::new().expect("handle");
    h.set_stream(&stream);

    let a_host = fixed_seed_matrix(M, M, seed);
    let b_host = fixed_seed_matrix(M, M, seed.wrapping_add(1));

    let a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload A");
    let b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload B");
    let c: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, M * M).expect("alloc C");

    unsafe {
        h.dgemm(
            Op::N, Op::N,
            M, M, M,
            1.0,
            a.as_raw().0 as *const f64, M,
            b.as_raw().0 as *const f64, M,
            0.0,
            c.as_raw().0 as *mut f64, M,
            OzakiSlices::S8,
        )
        .expect("dgemm");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0.0f64; M * M];
    c.copy_to_host(&mut got).expect("D2H");
    got
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn same_hardware_bit_exact_across_runs() {
    let a = run_once(SEED);
    let b = run_once(SEED);

    // Bit-exact equality, not "approximately equal".
    let mut differ = 0usize;
    for (x, y) in a.iter().zip(&b) {
        if x.to_bits() != y.to_bits() {
            differ += 1;
        }
    }
    assert_eq!(
        differ, 0,
        "ozIMMU is supposed to be deterministic across runs on the same hardware; \
         {differ} of {} cells differed",
        a.len()
    );

    // Print the first 8 cells' bit patterns. Capture this in CI logs
    // for both Linux and Windows runs; when we have both, hard-code
    // the expected pattern below.
    eprintln!("Phase 44b cross-platform bit-pattern fingerprint (M={}, SEED=0x{:x}, S=8):", M, SEED);
    for (i, &v) in a.iter().take(8).enumerate() {
        eprintln!("  cell[{i}] = 0x{:016x} ({v:e})", v.to_bits());
    }
}
