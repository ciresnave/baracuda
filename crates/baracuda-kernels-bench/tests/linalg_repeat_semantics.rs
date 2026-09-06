//! Does an UNRESTORED Cholesky loop actually enter a failure path?
//!
//! ⚠️ I have been asserting this in a commit message, a memory file and three
//! peer messages: "after call 1 the buffer holds `L`, which is not SPD, so
//! cuSOLVER halts at the failing minor and `info[i] != 0`." That is an
//! INFERENCE about cuSOLVER's behaviour, not something I measured, and it is
//! the whole justification for `measure_median_ns_restored`.
//!
//! A report measured in nine claims and inferred in the tenth does not vouch
//! for the tenth. So: measure it.
//!
//! Three outcomes, and I am publishing whichever occurs:
//!   info stays 0 and timings are flat  -> MY REASON IS WRONG. The restore is
//!                                         still correct hygiene (iterations
//!                                         2..N factor a different matrix), but
//!                                         the "failure path" claim must be
//!                                         RETRACTED everywhere I stated it.
//!   info != 0 from call 2 onward       -> claim confirmed as stated.
//!   info != 0 AND call 2 is much faster-> confirmed WITH the consequence that
//!                                         makes it dangerous: an early return
//!                                         pulls the median down.
//!
//! Drop into crates/baracuda-kernels-bench/tests/ and run:
//!   cargo test -p baracuda-kernels-bench --release --test repeat_destroys \
//!     -- --ignored --nocapture

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    CholeskyArgs, CholeskyDescriptor, CholeskyPlan, ElementKind, PlanPreference, TensorMut,
    Workspace, contiguous_stride,
};
use baracuda_kernels_bench::setup_device;

const N: i32 = 256;
const CALLS: usize = 4;

#[test]
#[ignore = "requires a CUDA device; run explicitly with --ignored"]
fn unrestored_cholesky_repeats_do_not_enter_a_failure_path() {
    let (ctx, stream) = setup_device();
    let nu = N as usize;

    // Same fixture as benches/linalg.rs: symmetric, diagonally dominant,
    // therefore SPD by Gershgorin.
    let mut host = vec![0.5_f32; nu * nu];
    for i in 0..nu {
        host[i * nu + i] = N as f32;
    }

    let mut work = DeviceBuffer::from_slice(&ctx, &host).expect("alloc work");
    let mut info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = CholeskyDescriptor {
        matrix_size: N,
        batch_size: 1,
        lower: true,
        element: ElementKind::F32,
    };
    let plan =
        CholeskyPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let ws_bytes = plan.query_workspace_size(&stream).unwrap_or(0).max(1);
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let sh = [1, N, N];
    let st = contiguous_stride(sh);

    let mut infos = Vec::new();
    let mut times = Vec::new();
    for call in 0..CALLS {
        // Reset `info` so each reading is about THIS call and not a leftover
        // from the previous one — otherwise a nonzero on call 3 is
        // indistinguishable from a nonzero on call 2 that was never cleared,
        // which is the same read-a-stale-value defect this test exists to find.
        info = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

        let t0 = std::time::Instant::now();
        {
            let args = CholeskyArgs::<f32> {
                a: TensorMut {
                    data: work.as_slice_mut(),
                    shape: sh,
                    stride: st,
                },
                info: TensorMut {
                    data: info.as_slice_mut(),
                    shape: [1],
                    stride: [1],
                },
            };
            plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                .expect("cholesky run");
        }
        stream.synchronize().expect("sync");
        let dt = t0.elapsed();

        let mut got = vec![0_i32; 1];
        info.copy_to_host(&mut got).expect("read info");
        println!("call {call}: info = {}, wall = {:?}", got[0], dt);
        infos.push(got[0]);
        times.push(dt);
    }

    // ---- ARM 2: the SAME loop, but restoring the input before each call. ----
    // One variable changed, same instrument, so the arms are comparable. This
    // is what decides whether the restore changes the published NUMBER or only
    // the MEANING of it.
    let pristine = DeviceBuffer::from_slice(&ctx, &host).expect("alloc pristine");
    work = DeviceBuffer::from_slice(&ctx, &host).expect("realloc work");
    let mut r_times = Vec::new();
    let mut r_infos = Vec::new();
    for call in 0..CALLS {
        pristine
            .copy_to_device_async(&work, &stream)
            .expect("restore");
        stream.synchronize().expect("sync restore");
        info = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");
        let t0 = std::time::Instant::now();
        {
            let args = CholeskyArgs::<f32> {
                a: TensorMut {
                    data: work.as_slice_mut(),
                    shape: sh,
                    stride: st,
                },
                info: TensorMut {
                    data: info.as_slice_mut(),
                    shape: [1],
                    stride: [1],
                },
            };
            plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                .expect("cholesky run (restored)");
        }
        stream.synchronize().expect("sync");
        let dt = t0.elapsed();
        let mut got = vec![0_i32; 1];
        info.copy_to_host(&mut got).expect("read info");
        println!("RESTORED call {call}: info = {}, wall = {:?}", got[0], dt);
        r_infos.push(got[0]);
        r_times.push(dt);
    }

    println!("\n--- verdict ---");
    println!("info sequence  unrestored: {infos:?}");
    println!("info sequence  restored:   {r_infos:?}");
    println!("steady state (calls 1..):");
    println!("  unrestored: {:?}", &times[1..]);
    println!("  restored:   {:?}", &r_times[1..]);
    // Call 1 MUST succeed - if it does not, the fixture is wrong and nothing
    // else this test says means anything. That is the positive control.
    assert_eq!(
        infos[0], 0,
        "POSITIVE CONTROL FAILED: the first factorization of an SPD matrix \
         returned info={} - the fixture is not SPD and no conclusion about \
         later calls is available",
        infos[0]
    );
    if infos[1..].iter().all(|&i| i == 0) {
        println!(
            "MY CLAIM IS WRONG: repeats returned info=0. The restore is still \
             correct (2..N factor a different matrix) but the 'failure path' \
             reason must be retracted."
        );
    } else {
        println!(
            "CLAIM CONFIRMED: repeats returned nonzero info at the failing \
             minor, and a timing loop never reads it."
        );
    }
}
