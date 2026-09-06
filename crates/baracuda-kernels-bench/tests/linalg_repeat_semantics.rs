//! Does an UNRESTORED Cholesky loop actually enter a failure path?
//!
//! ⚠️ I asserted this in a commit message, a bench module doc and three peer
//! messages: *"after call 1 the buffer holds `L`, which is not SPD, so cuSOLVER
//! halts at the failing minor and `info[i] != 0`."* That was an INFERENCE about
//! cuSOLVER's behaviour, never measured, and it was the whole stated
//! justification for `measure_median_ns_restored`.
//!
//! A report measured in nine claims and inferred in the tenth does not vouch
//! for the tenth. So this measures it, with two arms and one variable.
//!
//! **Result: the claim is FALSE.** `info` is zero in both arms and the steady
//! states interleave — the restore does not move the number at N=256. The
//! premise was wrong because `cusolverDnSpotrf(lower)` reads only the LOWER
//! TRIANGLE, and `L`'s lower triangle read as a symmetric matrix is still
//! diagonally dominant with a positive diagonal, hence still SPD. *"The buffer
//! holds `L`"* and *"the buffer holds a matrix that fails to factor"* are
//! different claims and only the first is true.
//!
//! ⚠️ The measurement is FIXTURE-DEPENDENT. It licenses deleting a reason I
//! invented; it does NOT license the converse *"repeats are always safe."* The
//! restore stays, on the reason that is true and dull: iterations 2..N factor a
//! DIFFERENT MATRIX, so the loop is not timing a fixed problem.
//!
//! Run:
//!   cargo test -p baracuda-kernels-bench --release \
//!     --test linalg_repeat_semantics -- --ignored --nocapture

use baracuda_driver::{Context, DeviceBuffer, Stream};
use baracuda_kernels::{
    CholeskyArgs, CholeskyDescriptor, CholeskyPlan, ElementKind, PlanPreference, TensorMut,
    Workspace, contiguous_stride,
};
use baracuda_kernels_bench::setup_device;
use std::time::Duration;

const N: i32 = 256;
const CALLS: usize = 4;

/// Everything one factorization needs, so each arm below is a short loop rather
/// than a second copy of the setup.
struct Fixture {
    ctx: Context,
    stream: Stream,
    plan: CholeskyPlan<f32>,
    work: DeviceBuffer<f32>,
    pristine: DeviceBuffer<f32>,
    info: DeviceBuffer<i32>,
    ws: DeviceBuffer<u8>,
}

impl Fixture {
    /// Symmetric and diagonally dominant, therefore SPD by Gershgorin — the
    /// same fixture `benches/linalg.rs` uses.
    fn new() -> Self {
        let (ctx, stream) = setup_device();
        let nu = N as usize;
        let mut host = vec![0.5_f32; nu * nu];
        for i in 0..nu {
            host[i * nu + i] = N as f32;
        }
        let desc = CholeskyDescriptor {
            matrix_size: N,
            batch_size: 1,
            lower: true,
            element: ElementKind::F32,
        };
        let plan =
            CholeskyPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
        let ws_bytes = plan.query_workspace_size(&stream).unwrap_or(0).max(1);
        Self {
            work: DeviceBuffer::from_slice(&ctx, &host).expect("alloc work"),
            pristine: DeviceBuffer::from_slice(&ctx, &host).expect("alloc pristine"),
            info: DeviceBuffer::zeros(&ctx, 1).expect("alloc info"),
            ws: DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws"),
            plan,
            ctx,
            stream,
        }
    }

    /// One factorization. Returns `(info, wall)`.
    ///
    /// `info` is reset first so each reading is about THIS call and not a
    /// leftover — otherwise a nonzero on call 3 is indistinguishable from an
    /// uncleared nonzero on call 2, which is the read-a-stale-value defect this
    /// test exists to find.
    fn factor_once(&mut self, restore: bool) -> (i32, Duration) {
        if restore {
            self.pristine
                .copy_to_device_async(&self.work, &self.stream)
                .expect("restore");
            self.stream.synchronize().expect("sync restore");
        }
        self.info = DeviceBuffer::zeros(&self.ctx, 1).expect("alloc info");
        let sh = [1, N, N];
        let st = contiguous_stride(sh);
        let t0 = std::time::Instant::now();
        {
            let args = CholeskyArgs::<f32> {
                a: TensorMut {
                    data: self.work.as_slice_mut(),
                    shape: sh,
                    stride: st,
                },
                info: TensorMut {
                    data: self.info.as_slice_mut(),
                    shape: [1],
                    stride: [1],
                },
            };
            self.plan
                .run(
                    &self.stream,
                    Workspace::Borrowed(self.ws.as_slice_mut()),
                    args,
                )
                .expect("cholesky run");
        }
        self.stream.synchronize().expect("sync");
        let dt = t0.elapsed();
        let mut got = vec![0_i32; 1];
        self.info.copy_to_host(&mut got).expect("read info");
        (got[0], dt)
    }

    /// `CALLS` factorizations, restoring or not. Returns `(infos, walls)`.
    fn arm(&mut self, restore: bool, label: &str) -> (Vec<i32>, Vec<Duration>) {
        let mut infos = Vec::new();
        let mut times = Vec::new();
        for call in 0..CALLS {
            let (info, dt) = self.factor_once(restore);
            println!("{label} call {call}: info = {info}, wall = {dt:?}");
            infos.push(info);
            times.push(dt);
        }
        (infos, times)
    }
}

#[test]
#[ignore = "requires a CUDA device; run explicitly with --ignored"]
fn unrestored_cholesky_repeats_do_not_enter_a_failure_path() {
    let mut f = Fixture::new();
    let (infos, times) = f.arm(false, "UNRESTORED");

    // Rebuild so arm 2 starts from the same state arm 1 did.
    let mut f = Fixture::new();
    let (r_infos, r_times) = f.arm(true, "RESTORED  ");

    println!("\n--- verdict ---");
    println!("info sequence  unrestored: {infos:?}");
    println!("info sequence  restored:   {r_infos:?}");
    println!("steady state (calls 1..):");
    println!("  unrestored: {:?}", &times[1..]);
    println!("  restored:   {:?}", &r_times[1..]);

    // Call 1 MUST succeed. If it does not, the fixture is not SPD and nothing
    // else this test says means anything. That is the positive control.
    assert_eq!(
        infos[0], 0,
        "POSITIVE CONTROL FAILED: the first factorization of an SPD matrix \
         returned info={} — the fixture is not SPD and no conclusion about \
         later calls is available",
        infos[0]
    );
    if infos[1..].iter().all(|&i| i == 0) {
        println!(
            "CLAIM REFUTED: repeats returned info=0. The restore is still correct \
             (2..N factor a different matrix) but the 'failure path' reason is retracted."
        );
    } else {
        println!(
            "Repeats returned nonzero info at the failing minor, and a timing loop \
             never reads it."
        );
    }
}
