//! NCCL smoke test. NCCL is Linux-only in practice; on other hosts or when
//! NCCL isn't installed the test skips gracefully instead of failing.
#![cfg(feature = "device-tests")]

use baracuda_nccl::{Communicator, version};

#[test]
fn nccl_loads_and_reports_version() {
    match version() {
        Ok(v) => {
            eprintln!("NCCL version (packed): {v}");
            assert!(v >= 22000, "unexpectedly old NCCL version");
        }
        Err(e) => {
            eprintln!("NCCL not available: {e}. Skipping.");
        }
    }
}

#[test]
fn single_device_comm_roundtrip() {
    // Single-device NCCL comm — degenerate but exercises the full loader path.
    match Communicator::init_all(&[0]) {
        Ok(comms) => {
            assert_eq!(comms.len(), 1);
            let rank = comms[0].rank();
            let n = comms[0].world_size();
            eprintln!("comm: rank={rank}, world_size={n}");
            assert_eq!(rank, 0);
            assert_eq!(n, 1);
        }
        Err(e) => {
            eprintln!("NCCL init_all failed: {e}. Skipping.");
        }
    }
}
