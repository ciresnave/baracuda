//! NcclUniqueId smoke tests — verifies `generate()` + byte-cast roundtrip.
//!
//! These tests require a working NCCL install (typically Linux multi-GPU
//! hosts; Windows NCCL is experimental and rarely shipped). On hosts
//! without NCCL the tests skip gracefully via `#[ignore]` — they would
//! otherwise fail at `NcclUniqueId::generate()` with `LoaderError::
//! LibraryNotFound`.
//!
//! Multi-rank validation of the id is out of scope for Phase 52 (needs
//! 2+ GPUs or a process-spawning harness).

use baracuda_nccl::NcclUniqueId;

#[test]
#[ignore = "requires NCCL installed (typically Linux multi-GPU hosts)"]
fn nccl_unique_id_generate_roundtrip() {
    let id = match NcclUniqueId::generate() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("NCCL not available: {e}. Skipping.");
            return;
        }
    };

    // Round-trip through the 128-byte wire form.
    let bytes = id.as_bytes();
    assert_eq!(bytes.len(), 128, "ncclUniqueId is a fixed 128-byte blob");

    let restored = NcclUniqueId::from_bytes(bytes);
    let bytes2 = restored.as_bytes();
    assert_eq!(
        bytes, bytes2,
        "NcclUniqueId::from_bytes(as_bytes()) must round-trip byte-for-byte"
    );
}

#[test]
#[ignore = "requires NCCL installed"]
fn nccl_unique_id_generate_is_nonzero() {
    let id = match NcclUniqueId::generate() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("NCCL not available: {e}. Skipping.");
            return;
        }
    };
    let bytes = id.as_bytes();
    // NCCL embeds a TCP listen address + nonce in the id — astronomically
    // unlikely to be all-zero (an all-zero id would indicate a stub).
    assert!(
        bytes.iter().any(|&b| b != 0),
        "freshly-generated NcclUniqueId must not be all zeros"
    );
}

#[test]
#[ignore = "requires NCCL installed"]
fn nccl_unique_id_generate_is_unique_per_call() {
    let a = match NcclUniqueId::generate() {
        Ok(id) => id,
        Err(e) => {
            eprintln!("NCCL not available: {e}. Skipping.");
            return;
        }
    };
    let b = NcclUniqueId::generate().expect("second generate should succeed");
    assert_ne!(
        a.as_bytes(),
        b.as_bytes(),
        "two successive ncclGetUniqueId calls should produce distinct ids"
    );
}
