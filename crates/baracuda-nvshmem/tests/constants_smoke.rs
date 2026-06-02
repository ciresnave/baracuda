//! Host-independent smoke tests — these do **not** load NVSHMEM, so they run
//! on every host (Linux / Windows / no-GPU CI). They pin the predefined team
//! handle values and the safe/raw `Team` mapping so a future refactor can't
//! silently change the ABI constants.

use baracuda_nvshmem::Team;
use baracuda_nvshmem_sys::nvshmem_team_t;

#[test]
fn predefined_team_handles_match_abi() {
    assert_eq!(nvshmem_team_t::WORLD.0, 0, "NVSHMEM_TEAM_WORLD");
    assert_eq!(nvshmem_team_t::SHARED.0, 1, "NVSHMEM_TEAM_SHARED");
    assert_eq!(nvshmem_team_t::INVALID.0, -1, "NVSHMEM_TEAM_INVALID");
}

#[test]
fn safe_team_wraps_raw_handles() {
    assert_eq!(Team::WORLD.as_raw(), nvshmem_team_t::WORLD);
    assert_eq!(Team::SHARED.as_raw(), nvshmem_team_t::SHARED);
    assert_ne!(Team::WORLD, Team::SHARED);
}
