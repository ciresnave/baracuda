//! NVSHMEM smoke tests. NVSHMEM is Linux-only in practice and requires the
//! NVSHMEM runtime installed *and* a multi-process launch (`nvshmrun` /
//! `mpirun`). On hosts without it, or when run as a plain `cargo test` single
//! process, these skip gracefully instead of failing — hence `#[ignore]`.

use baracuda_nvshmem::{version, Context, Team};

#[test]
#[ignore = "requires NVSHMEM runtime (Linux, sm_70+) + a PE launcher"]
fn nvshmem_loads_and_reports_version() {
    match version() {
        Ok((major, minor)) => {
            eprintln!("NVSHMEM version: {major}.{minor}");
            assert!(major >= 2, "unexpectedly old NVSHMEM");
        }
        Err(e) => eprintln!("NVSHMEM not available: {e}. Skipping."),
    }
}

#[test]
#[ignore = "requires NVSHMEM runtime + a PE launcher"]
fn context_init_and_pe_discovery() {
    match Context::init() {
        Ok(ctx) => {
            let me = ctx.my_pe();
            let world = ctx.n_pes();
            eprintln!("NVSHMEM: my_pe={me}, n_pes={world}");
            assert!(world >= 1);
            assert!((0..world).contains(&me));
        }
        Err(e) => eprintln!("NVSHMEM init failed: {e}. Skipping."),
    }
}

#[test]
#[ignore = "requires NVSHMEM runtime + a PE launcher"]
fn symmetric_heap_alloc_and_self_put() {
    let Ok(ctx) = Context::init() else {
        eprintln!("NVSHMEM init failed. Skipping.");
        return;
    };
    let src = ctx.malloc::<f32>(256).expect("symmetric malloc src");
    let dst = ctx.malloc::<f32>(256).expect("symmetric malloc dst");
    // Self-put (pe == my_pe) is a degenerate but valid exercise of the path.
    ctx.put(&dst, &src, 256, ctx.my_pe()).expect("put");
    ctx.barrier_all().expect("barrier");
    eprintln!("symmetric put + barrier round-trip ok");
}

#[test]
#[ignore = "requires NVSHMEM runtime + a PE launcher"]
fn world_team_matches_context() {
    let Ok(ctx) = Context::init() else {
        eprintln!("NVSHMEM init failed. Skipping.");
        return;
    };
    let world = ctx.world();
    assert_eq!(world, Team::WORLD);
    assert_eq!(world.n_pes().expect("team n_pes"), ctx.n_pes());
    assert_eq!(world.my_pe().expect("team my_pe"), ctx.my_pe());
}
