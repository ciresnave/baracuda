//! GPU-gated integration test for Wave-22 Driver-API additions:
//! user objects attached to graphs.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use baracuda_driver::user_object::UserObject;
use baracuda_driver::{Context, Device, Graph};

static DESTRUCTOR_CALLS: AtomicU32 = AtomicU32::new(0);

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12+"]
fn user_object_destructor_fires_when_graph_drops() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    DESTRUCTOR_CALLS.store(0, Ordering::SeqCst);

    // Capture a closure; the closure dropping implies destructor ran.
    let counter = Arc::new(());
    let _counter_clone = counter.clone();
    let uo = match UserObject::new(
        move || {
            DESTRUCTOR_CALLS.fetch_add(1, Ordering::SeqCst);
            drop(_counter_clone);
        },
        1,
    ) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("cuUserObjectCreate rejected: {e:?}");
            return;
        }
    };

    let graph = Graph::new(&ctx).unwrap();
    // Transfer our single reference to the graph — safer than retain+release.
    graph.retain_user_object(&uo, 1, 0).unwrap();
    // Drop our own reference so the graph is the sole owner.
    uo.release(1).unwrap();

    assert_eq!(
        DESTRUCTOR_CALLS.load(Ordering::SeqCst),
        0,
        "destructor should not fire while graph holds ref"
    );

    // Drop graph -> releases user object -> destructor runs.
    drop(graph);

    // The destructor runs synchronously on graph destroy.
    assert_eq!(
        DESTRUCTOR_CALLS.load(Ordering::SeqCst),
        1,
        "destructor should have fired on graph drop"
    );
    assert_eq!(Arc::strong_count(&counter), 1, "Arc clone was dropped");

    drop(ctx);
}
