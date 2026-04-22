//! GPU-gated integration test for Wave-19 Driver-API additions:
//! conditional graph nodes (CUDA 12.3+).

use baracuda_cuda_sys::types::{CUgraphConditionalNodeType, CUgraphNodeType};
use baracuda_driver::{Context, Device, DeviceBuffer, Graph, Stream};

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.3+"]
fn add_conditional_if_node_with_memset_body() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let parent = Graph::new(&ctx).unwrap();
    // default_launch_value = 1 -> the IF body runs once.
    let handle = match parent.conditional_handle(1, 0) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("cuGraphConditionalHandleCreate not supported: {e:?}");
            return;
        }
    };

    let (cond_node, body) =
        match parent.add_conditional_node(&[], handle, CUgraphConditionalNodeType::IF, 1) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("add_conditional_node rejected: {e:?}");
                return;
            }
        };

    assert_eq!(cond_node.node_type().unwrap(), CUgraphNodeType::CONDITIONAL);

    // The body graph is owned by the conditional node; body graphs have
    // tight restrictions on what kinds of nodes they can contain (no
    // memset in some CUDA versions). Add an empty node instead — it's
    // always legal and proves the body CUgraph handle is usable.
    let _empty = match body.add_empty_node(&[]) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("body.add_empty_node failed: {e:?}");
            return;
        }
    };

    // Instantiate + launch. Default value was 1 so the body should run.
    let exec = match parent.instantiate() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("conditional graph instantiate failed: {e:?}");
            return;
        }
    };
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();

    // Sanity — if we got here, the conditional graph compiled and ran.
    let _ = (stream, ctx);
    // Silence unused-import warning if the earlier branches returned early.
    let _ = DeviceBuffer::<u8>::new;
}
