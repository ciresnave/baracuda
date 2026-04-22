//! GPU-gated integration test for Wave-23 Driver-API additions:
//! profiler start/stop, func/kernel name + param info, device UUID,
//! context ID, module loading mode, graph dot-print.

use baracuda_driver::{profiler, Context, Device, Graph, Module};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn misc_metadata_and_profiler_controls() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();

    let uuid = device.uuid().unwrap();
    eprintln!("device UUID: {:02x?}", uuid);
    assert!(uuid.iter().any(|&b| b != 0), "UUID should be non-zero");

    let _luid = device.luid(); // may fail on Linux; just exercise the call

    let ctx = Context::new(&device).unwrap();
    let id = ctx.id().unwrap();
    eprintln!("context id = {id}");
    assert_ne!(id, 0);

    // Module loading mode reports eager (0x1) or lazy (0x2).
    let mode = Module::loading_mode().unwrap();
    eprintln!("module loading mode = {mode:#x}");
    assert!(mode == 1 || mode == 2);

    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();
    assert_eq!(kernel.name().unwrap(), "vector_add");
    let (off, sz) = kernel.param_info(0).unwrap();
    assert_eq!(off, 0);
    assert_eq!(sz, 8);

    // Profiler start/stop (external profiler will observe these if attached).
    {
        let _section = profiler::section().unwrap();
        // ... pretend work ...
    }

    // Graph dot-print to a temp file.
    let graph = Graph::new(&ctx).unwrap();
    let _ = graph.add_empty_node(&[]).unwrap();
    let tmp_path = std::env::temp_dir().join("baracuda_wave23_graph.dot");
    let tmp_path_str = tmp_path.to_string_lossy().into_owned();
    graph.debug_dot_print(&tmp_path_str, 0).unwrap();
    let contents = std::fs::read_to_string(&tmp_path).unwrap();
    assert!(
        contents.contains("digraph"),
        "dot output should declare a digraph, got: {contents:?}"
    );
    let _ = std::fs::remove_file(&tmp_path);
}
