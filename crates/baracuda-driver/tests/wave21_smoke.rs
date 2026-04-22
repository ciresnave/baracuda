//! GPU-gated integration test for Wave-21 Driver-API additions:
//! kernel-attribute extension (cuKernel*) on library-loaded kernels.

use baracuda_cuda_sys::types::CUfunction_attribute;
use baracuda_driver::library::Library;
use baracuda_driver::{Context, Device};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12+"]
fn library_kernel_attrs_and_name() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    // Library API needs the context current for many operations.
    drop(ctx);

    let ctx = Context::new(&device).unwrap();
    let _ = &ctx;

    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    let kernel = lib.get_kernel("vector_add").unwrap();

    let max_threads = kernel
        .attribute(CUfunction_attribute::MAX_THREADS_PER_BLOCK, &device)
        .unwrap();
    let num_regs = kernel
        .attribute(CUfunction_attribute::NUM_REGS, &device)
        .unwrap();
    eprintln!("library kernel: max_threads={max_threads}, regs={num_regs}");
    assert!((1..=1024).contains(&max_threads));
    assert!(num_regs > 0);

    match kernel.name() {
        Ok(name) => {
            eprintln!("kernel.name() = {name:?}");
            assert!(name.contains("vector_add"));
        }
        Err(e) => eprintln!("cuKernelGetName failed (some drivers): {e:?}"),
    }

    // Query first param offset/size — vector_add takes (float*, float*, float*, uint)
    // so param 0 is an 8-byte pointer at offset 0.
    match kernel.param_info(0) {
        Ok((off, sz)) => {
            eprintln!("param 0: offset={off}, size={sz}");
            assert_eq!(off, 0);
            assert_eq!(sz, 8);
        }
        Err(e) => eprintln!("cuKernelGetParamInfo failed (some drivers): {e:?}"),
    }
}
