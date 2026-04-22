//! GPU-gated integration tests for Wave-2 Driver-API additions:
//! cuFuncGetAttribute, cuModuleGetGlobal, cuDevicePrimaryCtxGetState.

use baracuda_cuda_sys::types::CUfunction_attribute;
use baracuda_driver::{Context, Device, DeviceBuffer, Module};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

/// CUDA C source for the `get_global` test — compiled at test time via NVRTC
/// to avoid hand-written-PTX fragility across CUDA versions.
const WITH_GLOBAL_SRC: &str = r#"
extern "C" __device__ unsigned int g_counter;

extern "C" __global__ void read_counter(unsigned int* out) {
    *out = g_counter;
}
"#;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn function_attributes() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    let max_threads = kernel.max_threads_per_block().unwrap();
    let num_regs = kernel.num_regs().unwrap();
    let shared = kernel.shared_size_bytes().unwrap();
    let local = kernel.local_size_bytes().unwrap();
    let ptx_v = kernel.ptx_version().unwrap();
    let bin_v = kernel.binary_version().unwrap();
    eprintln!(
        "vector_add: max_threads={max_threads}, regs={num_regs}, smem={shared}B, lmem={local}B, \
         ptx={ptx_v}, bin={bin_v}"
    );
    assert!((1..=1024).contains(&max_threads));
    assert!(num_regs > 0);
    assert!(bin_v >= 50, "unexpectedly low compute capability: {bin_v}");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn set_max_dynamic_shared_mem() {
    // Some GPUs allow raising the per-kernel dynamic shared-memory limit
    // above the per-block default. If the device rejects the request, the
    // call should surface an error instead of silently succeeding — we
    // just verify the safe wrapper doesn't panic.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let module = Module::load_ptx(&ctx, VECTOR_ADD_PTX).unwrap();
    let kernel = module.get_function("vector_add").unwrap();

    // Try to set a modest 16 KiB carveout — should succeed on basically
    // every post-Kepler device.
    let result = kernel.set_attribute(
        CUfunction_attribute::MAX_DYNAMIC_SHARED_SIZE_BYTES,
        16 * 1024,
    );
    match result {
        Ok(()) => eprintln!("set_attribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) = 16 KiB OK"),
        Err(e) => eprintln!("device refused the request, treating as non-fatal: {e:?}"),
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn module_get_global_reads_device_variable() {
    use baracuda_cuda_sys::driver;
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let (cc_major, cc_minor) = device.compute_capability().unwrap();
    let arch = format!("--gpu-architecture=compute_{cc_major}{cc_minor}");
    let ptx = baracuda_nvrtc::Program::compile(WITH_GLOBAL_SRC, "with_global.cu", &[&arch])
        .expect("NVRTC compile");
    let ctx = Context::new(&device).unwrap();
    let module = Module::load_ptx(&ctx, &ptx).unwrap();

    // Resolve `__device__ unsigned int g_counter`.
    let (dptr, bytes) = module.get_global("g_counter").unwrap();
    eprintln!("g_counter @ {:#x}, {} bytes", dptr.0, bytes);
    assert_eq!(bytes, 4, "g_counter should be 4 bytes (u32)");
    assert_ne!(dptr.0, 0);

    // Write a known value from host to the device global.
    let sentinel: u32 = 0xCAFE_BABE;
    let d = driver().unwrap();
    let cu_htod = d.cu_memcpy_htod().unwrap();
    // SAFETY: dptr is live; sentinel is valid for reads of 4 bytes.
    let rc = unsafe { cu_htod(dptr, &sentinel as *const u32 as *const core::ffi::c_void, 4) };
    assert!(rc.is_success(), "cuMemcpyHtoD failed: {rc:?}");

    // Launch the kernel to copy g_counter into a device output buffer.
    let out: DeviceBuffer<u32> = DeviceBuffer::new(&ctx, 1).unwrap();
    let kernel = module.get_function("read_counter").unwrap();
    let ptr = out.as_raw();
    // SAFETY: kernel signature is `(unsigned int* out)`.
    unsafe {
        kernel
            .launch()
            .grid((1u32, 1, 1))
            .block((1u32, 1, 1))
            .arg(&ptr)
            .launch()
            .unwrap();
    }
    ctx.synchronize().unwrap();

    let mut val = [0u32; 1];
    out.copy_to_host(&mut val).unwrap();
    assert_eq!(
        val[0], sentinel,
        "expected {sentinel:#x}, got {:#x}",
        val[0]
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn primary_ctx_state() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();

    // Initially no retained primary ctx.
    let (flags0, active0) = device.primary_ctx_state().unwrap();
    eprintln!("primary ctx state (before retain): flags={flags0}, active={active0}");

    // Retain a primary-ctx reference through a Context::primary if we had one,
    // or via cuDevicePrimaryCtxRetain directly. Our test just exercises
    // the query end; actual retain happens implicitly when other tests
    // create explicit contexts.
    //
    // Create an explicit context and re-query — the primary context state
    // shouldn't change because explicit contexts are separate.
    let _ctx = Context::new(&device).unwrap();
    let (flags1, _active1) = device.primary_ctx_state().unwrap();
    assert_eq!(
        flags0, flags1,
        "primary-ctx flags changed after creating an explicit context"
    );
}
