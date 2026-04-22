//! GPU-gated integration tests for Wave-28 Driver-API additions:
//! array introspection, context-level events, P2P query, exec-affinity
//! probe, core-dump settings, library-extras (enumerate kernels + module).

use baracuda_cuda_sys::types::{CUcoredumpSettings, CUdevice_P2PAttribute, CUexecAffinityType};
use baracuda_driver::array::{Array, ArrayFormat};
use baracuda_driver::library::Library;
use baracuda_driver::{coredump, Context, Device, Event};

const VECTOR_ADD_PTX: &str = include_str!("kernels/vector_add.ptx");

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn array_descriptor_and_memory_requirements() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let arr = Array::new(&ctx, 64, 48, ArrayFormat::F32, 1).unwrap();
    let desc = arr.descriptor().unwrap();
    assert_eq!(desc.width, 64);
    assert_eq!(desc.height, 48);
    assert_eq!(desc.num_channels, 1);

    // cuArrayGetMemoryRequirements is only defined for arrays created
    // with the DEFERRED_MAPPING flag — for plain arrays it returns
    // INVALID_VALUE. Accept either outcome.
    match arr.memory_requirements(&device) {
        Ok(req) => {
            eprintln!("array memreq: size={}, align={}", req.size, req.alignment);
            assert!(req.size >= 64 * 48 * 4);
            assert!(req.alignment > 0);
        }
        Err(e) => eprintln!("cuArrayGetMemoryRequirements rejected (plain array): {e:?}"),
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn ctx_record_and_wait_event() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();

    let event = Event::new(&ctx).unwrap();
    match ctx.record_event(&event) {
        Ok(()) => eprintln!("cuCtxRecordEvent OK"),
        Err(e) => {
            eprintln!("cuCtxRecordEvent not supported on this driver: {e:?}");
            return;
        }
    }
    // Wait + synchronize to complete the sequence.
    ctx.wait_event(&event).unwrap();
    event.synchronize().unwrap();
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn p2p_attribute_self_query() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    // Self-to-self P2P query isn't really meaningful (you already have
    // access to your own memory), but it exercises the API path.
    match device.p2p_attribute(&device, CUdevice_P2PAttribute::ACCESS_SUPPORTED) {
        Ok(v) => eprintln!("P2P ACCESS_SUPPORTED (self, self) = {v}"),
        Err(e) => eprintln!("cuDeviceGetP2PAttribute unsupported: {e:?}"),
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn exec_affinity_support_probe() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let supports = device
        .exec_affinity_support(CUexecAffinityType::SM_COUNT)
        .unwrap();
    eprintln!("SM_COUNT affinity supported: {supports}");
    // Result is just informational — assert we didn't error.
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn coredump_enable_on_exception_round_trip() {
    baracuda_driver::init().unwrap();
    // Per-context: need a current context first.
    let device = Device::get(0).unwrap();
    let _ctx = Context::new(&device).unwrap();

    match coredump::enable_on_exception() {
        Ok(before) => {
            eprintln!("coredump enable_on_exception before: {before}");
            if let Err(e) = coredump::set_enable_on_exception(!before) {
                eprintln!("set_enable_on_exception failed: {e:?}");
                return;
            }
            let after = coredump::enable_on_exception().unwrap();
            assert_eq!(after, !before);
            coredump::set_enable_on_exception(before).unwrap(); // restore
        }
        Err(e) => eprintln!("coredump attrs unsupported: {e:?}"),
    }
    // Silence unused-import on the attribute-constant path.
    let _ = CUcoredumpSettings::ENABLE_ON_EXCEPTION;
}

#[test]
#[ignore = "requires an NVIDIA GPU + CUDA 12.4+"]
fn library_enumerate_kernels_and_module() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let _ctx = Context::new(&device).unwrap();

    let lib = Library::load_ptx(VECTOR_ADD_PTX).unwrap();
    match lib.kernel_count() {
        Ok(n) => {
            assert!(n >= 1);
            let kernels = lib.enumerate_kernels().unwrap();
            assert_eq!(kernels.len() as u32, n);
            let names: Vec<_> = kernels
                .iter()
                .map(|k| k.name().unwrap_or_default())
                .collect();
            eprintln!("library exposes {n} kernel(s): {names:?}");
            assert!(names.iter().any(|s| s.contains("vector_add")));
        }
        Err(e) => eprintln!("cuLibraryGetKernelCount unsupported: {e:?}"),
    }

    // module_raw may return a null module on library-without-module images.
    match lib.module_raw() {
        Ok(m) => eprintln!("library module: {:?}", m),
        Err(e) => eprintln!("cuLibraryGetModule unsupported: {e:?}"),
    }
}
