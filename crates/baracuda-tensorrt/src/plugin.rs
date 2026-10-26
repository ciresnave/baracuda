//! TensorRT's plugin registry: list the plugin creators registered in the process,
//! look one up by name / version / namespace, register TensorRT's built-in plugins,
//! and load third-party plugin libraries.
//!
//! The registry is a process-wide singleton owned by `libnvinfer`; [`PluginRegistry`]
//! is a cheap handle to it. Writing a plugin in Rust (an `IPluginV3` implementation)
//! is not covered here.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use baracuda_tensorrt_sys::{self as sys, registry as ffi};

use crate::{Error, Logger, Result};

/// Which creator interface a registered plugin uses.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PluginCreatorKind {
    /// `IPluginCreator`, the older V2 plugin interface.
    V2,
    /// `IPluginCreatorV3One`, the V3 interface.
    V3One,
    /// An interface this crate cannot read the name of.
    Unknown,
}

/// The identity of one registered plugin creator.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PluginCreatorInfo {
    /// Plugin name, such as `"ROIAlign_TRT"`.
    pub name: String,
    /// Plugin version string.
    pub version: String,
    /// Namespace the creator is registered in (empty for the default).
    pub namespace: String,
    /// The creator interface.
    pub kind: PluginCreatorKind,
}

/// An opaque handle for a plugin library loaded through [`PluginRegistry::load_library`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PluginLibraryHandle(*mut core::ffi::c_void);

// SAFETY: the handle is an opaque token that TensorRT only reads inside
// `deregisterLibrary`; it carries no thread affinity.
unsafe impl Send for PluginLibraryHandle {}
unsafe impl Sync for PluginLibraryHandle {}

fn text(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: TensorRT returns NUL-terminated strings owned by the creator, which the
        // registry keeps alive; the string is copied before this function returns.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

fn info(creator: *mut core::ffi::c_void) -> PluginCreatorInfo {
    // SAFETY: `creator` is a live creator pointer taken from the registry just before this
    // call; the registry owns it until the process ends or it is deregistered.
    unsafe {
        PluginCreatorInfo {
            name: text(ffi::trtPluginCreatorName(creator)),
            version: text(ffi::trtPluginCreatorVersion(creator)),
            namespace: text(ffi::trtPluginCreatorNamespace(creator)),
            kind: match ffi::trtPluginCreatorKind(creator) {
                1 => PluginCreatorKind::V2,
                2 => PluginCreatorKind::V3One,
                _ => PluginCreatorKind::Unknown,
            },
        }
    }
}

/// A handle to TensorRT's process-wide plugin registry.
#[derive(Copy, Clone, Debug)]
pub struct PluginRegistry {
    raw: sys::trtIPluginRegistry_t,
}

// SAFETY: the registry is a process-wide singleton that TensorRT documents as thread-safe;
// the handle is a pointer that is never freed.
unsafe impl Send for PluginRegistry {}
unsafe impl Sync for PluginRegistry {}

impl PluginRegistry {
    /// The process-wide registry (`getPluginRegistry()`). Needs the shim.
    pub fn global() -> Result<Self> {
        if !sys::SHIM_BUILT {
            return Err(Error::ShimNotBuilt);
        }
        let t = sys::tensorrt()?;
        // SAFETY: `getPluginRegistry` was resolved from libnvinfer by the loader and takes no arguments.
        let raw = unsafe { (t.get_plugin_registry()?)() };
        if raw.is_null() {
            return Err(Error::NullHandle {
                op: "getPluginRegistry",
            });
        }
        Ok(Self { raw })
    }

    /// Every creator currently registered, in the registry's order.
    pub fn creators(&self) -> Result<Vec<PluginCreatorInfo>> {
        let mut count = 0i32;
        // SAFETY: `self.raw` is the live registry and `count` is a local the call writes.
        let list = unsafe { ffi::trtPluginRegistryGetAllCreators(self.raw, &mut count) };
        if list.is_null() || count <= 0 {
            return Ok(Vec::new());
        }
        // SAFETY: TensorRT returns `count` creator pointers, valid until the registry changes;
        // they are read here, before any other registry call.
        let slice = unsafe { core::slice::from_raw_parts(list, count as usize) };
        Ok(slice.iter().map(|&c| info(c)).collect())
    }

    /// Look up one creator. `None` when the registry has no such creator.
    pub fn creator(
        &self,
        name: &str,
        version: &str,
        namespace: &str,
    ) -> Result<Option<PluginCreatorInfo>> {
        let (name, version, namespace) = (
            CString::new(name)?,
            CString::new(version)?,
            CString::new(namespace)?,
        );
        // SAFETY: `self.raw` is the live registry; the three strings are NUL-terminated
        // `CString`s that outlive the call.
        let creator = unsafe {
            ffi::trtPluginRegistryGetCreator(
                self.raw,
                name.as_ptr(),
                version.as_ptr(),
                namespace.as_ptr(),
            )
        };
        Ok((!creator.is_null()).then(|| info(creator)))
    }

    /// Load a plugin shared library and register the creators it exports. Returns
    /// [`Error::NullHandle`] when the library cannot be loaded or does not export the
    /// plugin entry points.
    pub fn load_library(&self, path: &str) -> Result<PluginLibraryHandle> {
        let path = CString::new(path)?;
        // SAFETY: `self.raw` is the live registry and `path` is a NUL-terminated `CString`
        // that outlives the call.
        let handle = unsafe { ffi::trtPluginRegistryLoadLibrary(self.raw, path.as_ptr()) };
        if handle.is_null() {
            return Err(Error::NullHandle {
                op: "IPluginRegistry::loadLibrary",
            });
        }
        Ok(PluginLibraryHandle(handle))
    }

    /// Unregister the creators a library added and release it.
    pub fn deregister_library(&self, handle: PluginLibraryHandle) {
        // SAFETY: `handle` came from `load_library` on this registry, and consuming it by
        // value here means it is deregistered once.
        unsafe { ffi::trtPluginRegistryDeregisterLibrary(self.raw, handle.0) };
    }
}

/// Register TensorRT's built-in plugins (`initLibNvInferPlugins`) under `namespace`
/// (use `""` for the default namespace). Returns TensorRT's own success flag. Needs
/// the plugin library (`nvinfer_plugin_10.dll` / `libnvinfer_plugin.so.10`) next to
/// TensorRT.
pub fn init_builtin_plugins(logger: &Logger, namespace: &str) -> Result<bool> {
    let namespace = CString::new(namespace)?;
    let init = sys::plugin_lib()?.init_lib_nvinfer_plugins();
    // SAFETY: `init` was resolved from the plugin library by the loader; the logger is the
    // process-wide `ILogger*` and the namespace is a NUL-terminated `CString` alive for the call.
    Ok(unsafe { init(logger.as_raw(), namespace.as_ptr()) })
}
