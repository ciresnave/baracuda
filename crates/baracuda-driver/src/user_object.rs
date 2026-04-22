//! CUDA Graph user objects (CUDA 12.0+).
//!
//! Graphs frequently hold references to external resources (allocators,
//! file handles, RAII guards) that must be kept alive for the graph's
//! lifetime. A [`UserObject`] is a refcounted handle + destructor that
//! you can *attach* to a graph via [`Graph::retain_user_object`]; when
//! the graph releases the last reference, the destructor runs.
//!
//! The Rust safe wrapper owns a `Box<dyn FnOnce() + Send>` trampoline so
//! idiomatic `move`-closures work as destructors.

use core::ffi::c_void;

use baracuda_cuda_sys::{driver, CUuserObject};

use crate::error::{check, Result};

/// A refcounted user object. Drop releases one reference.
pub struct UserObject {
    handle: CUuserObject,
}

unsafe impl Send for UserObject {}
unsafe impl Sync for UserObject {}

impl core::fmt::Debug for UserObject {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("UserObject")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

unsafe extern "C" fn destroy_trampoline(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: `user_data` was `Box::into_raw`'d by `UserObject::new`.
    let f: Box<Box<dyn FnOnce() + Send>> =
        unsafe { Box::from_raw(user_data as *mut Box<dyn FnOnce() + Send>) };
    (*f)();
}

impl UserObject {
    /// Create a user object whose destructor is `destroy`. `initial_refcount`
    /// is typically 1; the CUDA API requires at least 1.
    pub fn new<F>(destroy: F, initial_refcount: u32) -> Result<Self>
    where
        F: FnOnce() + Send + 'static,
    {
        let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(Box::new(destroy));
        let raw = Box::into_raw(boxed) as *mut c_void;
        let d = driver()?;
        let cu = d.cu_user_object_create()?;
        let mut object: CUuserObject = core::ptr::null_mut();
        // CUDA currently requires flags == CU_USER_OBJECT_NO_DESTRUCTOR_SYNC (= 1);
        // the destructor may run on any CUDA-internal thread.
        const CU_USER_OBJECT_NO_DESTRUCTOR_SYNC: core::ffi::c_uint = 1;
        let rc = unsafe {
            cu(
                &mut object,
                raw,
                Some(destroy_trampoline),
                initial_refcount,
                CU_USER_OBJECT_NO_DESTRUCTOR_SYNC,
            )
        };
        if rc != baracuda_cuda_sys::CUresult::SUCCESS {
            // Reclaim the box so we don't leak the closure.
            drop(unsafe { Box::from_raw(raw as *mut Box<dyn FnOnce() + Send>) });
            return Err(crate::error::Error::Status { status: rc });
        }
        Ok(Self { handle: object })
    }

    /// Add `count` references to this user object's refcount.
    pub fn retain(&self, count: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_user_object_retain()?;
        check(unsafe { cu(self.handle, count) })
    }

    /// Drop `count` references (runs destructor if this was the last).
    pub fn release(&self, count: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_user_object_release()?;
        check(unsafe { cu(self.handle, count) })
    }

    #[inline]
    pub fn as_raw(&self) -> CUuserObject {
        self.handle
    }
}

impl Drop for UserObject {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_user_object_release() {
                let _ = unsafe { cu(self.handle, 1) };
            }
        }
    }
}

// Extend Graph with user-object retention.
impl crate::Graph {
    /// Have this graph take `count` references to `object`. When the
    /// graph is destroyed (or when [`release_user_object`](Self::release_user_object)
    /// is called), those references are dropped.
    ///
    /// `flags` is reserved (pass 0) in CUDA 12.x; CUDA 13 adds
    /// `CU_GRAPH_USER_OBJECT_MOVE` = 1.
    pub fn retain_user_object(&self, object: &UserObject, count: u32, flags: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_retain_user_object()?;
        check(unsafe { cu(self.as_raw(), object.as_raw(), count, flags) })
    }

    pub fn release_user_object(&self, object: &UserObject, count: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_release_user_object()?;
        check(unsafe { cu(self.as_raw(), object.as_raw(), count) })
    }
}
