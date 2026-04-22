//! CUDA Graphs (Runtime API).
//!
//! Two construction paths match the Driver-side API:
//!
//! 1. **Stream capture** ([`Stream::begin_capture`], [`Stream::end_capture`])
//!    — run your work on a stream; CUDA records it into a graph instead
//!    of executing. This is the usual path for applications.
//! 2. **Explicit construction** ([`Graph::new`]) — empty graph, add
//!    nodes by hand via the runtime's `cudaGraphAdd*Node` functions
//!    (baracuda-runtime doesn't expose typed builders for those yet —
//!    use stream capture, or drop down to the raw PFNs).
//!
//! Either way, [`Graph::instantiate`] compiles to [`GraphExec`], and
//! [`GraphExec::launch`] runs it on any stream.

use std::sync::Arc;

use baracuda_cuda_sys::runtime::{
    cudaGraphExec_t, cudaGraphNode_t, cudaGraph_t, runtime, types::cudaStreamCaptureStatus,
};

use crate::error::{check, Result};
use crate::stream::Stream;

/// Stream-capture mode (matches `cudaStreamCaptureMode`).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum CaptureMode {
    Global,
    #[default]
    ThreadLocal,
    Relaxed,
}

impl CaptureMode {
    #[inline]
    fn raw(self) -> i32 {
        match self {
            CaptureMode::Global => 0,
            CaptureMode::ThreadLocal => 1,
            CaptureMode::Relaxed => 2,
        }
    }
}

impl Stream {
    /// Begin recording operations submitted to this stream into a graph.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_stream_begin_capture()?;
        check(unsafe { cu(self.as_raw(), mode.raw()) })
    }

    /// Stop capture and return the graph of everything recorded.
    pub fn end_capture(&self) -> Result<Graph> {
        let r = runtime()?;
        let cu = r.cuda_stream_end_capture()?;
        let mut graph: cudaGraph_t = core::ptr::null_mut();
        check(unsafe { cu(self.as_raw(), &mut graph) })?;
        Ok(Graph {
            inner: Arc::new(GraphInner { handle: graph }),
        })
    }

    /// Convenience wrapper: run `f`, capturing its submissions to this
    /// stream, and return the resulting graph.
    pub fn capture<F>(&self, mode: CaptureMode, f: F) -> Result<Graph>
    where
        F: FnOnce(&Stream) -> Result<()>,
    {
        self.begin_capture(mode)?;
        let inner_result = f(self);
        let end_result = self.end_capture();
        match (inner_result, end_result) {
            (Ok(()), Ok(graph)) => Ok(graph),
            (Err(e), _) => Err(e),
            (Ok(()), Err(e)) => Err(e),
        }
    }

    /// `true` if this stream is currently recording into a graph.
    pub fn is_capturing(&self) -> Result<bool> {
        let r = runtime()?;
        let cu = r.cuda_stream_is_capturing()?;
        let mut status: core::ffi::c_int = 0;
        check(unsafe { cu(self.as_raw(), &mut status) })?;
        Ok(status == cudaStreamCaptureStatus::ACTIVE)
    }
}

/// A CUDA graph — DAG of operations, replayable via [`Graph::instantiate`].
#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
}

struct GraphInner {
    handle: cudaGraph_t,
}

unsafe impl Send for GraphInner {}
unsafe impl Sync for GraphInner {}

impl core::fmt::Debug for GraphInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Graph")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for Graph {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl Graph {
    /// Create an empty graph.
    pub fn new() -> Result<Self> {
        let r = runtime()?;
        let cu = r.cuda_graph_create()?;
        let mut graph: cudaGraph_t = core::ptr::null_mut();
        check(unsafe { cu(&mut graph, 0) })?;
        Ok(Self {
            inner: Arc::new(GraphInner { handle: graph }),
        })
    }

    /// Compile this graph into an executable form.
    pub fn instantiate(&self) -> Result<GraphExec> {
        let r = runtime()?;
        let cu = r.cuda_graph_instantiate()?;
        let mut exec: cudaGraphExec_t = core::ptr::null_mut();
        check(unsafe { cu(&mut exec, self.inner.handle, 0) })?;
        Ok(GraphExec {
            inner: Arc::new(GraphExecInner { handle: exec }),
        })
    }

    /// Approximate node count (for debugging).
    pub fn node_count(&self) -> Result<usize> {
        let r = runtime()?;
        let cu = r.cuda_graph_get_nodes()?;
        let mut count: usize = 0;
        check(unsafe { cu(self.inner.handle, core::ptr::null_mut(), &mut count) })?;
        Ok(count)
    }

    #[inline]
    pub fn as_raw(&self) -> cudaGraph_t {
        self.inner.handle
    }

    /// Add an empty "barrier" node with the given dependencies.
    pub fn add_empty_node(&self, dependencies: &[GraphNode]) -> Result<GraphNode> {
        let r = runtime()?;
        let cu = r.cuda_graph_add_empty_node()?;
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, dp, dl) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add a kernel-launch node.
    ///
    /// # Safety
    ///
    /// Same discipline as [`crate::LaunchBuilder::launch`]: argument
    /// count/order/types must match the kernel's C signature.
    pub unsafe fn add_kernel_node(
        &self,
        dependencies: &[GraphNode],
        kernel: &crate::Kernel,
        grid: crate::Dim3,
        block: crate::Dim3,
        shared_mem_bytes: u32,
        args: &mut [*mut core::ffi::c_void],
    ) -> Result<GraphNode> {
        use baracuda_cuda_sys::runtime::types::{cudaKernelNodeParams, dim3};
        let r = runtime()?;
        let cu = r.cuda_graph_add_kernel_node()?;
        let params = cudaKernelNodeParams {
            func: kernel.as_launch_ptr() as *mut core::ffi::c_void,
            grid_dim: dim3::new(grid.x, grid.y, grid.z),
            block_dim: dim3::new(block.x, block.y, block.z),
            shared_mem_bytes,
            kernel_params: if args.is_empty() {
                core::ptr::null_mut()
            } else {
                args.as_mut_ptr()
            },
            extra: core::ptr::null_mut(),
        };
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(cu(&mut node, self.inner.handle, dp, dl, &params))?;
        Ok(GraphNode { raw: node })
    }

    /// Add a memset node filling `count` 4-byte words at `dst` with `value`.
    pub fn add_memset_u32_node(
        &self,
        dependencies: &[GraphNode],
        dst: *mut core::ffi::c_void,
        value: u32,
        count: usize,
    ) -> Result<GraphNode> {
        use baracuda_cuda_sys::runtime::types::cudaMemsetParams;
        let r = runtime()?;
        let cu = r.cuda_graph_add_memset_node()?;
        let params = cudaMemsetParams {
            dst,
            pitch: 0,
            value,
            element_size: 4,
            width: count,
            height: 1,
        };
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, dp, dl, &params) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add a host-function node. `fn_` runs on a driver-owned thread
    /// when this node executes.
    ///
    /// # Safety
    ///
    /// `fn_` must remain callable with `user_data` as long as any
    /// `GraphExec` derived from this graph is alive.
    pub unsafe fn add_host_node(
        &self,
        dependencies: &[GraphNode],
        fn_: unsafe extern "C" fn(*mut core::ffi::c_void),
        user_data: *mut core::ffi::c_void,
    ) -> Result<GraphNode> {
        use baracuda_cuda_sys::runtime::types::cudaHostNodeParams;
        let r = runtime()?;
        let cu = r.cuda_graph_add_host_node()?;
        let params = cudaHostNodeParams {
            fn_: Some(fn_),
            user_data,
        };
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(cu(&mut node, self.inner.handle, dp, dl, &params))?;
        Ok(GraphNode { raw: node })
    }

    /// Add a child-graph node.
    pub fn add_child_graph_node(
        &self,
        dependencies: &[GraphNode],
        child: &Graph,
    ) -> Result<GraphNode> {
        let r = runtime()?;
        let cu = r.cuda_graph_add_child_graph_node()?;
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, dp, dl, child.as_raw()) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add an event-record node.
    pub fn add_event_record_node(
        &self,
        dependencies: &[GraphNode],
        event: &crate::Event,
    ) -> Result<GraphNode> {
        let r = runtime()?;
        let cu = r.cuda_graph_add_event_record_node()?;
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, dp, dl, event.as_raw()) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add an event-wait node.
    pub fn add_event_wait_node(
        &self,
        dependencies: &[GraphNode],
        event: &crate::Event,
    ) -> Result<GraphNode> {
        let r = runtime()?;
        let cu = r.cuda_graph_add_event_wait_node()?;
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, dp, dl, event.as_raw()) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add a stream-ordered mem-alloc node. Returns the node plus the
    /// device pointer the node will allocate at launch time.
    pub fn add_mem_alloc_node(
        &self,
        dependencies: &[GraphNode],
        device: &crate::Device,
        bytesize: usize,
    ) -> Result<(GraphNode, *mut core::ffi::c_void)> {
        use baracuda_cuda_sys::runtime::types::{
            cudaMemAllocNodeParams, cudaMemAllocationHandleType, cudaMemAllocationType,
            cudaMemLocation, cudaMemLocationType, cudaMemPoolProps,
        };
        let r = runtime()?;
        let cu = r.cuda_graph_add_mem_alloc_node()?;
        let mut params = cudaMemAllocNodeParams {
            pool_props: cudaMemPoolProps {
                alloc_type: cudaMemAllocationType::PINNED,
                handle_types: cudaMemAllocationHandleType::NONE,
                location: cudaMemLocation {
                    type_: cudaMemLocationType::DEVICE,
                    id: device.ordinal(),
                },
                ..Default::default()
            },
            access_descs: core::ptr::null(),
            access_desc_count: 0,
            bytesize,
            dptr: core::ptr::null_mut(),
        };
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, dp, dl, &mut params) })?;
        Ok((GraphNode { raw: node }, params.dptr))
    }

    /// Add a stream-ordered mem-free node for `dptr`.
    ///
    /// # Safety
    ///
    /// `dptr` must be a pointer returned by a prior mem-alloc node in
    /// this graph.
    pub unsafe fn add_mem_free_node(
        &self,
        dependencies: &[GraphNode],
        dptr: *mut core::ffi::c_void,
    ) -> Result<GraphNode> {
        let r = runtime()?;
        let cu = r.cuda_graph_add_mem_free_node()?;
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(cu(&mut node, self.inner.handle, dp, dl, dptr))?;
        Ok(GraphNode { raw: node })
    }

    /// Create a conditional-node handle (CUDA 12.3+). The returned u64
    /// is an opaque driver handle used by `cuGraphAddNode`-style
    /// conditional-node construction, which the runtime exposes via
    /// `cudaGraphAddNode`. `default_launch_value` is the handle's
    /// starting value (commonly 0 = "don't execute"), `flags` = 0 for
    /// the default. Returns [`crate::Error::FeatureNotSupported`] on
    /// older CUDA.
    pub fn conditional_handle_create(&self, default_launch_value: u32, flags: u32) -> Result<u64> {
        use baracuda_types::{supports, Feature};
        let installed = crate::init::driver_version()?;
        if !supports(installed, Feature::GraphConditionalNodes) {
            return Err(crate::error::Error::FeatureNotSupported {
                api: "cudaGraphConditionalHandleCreate",
                since: Feature::GraphConditionalNodes.required_version(),
            });
        }
        let r = runtime()?;
        let cu = r.cuda_graph_conditional_handle_create()?;
        let mut handle: u64 = 0;
        check(unsafe { cu(&mut handle, self.inner.handle, default_launch_value, flags) })?;
        Ok(handle)
    }

    /// Low-level `cudaGraphAddNode` — add a node from a tagged
    /// `cudaGraphNodeParams` struct. baracuda-runtime exposes typed
    /// builders for the common node types (kernel, memset, host, etc.);
    /// this escape hatch exists for the node types the typed API does
    /// not cover (notably conditional nodes on CUDA 12.3+).
    ///
    /// # Safety
    ///
    /// `node_params` must point at a correctly-tagged
    /// `cudaGraphNodeParams` whose union payload matches the `type`
    /// field.
    pub unsafe fn add_node_raw(
        &self,
        dependencies: &[GraphNode],
        node_params: *mut core::ffi::c_void,
    ) -> Result<GraphNode> {
        let r = runtime()?;
        let cu = r.cuda_graph_add_node()?;
        let deps: Vec<_> = dependencies.iter().map(|n| n.raw).collect();
        let (dp, dl) = deps_raw(&deps);
        let mut node: cudaGraphNode_t = core::ptr::null_mut();
        check(cu(&mut node, self.inner.handle, dp, dl, node_params))?;
        Ok(GraphNode { raw: node })
    }

    /// Add dependency edges `from[i] -> to[i]`.
    pub fn add_dependencies(&self, from: &[GraphNode], to: &[GraphNode]) -> Result<()> {
        assert_eq!(from.len(), to.len());
        if from.is_empty() {
            return Ok(());
        }
        let r = runtime()?;
        let cu = r.cuda_graph_add_dependencies()?;
        let f: Vec<_> = from.iter().map(|n| n.raw).collect();
        let t: Vec<_> = to.iter().map(|n| n.raw).collect();
        check(unsafe { cu(self.inner.handle, f.as_ptr(), t.as_ptr(), f.len()) })
    }
}

fn deps_raw(deps: &[cudaGraphNode_t]) -> (*const cudaGraphNode_t, usize) {
    if deps.is_empty() {
        (core::ptr::null(), 0)
    } else {
        (deps.as_ptr(), deps.len())
    }
}

/// A node inside a [`Graph`]. Lightweight `Copy` handle that borrows the
/// parent graph's storage.
#[derive(Copy, Clone, Debug)]
pub struct GraphNode {
    raw: cudaGraphNode_t,
}

impl GraphNode {
    #[inline]
    pub fn as_raw(&self) -> cudaGraphNode_t {
        self.raw
    }

    /// Return the `cudaGraphNodeType` integer. Compare against CUDA's
    /// enum values in the runtime API docs (0=Kernel, 1=Memcpy, 2=Memset,
    /// 3=Host, 4=Graph, 5=Empty, 6=WaitEvent, 7=EventRecord, 10=MemAlloc,
    /// 11=MemFree).
    pub fn node_type(&self) -> Result<i32> {
        let r = runtime()?;
        let cu = r.cuda_graph_node_get_type()?;
        let mut t: core::ffi::c_int = 0;
        check(unsafe { cu(self.raw, &mut t) })?;
        Ok(t)
    }

    /// For `MemFree` nodes: return the device pointer this node will free.
    pub fn mem_free_ptr(&self) -> Result<*mut core::ffi::c_void> {
        let r = runtime()?;
        let cu = r.cuda_graph_mem_free_node_get_params()?;
        let mut p: *mut core::ffi::c_void = core::ptr::null_mut();
        check(unsafe { cu(self.raw, &mut p) })?;
        Ok(p)
    }
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_graph_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// An instantiated (executable) CUDA graph.
#[derive(Clone)]
pub struct GraphExec {
    inner: Arc<GraphExecInner>,
}

struct GraphExecInner {
    handle: cudaGraphExec_t,
}

unsafe impl Send for GraphExecInner {}
unsafe impl Sync for GraphExecInner {}

impl core::fmt::Debug for GraphExecInner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GraphExec")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl core::fmt::Debug for GraphExec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl GraphExec {
    /// Launch this graph on `stream`.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let r = runtime()?;
        let cu = r.cuda_graph_launch()?;
        check(unsafe { cu(self.inner.handle, stream.as_raw()) })
    }

    /// Attempt to update this executable graph in place from a
    /// topology-identical template. On refusal the exec is left
    /// unchanged; inspect [`UpdateResult`] for the specific reason.
    pub fn update(&self, new_template: &Graph) -> Result<UpdateResult> {
        let r = runtime()?;
        let cu = r.cuda_graph_exec_update()?;
        let mut error_node: cudaGraphNode_t = core::ptr::null_mut();
        let mut result: core::ffi::c_int = 0;
        // cudaGraphExecUpdate may return non-SUCCESS even when the
        // `result` field has its own more informative value (per CUDA
        // docs). We propagate the returned rc only when `result` is
        // `SUCCESS` — otherwise we return the parsed result.
        let rc = unsafe {
            cu(
                self.inner.handle,
                new_template.as_raw(),
                &mut error_node,
                &mut result,
            )
        };
        if rc != baracuda_cuda_sys::runtime::cudaError_t::Success
            && result == baracuda_cuda_sys::runtime::types::cudaGraphExecUpdateResult::SUCCESS
        {
            return Err(crate::error::Error::Status { status: rc });
        }
        Ok(UpdateResult {
            result,
            error_node: if error_node.is_null() {
                None
            } else {
                Some(GraphNode { raw: error_node })
            },
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cudaGraphExec_t {
        self.inner.handle
    }
}

/// Outcome of [`GraphExec::update`]. `result` is a
/// `cudaGraphExecUpdateResult` code — `SUCCESS` (0) means the executable
/// graph was patched in place.
#[derive(Clone, Debug)]
pub struct UpdateResult {
    pub result: core::ffi::c_int,
    pub error_node: Option<GraphNode>,
}

impl UpdateResult {
    pub fn is_success(&self) -> bool {
        self.result == baracuda_cuda_sys::runtime::types::cudaGraphExecUpdateResult::SUCCESS
    }
}

impl Drop for GraphExecInner {
    fn drop(&mut self) {
        if let Ok(r) = runtime() {
            if let Ok(cu) = r.cuda_graph_exec_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}
