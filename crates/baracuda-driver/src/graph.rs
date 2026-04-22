//! CUDA Graphs — record a sequence of operations once, replay cheaply.
//!
//! Two construction paths:
//!
//! 1. **Stream capture** ([`Stream::begin_capture`], [`Stream::end_capture`])
//!    — run your work normally on a stream; the driver records it into a
//!    graph instead of executing it. This is the recommended entry point
//!    for most users.
//! 2. **Explicit construction** ([`Graph::new`]) — build a graph node by
//!    node. baracuda v0.1 does not yet expose typed node builders; use
//!    capture for now.
//!
//! Either way, instantiate the [`Graph`] to a [`GraphExec`] and launch it
//! on any stream as many times as you like.

use std::sync::Arc;

use baracuda_cuda_sys::types::{
    CUgraphConditionalHandle, CUgraphExecUpdateResultInfo, CUgraphNodeParams, CUgraphNodeType,
    CUmemAllocationHandleType, CUmemAllocationType, CUmemLocation, CUmemLocationType,
    CUmemPoolProps, CUDA_CONDITIONAL_NODE_PARAMS, CUDA_HOST_NODE_PARAMS, CUDA_KERNEL_NODE_PARAMS,
    CUDA_MEMCPY3D, CUDA_MEMSET_NODE_PARAMS, CUDA_MEM_ALLOC_NODE_PARAMS,
};
use baracuda_cuda_sys::{driver, CUdeviceptr, CUgraph, CUgraphExec, CUgraphNode};

use crate::context::Context;
use crate::error::{check, Result};
use crate::event::Event;
use crate::launch::Dim3;
use crate::module::Function;
use crate::stream::Stream;

/// Stream-capture mode, matching `CUstreamCaptureMode`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum CaptureMode {
    /// Captures operations on *any* stream in the process while this
    /// thread's stream is capturing. Discouraged — mostly present for
    /// compatibility with legacy code.
    Global,
    /// Captures only operations on streams whose capture was started from
    /// the current thread. **Recommended default.**
    #[default]
    ThreadLocal,
    /// Permissive mode — allows unsynchronized cross-stream activity
    /// without failing capture.
    Relaxed,
}

impl CaptureMode {
    #[inline]
    fn raw(self) -> u32 {
        match self {
            CaptureMode::Global => 0,
            CaptureMode::ThreadLocal => 1,
            CaptureMode::Relaxed => 2,
        }
    }
}

impl Stream {
    /// Begin recording operations submitted to this stream into a CUDA graph.
    ///
    /// Call [`Stream::end_capture`] to retrieve the resulting [`Graph`].
    /// Most operations (kernel launches, memcpys, event records) enqueued
    /// between these two calls are captured rather than executed.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_stream_begin_capture()?;
        check(unsafe { cu(self.as_raw(), mode.raw()) })
    }

    /// Stop capture and return the graph of everything that was recorded.
    pub fn end_capture(&self) -> Result<Graph> {
        let d = driver()?;
        let cu = d.cu_stream_end_capture()?;
        let mut graph: CUgraph = core::ptr::null_mut();
        check(unsafe { cu(self.as_raw(), &mut graph) })?;
        Ok(Graph {
            inner: Arc::new(GraphInner {
                handle: graph,
                context: self.context().clone(),
                owned: true,
            }),
        })
    }

    /// Convenience wrapper: run `f`, capturing everything it submits to
    /// this stream, and return the resulting graph.
    ///
    /// `f` should enqueue its work on `self`. If it errors out mid-capture,
    /// we still end the capture to avoid leaking the captured state.
    pub fn capture<F>(&self, mode: CaptureMode, f: F) -> Result<Graph>
    where
        F: FnOnce(&Stream) -> Result<()>,
    {
        self.begin_capture(mode)?;
        let inner_result = f(self);
        // End capture regardless of f's success so we don't leak state.
        let end_result = self.end_capture();
        match (inner_result, end_result) {
            (Ok(()), Ok(graph)) => Ok(graph),
            (Err(e), _) => Err(e),
            (Ok(()), Err(e)) => Err(e),
        }
    }

    /// `true` if this stream is currently in capture mode.
    pub fn is_capturing(&self) -> Result<bool> {
        let d = driver()?;
        let cu = d.cu_stream_is_capturing()?;
        let mut status: core::ffi::c_uint = 0;
        check(unsafe { cu(self.as_raw(), &mut status) })?;
        // CUstreamCaptureStatus::NONE = 0, ACTIVE = 1, INVALIDATED = 2.
        Ok(status == 1)
    }
}

/// A CUDA graph — a DAG of CUDA operations.
#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
}

struct GraphInner {
    handle: CUgraph,
    context: Context,
    /// When `false`, this `Graph` wraps a graph owned by something else
    /// (e.g. the body of a conditional node). Drop is a no-op.
    owned: bool,
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
    /// Create an empty graph in the given context. Use this as a starting
    /// point for explicit node construction — note that baracuda v0.1
    /// does not yet expose typed node builders, so in practice you'll
    /// almost always build graphs via stream capture instead.
    pub fn new(context: &Context) -> Result<Self> {
        context.set_current()?;
        let d = driver()?;
        let cu = d.cu_graph_create()?;
        let mut graph: CUgraph = core::ptr::null_mut();
        check(unsafe { cu(&mut graph, 0) })?;
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle: graph,
                context: context.clone(),
                owned: true,
            }),
        })
    }

    /// Compile this graph into an executable form that can be launched.
    pub fn instantiate(&self) -> Result<GraphExec> {
        self.instantiate_with_flags(0)
    }

    /// As [`Self::instantiate`] but passes `flags` to
    /// `cuGraphInstantiateWithFlags` (see [`instantiate_flags`]).
    pub fn instantiate_with_flags(&self, flags: u64) -> Result<GraphExec> {
        let d = driver()?;
        let cu = d.cu_graph_instantiate_with_flags()?;
        let mut exec: CUgraphExec = core::ptr::null_mut();
        check(unsafe { cu(&mut exec, self.inner.handle, flags) })?;
        Ok(GraphExec {
            inner: Arc::new(GraphExecInner {
                handle: exec,
                context: self.inner.context.clone(),
            }),
        })
    }

    /// Approximate number of nodes in the graph (useful for debugging).
    pub fn node_count(&self) -> Result<usize> {
        let d = driver()?;
        let cu = d.cu_graph_get_nodes()?;
        let mut count: usize = 0;
        check(unsafe { cu(self.inner.handle, core::ptr::null_mut(), &mut count) })?;
        Ok(count)
    }

    /// Raw `CUgraph`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUgraph {
        self.inner.handle
    }

    /// Add an empty "join / barrier" node with the given dependencies.
    /// Returns the new node so it can be used as a dependency of later nodes.
    pub fn add_empty_node(&self, dependencies: &[GraphNode]) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_empty_node()?;
        let mut node: CUgraphNode = core::ptr::null_mut();
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        check(unsafe { cu(&mut node, self.inner.handle, deps_ptr, deps_len) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add a kernel-launch node. `args` is the same `*mut c_void` array you'd
    /// pass to `cuLaunchKernel` — build it from [`baracuda_types::KernelArg`]
    /// pointers, same as the [`crate::LaunchBuilder`] does.
    ///
    /// # Safety
    ///
    /// Same responsibilities as [`crate::LaunchBuilder::launch`]: argument
    /// count, order, and types must match the kernel's C signature, and
    /// pointer-typed arguments must remain live as long as any executable
    /// derived from this graph is running.
    pub unsafe fn add_kernel_node(
        &self,
        dependencies: &[GraphNode],
        function: &Function,
        grid: impl Into<Dim3>,
        block: impl Into<Dim3>,
        shared_mem_bytes: u32,
        args: &mut [*mut core::ffi::c_void],
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_kernel_node()?;
        let grid = grid.into();
        let block = block.into();
        let params = CUDA_KERNEL_NODE_PARAMS {
            func: function.as_raw(),
            grid_dim_x: grid.x,
            grid_dim_y: grid.y,
            grid_dim_z: grid.z,
            block_dim_x: block.x,
            block_dim_y: block.y,
            block_dim_z: block.z,
            shared_mem_bytes,
            kernel_params: if args.is_empty() {
                core::ptr::null_mut()
            } else {
                args.as_mut_ptr()
            },
            extra: core::ptr::null_mut(),
            kern: core::ptr::null_mut(),
            ctx: core::ptr::null_mut(),
        };
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(cu(
            &mut node,
            self.inner.handle,
            deps_ptr,
            deps_len,
            &params,
        ))?;
        Ok(GraphNode { raw: node })
    }

    /// Add a 1-D memset node that fills `count` elements starting at `dst`
    /// with the 4-byte pattern `value`. Operates in the graph's parent
    /// context.
    pub fn add_memset_u32_node(
        &self,
        dependencies: &[GraphNode],
        dst: CUdeviceptr,
        value: u32,
        count: usize,
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_memset_node()?;
        let params = CUDA_MEMSET_NODE_PARAMS {
            dst,
            pitch: 0,
            value,
            element_size: 4,
            width: count,
            height: 1,
        };
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut node,
                self.inner.handle,
                deps_ptr,
                deps_len,
                &params,
                self.inner.context.as_raw(),
            )
        })?;
        Ok(GraphNode { raw: node })
    }

    /// Deep-copy this graph (including its topology). The clone is
    /// independent — destroying one does not affect the other.
    pub fn clone_graph(&self) -> Result<Self> {
        let d = driver()?;
        let cu = d.cu_graph_clone()?;
        let mut out: CUgraph = core::ptr::null_mut();
        check(unsafe { cu(&mut out, self.inner.handle) })?;
        Ok(Self {
            inner: Arc::new(GraphInner {
                handle: out,
                context: self.inner.context.clone(),
                owned: true,
            }),
        })
    }

    /// Add a memcpy node. `params` is a fully-populated [`CUDA_MEMCPY3D`].
    pub fn add_memcpy_node(
        &self,
        dependencies: &[GraphNode],
        params: &CUDA_MEMCPY3D,
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_memcpy_node()?;
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut node,
                self.inner.handle,
                deps_ptr,
                deps_len,
                params,
                self.inner.context.as_raw(),
            )
        })?;
        Ok(GraphNode { raw: node })
    }

    /// Add a host-function node. Hand the closure's trampoline address
    /// and a user-data pointer that remains valid for the lifetime of any
    /// executable graph derived from this graph.
    ///
    /// # Safety
    ///
    /// `fn_` will be invoked on a CUDA-internal host thread with
    /// `user_data` as its argument. The pointer must remain valid as long
    /// as any `GraphExec` containing this node is alive.
    pub unsafe fn add_host_node(
        &self,
        dependencies: &[GraphNode],
        fn_: unsafe extern "C" fn(*mut core::ffi::c_void),
        user_data: *mut core::ffi::c_void,
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_host_node()?;
        let params = CUDA_HOST_NODE_PARAMS {
            fn_: Some(fn_),
            user_data,
        };
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(cu(
            &mut node,
            self.inner.handle,
            deps_ptr,
            deps_len,
            &params,
        ))?;
        Ok(GraphNode { raw: node })
    }

    /// Add a child-graph node — executes `child` in its entirety when
    /// reached.
    pub fn add_child_graph_node(
        &self,
        dependencies: &[GraphNode],
        child: &Graph,
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_child_graph_node()?;
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut node,
                self.inner.handle,
                deps_ptr,
                deps_len,
                child.as_raw(),
            )
        })?;
        Ok(GraphNode { raw: node })
    }

    /// Add an event-record node — records `event` when executed.
    pub fn add_event_record_node(
        &self,
        dependencies: &[GraphNode],
        event: &Event,
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_event_record_node()?;
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut node,
                self.inner.handle,
                deps_ptr,
                deps_len,
                event.as_raw(),
            )
        })?;
        Ok(GraphNode { raw: node })
    }

    /// Add an event-wait node — blocks downstream nodes until `event` has
    /// been recorded.
    pub fn add_event_wait_node(
        &self,
        dependencies: &[GraphNode],
        event: &Event,
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_event_wait_node()?;
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut node,
                self.inner.handle,
                deps_ptr,
                deps_len,
                event.as_raw(),
            )
        })?;
        Ok(GraphNode { raw: node })
    }

    /// Add a stream-ordered memory allocation node. When the graph runs,
    /// the node allocates `bytesize` bytes on `device` (from the device's
    /// default pool). The resulting device pointer is returned in the
    /// output tuple alongside the new node.
    pub fn add_mem_alloc_node(
        &self,
        dependencies: &[GraphNode],
        device: &crate::Device,
        bytesize: usize,
    ) -> Result<(GraphNode, CUdeviceptr)> {
        let d = driver()?;
        let cu = d.cu_graph_add_mem_alloc_node()?;
        let mut params = CUDA_MEM_ALLOC_NODE_PARAMS {
            pool_props: CUmemPoolProps {
                alloc_type: CUmemAllocationType::PINNED,
                handle_types: CUmemAllocationHandleType::NONE,
                location: CUmemLocation {
                    type_: CUmemLocationType::DEVICE,
                    id: device.as_raw().0,
                },
                ..Default::default()
            },
            access_descs: core::ptr::null(),
            access_desc_count: 0,
            bytesize,
            dptr: CUdeviceptr(0),
        };
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut node,
                self.inner.handle,
                deps_ptr,
                deps_len,
                &mut params,
            )
        })?;
        Ok((GraphNode { raw: node }, params.dptr))
    }

    /// Add a stream-ordered memory-free node for `dptr` (which is
    /// typically the `dptr` returned by a prior
    /// [`Self::add_mem_alloc_node`] on the same graph).
    pub fn add_mem_free_node(
        &self,
        dependencies: &[GraphNode],
        dptr: CUdeviceptr,
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_mem_free_node()?;
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, deps_ptr, deps_len, dptr) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add a batch-memop node — a single node that performs a sequence of
    /// 32/64-bit wait/write value operations on device memory atomically
    /// wrt the graph's execution order.
    ///
    /// `ops` may include any mix of [`baracuda_cuda_sys::types::CUstreamBatchMemOpParams`]
    /// entries built with that type's `wait_value_*` / `write_value_*` helpers.
    pub fn add_batch_mem_op_node(
        &self,
        dependencies: &[GraphNode],
        ops: &mut [baracuda_cuda_sys::types::CUstreamBatchMemOpParams],
    ) -> Result<GraphNode> {
        let d = driver()?;
        let cu = d.cu_graph_add_batch_mem_op_node()?;
        let params = baracuda_cuda_sys::types::CUDA_BATCH_MEM_OP_NODE_PARAMS {
            ctx: self.inner.context.as_raw(),
            count: ops.len() as core::ffi::c_uint,
            param_array: ops.as_mut_ptr(),
            flags: 0,
        };
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe { cu(&mut node, self.inner.handle, deps_ptr, deps_len, &params) })?;
        Ok(GraphNode { raw: node })
    }

    /// Add dependency edges from each node in `from` to its counterpart in
    /// `to`. Both slices must have the same length.
    pub fn add_dependencies(&self, from: &[GraphNode], to: &[GraphNode]) -> Result<()> {
        assert_eq!(from.len(), to.len(), "add_dependencies: length mismatch");
        if from.is_empty() {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_graph_add_dependencies()?;
        let f: Vec<CUgraphNode> = from.iter().map(|n| n.raw).collect();
        let t: Vec<CUgraphNode> = to.iter().map(|n| n.raw).collect();
        check(unsafe { cu(self.inner.handle, f.as_ptr(), t.as_ptr(), f.len()) })
    }

    /// Remove previously-added dependency edges.
    pub fn remove_dependencies(&self, from: &[GraphNode], to: &[GraphNode]) -> Result<()> {
        assert_eq!(from.len(), to.len(), "remove_dependencies: length mismatch");
        if from.is_empty() {
            return Ok(());
        }
        let d = driver()?;
        let cu = d.cu_graph_remove_dependencies()?;
        let f: Vec<CUgraphNode> = from.iter().map(|n| n.raw).collect();
        let t: Vec<CUgraphNode> = to.iter().map(|n| n.raw).collect();
        check(unsafe { cu(self.inner.handle, f.as_ptr(), t.as_ptr(), f.len()) })
    }

    /// Dump a Graphviz-compatible representation of this graph to `path`.
    /// Pass `flags = 0` for the default verbose output.
    pub fn debug_dot_print(&self, path: &str, flags: u32) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_debug_dot_print()?;
        let c_path = std::ffi::CString::new(path).map_err(|_| {
            crate::error::Error::Loader(baracuda_core::LoaderError::SymbolNotFound {
                library: "cuda-driver",
                symbol: "cuGraphDebugDotPrint(path contained a NUL byte)",
            })
        })?;
        check(unsafe { cu(self.inner.handle, c_path.as_ptr(), flags) })
    }

    /// Create a conditional handle tied to this parent graph. Pass the
    /// handle's value from inside a kernel (via
    /// `cudaGraphSetConditional(handle, val)`) to drive whether or how
    /// many times the conditional body executes.
    pub fn conditional_handle(
        &self,
        default_launch_value: u32,
        flags: u32,
    ) -> Result<CUgraphConditionalHandle> {
        let d = driver()?;
        let cu = d.cu_graph_conditional_handle_create()?;
        let mut h: CUgraphConditionalHandle = 0;
        check(unsafe {
            cu(
                &mut h,
                self.inner.handle,
                self.inner.context.as_raw(),
                default_launch_value,
                flags,
            )
        })?;
        Ok(h)
    }

    /// Add a conditional node (IF / WHILE / SWITCH). Returns `(node, body)`
    /// — populate the `body` graph with the code to execute conditionally.
    ///
    /// `type_` is one of
    /// [`baracuda_cuda_sys::types::CUgraphConditionalNodeType`].
    /// `size` is the count of body graphs (1 for IF/WHILE; up to N for SWITCH).
    pub fn add_conditional_node(
        &self,
        dependencies: &[GraphNode],
        handle: CUgraphConditionalHandle,
        type_: i32,
        size: u32,
    ) -> Result<(GraphNode, Graph)> {
        let d = driver()?;
        let cu = d.cu_graph_add_node()?;
        let mut body: CUgraph = core::ptr::null_mut();
        let cond = CUDA_CONDITIONAL_NODE_PARAMS {
            handle,
            type_,
            size,
            body_graph_out: &mut body,
            ctx: self.inner.context.as_raw(),
        };
        let mut params = CUgraphNodeParams {
            type_: CUgraphNodeType::CONDITIONAL,
            ..Default::default()
        };
        // Write the CUDA_CONDITIONAL_NODE_PARAMS at the start of the payload.
        // SAFETY: payload is [u64; 29] = 232 bytes, 8-aligned; conditional
        // params fit in 32 bytes.
        unsafe {
            let dst = params.payload.as_mut_ptr() as *mut CUDA_CONDITIONAL_NODE_PARAMS;
            dst.write(cond);
        }
        let deps: Vec<CUgraphNode> = dependencies.iter().map(|n| n.raw).collect();
        let (deps_ptr, deps_len) = deps_raw(&deps);
        let mut node: CUgraphNode = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut node,
                self.inner.handle,
                deps_ptr,
                core::ptr::null(),
                deps_len,
                &mut params,
            )
        })?;
        // The body CUgraph is owned by the conditional node; wrap it
        // non-owning so our Drop doesn't double-free.
        let body_graph = Graph {
            inner: Arc::new(GraphInner {
                handle: body,
                context: self.inner.context.clone(),
                owned: false,
            }),
        };
        Ok((GraphNode { raw: node }, body_graph))
    }

    /// Return `(from, to)` vectors describing every edge in the graph.
    pub fn edges(&self) -> Result<(Vec<GraphNode>, Vec<GraphNode>)> {
        let d = driver()?;
        let cu = d.cu_graph_get_edges()?;
        // First call: ask for edge count.
        let mut count: usize = 0;
        check(unsafe {
            cu(
                self.inner.handle,
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &mut count,
            )
        })?;
        let mut from = vec![core::ptr::null_mut(); count];
        let mut to = vec![core::ptr::null_mut(); count];
        if count > 0 {
            check(unsafe {
                cu(
                    self.inner.handle,
                    from.as_mut_ptr(),
                    to.as_mut_ptr(),
                    &mut count,
                )
            })?;
        }
        Ok((
            from.into_iter().map(|raw| GraphNode { raw }).collect(),
            to.into_iter().map(|raw| GraphNode { raw }).collect(),
        ))
    }
}

fn deps_raw(deps: &[CUgraphNode]) -> (*const CUgraphNode, usize) {
    if deps.is_empty() {
        (core::ptr::null(), 0)
    } else {
        (deps.as_ptr(), deps.len())
    }
}

/// A node inside a [`Graph`]. Lightweight handle that borrows the parent
/// graph's storage — destroying the graph invalidates all of its nodes.
///
/// `GraphNode` is `Copy` so you can use a single node as a dependency of
/// many successors without cloning.
#[derive(Copy, Clone, Debug)]
pub struct GraphNode {
    raw: CUgraphNode,
}

impl GraphNode {
    /// Raw `CUgraphNode`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUgraphNode {
        self.raw
    }

    /// Return the `CUgraphNodeType` code for this node. Compare against
    /// constants in [`baracuda_cuda_sys::types::CUgraphNodeType`].
    pub fn node_type(&self) -> Result<core::ffi::c_int> {
        let d = driver()?;
        let cu = d.cu_graph_node_get_type()?;
        let mut t: core::ffi::c_int = 0;
        check(unsafe { cu(self.raw, &mut t) })?;
        Ok(t)
    }

    /// Return this node's upstream dependencies.
    pub fn dependencies(&self) -> Result<Vec<GraphNode>> {
        let d = driver()?;
        let cu = d.cu_graph_node_get_dependencies()?;
        let mut count: usize = 0;
        check(unsafe { cu(self.raw, core::ptr::null_mut(), &mut count) })?;
        let mut out = vec![core::ptr::null_mut(); count];
        if count > 0 {
            check(unsafe { cu(self.raw, out.as_mut_ptr(), &mut count) })?;
        }
        Ok(out.into_iter().map(|raw| GraphNode { raw }).collect())
    }

    /// Return nodes that depend on this node.
    pub fn dependent_nodes(&self) -> Result<Vec<GraphNode>> {
        let d = driver()?;
        let cu = d.cu_graph_node_get_dependent_nodes()?;
        let mut count: usize = 0;
        check(unsafe { cu(self.raw, core::ptr::null_mut(), &mut count) })?;
        let mut out = vec![core::ptr::null_mut(); count];
        if count > 0 {
            check(unsafe { cu(self.raw, out.as_mut_ptr(), &mut count) })?;
        }
        Ok(out.into_iter().map(|raw| GraphNode { raw }).collect())
    }

    /// Fetch current kernel-node params (kernel-node nodes only).
    pub fn kernel_params(&self) -> Result<CUDA_KERNEL_NODE_PARAMS> {
        let d = driver()?;
        let cu = d.cu_graph_kernel_node_get_params()?;
        let mut p = CUDA_KERNEL_NODE_PARAMS::default();
        check(unsafe { cu(self.raw, &mut p) })?;
        Ok(p)
    }

    /// Overwrite this kernel-node's params on the template graph (not the
    /// instantiated exec — use [`GraphExec::set_kernel_node_params`] for
    /// live edit).
    ///
    /// # Safety
    ///
    /// The caller ensures the new params describe a valid kernel launch
    /// — same kind of invariants as [`crate::LaunchBuilder::launch`].
    pub unsafe fn set_kernel_params(&self, params: &CUDA_KERNEL_NODE_PARAMS) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_kernel_node_set_params()?;
        check(cu(self.raw, params))
    }

    /// Fetch current memset-node params (memset-node nodes only).
    pub fn memset_params(&self) -> Result<CUDA_MEMSET_NODE_PARAMS> {
        let d = driver()?;
        let cu = d.cu_graph_memset_node_get_params()?;
        let mut p = CUDA_MEMSET_NODE_PARAMS::default();
        check(unsafe { cu(self.raw, &mut p) })?;
        Ok(p)
    }

    /// Overwrite this memset-node's params on the template graph.
    pub fn set_memset_params(&self, params: &CUDA_MEMSET_NODE_PARAMS) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_memset_node_set_params()?;
        check(unsafe { cu(self.raw, params) })
    }

    /// Fetch the device pointer this `MemFree` node will free. Only valid
    /// on nodes of type `CUgraphNodeType::MEM_FREE`.
    pub fn mem_free_ptr(&self) -> Result<CUdeviceptr> {
        let d = driver()?;
        let cu = d.cu_graph_mem_free_node_get_params()?;
        let mut p = CUdeviceptr(0);
        check(unsafe { cu(self.raw, &mut p) })?;
        Ok(p)
    }

    /// Fetch the full mem-alloc-node params (pool props + bytesize + the
    /// output `dptr` CUDA will write into at execute time).
    pub fn mem_alloc_params(&self) -> Result<CUDA_MEM_ALLOC_NODE_PARAMS> {
        let d = driver()?;
        let cu = d.cu_graph_mem_alloc_node_get_params()?;
        let mut p = CUDA_MEM_ALLOC_NODE_PARAMS::default();
        check(unsafe { cu(self.raw, &mut p) })?;
        Ok(p)
    }

    /// Fetch current memcpy-node params.
    pub fn memcpy_params(&self) -> Result<CUDA_MEMCPY3D> {
        let d = driver()?;
        let cu = d.cu_graph_memcpy_node_get_params()?;
        let mut p = CUDA_MEMCPY3D::default();
        check(unsafe { cu(self.raw, &mut p) })?;
        Ok(p)
    }

    /// Overwrite this memcpy-node's params on the template graph.
    pub fn set_memcpy_params(&self, params: &CUDA_MEMCPY3D) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_memcpy_node_set_params()?;
        check(unsafe { cu(self.raw, params) })
    }

    /// Explicitly destroy this node inside its parent graph. Usually you
    /// just drop the [`Graph`] to clean up everything at once; this is only
    /// useful for surgically editing a graph mid-construction.
    ///
    /// # Safety
    ///
    /// The caller must not use this `GraphNode` (or any dependency-list
    /// reference to it) after calling this function.
    pub unsafe fn destroy(self) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_destroy_node()?;
        check(cu(self.raw))
    }
}

/// Re-export of `cuGraphInstantiateWithFlags` flag constants.
pub mod instantiate_flags {
    pub use baracuda_cuda_sys::types::CUgraphInstantiate_flags::*;
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        if !self.owned || self.handle.is_null() {
            return;
        }
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_graph_destroy() {
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
    handle: CUgraphExec,
    #[allow(dead_code)]
    context: Context,
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
    /// Launch this graph on `stream`. Can be called repeatedly — that's the
    /// whole point of CUDA graphs.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_launch()?;
        check(unsafe { cu(self.inner.handle, stream.as_raw()) })
    }

    /// Raw `CUgraphExec`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> CUgraphExec {
        self.inner.handle
    }

    /// Try to update this executable graph in place based on a new
    /// template graph. Returns `Ok(info)` in all cases — inspect
    /// [`UpdateResult`] to distinguish success from the various
    /// reasons CUDA refused the update. On refusal, the executable graph
    /// is left unchanged (not corrupted), so the caller can fall back to
    /// re-instantiating.
    ///
    /// Only topology-invariant changes are allowed (same number + type of
    /// nodes, same dependency edges). Changes to kernel arguments, grid
    /// dims, memset values, memcpy params, etc. are supported.
    pub fn update(&self, new_template: &Graph) -> Result<UpdateResult> {
        let d = driver()?;
        let cu = d.cu_graph_exec_update()?;
        let mut info = CUgraphExecUpdateResultInfo::default();
        // cuGraphExecUpdate_v2 returns SUCCESS even when the update
        // failed for topology/type reasons — the info.result field
        // carries the real outcome.
        let rc = unsafe { cu(self.inner.handle, new_template.as_raw(), &mut info) };
        if rc != baracuda_cuda_sys::CUresult::SUCCESS
            && info.result == baracuda_cuda_sys::types::CUgraphExecUpdateResult::SUCCESS
        {
            return Err(crate::error::Error::Status { status: rc });
        }
        Ok(UpdateResult {
            result: info.result,
            error_node: if info.error_node.is_null() {
                None
            } else {
                Some(GraphNode {
                    raw: info.error_node,
                })
            },
            error_from_node: if info.error_from_node.is_null() {
                None
            } else {
                Some(GraphNode {
                    raw: info.error_from_node,
                })
            },
        })
    }

    /// Live-edit a kernel-node's parameters on the instantiated graph.
    /// Avoids re-instantiation when only arg values / grid dims / shmem
    /// change. The topology must match (no new nodes, no new edges).
    ///
    /// # Safety
    ///
    /// Same launch invariants as [`crate::LaunchBuilder::launch`].
    pub unsafe fn set_kernel_node_params(
        &self,
        node: GraphNode,
        params: &CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_exec_kernel_node_set_params()?;
        check(cu(self.inner.handle, node.raw, params))
    }

    /// Live-edit a memcpy-node's parameters on the instantiated graph.
    pub fn set_memcpy_node_params(&self, node: GraphNode, params: &CUDA_MEMCPY3D) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_exec_memcpy_node_set_params()?;
        check(unsafe {
            cu(
                self.inner.handle,
                node.raw,
                params,
                self.inner.context.as_raw(),
            )
        })
    }

    /// Live-edit a memset-node's parameters on the instantiated graph.
    pub fn set_memset_node_params(
        &self,
        node: GraphNode,
        params: &CUDA_MEMSET_NODE_PARAMS,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_exec_memset_node_set_params()?;
        check(unsafe {
            cu(
                self.inner.handle,
                node.raw,
                params,
                self.inner.context.as_raw(),
            )
        })
    }

    /// Live-edit a host-node's callback on the instantiated graph.
    ///
    /// # Safety
    ///
    /// `fn_` must remain callable with `user_data` for the lifetime of
    /// this `GraphExec`.
    pub unsafe fn set_host_node_params(
        &self,
        node: GraphNode,
        fn_: unsafe extern "C" fn(*mut core::ffi::c_void),
        user_data: *mut core::ffi::c_void,
    ) -> Result<()> {
        let d = driver()?;
        let cu = d.cu_graph_exec_host_node_set_params()?;
        let params = CUDA_HOST_NODE_PARAMS {
            fn_: Some(fn_),
            user_data,
        };
        check(cu(self.inner.handle, node.raw, &params))
    }
}

impl Drop for GraphExecInner {
    fn drop(&mut self) {
        if let Ok(d) = driver() {
            if let Ok(cu) = d.cu_graph_exec_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Outcome of [`GraphExec::update`]. Inspect [`result`](Self::result)
/// against the constants in
/// [`baracuda_cuda_sys::types::CUgraphExecUpdateResult`] —
/// `SUCCESS` (0) means the executable graph was patched in place.
#[derive(Clone, Debug)]
pub struct UpdateResult {
    pub result: core::ffi::c_int,
    /// Node in the *new* template that triggered the failure, if any.
    pub error_node: Option<GraphNode>,
    /// Corresponding node in the *old* (already-instantiated) template.
    pub error_from_node: Option<GraphNode>,
}

impl UpdateResult {
    /// `true` iff CUDA accepted the update.
    pub fn is_success(&self) -> bool {
        self.result == baracuda_cuda_sys::types::CUgraphExecUpdateResult::SUCCESS
    }
}

// ---- Graph-memory per-device attribute queries --------------------------

/// Trim the per-device graph-mem reserve back to the minimum footprint.
pub fn device_graph_mem_trim(device: &crate::Device) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_device_graph_mem_trim()?;
    check(unsafe { cu(device.as_raw()) })
}

/// Query a `CUgraphMem_attribute` (used / reserved memory, high-water
/// marks) — see [`baracuda_cuda_sys::types::CUgraphMem_attribute`].
pub fn device_graph_mem_attribute(device: &crate::Device, attr: i32) -> Result<u64> {
    let d = driver()?;
    let cu = d.cu_device_get_graph_mem_attribute()?;
    let mut v: u64 = 0;
    check(unsafe {
        cu(
            device.as_raw(),
            attr,
            &mut v as *mut u64 as *mut core::ffi::c_void,
        )
    })?;
    Ok(v)
}

/// Reset a `CUgraphMem_attribute` (typically the high-water marks).
pub fn device_set_graph_mem_attribute(device: &crate::Device, attr: i32, value: u64) -> Result<()> {
    let d = driver()?;
    let cu = d.cu_device_set_graph_mem_attribute()?;
    let mut v = value;
    check(unsafe {
        cu(
            device.as_raw(),
            attr,
            &mut v as *mut u64 as *mut core::ffi::c_void,
        )
    })
}
