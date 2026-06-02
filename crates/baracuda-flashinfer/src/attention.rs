//! Paged-KV attention plans (decode, append, cascade merge).
//!
//! These are the FlashInfer-backed attention primitives for KV-cache
//! based serving. The plan implementations live in
//! [`baracuda_kernels::attention`]; this module re-exports them under
//! the `baracuda-flashinfer` namespace and is the documented entry
//! point. See the crate root for the high-level overview.
//!
//! Typical serving loop wiring:
//!
//! 1. After the QKV projection of the current step, call
//!    [`PagedKvAppendPlan`] to write the new K/V slices into the paged
//!    store (one token per request).
//! 2. Call [`BatchPagedDecodePlan`] to attend each request's single
//!    query row against its paged history, producing one output row +
//!    per-row log-sum-exp.
//! 3. For prefix-sharing across requests (shared system prompt, RAG
//!    context reuse), attend the shared prefix once and the
//!    per-request suffix separately, then fuse the two partial states
//!    with [`CascadeAttentionPlan`].

pub use baracuda_kernels::{
    BatchPagedDecodeArgs, BatchPagedDecodeDescriptor, BatchPagedDecodeFp8Args,
    BatchPagedDecodeFp8Descriptor, BatchPagedDecodeFp8Plan, BatchPagedDecodePlan,
    BatchPagedPrefillArgs, BatchPagedPrefillDescriptor, BatchPagedPrefillPlan,
    BatchRaggedPrefillArgs, BatchRaggedPrefillDescriptor, BatchRaggedPrefillPlan,
    CascadeAttentionArgs, CascadeAttentionDescriptor, CascadeAttentionPlan, CascadeMergeStatesArgs,
    CascadeMergeStatesDescriptor, CascadeMergeStatesPlan, Fp8KvDtype, PagedKvAppendArgs,
    PagedKvAppendDescriptor, PagedKvAppendPlan, PagedKvCacheDescriptor,
};
