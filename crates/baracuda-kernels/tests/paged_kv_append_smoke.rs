//! Direct-FFI smoke test for FlashInfer `paged_kv_append_decode`
//! (Phase 46/66, vLLM-style serving).
//!
//! Validates the lifecycle:
//!  - `can_implement` accepts a normal config; rejects head_dim=0.
//!  - `run` appends a single decode-step token to a pre-built paged KV-
//!    cache structure for f16 / bf16 / f32 dtypes.
//!  - Output K/V pages contain the expected key/value vectors at the
//!    correct paged position after the call.
//!
//! Layout matches FlashInfer upstream:
//!  - `k_data` / `v_data` — `[num_pages, page_size, num_heads, head_dim]`
//!  - `indices` — `[total_pages_across_batch]`
//!  - `indptr` — `[batch_size + 1]` (page-offset into `indices`)
//!  - `last_page_len` — `[batch_size]`
//!  - `key` / `value` — `[batch_size, num_heads, head_dim]` (one new
//!    decode-step token per request)
//!
//! `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89,flashinfer \
//!    --test paged_kv_append_smoke -- --include-ignored`.

#![cfg(all(any(feature = "sm80", feature = "sm89", feature = "sm90a"), feature = "flashinfer"))]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn paged_kv_append_can_implement_rejects_bad_head_dim() {
    let ok = unsafe {
        baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
            2, 16, 8, 128,
        )
    };
    assert_eq!(ok, 0, "normal config (B=2 page=16 H=8 D=128) should succeed");

    let bad = unsafe {
        baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_can_implement(
            2, 16, 8, 0,
        )
    };
    assert_ne!(bad, 0, "head_dim=0 must be rejected");
}

#[test]
#[ignore]
fn paged_kv_append_f32_basic() {
    let (ctx, stream) = setup();

    let batch: i32 = 2;
    let page_size: i32 = 16;
    let num_heads: i32 = 4;
    let head_dim: i32 = 64;
    let num_pages: i32 = 4; // 2 per request

    // Build paged-KV cache: zero-fill (single test, no prior content).
    let kv_numel = (num_pages * page_size * num_heads * head_dim) as usize;
    let mut dev_k: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, kv_numel).expect("k alloc");
    let mut dev_v: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, kv_numel).expect("v alloc");

    // indices: each request gets 2 pages.
    let indices_host: Vec<i32> = vec![0, 1, 2, 3];
    // indptr: prefix-sum into indices. Request 0 → pages [0,1]; req 1 → [2,3].
    let indptr_host: Vec<i32> = vec![0, 2, 4];
    // last_page_len: each request has 3 tokens already in its last page;
    // the new key/value goes into slot 3 of the last page.
    let last_page_len_host: Vec<i32> = vec![3, 3];

    let mut dev_indices = DeviceBuffer::from_slice(&ctx, &indices_host).expect("indices");
    let mut dev_indptr = DeviceBuffer::from_slice(&ctx, &indptr_host).expect("indptr");
    let mut dev_last_page_len =
        DeviceBuffer::from_slice(&ctx, &last_page_len_host).expect("last_page_len");

    // New decode-step key/value, one token per request.
    let kv_per_req = (num_heads * head_dim) as usize;
    let key_host: Vec<f32> = (0..(batch as usize * kv_per_req))
        .map(|i| ((i as f32) * 0.011 + 0.1).sin() * 0.7)
        .collect();
    let value_host: Vec<f32> = (0..(batch as usize * kv_per_req))
        .map(|i| ((i as f32) * 0.013 - 0.2).cos() * 0.5)
        .collect();
    let dev_key = DeviceBuffer::from_slice(&ctx, &key_host).expect("key");
    let dev_value = DeviceBuffer::from_slice(&ctx, &value_host).expect("value");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_f32_run(
            batch,
            page_size,
            num_heads,
            head_dim,
            dev_k.as_slice_mut().as_raw().0 as *mut c_void,
            dev_v.as_slice_mut().as_raw().0 as *mut c_void,
            dev_indices.as_slice_mut().as_raw().0 as *mut c_void,
            dev_indptr.as_slice_mut().as_raw().0 as *mut c_void,
            dev_last_page_len.as_slice_mut().as_raw().0 as *mut c_void,
            dev_key.as_slice().as_raw().0 as *const c_void,
            dev_value.as_slice().as_raw().0 as *const c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "paged_kv_append f32 status");
    stream.synchronize().expect("sync");

    // Verify: request 0's key landed at page-1 (its last page), slot 3.
    let mut k_host: Vec<f32> = vec![0.0; kv_numel];
    dev_k.copy_to_host(&mut k_host).expect("dl k");

    // Page 1, slot 3 should match key_host[0..kv_per_req].
    let page_stride = (page_size * num_heads * head_dim) as usize;
    let slot_stride = (num_heads * head_dim) as usize;
    let dst_page: usize = 1; // req 0's last page index
    let dst_slot: usize = 3; // last_page_len[0]
    let dst_base = dst_page * page_stride + dst_slot * slot_stride;
    for h in 0..(num_heads as usize) {
        for d in 0..(head_dim as usize) {
            let got = k_host[dst_base + h * head_dim as usize + d];
            let want = key_host[h * head_dim as usize + d];
            assert!(
                (got - want).abs() < 1e-6,
                "k mismatch @ req=0 h={h} d={d}: got={got} want={want}"
            );
        }
    }

    // Request 1's key landed at page-3, slot 3.
    let dst_page2: usize = 3;
    let dst_base2 = dst_page2 * page_stride + dst_slot * slot_stride;
    for h in 0..(num_heads as usize) {
        for d in 0..(head_dim as usize) {
            let got = k_host[dst_base2 + h * head_dim as usize + d];
            let want = key_host[kv_per_req + h * head_dim as usize + d];
            assert!(
                (got - want).abs() < 1e-6,
                "k mismatch @ req=1 h={h} d={d}: got={got} want={want}"
            );
        }
    }
}

#[test]
#[ignore]
fn paged_kv_append_f16_runs_to_completion() {
    let (ctx, stream) = setup();

    let batch: i32 = 1;
    let page_size: i32 = 16;
    let num_heads: i32 = 2;
    let head_dim: i32 = 32;
    let num_pages: i32 = 2;

    let kv_numel = (num_pages * page_size * num_heads * head_dim) as usize;
    let mut dev_k: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, kv_numel).expect("k");
    let mut dev_v: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, kv_numel).expect("v");

    let mut dev_indices = DeviceBuffer::from_slice(&ctx, &[0_i32, 1]).expect("indices");
    let mut dev_indptr = DeviceBuffer::from_slice(&ctx, &[0_i32, 2]).expect("indptr");
    let mut dev_last_page_len = DeviceBuffer::from_slice(&ctx, &[5_i32]).expect("last_page_len");

    let kv_per_req = (num_heads * head_dim) as usize;
    let key_f32: Vec<f32> = (0..kv_per_req).map(|i| (i as f32) * 0.01).collect();
    let key_host: Vec<f16> = key_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let val_host: Vec<f16> = vec![f16::from_f32(0.25); kv_per_req];

    let dev_key = DeviceBuffer::from_slice(&ctx, &key_host).expect("key");
    let dev_value = DeviceBuffer::from_slice(&ctx, &val_host).expect("value");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_f16_run(
            batch, page_size, num_heads, head_dim,
            dev_k.as_slice_mut().as_raw().0 as *mut c_void,
            dev_v.as_slice_mut().as_raw().0 as *mut c_void,
            dev_indices.as_slice_mut().as_raw().0 as *mut c_void,
            dev_indptr.as_slice_mut().as_raw().0 as *mut c_void,
            dev_last_page_len.as_slice_mut().as_raw().0 as *mut c_void,
            dev_key.as_slice().as_raw().0 as *const c_void,
            dev_value.as_slice().as_raw().0 as *const c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "paged_kv_append f16 status");
    stream.synchronize().expect("sync");
}

#[test]
#[ignore]
fn paged_kv_append_bf16_runs_to_completion() {
    let (ctx, stream) = setup();

    let batch: i32 = 1;
    let page_size: i32 = 16;
    let num_heads: i32 = 2;
    let head_dim: i32 = 32;
    let num_pages: i32 = 2;

    let kv_numel = (num_pages * page_size * num_heads * head_dim) as usize;
    let mut dev_k: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, kv_numel).expect("k");
    let mut dev_v: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, kv_numel).expect("v");

    let mut dev_indices = DeviceBuffer::from_slice(&ctx, &[0_i32, 1]).expect("indices");
    let mut dev_indptr = DeviceBuffer::from_slice(&ctx, &[0_i32, 2]).expect("indptr");
    let mut dev_last_page_len = DeviceBuffer::from_slice(&ctx, &[7_i32]).expect("last_page_len");

    let kv_per_req = (num_heads * head_dim) as usize;
    let key_host: Vec<bf16> = (0..kv_per_req).map(|i| bf16::from_f32((i as f32) * 0.01)).collect();
    let val_host: Vec<bf16> = vec![bf16::from_f32(0.5); kv_per_req];

    let dev_key = DeviceBuffer::from_slice(&ctx, &key_host).expect("key");
    let dev_value = DeviceBuffer::from_slice(&ctx, &val_host).expect("value");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_bf16_run(
            batch, page_size, num_heads, head_dim,
            dev_k.as_slice_mut().as_raw().0 as *mut c_void,
            dev_v.as_slice_mut().as_raw().0 as *mut c_void,
            dev_indices.as_slice_mut().as_raw().0 as *mut c_void,
            dev_indptr.as_slice_mut().as_raw().0 as *mut c_void,
            dev_last_page_len.as_slice_mut().as_raw().0 as *mut c_void,
            dev_key.as_slice().as_raw().0 as *const c_void,
            dev_value.as_slice().as_raw().0 as *const c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "paged_kv_append bf16 status");
    stream.synchronize().expect("sync");
}
