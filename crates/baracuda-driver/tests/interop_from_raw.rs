//! Interop ownership tests for `from_raw` (transfer) / `borrow_raw` (borrow)
//! on the cross-framework driver handles (Context / Stream / Event / Device).
//!
//! GPU-gated (`#[ignore]`). Run on a machine with an NVIDIA GPU via:
//!
//! ```text
//! cargo test -p baracuda-driver --test interop_from_raw -- --ignored
//! ```
//!
//! The load-bearing invariant is that a **borrowed** wrapper must NOT destroy
//! the handle it wraps — otherwise adopting another library's context / stream
//! / event would corrupt that library. Each `borrow_*` test drops the borrowed
//! view and then keeps using the original handle: if `Drop` wrongly destroyed
//! it, the subsequent use would fail.

use baracuda_driver::{Context, Device, Event, Stream};

fn setup() -> baracuda_driver::Result<(Device, Context, Stream)> {
    baracuda_driver::init()?;
    let device = Device::get(0)?;
    let ctx = Context::new(&device)?;
    let stream = Stream::new(&ctx)?;
    Ok((device, ctx, stream))
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn borrow_raw_stream_does_not_destroy() {
    let (_dev, ctx, _stream) = setup().expect("setup");
    let owned = Stream::new(&ctx).expect("create stream");
    let raw = owned.as_raw();

    // Borrow the same raw stream; dropping the borrow must NOT cuStreamDestroy.
    {
        // SAFETY: `raw` is a live CUstream on `ctx`, owned by `owned` which
        // outlives this borrowed view.
        let borrowed = unsafe { Stream::borrow_raw(raw, &ctx) };
        assert_eq!(borrowed.as_raw(), raw);
        borrowed.synchronize().expect("borrowed stream is usable");
    } // <- borrowed dropped here

    // If the borrow had destroyed the stream, this would error.
    owned
        .synchronize()
        .expect("original stream still valid after the borrow dropped");
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn borrow_raw_context_does_not_destroy() {
    let (dev, ctx, _stream) = setup().expect("setup");
    let raw = ctx.as_raw();

    {
        // SAFETY: `raw` is a live CUcontext on `dev`, owned by `ctx` which
        // outlives this borrowed view.
        let borrowed = unsafe { Context::borrow_raw(raw, &dev) };
        assert_eq!(borrowed.as_raw(), raw);
        borrowed.api_version().expect("borrowed context is usable");
    } // <- borrowed dropped here (must NOT cuCtxDestroy)

    // If the borrow had destroyed the context, creating a stream on it fails.
    Stream::new(&ctx).expect("original context still valid after the borrow dropped");
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn borrow_raw_event_does_not_destroy() {
    let (_dev, ctx, stream) = setup().expect("setup");
    let owned = Event::new(&ctx).expect("create event");
    let raw = owned.as_raw();

    {
        // SAFETY: `raw` is a live CUevent on `ctx`, owned by `owned`.
        let borrowed = unsafe { Event::borrow_raw(raw, &ctx) };
        assert_eq!(borrowed.as_raw(), raw);
        borrowed.record(&stream).expect("borrowed event records");
        borrowed.synchronize().expect("borrowed event syncs");
    } // <- borrowed dropped here (must NOT cuEventDestroy)

    owned
        .synchronize()
        .expect("original event still valid after the borrow dropped");
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn from_raw_stream_transfer_destroys_on_drop() {
    let (_dev, ctx, _stream) = setup().expect("setup");
    let created = Stream::new(&ctx).expect("create stream");
    let raw = created.as_raw();
    // Simulate a foreign owner handing us the raw handle: suppress the original
    // wrapper's destroy so ownership is single.
    std::mem::forget(created);

    // SAFETY: `raw` is a live CUstream on `ctx`; we take sole ownership.
    let adopted = unsafe { Stream::from_raw(raw, &ctx) };
    adopted.synchronize().expect("adopted stream is usable");
    drop(adopted); // must cuStreamDestroy here

    // Prove the destroy actually ran: re-wrapping the now-dangling handle and
    // using it must fail. No stream is created between the destroy and this
    // check, so `raw` cannot have been reused for a different object.
    // SAFETY: borrow_raw only stores the pointer; the driver validates the
    // destroyed handle on the call and returns an error rather than UB.
    let ghost = unsafe { Stream::borrow_raw(raw, &ctx) };
    assert!(
        ghost.synchronize().is_err(),
        "stream must be destroyed after transfer-drop (from_raw owns it)"
    );

    // The context is healthy: a fresh stream still creates.
    Stream::new(&ctx).expect("context healthy after transfer-destroy");
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn from_raw_event_transfer_destroys_on_drop() {
    let (_dev, ctx, _stream) = setup().expect("setup");
    let created = Event::new(&ctx).expect("create event");
    let raw = created.as_raw();
    std::mem::forget(created);

    // SAFETY: live CUevent on `ctx`; sole-ownership transfer.
    let adopted = unsafe { Event::from_raw(raw, &ctx) };
    adopted.is_complete().expect("adopted event is usable");
    drop(adopted); // must cuEventDestroy here

    // SAFETY: as above — the driver call on the destroyed handle returns an error.
    let ghost = unsafe { Event::borrow_raw(raw, &ctx) };
    assert!(
        ghost.is_complete().is_err(),
        "event must be destroyed after transfer-drop"
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn from_raw_context_transfer_destroys_on_drop() {
    let (dev, ctx, _stream) = setup().expect("setup");
    // A fresh, explicitly-created context to transfer (creating it makes it
    // current on this thread, replacing `ctx`).
    let created = Context::new(&dev).expect("create context");
    let raw = created.as_raw();
    std::mem::forget(created);

    // SAFETY: live explicitly-created CUcontext on `dev` (never a primary
    // context); sole-ownership transfer.
    let adopted = unsafe { Context::from_raw(raw, &dev) };
    adopted.api_version().expect("adopted context is usable");
    drop(adopted); // must cuCtxDestroy here

    // Prove destruction: api_version() (takes the handle explicitly) now fails.
    // SAFETY: borrow_raw only stores the pointer; the driver validates it.
    let ghost = unsafe { Context::borrow_raw(raw, &dev) };
    assert!(
        ghost.api_version().is_err(),
        "context must be destroyed after transfer-drop"
    );

    // Restore a valid current context (the destroyed one was current here).
    ctx.set_current().expect("restore current context");
}

#[test]
#[ignore = "requires an NVIDIA GPU; run with `cargo test -- --ignored`"]
fn device_from_raw_round_trips() {
    baracuda_driver::init().expect("init");
    let dev = Device::get(0).expect("device 0");
    let raw = dev.as_raw();
    assert_eq!(Device::from_raw(raw).as_raw(), raw);
    assert_eq!(Device::from_raw(raw), dev);
}
