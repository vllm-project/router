//! Event decoding, admission, and subscriber transport.

pub(crate) const MAX_KV_EVENT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) mod apply;
pub(crate) mod decode;
pub(crate) mod event_stream;
mod live_event_buffer;
pub mod subscriber;
pub(crate) mod transport;
pub(crate) mod validate;
pub mod wire;
