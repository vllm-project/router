//! Pure decoding and admission helpers; these do not activate subscriptions.
mod decoder;
mod endpoints;

pub use decoder::{decode_batch, validate_device_dp1_batch};
pub use endpoints::{parse_endpoint_mapping, resolve_endpoints};
