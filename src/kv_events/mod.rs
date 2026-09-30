//! Static DP=1, normal-dense KV Events ingestion.

mod decoder;
mod endpoints;
mod pool;

pub use endpoints::{parse_endpoint_mapping, resolve_endpoints};
pub use pool::KVEventPool;
