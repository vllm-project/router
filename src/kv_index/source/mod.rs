//! Worker discovery and event ingestion.

pub(crate) mod binding;
pub mod discovery;
pub(crate) mod endpoints;
pub(crate) mod events;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceError {
    InvalidLayout,
    Untrusted,
    Sequence,
    InvalidBatch,
    /// The publisher's integer hashes cannot address the declared SHA256 chain.
    IntBlockHashes,
    /// A first-block event disagrees with the inferred SHA256-CBOR root.
    RootHashMismatch,
    Capacity,
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod test_support;
