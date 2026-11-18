use std::sync::Arc;

/// One request's query against the index.
#[derive(Debug, Clone)]
pub struct MatchQuery {
    /// kv-cache-spec group to query — selects the shard within the indexer.
    pub group_idx: u32,
    /// Block-local hashes, walked from the shard root in prefix order.
    pub local_hashes: Vec<Arc<str>>,
    /// Tiers to report; the cost model requests the tiers it scores.
    pub tiers_of_interest: Vec<StorageTier>,
}

/// One hit: the block intervals `[start, end)` a target holds at this tier,
/// each maximal. Flat — one row per `(target, tier)` — so the cost model
/// applies two-segment reuse over a target's tiers in one pass.
#[derive(Debug, Clone)]
pub struct TierMatch {
    pub target: RoutableTarget,
    pub tier: StorageTier,
    /// Held block intervals `[start, end)`, each maximal; a gap closes a
    /// segment but later segments survive.
    pub segments: Vec<(u32, u32)>,
    pub locality: Locality,
}

/// Append a block to the last contiguous segment or start a new segment.
pub(crate) fn push_block_segment(segments: &mut Vec<(u32, u32)>, start: u32, end: u32) {
    match segments.last_mut() {
        Some((_, previous_end)) if *previous_end == start => *previous_end = end,
        _ => segments.push((start, end)),
    }
}

/// Where a request goes — one (worker, rank) target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RoutableTarget {
    /// Engine-reported instance identity, stable across restarts.
    pub instance_id: Arc<str>,
    /// Data-parallel rank this target routes to.
    pub dp_rank: u32,
}

/// Names a shared pool holding KV blocks across workers (Mooncake).
pub type CacheOwnerId = Arc<str>;

/// What holds a cached block — one of two domains. A `Worker` is ephemeral
/// (cleared on attachment restart); a `CacheOwner` is a shared pool (Mooncake)
/// that survives worker restarts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResidencyOwner {
    /// A worker's own device/host cache. Cleared on restart via `incarnation`.
    Worker {
        source: SourceId,
        dp_rank: u32,
        incarnation: u64,
    },
    /// A shared-pool copy attributed to the worker that pushed it.
    CacheOwner {
        pool_id: CacheOwnerId,
        source: SourceId,
        dp_rank: u32,
    },
}

impl ResidencyOwner {
    /// The target a hit on this owner routes to.
    pub fn target(&self) -> RoutableTarget {
        let (source, dp_rank) = match self {
            ResidencyOwner::Worker {
                source, dp_rank, ..
            }
            | ResidencyOwner::CacheOwner {
                source, dp_rank, ..
            } => (source, dp_rank),
        };
        RoutableTarget {
            instance_id: source.clone(),
            dp_rank: *dp_rank,
        }
    }

    /// Locality derived from the domain: Worker = Local, CacheOwner = Remote.
    pub fn locality(&self) -> Locality {
        match self {
            ResidencyOwner::Worker { .. } => Locality::Local,
            ResidencyOwner::CacheOwner { .. } => Locality::Remote,
        }
    }

    /// Which domain this owner belongs to — drives `clear` scoping.
    pub fn domain(&self) -> ClearScope {
        match self {
            ResidencyOwner::Worker { .. } => ClearScope::Worker,
            ResidencyOwner::CacheOwner { .. } => ClearScope::CacheOwner,
        }
    }
}

/// Names one event source — a worker publisher.
pub type SourceId = Arc<str>;

/// Engine block-hash mode. Router-side config, pinned per cache identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashMode {
    PythonHashSeed,
    Sha256,
    XxhashCbor,
}

impl HashMode {
    /// Parse a router-side config string. Unknown → Sha256 (safe default).
    pub fn parse_config(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "python_hash_seed" | "pythonhashseed" => HashMode::PythonHashSeed,
            "xxhash_cbor" | "xxhashcbor" => HashMode::XxhashCbor,
            _ => HashMode::Sha256,
        }
    }
}

/// Storage medium holding a residency. Decoded from the engine `medium` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    Device,
    HostPinned,
    Disk,
    External,
}

impl StorageTier {
    pub(crate) fn rank(self) -> u8 {
        match self {
            Self::Device => 0,
            Self::HostPinned => 1,
            Self::Disk => 2,
            Self::External => 3,
        }
    }

    /// Map a worker's `medium` field to its tier. vLLM emits uppercase
    /// `GPU`/`CPU`/`STORAGE`; matched case-insensitively. Unknown/None → Device.
    /// `External` is not produced here — it derives from the owner domain.
    pub fn from_medium(s: Option<&str>) -> Self {
        match s.map(|m| m.to_ascii_uppercase()).as_deref() {
            None | Some("GPU") | Some("DEVICE") => StorageTier::Device,
            Some("CPU") | Some("HOST") | Some("HOST_PINNED") => StorageTier::HostPinned,
            Some("STORAGE") | Some("DISK") => StorageTier::Disk,
            _ => StorageTier::Device,
        }
    }
}

/// Whether a residency is local or remote relative to the reading target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Locality {
    Local,
    Remote,
}

/// Scope of a `clear`. `Worker` clears a worker's device/host residency;
/// `CacheOwner` clears shared-pool residency; `All` clears both domains.
/// A `Worker` clear never crosses into `CacheOwner`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClearScope {
    Worker,
    CacheOwner,
    All,
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn medium_none_maps_to_device() {
        assert_eq!(StorageTier::from_medium(None), StorageTier::Device);
    }

    #[test]
    fn medium_uppercase_vllm_values() {
        assert_eq!(StorageTier::from_medium(Some("GPU")), StorageTier::Device);
        assert_eq!(
            StorageTier::from_medium(Some("CPU")),
            StorageTier::HostPinned
        );
        assert_eq!(StorageTier::from_medium(Some("STORAGE")), StorageTier::Disk);
    }

    #[test]
    fn medium_case_insensitive_and_aliases() {
        assert_eq!(StorageTier::from_medium(Some("gpu")), StorageTier::Device);
        assert_eq!(
            StorageTier::from_medium(Some("device")),
            StorageTier::Device
        );
        assert_eq!(
            StorageTier::from_medium(Some("cpu")),
            StorageTier::HostPinned
        );
        assert_eq!(
            StorageTier::from_medium(Some("host")),
            StorageTier::HostPinned
        );
        assert_eq!(StorageTier::from_medium(Some("disk")), StorageTier::Disk);
    }

    #[test]
    fn medium_never_produces_external() {
        // External derives from the owner domain, not `medium`: pool-backend
        // medium values fall through to Device, never External.
        assert_eq!(
            StorageTier::from_medium(Some("external")),
            StorageTier::Device
        );
        assert_eq!(StorageTier::from_medium(Some("pool")), StorageTier::Device);
        assert_eq!(
            StorageTier::from_medium(Some("shared")),
            StorageTier::Device
        );
    }

    #[test]
    fn medium_unknown_defaults_to_device() {
        assert_eq!(
            StorageTier::from_medium(Some("nonsense")),
            StorageTier::Device
        );
    }

    #[test]
    fn locality_and_domain_derived_from_owner() {
        let w = ResidencyOwner::Worker {
            source: Arc::from("w"),
            dp_rank: 0,
            incarnation: 0,
        };
        let p = ResidencyOwner::CacheOwner {
            pool_id: Arc::from("p"),
            source: Arc::from("w"),
            dp_rank: 0,
        };
        assert_eq!(w.locality(), Locality::Local);
        assert_eq!(p.locality(), Locality::Remote);
        assert_eq!(w.domain(), ClearScope::Worker);
        assert_eq!(p.domain(), ClearScope::CacheOwner);
    }

    #[test]
    fn target_routes_to_pushing_worker() {
        let w = ResidencyOwner::Worker {
            source: Arc::from("w0"),
            dp_rank: 1,
            incarnation: 3,
        };
        assert_eq!(
            w.target(),
            RoutableTarget {
                instance_id: Arc::from("w0"),
                dp_rank: 1
            }
        );
        let p = ResidencyOwner::CacheOwner {
            pool_id: Arc::from("pool"),
            source: Arc::from("w0"),
            dp_rank: 1,
        };
        // A pool hit is attributed to the worker that pushed it.
        assert_eq!(
            p.target(),
            RoutableTarget {
                instance_id: Arc::from("w0"),
                dp_rank: 1
            }
        );
    }
}
