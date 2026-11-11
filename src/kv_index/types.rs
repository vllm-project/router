use std::sync::Arc;

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

/// Warn once per unknown medium value: a connector that invents a new name must
/// not silently gain local-device credit.
fn warn_unknown_medium(medium: &str) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    let mut seen = seen.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if seen.insert(medium.to_owned()) {
        tracing::warn!(
            medium,
            "unknown KV cache medium; treating it as unaddressable"
        );
    }
}

impl StorageTier {
    /// Map a worker's `medium` field to its tier. vLLM emits uppercase
    /// `GPU`/`CPU`/`STORAGE`; accelerator backends name their device instead
    /// (`NPU`, `ASCEND`, `XPU`). Those are the same local device tier, so the
    /// mapping is by role, not by vendor. An unknown value is unaddressable:
    /// returning `None` keeps a new connector from being priced as local HBM.
    pub fn from_medium(s: Option<&str>) -> Option<Self> {
        match s.map(|m| m.to_ascii_uppercase()).as_deref() {
            // The field is optional in the wire format; its absence means the
            // engine's default device residency.
            None | Some("GPU") | Some("DEVICE") | Some("NPU") | Some("ASCEND") | Some("XPU") => {
                Some(Self::Device)
            }
            Some("CPU") | Some("HOST") | Some("HOST_PINNED") => Some(Self::HostPinned),
            Some("STORAGE") | Some("DISK") => Some(Self::Disk),
            Some(other) => {
                warn_unknown_medium(other);
                None
            }
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
mod tests {
    use super::*;

    #[test]
    fn medium_none_maps_to_device() {
        assert_eq!(StorageTier::from_medium(None), Some(StorageTier::Device));
    }

    #[test]
    fn medium_uppercase_vllm_values() {
        assert_eq!(
            StorageTier::from_medium(Some("GPU")),
            Some(StorageTier::Device)
        );
        assert_eq!(
            StorageTier::from_medium(Some("CPU")),
            Some(StorageTier::HostPinned)
        );
        assert_eq!(
            StorageTier::from_medium(Some("STORAGE")),
            Some(StorageTier::Disk)
        );
    }

    #[test]
    fn medium_case_insensitive_and_aliases() {
        assert_eq!(
            StorageTier::from_medium(Some("gpu")),
            Some(StorageTier::Device)
        );
        assert_eq!(
            StorageTier::from_medium(Some("device")),
            Some(StorageTier::Device)
        );
        assert_eq!(
            StorageTier::from_medium(Some("cpu")),
            Some(StorageTier::HostPinned)
        );
        assert_eq!(
            StorageTier::from_medium(Some("host")),
            Some(StorageTier::HostPinned)
        );
        assert_eq!(
            StorageTier::from_medium(Some("disk")),
            Some(StorageTier::Disk)
        );
    }

    #[test]
    fn accelerator_mediums_are_the_device_tier() {
        for medium in ["GPU", "gpu", "DEVICE", "NPU", "npu", "ASCEND", "XPU"] {
            assert_eq!(
                StorageTier::from_medium(Some(medium)),
                Some(StorageTier::Device),
                "{medium}"
            );
        }
    }

    #[test]
    fn unknown_medium_is_unaddressable() {
        // External derives from the owner domain, and a new connector name must
        // not be priced as local HBM.
        for medium in ["external", "pool", "shared", "nonsense", "MOONCAKE"] {
            assert_eq!(StorageTier::from_medium(Some(medium)), None, "{medium}");
        }
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
