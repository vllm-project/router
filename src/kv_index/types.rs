use std::sync::Arc;

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
    pub(crate) fn target_parts(&self) -> (&SourceId, u32) {
        match self {
            ResidencyOwner::Worker {
                source, dp_rank, ..
            }
            | ResidencyOwner::CacheOwner {
                source, dp_rank, ..
            } => (source, *dp_rank),
        }
    }

    /// The target a hit on this owner routes to.
    pub fn target(&self) -> RoutableTarget {
        let (source, dp_rank) = self.target_parts();
        RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
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

/// Storage medium holding a residency. Decoded from the engine `medium` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    Device,
    HostPinned,
    Disk,
    External,
}

/// Warn once when an unrecognized medium cannot be indexed.
fn warn_unknown_medium(medium: &str) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            medium,
            "unknown KV cache medium; treating it as unaddressable"
        );
    }
}

impl StorageTier {
    /// Shared ordering for selecting the fastest copy and pricing a restore.
    pub(crate) fn rank(self) -> u8 {
        match self {
            Self::Device => 0,
            Self::HostPinned => 1,
            Self::Disk => 2,
            Self::External => 3,
        }
    }

    /// Map a worker's `medium` field to its tier. vLLM emits uppercase
    /// `GPU`/`CPU`/`STORAGE`; accelerator backends name their device instead
    /// (`NPU`, `ASCEND`, `XPU`). Those are the same local device tier, so the
    /// mapping is by role, not by vendor. An unknown value is unaddressable:
    /// returning `None` keeps a new connector from being priced as local HBM.
    pub fn from_medium(s: Option<&str>) -> Option<Self> {
        // An omitted medium denotes the engine's default device residency.
        let medium = s.unwrap_or("GPU");
        for (name, tier) in [
            ("GPU", Self::Device),
            ("CPU", Self::HostPinned),
            ("STORAGE", Self::Disk),
            ("DEVICE", Self::Device),
            ("NPU", Self::Device),
            ("ASCEND", Self::Device),
            ("XPU", Self::Device),
            ("HOST", Self::HostPinned),
            ("HOST_PINNED", Self::HostPinned),
            ("DISK", Self::Disk),
        ] {
            if medium.eq_ignore_ascii_case(name) {
                return Some(tier);
            }
        }
        warn_unknown_medium(medium);
        None
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

/// Trust state for one independently addressed KV event source.
///
/// `Observed` is compatibility-mode credit for fresh observations, not a claim
/// that the publisher's complete history has been reconstructed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SourceRecoveryState {
    /// Worker history is complete; the external pool has no proven baseline.
    WorkerExact,
    Exact,
    Observed,
    Degraded,
    Fallback,
}

impl SourceRecoveryState {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::WorkerExact => "worker_exact",
            Self::Exact => "exact",
            Self::Observed => "observed",
            Self::Degraded => "degraded",
            Self::Fallback => "fallback",
        }
    }
}

/// Stable, bounded reasons for source recovery state transitions.
/// Each variant is a metric label; add a reason only when a distinct alert needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SourceRecoveryReason {
    Registered,
    InitialSequence,
    SequenceGap,
    ReplayApplied,
    RetryExhausted,
    ReplayFailed,
    IncarnationReset,
    ValidationFailed,
    ClearBarrier,
    FreshObservation,
}

impl SourceRecoveryReason {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Registered => "registered",
            Self::InitialSequence => "initial_sequence",
            Self::SequenceGap => "sequence_gap",
            Self::ReplayApplied => "replay_applied",
            Self::RetryExhausted => "retry_exhausted",
            Self::ReplayFailed => "replay_failed",
            Self::IncarnationReset => "incarnation_reset",
            Self::ValidationFailed => "validation_failed",
            Self::ClearBarrier => "clear_barrier",
            Self::FreshObservation => "fresh_observation",
        }
    }
}

/// Current externally observable recovery status for one source/rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SourceRecoveryStatus {
    pub(crate) state: SourceRecoveryState,
    pub(crate) reason: SourceRecoveryReason,
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
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

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct TierBits(pub(crate) u8);

impl TierBits {
    pub(crate) fn bit(tier: StorageTier) -> u8 {
        1 << tier.rank()
    }

    pub(crate) fn contains(self, tier: StorageTier) -> bool {
        self.0 & Self::bit(tier) != 0
    }

    pub(crate) fn add(&mut self, tier: StorageTier) {
        self.0 |= Self::bit(tier);
    }

    pub(crate) fn best(self) -> Option<StorageTier> {
        [
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::Disk,
            StorageTier::External,
        ]
        .into_iter()
        .find(|tier| self.contains(*tier))
    }
}
