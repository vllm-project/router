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
}

/// Where a request goes — one (worker, rank) target.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RoutableTarget {
    /// Engine-reported instance identity, stable across restarts.
    pub instance_id: Arc<str>,
    /// Data-parallel rank this target routes to.
    pub dp_rank: u32,
}

/// A worker rank's cache, scoped to one publisher incarnation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidencyOwner {
    pub(crate) source: SourceId,
    pub(crate) dp_rank: u32,
    pub(crate) incarnation: u64,
}

impl ResidencyOwner {
    pub(crate) fn target_parts(&self) -> (&SourceId, u32) {
        (&self.source, self.dp_rank)
    }

    /// The target a hit on this owner routes to.
    pub fn target(&self) -> RoutableTarget {
        let (source, dp_rank) = self.target_parts();
        RoutableTarget {
            instance_id: source.clone(),
            dp_rank,
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

/// Stable, bounded reasons for source recovery events.
/// Each variant is a metric label; add a reason only when a distinct alert needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceRecoveryReason {
    Registered,
    InitialSequence,
    SequenceGap,
    IdleCatchup,
    LiveBufferFull,
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
            Self::IdleCatchup => "idle_catchup",
            Self::LiveBufferFull => "live_buffer_full",
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
        // A worker's medium cannot attest to shared-pool residency.
        for medium in ["external", "pool", "shared", "nonsense", "MOONCAKE"] {
            assert_eq!(StorageTier::from_medium(Some(medium)), None, "{medium}");
        }
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
