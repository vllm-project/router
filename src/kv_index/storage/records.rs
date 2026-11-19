//! Residency tiers and incremental device replica counts.

use crate::kv_index::types::TierBits;
use crate::kv_index::StorageTier;

/// One authoritative observation, including native incremental GPU copies.
/// Keeping the count here avoids a second hash table with a separate lifetime.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(super) struct Residency {
    pub(super) tiers: TierBits,
    device_copies: u32,
}

impl Residency {
    pub(super) fn observe(&mut self, tier: StorageTier, incremental: bool) -> bool {
        if incremental {
            if tier != StorageTier::Device {
                return false;
            }
            let Some(next) = self.device_copies.checked_add(1) else {
                return false;
            };
            self.device_copies = next;
        }
        self.tiers.add(tier);
        true
    }

    /// Return true when the last observation in this tier was removed.
    pub(super) fn withdraw(&mut self, tier: StorageTier, incremental: bool) -> bool {
        if incremental {
            if tier != StorageTier::Device {
                return false;
            }
            if self.device_copies > 1 {
                self.device_copies -= 1;
                return false;
            }
        }
        if tier == StorageTier::Device {
            self.device_copies = 0;
        }
        let removed = self.contains(tier);
        self.tiers.0 &= !TierBits::bit(tier);
        removed
    }

    pub(super) fn contains(self, tier: StorageTier) -> bool {
        self.tiers.contains(tier)
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;

    #[test]
    fn residency_counts_only_incremental_gpu_copies_and_keeps_other_tiers() {
        let mut record = Residency::default();
        for _ in 0..2 {
            assert!(record.observe(StorageTier::Device, true));
        }
        assert!(!record.observe(StorageTier::HostPinned, true));
        assert!(record.observe(StorageTier::HostPinned, false));
        assert!(!record.withdraw(StorageTier::HostPinned, true));
        assert_eq!(record.device_copies, 2);
        assert!(!record.withdraw(StorageTier::Device, true));
        assert!(record.contains(StorageTier::Device));
        assert!(record.withdraw(StorageTier::Device, true));
        assert!(!record.contains(StorageTier::Device));
        assert!(record.contains(StorageTier::HostPinned));
        assert_eq!(record.device_copies, 0);
        record.withdraw(StorageTier::Device, false);
        record.withdraw(StorageTier::HostPinned, false);
        assert_eq!(record, Residency::default());
    }

    #[test]
    fn replica_overflow_does_not_change_the_observation() {
        let mut record = Residency {
            tiers: TierBits(TierBits::bit(StorageTier::Device)),
            device_copies: u32::MAX,
        };
        let before = record;
        assert!(!record.observe(StorageTier::Device, true));
        assert_eq!(record, before);
        assert!(record.observe(StorageTier::HostPinned, false));
        record.withdraw(StorageTier::Device, false);
        record.withdraw(StorageTier::HostPinned, false);
        assert_eq!(record, Residency::default());
    }
}
