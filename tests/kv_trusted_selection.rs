use std::sync::Arc;
use std::time::{Duration, Instant};
use vllm_router_rs::kv_index::routing::{
    KvRoutingCandidate, KvRoutingMode, KvTrustRegistry, TrustedKvSelector,
};
use vllm_router_rs::kv_index::{
    ClearScope, KvBlockIndexer, Locality, MatchQuery, ResidencyOwner, RoutableTarget, StorageTier,
    TierMatch, TieredMatchProvider,
};

fn target(rank: u32) -> RoutableTarget {
    RoutableTarget {
        instance_id: Arc::from("worker"),
        dp_rank: rank,
    }
}
fn owner(rank: u32) -> ResidencyOwner {
    ResidencyOwner::Worker {
        source: Arc::from("worker"),
        dp_rank: rank,
        incarnation: 0,
    }
}
fn query() -> MatchQuery {
    MatchQuery {
        group_idx: 0,
        local_hashes: vec![Arc::from("a"), Arc::from("b")],
        tiers_of_interest: vec![
            StorageTier::Device,
            StorageTier::HostPinned,
            StorageTier::External,
        ],
    }
}
fn candidates() -> Vec<KvRoutingCandidate> {
    vec![
        KvRoutingCandidate {
            target: target(0),
            active_requests: 0,
            healthy: true,
        },
        KvRoutingCandidate {
            target: target(1),
            active_requests: 1,
            healthy: true,
        },
    ]
}
fn trust(now: Instant) -> KvTrustRegistry {
    let mut t = KvTrustRegistry::new(2).unwrap();
    for rank in 0..2 {
        t.register(target(rank), 0, now);
        assert!(t.reconcile(&target(rank), 0, 5, now));
    }
    t
}
fn selector() -> TrustedKvSelector {
    TrustedKvSelector::new(0.8, 0.5, Duration::from_secs(10)).unwrap()
}
fn index() -> KvBlockIndexer {
    let idx = KvBlockIndexer::new();
    idx.store(
        0,
        owner(1),
        StorageTier::Device,
        None,
        &[
            (Arc::from("seq-a"), Arc::from("a")),
            (Arc::from("seq-b"), Arc::from("b")),
        ],
    );
    idx
}

#[test]
fn reconciled_prefix_selects_the_actual_holding_dp_rank() {
    let now = Instant::now();
    let idx = index();
    let d = selector().select(&idx, &query(), &candidates(), &trust(now), now);
    assert_eq!(d.mode, KvRoutingMode::Exact);
    assert_eq!(d.target, Some(target(1)));
    assert_eq!(d.scores[0].credited_blocks, 2.0);
}
#[test]
fn eviction_and_clear_change_real_index_selection() {
    let now = Instant::now();
    let idx = index();
    let t = trust(now);
    idx.remove(0, &owner(1), StorageTier::Device, &[Arc::from("seq-a")]);
    let d = selector().select(&idx, &query(), &candidates(), &t, now);
    assert_eq!(d.target, Some(target(0)));
    assert!(d.scores.iter().all(|s| s.credited_blocks == 0.0));
    idx.clear(&owner(1), ClearScope::Worker);
    assert_eq!(
        selector()
            .select(&idx, &query(), &candidates(), &t, now)
            .target,
        Some(target(0))
    );
}
struct NeverQuery;
impl TieredMatchProvider for NeverQuery {
    fn find_tiered_matches(&self, _: &MatchQuery) -> Vec<TierMatch> {
        panic!("untrusted KV must not be queried")
    }
}
#[test]
fn cold_start_and_live_deltas_do_not_establish_exact_trust() {
    let now = Instant::now();
    let mut t = KvTrustRegistry::new(2).unwrap();
    for rank in 0..2 {
        t.register(target(rank), 0, now);
        t.advance(&target(rank), 0, 6, now);
    }
    let d = selector().select(&NeverQuery, &query(), &candidates(), &t, now);
    assert_eq!(d.mode, KvRoutingMode::Fallback);
    assert_eq!(d.target, Some(target(0)));
}
#[test]
fn a_gap_requires_reconciliation_even_after_replay_deltas() {
    let now = Instant::now();
    let mut t = trust(now);
    t.advance(&target(1), 0, 8, now);
    t.advance(&target(1), 0, 9, now);
    assert_eq!(
        selector()
            .select(&NeverQuery, &query(), &candidates(), &t, now)
            .mode,
        KvRoutingMode::Fallback
    );
    assert!(!t.reconcile(&target(1), 0, 8, now));
    assert!(t.reconcile(&target(1), 0, 9, now));
    assert_eq!(
        selector()
            .select(&index(), &query(), &candidates(), &t, now)
            .mode,
        KvRoutingMode::Exact
    );
}
#[test]
fn restarting_and_old_epoch_events_cannot_credit_prior_residency() {
    let now = Instant::now();
    let mut t = trust(now);
    t.register(target(1), 1, now);
    assert!(!t.reconcile(&target(1), 0, 5, now));
    assert!(!t.advance(&target(1), 0, 6, now));
    assert_eq!(
        selector()
            .select(&NeverQuery, &query(), &candidates(), &t, now)
            .mode,
        KvRoutingMode::Fallback
    );
}
#[test]
fn duplicate_messages_do_not_keep_stale_sources_fresh() {
    let now = Instant::now();
    let mut t = trust(now);
    let later = now + Duration::from_secs(11);
    assert!(!t.advance(&target(0), 0, 5, later));
    assert!(!t.advance(&target(1), 0, 5, later));
    let d = selector().select(&NeverQuery, &query(), &candidates(), &t, later);
    assert_eq!(d.fallback_reason, Some("stale"));
}
#[test]
fn a_rollback_cannot_lower_the_reconciliation_watermark() {
    let now = Instant::now();
    let mut t = trust(now);
    assert!(!t.advance(&target(1), 0, 3, now));
    assert!(!t.reconcile(&target(1), 0, 3, now));
    assert_eq!(
        selector()
            .select(&NeverQuery, &query(), &candidates(), &t, now)
            .mode,
        KvRoutingMode::Fallback
    );
    assert!(t.reconcile(&target(1), 0, 5, now));
}
#[test]
fn unhealthy_targets_are_never_selected_or_required_for_trust() {
    let now = Instant::now();
    let mut c = candidates();
    c[1].healthy = false;
    let d = selector().select(&index(), &query(), &c, &trust(now), now);
    assert_eq!(d.target, Some(target(0)));
}
#[test]
fn trust_capacity_and_removal_are_bounded() {
    let now = Instant::now();
    let mut t = KvTrustRegistry::new(1).unwrap();
    assert!(t.register(target(0), 0, now));
    assert!(!t.register(target(1), 0, now));
    t.remove(&target(0));
    assert!(t.register(target(1), 0, now));
    assert!(!t.reconcile(&target(0), 0, 5, now));
}
#[test]
fn invalid_cost_configuration_is_rejected() {
    for v in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(TrustedKvSelector::new(v, 0.5, Duration::from_secs(1)).is_err());
        assert!(TrustedKvSelector::new(0.5, v, Duration::from_secs(1)).is_err());
    }
    assert!(TrustedKvSelector::new(0.5, 0.5, Duration::ZERO).is_err());
    assert!(KvTrustRegistry::new(0).is_err());
}
#[test]
fn disconnected_or_removed_sources_force_load_fallback() {
    let now = Instant::now();
    let mut t = trust(now);
    t.mark_unavailable(&target(1));
    assert_eq!(
        selector()
            .select(&NeverQuery, &query(), &candidates(), &t, now)
            .fallback_reason,
        Some("unavailable")
    );
    t.remove(&target(1));
    assert_eq!(
        selector()
            .select(&NeverQuery, &query(), &candidates(), &t, now)
            .fallback_reason,
        Some("unregistered")
    );
}
#[test]
fn empty_pools_and_queries_fall_back_without_querying_the_provider() {
    let now = Instant::now();
    let t = trust(now);
    assert_eq!(
        selector()
            .select(&NeverQuery, &query(), &[], &t, now)
            .target,
        None
    );
    let mut q = query();
    q.local_hashes.clear();
    assert_eq!(
        selector()
            .select(&NeverQuery, &q, &candidates(), &t, now)
            .fallback_reason,
        Some("empty_query")
    );
}
struct RemoteMatches;
impl TieredMatchProvider for RemoteMatches {
    fn find_tiered_matches(&self, _: &MatchQuery) -> Vec<TierMatch> {
        vec![
            TierMatch {
                target: target(0),
                tier: StorageTier::Device,
                segments: vec![(0, 1)],
                locality: Locality::Local,
            },
            TierMatch {
                target: target(1),
                tier: StorageTier::External,
                segments: vec![(0, 2)],
                locality: Locality::Remote,
            },
        ]
    }
}
#[test]
fn remote_tier_credit_is_discounted_instead_of_counted_as_device_hit() {
    let now = Instant::now();
    let mut c = candidates();
    c[1].active_requests = 0;
    let d = selector().select(&RemoteMatches, &query(), &c, &trust(now), now);
    assert_eq!(d.target, Some(target(0)));
    assert_eq!(
        d.scores
            .iter()
            .find(|s| s.target == target(1))
            .unwrap()
            .credited_blocks,
        0.125
    );
}

#[test]
fn active_load_can_outweigh_cache_affinity_and_receipt_explains_the_choice() {
    let now = Instant::now();
    let mut c = candidates();
    c[1].active_requests = 100;
    let load_weighted = TrustedKvSelector::new(0.1, 0.5, Duration::from_secs(10)).unwrap();
    let d = load_weighted.select(&index(), &query(), &c, &trust(now), now);
    assert_eq!(d.target, Some(target(0)));
    let receipt = d.to_json();
    assert_eq!(receipt["mode"], "exact");
    assert_eq!(receipt["target"]["dp_rank"], 0);
    assert!(receipt["scores"].is_array());
    assert!(receipt.get("token_ids").is_none());
    assert!(receipt.get("prompt").is_none());
}

#[test]
fn timestamps_ahead_of_the_query_clock_do_not_establish_freshness() {
    let now = Instant::now();
    let future = now + Duration::from_secs(1);
    let d = selector().select(&NeverQuery, &query(), &candidates(), &trust(future), now);
    assert_eq!(d.mode, KvRoutingMode::Fallback);
    assert_eq!(d.fallback_reason, Some("stale"));
}
