//! Source-aware token-fact resolution and shadow admission accounting.
//!
//! The Router decides *which* token count a Program admission uses; this module
//! owns that policy so the scheduling modules keep a single numeric demand.
//! Resolution is intentionally cheap: one lock, no tokenizer work, and no
//! additional encoding. Prepared counts arrive with the request.

use super::scheduler::ProgramScheduler;
use super::scheduler_state::{ProgramPauseReason, ProgramSchedulerState};
use super::{ArrivalTokenFacts, ContextTokens, ProgramRef, TokenCountMode, TokenizerFingerprint};
use crate::metrics::RouterMetrics;
use std::time::Instant;
use tracing::warn;

/// Maximum revoked fingerprints retained in scheduler state.
///
/// Revocation is a correctness fallback, not an audit log: the counter keeps
/// the total. The bound only prevents unbounded scheduler memory.
const MAX_REVOKED_FINGERPRINTS: usize = 64;
/// Maximum fingerprints whose verification is remembered.
///
/// Evicting an entry only re-subjects that class to a fresh probation window;
/// it can never make an unverified count trusted.
const MAX_VERIFIED_FINGERPRINTS: usize = 1024;
/// Stop optional preparation after three successful responses without usage.
pub(super) const MAX_TOKEN_USAGE_MISSES: u8 = 3;
const MAX_TOKEN_USAGE_POOLS: usize = 64;

impl ProgramScheduler {
    /// Record the active token-count mode once at startup.
    pub fn publish_token_count_mode(&self) {
        RouterMetrics::set_agent_aware_context_token_mode(self.config.token_count_mode.as_str());
    }

    /// Whether prepared counts may be applied for this model pool.
    ///
    /// Operators may list either the internal pool key
    /// (`program:model:<model>`, `program:all`, `program:fallback:unlabeled`)
    /// or the plain model name documented for `token_count_model_pools`.
    pub(crate) fn prepared_token_pool_allowed(&self, model_pool: &str) -> bool {
        if self.config.token_count_model_pools.is_empty() {
            return true;
        }
        let model = model_pool.strip_prefix("program:model:");
        self.config
            .token_count_model_pools
            .iter()
            .any(|allowed| allowed == model_pool || model.is_some_and(|model| allowed == model))
    }

    /// Gate optional preparation/usage injection before paying their CPU cost.
    pub(crate) fn prepared_token_pool_enabled(&self, model_pool: &str) -> bool {
        self.config.token_count_mode.uses_prepared_facts()
            && self.prepared_token_pool_allowed(model_pool)
            && !Self::usage_unavailable(&self.state.lock(), model_pool)
    }

    fn usage_unavailable(state: &ProgramSchedulerState, model_pool: &str) -> bool {
        state.token_usage_tracking_saturated
            || state
                .token_usage_misses
                .get(model_pool)
                .is_some_and(|misses| *misses >= MAX_TOKEN_USAGE_MISSES)
    }

    fn fingerprint_revoked(
        &self,
        state: &ProgramSchedulerState,
        fingerprint: Option<TokenizerFingerprint>,
    ) -> bool {
        fingerprint
            .is_some_and(|fingerprint| state.revoked_token_fingerprints.contains_key(&fingerprint))
    }

    pub(crate) fn revalidate_context(
        &self,
        state: &ProgramSchedulerState,
        model_pool: &str,
        context: ContextTokens,
    ) -> ContextTokens {
        context
            .fallback_if_revoked(
                state.token_fingerprint_saturated_pools.contains(model_pool)
                    || self.fingerprint_revoked(state, context.fingerprint()),
            )
            .fallback_if_usage_unavailable(Self::usage_unavailable(state, model_pool))
    }

    /// Revocation and all retained-fact updates share the scheduler lock. Scan
    /// only on a new revocation, including every fingerprint in a saturated pool.
    fn refresh_retained_token_facts(&self, state: &mut ProgramSchedulerState) {
        let revoked = &state.revoked_token_fingerprints;
        let saturated = &state.token_fingerprint_saturated_pools;
        let usage_misses = &state.token_usage_misses;
        let usage_tracking_saturated = state.token_usage_tracking_saturated;
        let changes = state.runtime.update_retained_contexts(|program, context| {
            context
                .fallback_if_revoked(
                    saturated.contains(program.model_pool())
                        || context
                            .fingerprint()
                            .is_some_and(|fingerprint| revoked.contains_key(&fingerprint)),
                )
                .fallback_if_usage_unavailable(
                    usage_tracking_saturated
                        || usage_misses
                            .get(program.model_pool())
                            .is_some_and(|misses| *misses >= MAX_TOKEN_USAGE_MISSES),
                )
        });
        let now = Instant::now();
        for (program, context) in changes {
            // Recheck committed but unclaimed admissions. Once admit_front has
            // claimed an attempt, it is in flight and cannot be recalled here,
            // even if the transport has not sent its request yet.
            if !self.config.binding_only
                && self.pause_idle(state, &program, ProgramPauseReason::CapacityRepair, now)
            {
                if let Some(target) = state
                    .decisions
                    .get(&program)
                    .and_then(|decision| decision.last_target.clone())
                {
                    let queue = if self.config.global_queue {
                        state
                            .global_queues
                            .entry(program.model_pool().to_string())
                            .or_default()
                    } else {
                        state.rank_queues.entry(target.clone()).or_default()
                    };
                    if !queue.contains(&program) {
                        queue.push_back(program.clone());
                    }
                    if let Some(decision) = state.decisions.get_mut(&program) {
                        decision.queued_at.get_or_insert(now);
                    }
                    self.set_force_resume_deadline(state, &target, &program, now);
                }
            }
            let placement = state.runtime.placement(&program).map(str::to_string);
            if let Some(decision) = state.decisions.get_mut(&program) {
                let before = decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens);
                // Preserve larger historical/in-flight demand. On a tie the
                // fallback must replace the old provenance after revocation.
                if context.tokens() >= decision.context.tokens() {
                    decision.context = context;
                    decision.context_shrink_observations = 0;
                }
                let after = decision.private_tokens(self.config.progress_ttl.decode_buffer_tokens);
                if let Some(target) = placement {
                    Self::adjust_usage(state, &target, after - before);
                }
            }
            state.runtime.notify_front(&program);
        }
    }

    /// Resolve the applied context demand for one arrival.
    ///
    /// In `estimated` mode this returns the legacy byte estimate unchanged. In
    /// `shadow` mode it also records the prepared counterfactual without
    /// changing any decision. In `exact` mode it applies the
    /// prepared count unless the model pool is not allowlisted or the
    /// fingerprint was revoked after an Engine disagreement.
    pub fn resolve_arrival_tokens(
        &self,
        model_pool: &str,
        arrival: ArrivalTokenFacts,
    ) -> ContextTokens {
        let allowed = self.prepared_token_pool_allowed(model_pool);
        let threshold = self.config.token_count_verification_settlements;
        let (revoked, verified, usage_unavailable) = {
            let state = self.state.lock();
            let fingerprint = arrival.prepared().and_then(|facts| facts.fingerprint());
            let revoked = state.token_fingerprint_saturated_pools.contains(model_pool)
                || self.fingerprint_revoked(&state, fingerprint);
            // `0` disables probation (legacy trust-on-first-use). Otherwise a
            // fingerprint must have accumulated enough matching settlements.
            let verified = threshold == 0
                || fingerprint.is_none_or(|fingerprint| {
                    state
                        .verified_token_fingerprints
                        .get(&fingerprint)
                        .is_some_and(|(count, _)| *count >= threshold)
                });
            (
                revoked,
                verified,
                Self::usage_unavailable(&state, model_pool),
            )
        };
        super::resolve_context_tokens(
            self.config.token_count_mode,
            arrival,
            allowed,
            revoked,
            verified,
        )
        .fallback_if_usage_unavailable(
            self.config.token_count_mode.uses_prepared_facts() && usage_unavailable,
        )
    }

    /// Record one arrival-time shadow admission counterfactual.
    ///
    /// The sample answers a single, reproducible question: with the current
    /// ledger on the Program's chosen rank, would the prepared count have been
    /// admitted by the same capacity and Program-count checks that the applied
    /// count uses? It never changes the decision.
    pub(crate) fn record_shadow_admission(
        &self,
        state: &ProgramSchedulerState,
        program: &ProgramRef,
        target_id: &str,
        now: Instant,
        shadow_tokens: usize,
    ) {
        if self.config.token_count_mode != TokenCountMode::Shadow {
            return;
        }
        if let Some(outcome) =
            self.shadow_admission_outcome(state, program, target_id, now, shadow_tokens)
        {
            RouterMetrics::record_agent_aware_context_token_shadow(target_id, outcome);
        }
    }

    /// Counterfactual outcome for one sampled admission, split out so tests can
    /// assert it without scraping metrics.
    ///
    /// Only an actual queued admission is sampled: the Program must be
    /// paused/reasoning with exactly one retained front turn. Both the applied
    /// and the prepared demand are then evaluated by the same rank admission
    /// plan the dispatcher uses (count, low watermark, growth reserve, and
    /// batch-gain admission), so the counterfactual cannot disagree with the
    /// real gate by construction.
    pub(crate) fn shadow_admission_outcome(
        &self,
        state: &ProgramSchedulerState,
        program: &ProgramRef,
        target_id: &str,
        now: Instant,
        shadow_tokens: usize,
    ) -> Option<&'static str> {
        state.decisions.get(program)?;
        if self.config.pd_mode || shadow_tokens == 0 {
            return None;
        }
        let runtime = state.runtime.view(program)?;
        if runtime.state != super::ProgramState::Paused
            || runtime.status != super::ProgramStatus::Reasoning
            || runtime.waiting_requests != 1
        {
            return None;
        }
        let applied_fits = self
            .rank_admission_plan_for_context(state, program, target_id, now, None)
            .is_some();
        let shadow_fits = self
            .rank_admission_plan_for_context(state, program, target_id, now, Some(shadow_tokens))
            .is_some();
        Some(match (applied_fits, shadow_fits) {
            (true, true) => "both_admit",
            (false, false) => "both_deny",
            (true, false) => "estimate_admits_exact_denies",
            (false, true) => "estimate_denies_exact_admits",
        })
    }

    /// Record one matching Engine settlement for a fingerprint.
    ///
    /// `token_count_verification_settlements` matching settlements are required
    /// before `exact` applies a count; `0` disables probation.
    fn mark_fingerprint_verified(
        &self,
        state: &mut ProgramSchedulerState,
        fingerprint: TokenizerFingerprint,
    ) {
        let threshold = self.config.token_count_verification_settlements;
        if threshold == 0 {
            return;
        }
        state.token_fingerprint_verification_sequence = state
            .token_fingerprint_verification_sequence
            .saturating_add(1);
        let sequence = state.token_fingerprint_verification_sequence;
        let entry = state
            .verified_token_fingerprints
            .entry(fingerprint)
            .or_insert((0, sequence));
        entry.0 = entry.0.saturating_add(1).min(threshold);
        entry.1 = sequence;
        while state.verified_token_fingerprints.len() > MAX_VERIFIED_FINGERPRINTS {
            let Some(oldest) = state
                .verified_token_fingerprints
                .iter()
                .min_by_key(|(_, (_, sequence))| *sequence)
                .map(|(fingerprint, _)| *fingerprint)
            else {
                break;
            };
            state.verified_token_fingerprints.remove(&oldest);
        }
    }

    /// Compare one prepared arrival count with the Engine-observed count.
    ///
    /// A disagreement revokes the fingerprint so later requests for the same
    /// model/template/options return to estimation instead of repeating an
    /// equivalence error. The comparison is bounded and never panics.
    pub(crate) fn reconcile_observed_prompt_tokens(
        &self,
        state: &mut ProgramSchedulerState,
        model_pool: &str,
        source: super::TokenFactSource,
        fingerprint: Option<TokenizerFingerprint>,
        prepared_tokens: Option<usize>,
        observed_tokens: Option<usize>,
    ) {
        if !source.is_prepared() || prepared_tokens.is_none() {
            return;
        }
        let observed_tokens = observed_tokens.filter(|tokens| *tokens > 0);
        let (Some(prepared), Some(observed)) = (prepared_tokens, observed_tokens) else {
            RouterMetrics::record_agent_aware_context_token_unobserved(model_pool, source.as_str());
            if Self::usage_unavailable(state, model_pool) {
                return;
            }
            if !state.token_usage_misses.contains_key(model_pool)
                && state.token_usage_misses.len() >= MAX_TOKEN_USAGE_POOLS
            {
                state.token_usage_tracking_saturated = true;
                warn!(event = "token_fact_usage_tracking_saturated", bound = MAX_TOKEN_USAGE_POOLS,
                    "usage tracking reached its pool bound; optional token facts disabled until restart");
            } else {
                let misses = state
                    .token_usage_misses
                    .entry(model_pool.to_string())
                    .or_default();
                *misses += 1;
                if *misses < MAX_TOKEN_USAGE_MISSES {
                    return;
                }
                warn!(event = "token_fact_usage_unavailable", model_pool, misses = *misses,
                    "Engine prompt usage repeatedly absent; optional preparation and usage injection disabled for this pool until restart");
            }
            RouterMetrics::set_agent_aware_context_token_usage_unavailable(model_pool);
            self.refresh_retained_token_facts(state);
            return;
        };
        // Reset a partial streak; late responses cannot reopen a disabled pool.
        if !Self::usage_unavailable(state, model_pool) {
            state.token_usage_misses.remove(model_pool);
        }
        RouterMetrics::record_agent_aware_context_token_observation_delta(
            model_pool,
            source.as_str(),
            observed as i64 - prepared as i64,
            observed,
        );
        let Some(fingerprint) = fingerprint else {
            return;
        };
        if prepared == observed {
            // In-flight matches still supply usage evidence, but cannot put a
            // revoked fingerprint (or a disabled pool) back in the verified LRU.
            if !self.fingerprint_revoked(state, Some(fingerprint))
                && !state.token_fingerprint_saturated_pools.contains(model_pool)
            {
                self.mark_fingerprint_verified(state, fingerprint);
            }
            return;
        }
        // A mismatch revokes this class until restart; matching later responses
        // cannot clear the revocation.
        state.verified_token_fingerprints.remove(&fingerprint);
        if state.token_fingerprint_saturated_pools.contains(model_pool) {
            // Every prepared fact for this pool already falls back to estimation.
            return;
        }
        let newly_revoked = state
            .revoked_token_fingerprints
            .insert(fingerprint, Instant::now())
            .is_none();
        if newly_revoked {
            state.token_fingerprint_revocations =
                state.token_fingerprint_revocations.saturating_add(1);
            RouterMetrics::record_agent_aware_context_token_revocation("usage_mismatch");
            warn!(
                event = "token_fact_fingerprint_revoked",
                model_pool,
                source = source.as_str(),
                fingerprint = %fingerprint,
                prepared_tokens,
                observed_tokens,
                "prepared input-token count disagreed with Engine usage; \
                 falling back to estimation for this fingerprint"
            );
        }
        if state.revoked_token_fingerprints.len() > MAX_REVOKED_FINGERPRINTS {
            // Keeping the bound by eviction would make an old, known-bad
            // fingerprint trusted again. Fail closed instead, but only for the
            // pool that overflowed: its prepared facts fall back to estimation
            // until the Router restarts, while other pools stay exact.
            state.revoked_token_fingerprints.remove(&fingerprint);
            state
                .token_fingerprint_saturated_pools
                .insert(model_pool.to_string());
            RouterMetrics::set_agent_aware_context_token_pool_saturated(model_pool, true);
            RouterMetrics::record_agent_aware_context_token_revocation("saturation");
            warn!(
                event = "token_fact_fingerprint_revocation_saturated",
                model_pool,
                bound = MAX_REVOKED_FINGERPRINTS,
                "revoked-fingerprint set reached its bound; prepared token counts now \
                 fall back to estimation for every fingerprint until restart"
            );
        }
        RouterMetrics::set_agent_aware_context_token_revoked(
            state.revoked_token_fingerprints.len(),
        );
        if newly_revoked {
            self.refresh_retained_token_facts(state);
        }
    }

    /// Whether Router-side text tokenization may supply exact facts for one
    /// endpoint. Only consulted when a prepared count exists; any Engine
    /// disagreement later revokes the fingerprint.
    pub fn exact_text_endpoint_enabled(&self, endpoint: &str) -> bool {
        self.config.token_count_mode.uses_prepared_facts()
            && self
                .config
                .exact_text_endpoints
                .iter()
                .any(|allowed| allowed == endpoint)
    }

    /// Whether any configured text endpoint may use Router-tokenized facts.
    pub fn exact_text_enabled(&self) -> bool {
        self.config.token_count_mode.uses_prepared_facts()
            && !self.config.exact_text_endpoints.is_empty()
    }

    /// HTTP metrics endpoint overrides configured for Program targets.
    pub fn metrics_base_urls(&self) -> std::collections::BTreeMap<String, String> {
        self.config.metrics_base_urls.clone()
    }

    /// Lifetime count of fingerprint revocations, for diagnostics.
    pub fn token_fingerprint_revocations(&self) -> u64 {
        self.state.lock().token_fingerprint_revocations
    }
}

/// Token-count mode of this scheduler, for diagnostics and startup logging.
impl ProgramScheduler {
    pub fn token_count_mode(&self) -> TokenCountMode {
        self.config.token_count_mode
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::program_scheduling::{
        ProgramSchedulerConfig, ProgramTarget, RequestSample, TokenFactSource, TokenFacts,
    };
    use serde_json::json;
    use std::time::Duration;

    fn target() -> ProgramTarget {
        ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: Some(0),
        }
    }

    pub(super) fn identity(name: &str) -> super::super::ProgramIdentity {
        super::super::ProgramIdentity::from_request(
            None,
            Some(&json!({
                "vllm_xargs": {"agentic_context": {
                    "program_id": name,
                    "expected_resume": true
                }}
            })),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    fn config(mode: TokenCountMode, capacity: usize) -> ProgramSchedulerConfig {
        let mut config = ProgramSchedulerConfig {
            token_count_mode: mode,
            ..Default::default()
        };
        config.progress_ttl.token_capacity = Some(capacity);
        config
    }

    #[tokio::test]
    async fn exact_mode_admits_with_engine_scale_demand() {
        let mut config = config(TokenCountMode::Exact, 100_000);
        config.token_count_verification_settlements = 0;
        let scheduler = ProgramScheduler::new(config);
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "client"]);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(4, Some(TokenFacts::client_token_ids(98_304, fingerprint))),
        );
        assert_eq!(context.tokens(), 98_304);
        assert_eq!(context.source(), TokenFactSource::ClientTokenIds);
        let dispatch = scheduler
            .acquire(
                identity("p"),
                context,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        assert_eq!(dispatch.context_tokens(), 98_304);
        assert_eq!(
            dispatch.context_token_source(),
            TokenFactSource::ClientTokenIds
        );
    }

    #[tokio::test]
    async fn shadow_arrival_reconciles_its_prepared_fact_at_settlement() {
        // A shadow request applies the estimate but still carries the prepared
        // fact; settlement must reconcile that fact so drift is visible before
        // exact mode is enabled.
        let config = ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Shadow,
            ..ProgramSchedulerConfig::default()
        };
        let scheduler = ProgramScheduler::new(config);
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "shadow"]);
        let arrival =
            ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(120, fingerprint)));
        let context = scheduler.resolve_arrival_tokens("model", arrival);
        assert_eq!(context.tokens(), 100);
        assert_eq!(context.shadow_tokens(), Some(120));
        let dispatch = scheduler
            .acquire(
                identity("p"),
                context,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        scheduler.complete(
            &dispatch,
            true,
            false,
            crate::program_scheduling::ProgramUsageObservation {
                prompt_tokens: Some(121),
                completion_tokens: Some(3),
                cached_prompt_tokens: None,
            },
            None,
        );
        let state = scheduler.state.lock();
        assert!(state.revoked_token_fingerprints.contains_key(&fingerprint));
    }

    #[test]
    fn revocation_saturation_falls_back_instead_of_forgetting() {
        // Eviction would make an old known-bad fingerprint trusted again; the
        // bound must fail closed for every prepared fact until restart.
        let config = ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            token_count_verification_settlements: 0,
            ..ProgramSchedulerConfig::default()
        };
        let scheduler = ProgramScheduler::new(config);
        let mut state = scheduler.state.lock();
        for index in 0..MAX_REVOKED_FINGERPRINTS {
            let label = format!("bad-{index}");
            state
                .revoked_token_fingerprints
                .insert(TokenizerFingerprint::from_parts(&[&label]), Instant::now());
        }
        let fresh = TokenizerFingerprint::from_parts(&["fresh"]);
        scheduler.reconcile_observed_prompt_tokens(
            &mut state,
            "model",
            TokenFactSource::PreparedTokenIds,
            Some(fresh),
            Some(10),
            Some(11),
        );
        assert!(state.token_fingerprint_saturated_pools.contains("model"));
        assert_eq!(
            state.revoked_token_fingerprints.len(),
            MAX_REVOKED_FINGERPRINTS
        );
        drop(state);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(5, Some(TokenFacts::prepared_token_ids(100, fresh))),
        );
        assert_eq!(context.tokens(), 5);
        assert_eq!(
            context.fallback(),
            Some(crate::program_scheduling::TokenFactFallback::FingerprintRevoked)
        );
        // Another pool keeps its exact facts: saturation is per model pool.
        let other = scheduler.resolve_arrival_tokens(
            "other-model",
            ArrivalTokenFacts::new(5, Some(TokenFacts::prepared_token_ids(100, fresh))),
        );
        assert_eq!(other.tokens(), 100);
        assert_eq!(other.source(), TokenFactSource::PreparedTokenIds);
    }

    #[tokio::test]
    async fn shadow_outcome_is_only_sampled_for_a_queued_front_turn() {
        // An already-active Program has no admission gate to counterfactually
        // evaluate, so shadow sampling must not report an outcome for it.
        let config = config(TokenCountMode::Shadow, 100_000);
        let scheduler = ProgramScheduler::new(config);
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "active"]);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(
                60_000,
                Some(TokenFacts::prepared_token_ids(40_000, fingerprint)),
            ),
        );
        let dispatch = scheduler
            .acquire(
                identity("p"),
                context,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        assert_eq!(dispatch.context_tokens(), 60_000);
        let state = scheduler.state.lock();
        assert_eq!(
            scheduler.shadow_admission_outcome(
                &state,
                dispatch.program(),
                "rank-0",
                Instant::now(),
                40_000,
            ),
            None
        );
    }

    #[test]
    fn shadow_outcome_uses_the_real_rank_admission_gate() {
        // The applied 45k demand exceeds a 40k rank while the prepared 30k
        // fits, and the queue has no active Program (batch-gain admission is
        // disabled), so the real gate yields the exact-deny counterfactual.
        let mut config = config(TokenCountMode::Shadow, 40_000);
        config.progress_ttl.decode_buffer_tokens = 10;
        let scheduler = ProgramScheduler::new(config);
        scheduler.sync_targets("model", std::slice::from_ref(&target()));
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "queued"]);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(
                45_000,
                Some(TokenFacts::prepared_token_ids(30_000, fingerprint)),
            ),
        );
        assert_eq!(context.tokens(), 45_000);
        assert_eq!(context.shadow_tokens(), Some(30_000));
        let now = Instant::now();
        let mut state = scheduler.state.lock();
        let handle = state
            .runtime
            .retain_request(&identity("p"), context, None, now);
        scheduler.ensure_decision_state(
            &mut state,
            &identity("p"),
            handle.program().clone(),
            context,
            now,
            None,
        );
        assert_eq!(
            scheduler.shadow_admission_outcome(&state, handle.program(), "rank-0", now, 30_000),
            Some("estimate_denies_exact_admits")
        );
    }

    #[test]
    fn shadow_outcome_includes_the_growth_reserve() {
        // A 1000-token rank with a 600-token growth reserve: the 100-token
        // estimate fits both the low watermark and the reserve, while the
        // prepared 800-token count fits immediate capacity but not the reserve.
        // The old comparison (immediate headroom only) reported both_admit.
        let mut config = config(TokenCountMode::Shadow, 1_000);
        config.progress_ttl.low_watermark_ratio = 1.0;
        config.progress_ttl.decode_buffer_tokens = 10;
        config.progress_ttl.stats_window_size = 2;
        let scheduler = ProgramScheduler::new(config);
        scheduler.sync_targets("model", std::slice::from_ref(&target()));
        let now = Instant::now();
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "reserve"]);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(800, fingerprint))),
        );
        let mut state = scheduler.state.lock();
        {
            let factors = state.rank_factors.entry("rank-0".to_string()).or_default();
            for index in 0..2 {
                factors.push_request(
                    RequestSample {
                        prompt_tokens: 100,
                        completion_tokens: 10,
                        cached_prompt_tokens: 0,
                        context_growth_tokens: Some(100),
                        e2e_seconds: 1.0,
                        decode_seconds: Some(1.0),
                        queue_seconds: 0.0,
                        rounds_since_ttl_pause: 3,
                        finished_at: now + Duration::from_millis(index),
                    },
                    2,
                );
            }
        }
        let handle = state
            .runtime
            .retain_request(&identity("p"), context, None, now);
        scheduler.ensure_decision_state(
            &mut state,
            &identity("p"),
            handle.program().clone(),
            context,
            now,
            None,
        );
        let applied = scheduler
            .rank_admission_plan(&state, handle.program(), "rank-0", now)
            .expect("the estimate fits the reserve");
        assert_eq!(applied.reserve_tokens, 600.0);
        assert!(
            scheduler
                .rank_admission_plan_for_context(&state, handle.program(), "rank-0", now, Some(800))
                .is_none(),
            "the prepared count exceeds the reserve"
        );
        assert_eq!(
            scheduler.shadow_admission_outcome(&state, handle.program(), "rank-0", now, 800),
            Some("estimate_admits_exact_denies")
        );
    }

    #[tokio::test]
    async fn exact_mode_applies_only_after_a_matching_settlement() {
        let config = config(TokenCountMode::Exact, 1_000_000);
        // Default probation is one matching settlement.
        assert_eq!(config.token_count_verification_settlements, 1);
        let scheduler = ProgramScheduler::new(config);
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "probation"]);
        let arrival =
            ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(500, fingerprint)));
        let pending = scheduler.resolve_arrival_tokens("model", arrival);
        assert_eq!(pending.tokens(), 100);
        assert_eq!(pending.shadow_tokens(), Some(500));
        assert_eq!(
            pending.fallback(),
            Some(crate::program_scheduling::TokenFactFallback::FingerprintUnverified)
        );

        let dispatch = scheduler
            .acquire(
                identity("p"),
                pending,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        scheduler.complete(
            &dispatch,
            true,
            false,
            crate::program_scheduling::ProgramUsageObservation {
                prompt_tokens: Some(500),
                completion_tokens: Some(1),
                cached_prompt_tokens: None,
            },
            None,
        );

        let verified = scheduler.resolve_arrival_tokens("model", arrival);
        assert_eq!(verified.tokens(), 500);
        assert_eq!(
            verified.source(),
            crate::program_scheduling::TokenFactSource::PreparedTokenIds
        );
        assert_eq!(verified.fallback(), None);
    }

    #[tokio::test]
    async fn exact_mode_mismatch_keeps_the_fingerprint_on_probation() {
        let config = config(TokenCountMode::Exact, 1_000_000);
        let scheduler = ProgramScheduler::new(config);
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "mismatch"]);
        let arrival =
            ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(500, fingerprint)));
        let pending = scheduler.resolve_arrival_tokens("model", arrival);
        let dispatch = scheduler
            .acquire(
                identity("p"),
                pending,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        scheduler.complete(
            &dispatch,
            true,
            false,
            crate::program_scheduling::ProgramUsageObservation {
                prompt_tokens: Some(400),
                completion_tokens: Some(1),
                cached_prompt_tokens: None,
            },
            None,
        );

        let revoked = scheduler.resolve_arrival_tokens("model", arrival);
        assert_eq!(revoked.tokens(), 100);
        assert_eq!(
            revoked.fallback(),
            Some(crate::program_scheduling::TokenFactFallback::FingerprintRevoked)
        );
    }

    #[test]
    fn prepared_token_pool_allowlist_gates_pools() {
        let mut config = ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            ..ProgramSchedulerConfig::default()
        };
        config.token_count_model_pools = vec!["program:model:a".to_string(), "model-c".to_string()];
        let scheduler = ProgramScheduler::new(config);
        assert!(scheduler.prepared_token_pool_allowed("program:model:a"));
        assert!(!scheduler.prepared_token_pool_allowed("program:model:b"));
        assert!(!scheduler.prepared_token_pool_allowed("program:all"));
        // Documented plain model names match their scheduler key too.
        assert!(scheduler.prepared_token_pool_allowed("program:model:model-c"));

        // An empty allowlist keeps the legacy behaviour: every pool may apply
        // a prepared count after verification.
        let open = ProgramScheduler::new(ProgramSchedulerConfig::default());
        assert!(open.prepared_token_pool_allowed("program:model:b"));
        assert!(open.prepared_token_pool_allowed("program:all"));
    }

    #[test]
    fn exact_text_endpoints_are_gated_by_mode_and_allowlist() {
        let mut config = ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            ..ProgramSchedulerConfig::default()
        };
        config.exact_text_endpoints = vec!["/v1/chat/completions".to_string()];
        let scheduler = ProgramScheduler::new(config.clone());
        assert!(scheduler.exact_text_endpoint_enabled("/v1/chat/completions"));
        assert!(!scheduler.exact_text_endpoint_enabled("/v1/completions"));

        let mut estimated = config.clone();
        estimated.token_count_mode = TokenCountMode::Estimated;
        assert!(
            !ProgramScheduler::new(estimated).exact_text_endpoint_enabled("/v1/chat/completions")
        );
    }

    #[tokio::test]
    async fn shadow_mode_records_counterfactual_and_keeps_estimated_admission() {
        let config = config(TokenCountMode::Shadow, 200_000);
        let scheduler = ProgramScheduler::new(config);
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "shadow"]);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(4, Some(TokenFacts::client_token_ids(98_304, fingerprint))),
        );
        let dispatch = scheduler
            .acquire(
                identity("p"),
                context,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        // Decisions keep the estimated demand ...
        assert_eq!(dispatch.context_tokens(), 4);
        assert_eq!(dispatch.context_token_source(), TokenFactSource::Estimated);
        // ... while the prepared counterfactual is visible to operators.
        let snapshot = scheduler.diagnostics();
        let program = &snapshot.ranks[0].programs[0];
        assert_eq!(program.context_tokens, 4);
        assert_eq!(program.shadow_context_tokens, Some(98_304));
    }

    #[tokio::test]
    async fn exact_mode_queues_a_program_the_estimate_would_admit() {
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "queue"]);
        let mut exact_config = config(TokenCountMode::Exact, 150_000);
        exact_config.token_count_verification_settlements = 0;
        let exact_scheduler = ProgramScheduler::new(exact_config);
        let exact_context = exact_scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(4, Some(TokenFacts::client_token_ids(98_304, fingerprint))),
        );
        assert_eq!(exact_context.tokens(), 98_304);
        // The first Program consumes over half the rank capacity ...
        exact_scheduler
            .acquire(
                identity("p"),
                exact_context,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        let second = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            exact_scheduler.acquire(
                identity("q"),
                exact_context,
                std::slice::from_ref(&target()),
                None,
            ),
        )
        .await;
        assert!(
            second.is_err(),
            "exact admission must retain the second Program"
        );

        // ... while the estimated demand admits it immediately.
        let estimated_scheduler = ProgramScheduler::new(config(TokenCountMode::Estimated, 150_000));
        let estimated_context = estimated_scheduler
            .resolve_arrival_tokens("model", ArrivalTokenFacts::estimated_only(4));
        estimated_scheduler
            .acquire(
                identity("p"),
                estimated_context,
                std::slice::from_ref(&target()),
                None,
            )
            .await
            .unwrap();
        let admitted = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            estimated_scheduler.acquire(
                identity("q"),
                estimated_context,
                std::slice::from_ref(&target()),
                None,
            ),
        )
        .await;
        assert!(admitted.is_ok(), "estimated admission should fit easily");
    }

    #[tokio::test]
    async fn revocation_falls_back_to_estimation() {
        let config = config(TokenCountMode::Exact, 100_000);
        let scheduler = ProgramScheduler::new(config);
        let fingerprint = TokenizerFingerprint::from_parts(&["model", "client"]);
        {
            let mut state = scheduler.state.lock();
            scheduler.reconcile_observed_prompt_tokens(
                &mut state,
                "model",
                TokenFactSource::ClientTokenIds,
                Some(fingerprint),
                Some(98_304),
                Some(98_300),
            );
            assert_eq!(state.revoked_token_fingerprints.len(), 1);
        }
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(4, Some(TokenFacts::client_token_ids(98_304, fingerprint))),
        );
        assert_eq!(context.tokens(), 4);
        assert_eq!(
            context.fallback(),
            Some(super::super::TokenFactFallback::FingerprintRevoked)
        );
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod revocation_tests {
    use super::tests::identity;
    use super::*;
    use crate::program_scheduling::*;
    use std::time::Duration;

    #[test]
    fn late_matches_do_not_reverify_revoked_or_saturated_fingerprints() {
        for saturated in [false, true] {
            let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
            let fingerprint = TokenizerFingerprint::from_parts(&["late-match"]);
            let mut state = scheduler.state.lock();
            scheduler.reconcile_observed_prompt_tokens(
                &mut state,
                "model",
                TokenFactSource::PreparedTokenIds,
                Some(fingerprint),
                Some(10),
                Some(11),
            );
            if saturated {
                // An overflow entry is removed from the revoked map, while
                // its entire model pool remains disabled until restart.
                state.revoked_token_fingerprints.remove(&fingerprint);
                state
                    .token_fingerprint_saturated_pools
                    .insert("model".into());
            }
            for index in 0..MAX_VERIFIED_FINGERPRINTS {
                scheduler.reconcile_observed_prompt_tokens(
                    &mut state,
                    "other-model",
                    TokenFactSource::PreparedTokenIds,
                    Some(TokenizerFingerprint::from_parts(&[&format!(
                        "valid-{index}"
                    )])),
                    Some(10),
                    Some(10),
                );
            }
            let verified_before = state.verified_token_fingerprints.clone();
            let sequence_before = state.token_fingerprint_verification_sequence;
            // A previously dispatched attempt reports a match after revocation.
            scheduler.reconcile_observed_prompt_tokens(
                &mut state,
                "model",
                TokenFactSource::PreparedTokenIds,
                Some(fingerprint),
                Some(10),
                Some(10),
            );
            assert!(
                state.verified_token_fingerprints == verified_before,
                "a late match changed the verified LRU after revocation (saturated={saturated})"
            );
            assert_eq!(
                state.token_fingerprint_verification_sequence,
                sequence_before
            );
            drop(state);
            assert_eq!(
                scheduler.diagnostics().verified_token_fingerprints,
                MAX_VERIFIED_FINGERPRINTS
            );
            let context = scheduler.resolve_arrival_tokens(
                "model",
                ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(10, fingerprint))),
            );
            assert_eq!(
                context.fallback(),
                Some(TokenFactFallback::FingerprintRevoked)
            );
        }
    }

    #[test]
    fn revoked_gauge_matches_the_bounded_map_after_overflow() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, || {
            let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
            let mut state = scheduler.state.lock();
            for index in 0..=MAX_REVOKED_FINGERPRINTS {
                let fingerprint = TokenizerFingerprint::from_parts(&[&index.to_string()]);
                scheduler.reconcile_observed_prompt_tokens(
                    &mut state,
                    "model",
                    TokenFactSource::PreparedTokenIds,
                    Some(fingerprint),
                    Some(10),
                    Some(11),
                );
            }
            assert_eq!(
                state.revoked_token_fingerprints.len(),
                MAX_REVOKED_FINGERPRINTS
            );
            assert!(handle.render().lines().any(|line| line == format!(
                "vllm_router_agent_aware_context_token_revoked_fingerprints {MAX_REVOKED_FINGERPRINTS}")));
        });
    }

    #[tokio::test]
    async fn failed_and_duplicate_settlements_do_not_trip_missing_usage_breaker() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            ..Default::default()
        });
        let fingerprint = TokenizerFingerprint::from_parts(&["missing-usage"]);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(10, fingerprint))),
        );
        let targets = [ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: None,
        }];
        let first = scheduler
            .acquire(identity("same"), context, &targets, None)
            .await
            .unwrap();
        let second = scheduler
            .acquire(identity("same"), context, &targets, None)
            .await
            .unwrap();
        let third = scheduler
            .acquire(identity("same"), context, &targets, None)
            .await
            .unwrap();
        scheduler.complete(&first, true, false, None, None);
        for _ in 0..MAX_TOKEN_USAGE_MISSES {
            scheduler.complete(&first, true, false, None, None);
        }
        scheduler.complete(&second, false, false, None, None);
        scheduler.complete(&third, false, false, Some(0), None);
        assert_eq!(scheduler.state.lock().token_usage_misses["model"], 1);
        assert!(scheduler.prepared_token_pool_enabled("model"));
    }

    #[tokio::test]
    async fn missing_usage_breaker_resets_partial_streaks_and_downgrades_retained_facts() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            token_count_verification_settlements: 0,
            ..Default::default()
        });
        let fingerprint = TokenizerFingerprint::from_parts(&["usage"]);
        let arrival =
            ArrivalTokenFacts::new(1000, Some(TokenFacts::prepared_token_ids(10, fingerprint)));
        let context = scheduler.resolve_arrival_tokens("model", arrival);
        let mut state = scheduler.state.lock();
        let handle =
            state
                .runtime
                .retain_request(&identity("waiting"), context, None, Instant::now());
        for observed in [None, None, Some(10)] {
            scheduler.reconcile_observed_prompt_tokens(
                &mut state,
                "model",
                TokenFactSource::PreparedTokenIds,
                Some(fingerprint),
                Some(10),
                observed,
            );
        }
        assert!(!ProgramScheduler::usage_unavailable(&state, "model"));
        for _ in 0..MAX_TOKEN_USAGE_MISSES {
            scheduler.reconcile_observed_prompt_tokens(
                &mut state,
                "model",
                TokenFactSource::PreparedTokenIds,
                Some(fingerprint),
                Some(10),
                None,
            );
        }
        assert_eq!(
            state.runtime.front_request_context(handle.program()),
            Some(1000)
        );
        // A late matching response must not re-enable preparation.
        scheduler.reconcile_observed_prompt_tokens(
            &mut state,
            "model",
            TokenFactSource::PreparedTokenIds,
            Some(fingerprint),
            Some(10),
            Some(10),
        );
        drop(state);
        assert!(!scheduler.prepared_token_pool_enabled("model"));
        assert!(scheduler.prepared_token_pool_enabled("other-model"));
        let context = scheduler.resolve_arrival_tokens("model", arrival);
        assert_eq!(context.tokens(), 1000);
        assert_eq!(
            context.fallback(),
            Some(TokenFactFallback::UsageUnavailable)
        );
        assert_eq!(
            scheduler.diagnostics().usage_unavailable_model_pools,
            ["model"]
        );
    }

    #[test]
    fn missing_usage_tracking_is_bounded_and_fails_closed() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            ..Default::default()
        });
        let mut state = scheduler.state.lock();
        for index in 0..=MAX_TOKEN_USAGE_POOLS {
            scheduler.reconcile_observed_prompt_tokens(
                &mut state,
                &format!("pool-{index}"),
                TokenFactSource::PreparedTokenIds,
                None,
                Some(10),
                None,
            );
        }
        assert_eq!(state.token_usage_misses.len(), MAX_TOKEN_USAGE_POOLS);
        assert!(state.token_usage_tracking_saturated);
        drop(state);
        assert!(!scheduler.prepared_token_pool_enabled("another-pool"));
        assert!(scheduler.diagnostics().usage_tracking_saturated);
    }

    #[tokio::test]
    async fn claimed_dispatch_keeps_its_facts_but_queued_successor_is_revoked() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            token_count_verification_settlements: 0,
            ..Default::default()
        });
        let fingerprint = TokenizerFingerprint::from_parts(&["claimed"]);
        let arrival =
            ArrivalTokenFacts::new(1000, Some(TokenFacts::prepared_token_ids(10, fingerprint)));
        let context = scheduler.resolve_arrival_tokens("model", arrival);
        let targets = [ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: None,
        }];
        // acquire claims an attempt; no transport has sent it in this test.
        let dispatch = scheduler
            .acquire(identity("claimed"), context, &targets, None)
            .await
            .unwrap();
        let mut state = scheduler.state.lock();
        let successor =
            state
                .runtime
                .retain_request(&identity("claimed"), context, None, Instant::now());
        scheduler.reconcile_observed_prompt_tokens(
            &mut state,
            "model",
            TokenFactSource::PreparedTokenIds,
            Some(fingerprint),
            Some(10),
            Some(1000),
        );
        assert!(state.runtime.is_live_attempt(&dispatch));
        assert_eq!(dispatch.context_tokens(), 10);
        assert_eq!(
            state.runtime.front_request_context(successor.program()),
            Some(1000)
        );
        assert_eq!(
            state
                .runtime
                .view(dispatch.program())
                .unwrap()
                .in_flight_requests,
            1
        );
    }

    #[tokio::test]
    async fn revocation_between_resolution_and_retention_uses_the_original_estimate() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            token_count_verification_settlements: 0,
            ..Default::default()
        });
        let fingerprint = TokenizerFingerprint::from_parts(&["race"]);
        let context = scheduler.resolve_arrival_tokens(
            "model",
            ArrivalTokenFacts::new(1000, Some(TokenFacts::prepared_token_ids(10, fingerprint))),
        );
        {
            let mut state = scheduler.state.lock();
            scheduler.reconcile_observed_prompt_tokens(
                &mut state,
                "model",
                TokenFactSource::PreparedTokenIds,
                Some(fingerprint),
                Some(10),
                Some(1000),
            );
        }
        let target = ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: None,
        };
        let dispatch = scheduler
            .acquire(identity("race"), context, &[target], None)
            .await
            .unwrap();
        assert_eq!(dispatch.context_tokens(), 1000);
        assert_eq!(dispatch.context_token_source(), TokenFactSource::Estimated);
        assert_eq!(
            dispatch.context_facts().fallback(),
            Some(TokenFactFallback::FingerprintRevoked)
        );
    }

    #[test]
    fn revoked_or_saturated_facts_recheck_admission_and_update_the_entire_fifo() {
        for global_queue in [false, true] {
            for saturate in [false, true] {
                let mut config = ProgramSchedulerConfig {
                    token_count_mode: TokenCountMode::Exact,
                    token_count_verification_settlements: 0,
                    global_queue,
                    ..Default::default()
                };
                config.progress_ttl.token_capacity = Some(500);
                config.progress_ttl.decode_buffer_tokens = 10;
                let scheduler = ProgramScheduler::new(config);
                let target = ProgramTarget {
                    id: "rank-0".into(),
                    base_url: "http://rank-0".into(),
                    dp_rank: None,
                };
                scheduler.sync_targets("model", &[target]);
                let fingerprint = TokenizerFingerprint::from_parts(&["queued"]);
                let front = scheduler.resolve_arrival_tokens(
                    "model",
                    ArrivalTokenFacts::new(
                        1000,
                        Some(TokenFacts::prepared_token_ids(10, fingerprint)),
                    ),
                );
                let tail = scheduler.resolve_arrival_tokens(
                    "model",
                    ArrivalTokenFacts::new(
                        1200,
                        Some(TokenFacts::prepared_token_ids(12, fingerprint)),
                    ),
                );
                let now = Instant::now();
                let mut state = scheduler.state.lock();
                let handle = state
                    .runtime
                    .retain_request(&identity("queued"), front, None, now);
                let reference = handle.program().clone();
                scheduler.ensure_decision_state(
                    &mut state,
                    &identity("queued"),
                    reference.clone(),
                    front,
                    now,
                    None,
                );
                let plan = scheduler
                    .rank_admission_plan(&state, &reference, "rank-0", now)
                    .unwrap();
                assert!(scheduler.commit_admission(&mut state, &reference, "rank-0", plan, now));
                state
                    .runtime
                    .retain_request(&identity("queued"), tail, None, now);

                // A different pool's retained fact must remain exact on saturation.
                let other_identity = ProgramIdentity::from_request(
                    None,
                    Some(&serde_json::json!({
                        "vllm_xargs":{"agentic_context":{"program_id":"other"}}
                    })),
                    Some("other-model"),
                )
                .unwrap()
                .unwrap();
                let other = state
                    .runtime
                    .retain_request(&other_identity, front, None, now);
                let mismatch = if saturate {
                    for index in 0..MAX_REVOKED_FINGERPRINTS {
                        state.revoked_token_fingerprints.insert(
                            TokenizerFingerprint::from_parts(&[&format!("bad-{index}")]),
                            now,
                        );
                    }
                    TokenizerFingerprint::from_parts(&["overflow"])
                } else {
                    fingerprint
                };
                scheduler.reconcile_observed_prompt_tokens(
                    &mut state,
                    "model",
                    TokenFactSource::PreparedTokenIds,
                    Some(mismatch),
                    Some(10),
                    Some(1000),
                );
                assert_eq!(state.runtime.front_request_context(&reference), Some(1000));
                assert_eq!(
                    state.runtime.view(&reference).unwrap().state,
                    ProgramState::Paused
                );
                assert!(scheduler
                    .rank_admission_plan(&state, &reference, "rank-0", now)
                    .is_none());
                assert_eq!(state.observations["rank-0"].active_program_token_delta, 0.0);
                if saturate {
                    assert!(state.token_fingerprint_saturated_pools.contains("model"));
                    assert_eq!(
                        state.runtime.front_request_context(other.program()),
                        Some(10)
                    );
                }
                drop(state);
                scheduler.cancel_retained_request(&handle);
                let state = scheduler.state.lock();
                assert_eq!(state.runtime.front_request_context(&reference), Some(1200));
            }
        }
    }
    #[tokio::test]
    async fn queued_request_must_stop_using_a_revoked_fact() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            max_active_programs_per_target: 1,
            ..Default::default()
        });
        let targets = [ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: None,
        }];
        let fp = TokenizerFingerprint::from_parts(&["review"]);
        let arrival = ArrivalTokenFacts::new(1000, Some(TokenFacts::prepared_token_ids(10, fp)));
        let probation = scheduler.resolve_arrival_tokens("model", arrival);
        let warm = scheduler
            .acquire(identity("warm"), probation, &targets, None)
            .await
            .unwrap();
        scheduler.complete(&warm, true, true, Some(10), None);
        let blocker = scheduler
            .acquire(
                identity("blocker"),
                scheduler.resolve_arrival_tokens("model", arrival),
                &targets,
                None,
            )
            .await
            .unwrap();
        let exact = scheduler.resolve_arrival_tokens("model", arrival);
        assert_eq!(exact.source(), TokenFactSource::PreparedTokenIds);
        let waiting = scheduler.acquire(identity("waiting"), exact, &targets, None);
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut waiting)
                .await
                .is_err()
        );
        // A real in-flight attempt reveals the mismatch before the queued request is admitted.
        scheduler.complete(&blocker, true, true, Some(1000), None);
        assert_eq!(
            scheduler.resolve_arrival_tokens("model", arrival).source(),
            TokenFactSource::Estimated
        );
        let dispatched = tokio::time::timeout(Duration::from_millis(100), &mut waiting)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            dispatched.context_token_source(),
            TokenFactSource::Estimated,
            "known bad fact remained queued and was dispatched at {} tokens",
            dispatched.context_tokens()
        );
        assert_eq!(dispatched.context_tokens(), 1000);
    }
    #[tokio::test]
    async fn duplicate_settlement_cannot_complete_probation() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            token_count_verification_settlements: 2,
            ..Default::default()
        });
        let fp = TokenizerFingerprint::from_parts(&["review-duplicate"]);
        let arrival = ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(10, fp)));
        let targets = [ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: None,
        }];
        let context = scheduler.resolve_arrival_tokens("model", arrival);
        let first = scheduler
            .acquire(identity("same"), context, &targets, None)
            .await
            .unwrap();
        let second = scheduler
            .acquire(identity("same"), context, &targets, None)
            .await
            .unwrap();
        scheduler.complete(&first, true, false, Some(10), None);
        assert_eq!(scheduler.diagnostics().verified_token_fingerprints, 0);
        assert_eq!(
            scheduler.resolve_arrival_tokens("model", arrival).source(),
            TokenFactSource::Estimated
        );
        scheduler.complete(&first, true, false, Some(10), None);
        assert_eq!(
            scheduler.resolve_arrival_tokens("model", arrival).source(),
            TokenFactSource::Estimated,
            "one real completion plus its duplicate must not satisfy a two-settlement probation"
        );
        assert_eq!(scheduler.diagnostics().verified_token_fingerprints, 0);
        scheduler.complete(&second, true, false, Some(10), None);
        assert_eq!(scheduler.diagnostics().verified_token_fingerprints, 1);
        assert!(scheduler
            .resolve_arrival_tokens("model", arrival)
            .source()
            .is_prepared());
    }

    #[tokio::test]
    async fn duplicate_settlement_cannot_revoke_a_fingerprint() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig {
            token_count_mode: TokenCountMode::Exact,
            ..Default::default()
        });
        let fp = TokenizerFingerprint::from_parts(&["review-stale"]);
        let arrival = ArrivalTokenFacts::new(100, Some(TokenFacts::prepared_token_ids(10, fp)));
        let targets = [ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://rank-0".into(),
            dp_rank: None,
        }];
        let context = scheduler.resolve_arrival_tokens("model", arrival);
        let first = scheduler
            .acquire(identity("same"), context, &targets, None)
            .await
            .unwrap();
        let _second = scheduler
            .acquire(identity("same"), context, &targets, None)
            .await
            .unwrap();
        scheduler.complete(&first, true, false, Some(10), None);
        assert_eq!(scheduler.diagnostics().revoked_token_fingerprints, 0);
        scheduler.complete(&first, true, false, Some(11), None);
        assert_eq!(scheduler.diagnostics().revoked_token_fingerprints, 0,
            "a stale duplicate callback must not mutate another in-flight attempt's verification state");
    }
}
