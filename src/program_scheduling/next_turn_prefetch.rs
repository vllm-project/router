//! Read-only scoring of next-turn KV prefetch during tool waits.
//!
//! For each wait this module decides what a prefetch policy would have done and,
//! once the next turn arrives, whether that load would have landed in time.
//! Nothing here reaches a worker or changes a scheduling decision: a plan has no
//! engine operation behind it, earns no KV credit, and charges no capacity.

use super::identity::{NextTurnHint, NextTurnHintInput};
use super::prediction_profile::ReturnTimeKey;
use super::request_pool::ProgramRequestHandle;
use super::return_time;
use super::scheduler_state::ProgramSchedulerState;
use super::{ProgramDispatch, ProgramRef, ProgramScheduler, ProgramState};
use crate::config::types::{
    KvLoadCostModel, NextTurnPrefetchConfig, NextTurnPrefetchMode, PredictionMode,
};
use crate::metrics::RouterMetrics;
use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};
use tracing::debug;

/// Why a wait has no plan, as of its latest evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefetchSkip {
    /// The wait ended before the prefix had been idle for `min_idle_seconds`.
    ShortWait,
    /// The wait ended after that floor, but no tick evaluated it in between.
    NotEvaluated,
    PrefixTooSmall,
    ObservationStale,
    NoRestoreEvidence,
    CapacityPressure,
    DemandWaiting,
    BudgetExhausted,
    PredictionUnavailable,
    LowProbability,
}

impl PrefetchSkip {
    fn as_str(self) -> &'static str {
        match self {
            Self::ShortWait => "short_wait",
            Self::NotEvaluated => "not_evaluated",
            Self::PrefixTooSmall => "prefix_too_small",
            Self::ObservationStale => "observation_stale",
            Self::NoRestoreEvidence => "no_restore_evidence",
            Self::CapacityPressure => "capacity_pressure",
            Self::DemandWaiting => "demand_waiting",
            Self::BudgetExhausted => "budget_exhausted",
            Self::PredictionUnavailable => "prediction_unavailable",
            Self::LowProbability => "low_probability",
        }
    }
}

/// What became of one plan. Every plan ends in exactly one outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefetchOutcome {
    /// The next turn arrived inside the useful window, ran on the planned target,
    /// and its usage reports a prefix worth loading.
    InTime,
    /// As `InTime`, but the next turn's usage reports no `cached_tokens`.
    Unverified,
    /// In time, but the next turn's usage shows no prefix worth loading.
    NoPrefix,
    /// The next turn arrived before the load would have finished.
    Late,
    /// No next turn arrived inside the useful window, so the load was wasted.
    Expired,
    /// The load would have finished, but the next turn failed, was cancelled,
    /// ran on another target, or will never come.
    Unused,
    /// The Program or target went away before the load would have finished.
    Cancelled,
}

impl PrefetchOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::InTime => "in_time",
            Self::Unverified => "unverified",
            Self::NoPrefix => "no_prefix",
            Self::Late => "late",
            Self::Expired => "expired",
            Self::Unused => "unused",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ForecastSource {
    /// A fresh `next_turn_hint` on the completed request.
    Hint,
    /// The workflow return-time profile.
    Profile,
}

impl ForecastSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hint => "hint",
            Self::Profile => "profile",
        }
    }
}

/// Forecast that the next turn arrives inside a load's useful window.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Forecast {
    source: ForecastSource,
    probability: f64,
    /// Samples behind the forecast, when known.
    support: Option<u64>,
}

/// One target's state at one evaluation, read from a fresh observation.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TargetFacts {
    /// Referenced blocks only: an idle cached prefix counts as free here.
    kv_cache_usage: f64,
    waiting_requests: usize,
    restored_recently: bool,
    /// Logical capacity free for a Program that holds none, when capacity is configured.
    paused_headroom_tokens: Option<f64>,
}

/// Everything one evaluation reads apart from the forecast.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PlanFacts {
    elapsed_seconds: f64,
    prefix_tokens: usize,
    /// `None` when no fresh observation describes the target.
    target: Option<TargetFacts>,
    speculative_tokens: usize,
    queue_delay_seconds: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PlanChoice {
    forecast: Forecast,
    /// Time from now until the load would have finished, queueing included.
    lead_seconds: f64,
}

/// Longest modeled time, so an extreme configuration cannot overflow an `Instant`.
const MAX_MODELED_SECONDS: f64 = 1e6;

fn modeled_duration(seconds: f64) -> Duration {
    Duration::from_secs_f64(seconds.min(MAX_MODELED_SECONDS))
}

fn load_seconds(model: &KvLoadCostModel, tokens: usize) -> f64 {
    model.intercept_seconds + model.seconds_per_1k_tokens * tokens as f64 / 1000.0
}

/// One evaluation of a wait, cheapest gates first. `forecast` receives the useful
/// window `(start, end]` in seconds since the turn finished.
fn choose_plan(
    config: &NextTurnPrefetchConfig,
    facts: &PlanFacts,
    forecast: impl FnOnce(f64, f64) -> Option<Forecast>,
) -> Result<PlanChoice, PrefetchSkip> {
    let target = facts.target.ok_or(PrefetchSkip::ObservationStale)?;
    if !target.restored_recently {
        return Err(PrefetchSkip::NoRestoreEvidence);
    }
    if target.kv_cache_usage > config.max_kv_cache_usage
        || target
            .paused_headroom_tokens
            .is_some_and(|headroom| headroom < facts.prefix_tokens as f64)
    {
        return Err(PrefetchSkip::CapacityPressure);
    }
    if target.waiting_requests > 0 {
        return Err(PrefetchSkip::DemandWaiting);
    }
    if facts.speculative_tokens.saturating_add(facts.prefix_tokens)
        > config.max_speculative_tokens_per_target
    {
        return Err(PrefetchSkip::BudgetExhausted);
    }
    let lead_seconds =
        facts.queue_delay_seconds + load_seconds(&config.load_cost_model, facts.prefix_tokens);
    let start = facts.elapsed_seconds + lead_seconds;
    let forecast = forecast(start, start + config.validity_seconds)
        .ok_or(PrefetchSkip::PredictionUnavailable)?;
    if forecast.probability < config.min_useful_probability {
        return Err(PrefetchSkip::LowProbability);
    }
    Ok(PlanChoice {
        forecast,
        lead_seconds,
    })
}

/// A fresh hint's forecast for the window `(start, end]`. `None` once the hint is
/// stale: past its expiry, or past the return it expected.
fn hint_forecast(
    hint: &NextTurnHint,
    elapsed_seconds: f64,
    start: f64,
    end: f64,
) -> Option<Forecast> {
    let expected = hint.expected_return.as_secs_f64();
    if expected <= elapsed_seconds
        || hint
            .valid_for
            .is_some_and(|valid_for| elapsed_seconds > valid_for.as_secs_f64())
    {
        return None;
    }
    Some(Forecast {
        source: ForecastSource::Hint,
        probability: if start < expected && expected <= end {
            f64::from(hint.confidence_permille) / 1000.0
        } else {
            0.0
        },
        support: hint.support,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct PrefetchPlan {
    source: ForecastSource,
    tokens: usize,
    start_at: Instant,
    finish_at: Instant,
    expires_at: Instant,
}

impl PrefetchPlan {
    /// Tokens the modeled load would have moved by `now`.
    fn moved_tokens(&self, now: Instant) -> usize {
        let load = self.finish_at.saturating_duration_since(self.start_at);
        let moved = now.saturating_duration_since(self.start_at);
        if moved >= load {
            return self.tokens;
        }
        (self.tokens as f64 * moved.as_secs_f64() / load.as_secs_f64()) as usize
    }
}

#[derive(Debug, Clone, PartialEq)]
enum WaitPhase {
    Undecided(PrefetchSkip),
    Planned(PrefetchPlan),
    /// The request that arrived at this instant ended the wait, inside the plan's
    /// window. Only that request settles the plan: a Program's requests can overlap.
    AwaitingTurn(PrefetchPlan, Instant),
}

/// One Program's wait between a finished turn and the next one's arrival.
#[derive(Debug, Clone)]
pub(crate) struct NextTurnWait {
    started_at: Instant,
    target_id: String,
    prefix_tokens: usize,
    keys: Vec<ReturnTimeKey>,
    hint: Option<NextTurnHint>,
    phase: WaitPhase,
}

/// One modeled load on a target's stream.
#[derive(Debug)]
struct ModeledLoad {
    program: ProgramRef,
    planned_at: Instant,
    start_at: Instant,
    finish_at: Instant,
}

/// One target's modeled load stream, its budget, and its counters.
#[derive(Debug, Default)]
pub(crate) struct NextTurnPrefetchLedger {
    /// Modeled loads, in order; a target serves one at a time.
    load_stream: VecDeque<ModeledLoad>,
    /// Tokens of plans not settled yet. A finished load stays charged until then,
    /// because the idle prefixes it displaced are gone either way.
    pub(crate) speculative_tokens: usize,
    pub(crate) plans: BTreeMap<&'static str, u64>,
    pub(crate) skips: BTreeMap<&'static str, u64>,
    pub(crate) outcomes: BTreeMap<&'static str, u64>,
    pub(crate) outcome_tokens: BTreeMap<&'static str, u64>,
}

impl NextTurnPrefetchLedger {
    fn queue_delay_seconds(&self, now: Instant) -> f64 {
        self.load_stream.back().map_or(0.0, |load| {
            load.finish_at.saturating_duration_since(now).as_secs_f64()
        })
    }

    fn commit(&mut self, now: Instant, program: &ProgramRef, plan: &PrefetchPlan) {
        while self
            .load_stream
            .front()
            .is_some_and(|load| load.finish_at <= now)
        {
            self.load_stream.pop_front();
        }
        self.load_stream.push_back(ModeledLoad {
            program: program.clone(),
            planned_at: now,
            start_at: plan.start_at,
            finish_at: plan.finish_at,
        });
        self.speculative_tokens = self.speculative_tokens.saturating_add(plan.tokens);
    }

    /// A load that stops at `at` frees the stream from then, or from its start if it
    /// had not started, so each load queued behind it starts as soon as the stream is
    /// free and its own plan was made. Returns each moved load's Program and how much
    /// sooner it now starts and finishes.
    fn release_load(&mut self, program: &ProgramRef, at: Instant) -> Vec<(ProgramRef, Duration)> {
        let Some(index) = self
            .load_stream
            .iter()
            .position(|load| load.program == *program)
        else {
            return Vec::new();
        };
        let Some(released) = self.load_stream.remove(index) else {
            return Vec::new();
        };
        let mut free_at = released.start_at.max(at).min(released.finish_at);
        let mut moved = Vec::new();
        for load in self.load_stream.range_mut(index..) {
            let start_at = free_at.max(load.planned_at);
            if start_at >= load.start_at {
                // Nothing after it moves either.
                break;
            }
            let sooner = load.start_at.duration_since(start_at);
            load.start_at = start_at;
            load.finish_at -= sooner;
            free_at = load.finish_at;
            moved.push((load.program.clone(), sooner));
        }
        moved
    }
}

impl ProgramScheduler {
    fn next_turn_prefetch_shadow(&self) -> bool {
        self.config.next_turn_prefetch.mode == NextTurnPrefetchMode::Shadow
    }

    /// Open a wait for a Program that finished a turn and expects another.
    pub(crate) fn begin_next_turn_wait(
        &self,
        state: &mut ProgramSchedulerState,
        dispatch: &ProgramDispatch,
        prefix_tokens: usize,
        now: Instant,
    ) {
        if !self.next_turn_prefetch_shadow() {
            return;
        }
        let hint = match &dispatch.request_hints().next_turn_hint {
            NextTurnHintInput::Absent => None,
            NextTurnHintInput::Usable(hint) => Some(hint.clone()),
            NextTurnHintInput::Ignored(reason) => {
                RouterMetrics::record_next_turn_hint_ignored(reason);
                None
            }
        };
        if prefix_tokens < self.config.next_turn_prefetch.min_prefix_tokens {
            // No later tick can change this, so no wait is kept for it.
            Self::record_skip(
                state,
                &dispatch.target_id,
                PrefetchSkip::PrefixTooSmall,
                false,
            );
            return;
        }
        let Some(decision) = state.decisions.get_mut(dispatch.program()) else {
            return;
        };
        let keys = if self.config.prediction_profile.mode == PredictionMode::Off {
            Vec::new()
        } else {
            return_time::keys_for_decision(
                decision,
                dispatch.program().model_pool(),
                &self.config.prediction_profile,
            )
        };
        decision.next_turn_wait = Some(NextTurnWait {
            started_at: now,
            target_id: dispatch.target_id.clone(),
            prefix_tokens,
            keys,
            hint,
            phase: WaitPhase::Undecided(PrefetchSkip::NotEvaluated),
        });
    }

    /// A new request for the Program ends its wait.
    pub(crate) fn next_turn_arrived(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        arrived_at: Instant,
    ) {
        let Some(decision) = state.decisions.get_mut(program) else {
            return;
        };
        let Some(mut wait) = decision.next_turn_wait.take() else {
            return;
        };
        let waited = arrived_at
            .saturating_duration_since(wait.started_at)
            .as_secs_f64();
        match wait.phase {
            WaitPhase::Undecided(skip) => {
                let config = &self.config.next_turn_prefetch;
                let skip = if skip == PrefetchSkip::NotEvaluated && waited < config.min_idle_seconds
                {
                    PrefetchSkip::ShortWait
                } else {
                    skip
                };
                let fit = self.would_have_fit(wait.prefix_tokens, waited);
                Self::record_skip(state, &wait.target_id, skip, fit);
            }
            WaitPhase::Planned(ref plan) if arrived_at < plan.finish_at => {
                RouterMetrics::record_next_turn_prefetch_margin(
                    &wait.target_id,
                    PrefetchOutcome::Late.as_str(),
                    plan.finish_at.saturating_duration_since(arrived_at),
                );
                Self::record_outcome(state, &wait.target_id, plan, PrefetchOutcome::Late);
                Self::release_load(state, &wait.target_id, program, arrived_at);
            }
            WaitPhase::Planned(ref plan) if arrived_at > plan.expires_at => {
                Self::record_outcome(state, &wait.target_id, plan, PrefetchOutcome::Expired);
            }
            WaitPhase::Planned(plan) => {
                wait.phase = WaitPhase::AwaitingTurn(plan, arrived_at);
                decision.next_turn_wait = Some(wait);
            }
            WaitPhase::AwaitingTurn(..) => decision.next_turn_wait = Some(wait),
        }
    }

    /// The completion of the request that ended the wait knows the target that served
    /// the turn and how much of its prompt came from a cache tier.
    pub(crate) fn settle_next_turn(
        &self,
        state: &mut ProgramSchedulerState,
        dispatch: &ProgramDispatch,
        success: bool,
        cached_prompt_tokens: Option<usize>,
    ) {
        let Some(decision) = state.decisions.get_mut(dispatch.program()) else {
            return;
        };
        let Some(wait) = decision.next_turn_wait.take() else {
            return;
        };
        let (plan, turn_arrived_at) = match &wait.phase {
            WaitPhase::AwaitingTurn(plan, turn) if *turn == dispatch.request_arrived_at() => {
                (plan, *turn)
            }
            _ => {
                decision.next_turn_wait = Some(wait);
                return;
            }
        };
        let min_prefix = self.config.next_turn_prefetch.min_prefix_tokens;
        let (outcome, tokens) = if !success || dispatch.target_id != wait.target_id {
            (PrefetchOutcome::Unused, plan.tokens)
        } else {
            let outcome = match cached_prompt_tokens {
                Some(cached) if cached < min_prefix => (PrefetchOutcome::NoPrefix, plan.tokens),
                Some(cached) => (PrefetchOutcome::InTime, cached.min(plan.tokens)),
                None => (PrefetchOutcome::Unverified, plan.tokens),
            };
            RouterMetrics::record_next_turn_prefetch_margin(
                &wait.target_id,
                outcome.0.as_str(),
                turn_arrived_at.saturating_duration_since(plan.finish_at),
            );
            outcome
        };
        Self::record_outcome_tokens(state, &wait.target_id, plan, outcome, tokens);
    }

    /// A next turn the client gave up on leaves its in-time plan with nothing to serve.
    pub(crate) fn next_turn_cancelled(
        &self,
        state: &mut ProgramSchedulerState,
        handle: &ProgramRequestHandle,
    ) {
        let Some(decision) = state.decisions.get_mut(handle.program()) else {
            return;
        };
        let Some(wait) = decision.next_turn_wait.take() else {
            return;
        };
        match &wait.phase {
            WaitPhase::AwaitingTurn(plan, turn) if *turn == handle.arrived_at() => {
                Self::record_outcome(state, &wait.target_id, plan, PrefetchOutcome::Unused);
            }
            _ => decision.next_turn_wait = Some(wait),
        }
    }

    /// Evaluate every open wait; called from the periodic tick.
    pub(crate) fn evaluate_next_turn_waits(&self, state: &mut ProgramSchedulerState, now: Instant) {
        if !self.next_turn_prefetch_shadow() {
            return;
        }
        let evaluation_started_at = Instant::now();
        let mut open = state
            .decisions
            .iter()
            .filter_map(|(program, decision)| {
                let wait = decision.next_turn_wait.as_ref()?;
                (!matches!(wait.phase, WaitPhase::AwaitingTurn(..)))
                    .then(|| (wait.started_at, program.clone()))
            })
            .collect::<Vec<_>>();
        // Oldest first, so a contended budget serves waits in arrival order.
        open.sort_by_key(|(started_at, _)| *started_at);
        for (_, program) in open {
            let Some(mut wait) = state
                .decisions
                .get_mut(&program)
                .and_then(|decision| decision.next_turn_wait.take())
            else {
                continue;
            };
            if let WaitPhase::Planned(plan) = &wait.phase {
                if now > plan.expires_at {
                    Self::record_outcome(state, &wait.target_id, plan, PrefetchOutcome::Expired);
                    continue;
                }
            } else {
                self.evaluate_wait(state, &program, &mut wait, now);
            }
            if let Some(decision) = state.decisions.get_mut(&program) {
                decision.next_turn_wait = Some(wait);
            }
        }
        RouterMetrics::record_next_turn_prefetch_evaluation(evaluation_started_at.elapsed());
    }

    /// Settle a wait whose Program is being released.
    pub(crate) fn release_next_turn_wait(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        now: Instant,
    ) {
        if let Some(wait) = state
            .decisions
            .get_mut(program)
            .and_then(|decision| decision.next_turn_wait.take())
        {
            Self::close_next_turn_wait(state, program, &wait, now);
        }
    }

    /// Settle every wait on a target that is being removed.
    pub(crate) fn close_next_turn_waits_on_target(
        &self,
        state: &mut ProgramSchedulerState,
        target_id: &str,
        now: Instant,
    ) {
        if !self.next_turn_prefetch_shadow() {
            return;
        }
        let waits = state
            .decisions
            .iter_mut()
            .filter(|(_, decision)| {
                decision
                    .next_turn_wait
                    .as_ref()
                    .is_some_and(|wait| wait.target_id == target_id)
            })
            .filter_map(|(program, decision)| {
                Some((program.clone(), decision.next_turn_wait.take()?))
            })
            .collect::<Vec<_>>();
        for (program, wait) in &waits {
            Self::close_next_turn_wait(state, program, wait, now);
        }
    }

    /// Settle a wait that no next turn will end.
    fn close_next_turn_wait(
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        wait: &NextTurnWait,
        now: Instant,
    ) {
        match &wait.phase {
            WaitPhase::Undecided(skip) => {
                Self::record_skip(state, &wait.target_id, *skip, false);
            }
            WaitPhase::Planned(plan) if now < plan.finish_at => {
                Self::record_outcome_tokens(
                    state,
                    &wait.target_id,
                    plan,
                    PrefetchOutcome::Cancelled,
                    plan.moved_tokens(now),
                );
                Self::release_load(state, &wait.target_id, program, now);
            }
            WaitPhase::Planned(plan) if now > plan.expires_at => {
                Self::record_outcome(state, &wait.target_id, plan, PrefetchOutcome::Expired);
            }
            WaitPhase::Planned(plan) | WaitPhase::AwaitingTurn(plan, _) => {
                Self::record_outcome(state, &wait.target_id, plan, PrefetchOutcome::Unused);
            }
        }
    }

    fn evaluate_wait(
        &self,
        state: &mut ProgramSchedulerState,
        program: &ProgramRef,
        wait: &mut NextTurnWait,
        now: Instant,
    ) {
        let config = &self.config.next_turn_prefetch;
        let elapsed_seconds = now.saturating_duration_since(wait.started_at).as_secs_f64();
        if elapsed_seconds < config.min_idle_seconds {
            return;
        }
        let target = state
            .observations
            .get(&wait.target_id)
            .filter(|observation| self.observation_is_fresh(observation, now))
            .and_then(|observation| {
                let kv_cache_usage = observation.kv_cache_usage?;
                let waiting_requests = observation.waiting_requests?;
                // Read after the fresh usage above: this is what keeps it a lookup,
                // not a scan of every Program, on the scheduler's lock.
                let paused_headroom_tokens = self
                    .config
                    .progress_ttl
                    .token_capacity
                    .filter(|_| {
                        state.runtime.state(program).map(|state| state.0)
                            == Some(ProgramState::Paused)
                    })
                    .map(|capacity| {
                        capacity as f64 * self.config.progress_ttl.low_watermark_ratio
                            - self.target_usage(state, &wait.target_id, now)
                    });
                Some(TargetFacts {
                    kv_cache_usage,
                    waiting_requests,
                    restored_recently: observation.last_restore_at.is_some_and(|restored_at| {
                        now.saturating_duration_since(restored_at).as_secs_f64()
                            <= config.restore_evidence_window_seconds
                    }),
                    paused_headroom_tokens,
                })
            });
        let ledger = state.next_turn_prefetch.get(&wait.target_id);
        let facts = PlanFacts {
            elapsed_seconds,
            prefix_tokens: wait.prefix_tokens,
            target,
            speculative_tokens: ledger.map_or(0, |ledger| ledger.speculative_tokens),
            queue_delay_seconds: ledger.map_or(0.0, |ledger| ledger.queue_delay_seconds(now)),
        };
        let prediction = &self.config.prediction_profile;
        let choice = choose_plan(config, &facts, |start, end| {
            wait.hint
                .as_ref()
                .and_then(|hint| hint_forecast(hint, elapsed_seconds, start, end))
                .or_else(|| {
                    let forecast = state.prediction_profiles.predict_return_window(
                        &wait.keys,
                        elapsed_seconds,
                        start,
                        end,
                        prediction.return_time_min_support,
                        prediction.prior_weight,
                    )?;
                    Some(Forecast {
                        source: ForecastSource::Profile,
                        probability: forecast.probability,
                        support: Some(forecast.support as u64),
                    })
                })
        });
        match choice {
            Err(skip) => wait.phase = WaitPhase::Undecided(skip),
            Ok(choice) => {
                let finish_at = now + modeled_duration(choice.lead_seconds);
                let plan = PrefetchPlan {
                    source: choice.forecast.source,
                    tokens: wait.prefix_tokens,
                    start_at: now + modeled_duration(facts.queue_delay_seconds),
                    finish_at,
                    expires_at: finish_at + modeled_duration(config.validity_seconds),
                };
                let ledger = state
                    .next_turn_prefetch
                    .entry(wait.target_id.clone())
                    .or_default();
                ledger.commit(now, program, &plan);
                *ledger.plans.entry(plan.source.as_str()).or_default() += 1;
                let kv_cache_usage = facts.target.map_or(0.0, |target| target.kv_cache_usage);
                RouterMetrics::record_next_turn_prefetch_plan(
                    &wait.target_id,
                    plan.source.as_str(),
                    kv_cache_usage,
                );
                debug!(
                    event = "next_turn_prefetch_planned",
                    program = %program.redacted_id(),
                    target = %wait.target_id,
                    source = plan.source.as_str(),
                    probability = choice.forecast.probability,
                    support = choice.forecast.support,
                    tokens = plan.tokens,
                    lead_seconds = choice.lead_seconds,
                    provenance = wait
                        .hint
                        .as_ref()
                        .and_then(|hint| hint.provenance.as_deref()),
                    "Next-turn prefetch shadow plan"
                );
                wait.phase = WaitPhase::Planned(plan);
            }
        }
    }

    /// Whether a plan made at the idle floor would have finished before the
    /// wait ended and still been useful when it did.
    fn would_have_fit(&self, prefix_tokens: usize, waited_seconds: f64) -> bool {
        let config = &self.config.next_turn_prefetch;
        let finish = config.min_idle_seconds + load_seconds(&config.load_cost_model, prefix_tokens);
        waited_seconds >= finish && waited_seconds <= finish + config.validity_seconds
    }

    fn record_skip(
        state: &mut ProgramSchedulerState,
        target_id: &str,
        skip: PrefetchSkip,
        would_have_fit: bool,
    ) {
        RouterMetrics::record_next_turn_prefetch_skip(target_id, skip.as_str(), would_have_fit);
        *state
            .next_turn_prefetch
            .entry(target_id.to_string())
            .or_default()
            .skips
            .entry(skip.as_str())
            .or_default() += 1;
    }

    fn record_outcome(
        state: &mut ProgramSchedulerState,
        target_id: &str,
        plan: &PrefetchPlan,
        outcome: PrefetchOutcome,
    ) {
        Self::record_outcome_tokens(state, target_id, plan, outcome, plan.tokens);
    }

    fn record_outcome_tokens(
        state: &mut ProgramSchedulerState,
        target_id: &str,
        plan: &PrefetchPlan,
        outcome: PrefetchOutcome,
        tokens: usize,
    ) {
        RouterMetrics::record_next_turn_prefetch_outcome(
            target_id,
            plan.source.as_str(),
            outcome.as_str(),
            tokens,
        );
        let ledger = state
            .next_turn_prefetch
            .entry(target_id.to_string())
            .or_default();
        ledger.speculative_tokens = ledger.speculative_tokens.saturating_sub(plan.tokens);
        *ledger.outcomes.entry(outcome.as_str()).or_default() += 1;
        *ledger.outcome_tokens.entry(outcome.as_str()).or_default() += tokens as u64;
    }

    /// A load that stopped at `at` gives the rest of its time to the loads queued
    /// behind it, and their plans move with them.
    fn release_load(
        state: &mut ProgramSchedulerState,
        target_id: &str,
        program: &ProgramRef,
        at: Instant,
    ) {
        let Some(ledger) = state.next_turn_prefetch.get_mut(target_id) else {
            return;
        };
        let moved = ledger.release_load(program, at);
        for (program, sooner) in &moved {
            if let Some(NextTurnWait {
                phase: WaitPhase::Planned(plan),
                ..
            }) = state
                .decisions
                .get_mut(program)
                .and_then(|decision| decision.next_turn_wait.as_mut())
            {
                plan.start_at -= *sooner;
                plan.finish_at -= *sooner;
                plan.expires_at -= *sooner;
            }
        }
        if let Some((_, sooner)) = moved.first() {
            debug!(
                event = "next_turn_prefetch_queue_moved_up",
                target = %target_id,
                plans = moved.len(),
                sooner_seconds = sooner.as_secs_f64(),
                "Next-turn prefetch shadow loads moved up"
            );
        }
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::program_scheduling::prediction_profile::PredictionProfileStore;
    use crate::program_scheduling::{
        BackendObservation, ProgramIdentity, ProgramSchedulerConfig, ProgramTarget,
        ProgramUsageObservation,
    };
    use serde_json::json;

    const TOKENS: usize = 8_000;

    fn config() -> NextTurnPrefetchConfig {
        NextTurnPrefetchConfig {
            mode: NextTurnPrefetchMode::Shadow,
            load_cost_model: KvLoadCostModel {
                intercept_seconds: 0.0,
                seconds_per_1k_tokens: 0.125,
            },
            ..NextTurnPrefetchConfig::default()
        }
    }

    fn facts() -> PlanFacts {
        PlanFacts {
            elapsed_seconds: 2.0,
            prefix_tokens: TOKENS,
            target: Some(TargetFacts {
                kv_cache_usage: 0.5,
                waiting_requests: 0,
                restored_recently: true,
                paused_headroom_tokens: None,
            }),
            speculative_tokens: 0,
            queue_delay_seconds: 0.0,
        }
    }

    fn certain(_: f64, _: f64) -> Option<Forecast> {
        Some(Forecast {
            source: ForecastSource::Profile,
            probability: 1.0,
            support: None,
        })
    }

    #[test]
    fn plan_reports_lead_and_the_useful_window() {
        let mut window = None;
        let choice = choose_plan(&config(), &facts(), |start, end| {
            window = Some((start, end));
            certain(start, end)
        })
        .unwrap();
        assert_eq!(choice.lead_seconds, 1.0);
        assert_eq!(window, Some((3.0, 13.0)));
    }

    #[test]
    fn each_gate_refuses_with_its_own_reason() {
        type Change = fn(&mut PlanFacts);
        let cases: [(PrefetchSkip, Change); 5] = [
            (PrefetchSkip::ObservationStale, |facts| facts.target = None),
            (PrefetchSkip::NoRestoreEvidence, |facts| {
                facts.target.as_mut().unwrap().restored_recently = false
            }),
            (PrefetchSkip::CapacityPressure, |facts| {
                facts.target.as_mut().unwrap().kv_cache_usage = 0.91
            }),
            (PrefetchSkip::DemandWaiting, |facts| {
                facts.target.as_mut().unwrap().waiting_requests = 1
            }),
            (PrefetchSkip::BudgetExhausted, |facts| {
                facts.speculative_tokens = 65536 - TOKENS + 1
            }),
        ];
        for (expected, change) in cases {
            let mut facts = facts();
            change(&mut facts);
            assert_eq!(
                choose_plan(&config(), &facts, certain),
                Err(expected),
                "{expected:?}"
            );
        }
    }

    #[test]
    fn paused_program_needs_logical_headroom_for_its_prefix() {
        let mut short = facts();
        short.target.as_mut().unwrap().paused_headroom_tokens = Some(TOKENS as f64 - 1.0);
        assert_eq!(
            choose_plan(&config(), &short, certain),
            Err(PrefetchSkip::CapacityPressure)
        );
        let mut enough = facts();
        enough.target.as_mut().unwrap().paused_headroom_tokens = Some(TOKENS as f64);
        assert!(choose_plan(&config(), &enough, certain).is_ok());
    }

    #[test]
    fn gates_pass_exactly_at_their_limits() {
        let mut at_usage = facts();
        at_usage.target.as_mut().unwrap().kv_cache_usage = 0.9;
        assert!(choose_plan(&config(), &at_usage, certain).is_ok());
        let at_probability = choose_plan(&config(), &facts(), |_, _| {
            Some(Forecast {
                source: ForecastSource::Profile,
                probability: 0.5,
                support: None,
            })
        });
        assert!(at_probability.is_ok());
        let mut at_budget = facts();
        at_budget.speculative_tokens = 65536 - TOKENS;
        assert!(choose_plan(&config(), &at_budget, certain).is_ok());
    }

    #[test]
    fn queue_delay_moves_the_window_later() {
        let mut facts = facts();
        facts.queue_delay_seconds = 4.0;
        let mut window = None;
        let choice = choose_plan(&config(), &facts, |start, end| {
            window = Some((start, end));
            certain(start, end)
        })
        .unwrap();
        assert_eq!(choice.lead_seconds, 5.0);
        assert_eq!(window, Some((7.0, 17.0)));
    }

    #[test]
    fn forecast_gates_come_last() {
        assert_eq!(
            choose_plan(&config(), &facts(), |_, _| None),
            Err(PrefetchSkip::PredictionUnavailable)
        );
        assert_eq!(
            choose_plan(&config(), &facts(), |_, _| Some(Forecast {
                source: ForecastSource::Profile,
                probability: 0.49,
                support: None,
            })),
            Err(PrefetchSkip::LowProbability)
        );
    }

    fn hint(expected: f64, valid_for: Option<f64>, confidence_permille: u16) -> NextTurnHint {
        NextTurnHint {
            provenance: None,
            expected_return: Duration::from_secs_f64(expected),
            confidence_permille,
            support: None,
            valid_for: valid_for.map(Duration::from_secs_f64),
        }
    }

    #[test]
    fn hint_counts_only_inside_the_window_and_while_fresh() {
        let probability = |expected: f64| {
            hint_forecast(&hint(expected, None, 800), 2.0, 3.0, 13.0)
                .unwrap()
                .probability
        };
        assert_eq!(probability(5.0), 0.8);
        assert_eq!(probability(13.0), 0.8, "the window includes its end");
        assert_eq!(probability(3.0), 0.0, "due before the load finishes");
        assert_eq!(probability(14.0), 0.0, "due after the window");
        assert!(hint_forecast(&hint(5.0, Some(1.0), 800), 2.0, 3.0, 13.0).is_none());
        assert!(hint_forecast(&hint(5.0, Some(2.0), 800), 2.0, 3.0, 13.0).is_some());
        assert!(
            hint_forecast(&hint(2.0, None, 800), 2.0, 3.0, 13.0).is_none(),
            "a return that did not come by its time says nothing more"
        );
    }

    fn profile_scheduler() -> ProgramScheduler {
        let mut config = ProgramSchedulerConfig::default();
        config.next_turn_prefetch = self::config();
        config.prediction_profile.return_time_bucket_bounds_seconds = vec![1.0, 2.0, 5.0, 10.0];
        config.prediction_profile.return_time_min_support = 1;
        ProgramScheduler::new(config)
    }

    fn target() -> ProgramTarget {
        ProgramTarget {
            id: "rank-0".into(),
            base_url: "http://worker".into(),
            dp_rank: Some(0),
        }
    }

    fn identity(context: serde_json::Value) -> ProgramIdentity {
        let mut base = json!({"program_id": "program", "task_id": null, "expected_resume": true});
        base.as_object_mut()
            .unwrap()
            .extend(context.as_object().unwrap().clone());
        ProgramIdentity::from_request(
            None,
            Some(&json!({"vllm_xargs": {"agentic_context": base}})),
            Some("model"),
        )
        .unwrap()
        .unwrap()
    }

    fn observe(scheduler: &ProgramScheduler, usage: f64, restored_total: f64) {
        let targets = scheduler.all_targets();
        let epoch = scheduler.begin_observation(&targets);
        scheduler.apply_observations(
            epoch,
            targets.into_iter().map(|target| BackendObservation {
                target_id: target.id,
                base_url: target.base_url,
                dp_rank: target.dp_rank,
                kv_cache_usage: Some(usage),
                running_requests: Some(0),
                waiting_requests: Some(0),
                external_kv_prompt_tokens_total: Some(restored_total),
                observed_at: Instant::now(),
            }),
        );
    }

    /// Two observations, so the restore counter moves and the target has evidence.
    fn observe_restoring(scheduler: &ProgramScheduler, usage: f64) {
        observe(scheduler, usage, 100.0);
        observe(scheduler, usage, 200.0);
    }

    fn usage(cached: Option<usize>) -> ProgramUsageObservation {
        ProgramUsageObservation {
            prompt_tokens: Some(TOKENS),
            completion_tokens: Some(10),
            cached_prompt_tokens: cached,
        }
    }

    async fn finished_turn(
        scheduler: &ProgramScheduler,
        context: serde_json::Value,
    ) -> ProgramDispatch {
        let dispatch = scheduler
            .acquire(identity(context), TOKENS, &[target()], None)
            .await
            .unwrap();
        scheduler.complete(&dispatch, true, false, usage(None), None);
        dispatch
    }

    fn evaluate_after(scheduler: &ProgramScheduler, seconds: f64) -> Instant {
        let now = Instant::now() + Duration::from_secs_f64(seconds);
        let mut state = scheduler.state.lock();
        scheduler.evaluate_next_turn_waits(&mut state, now);
        now
    }

    fn outcomes(scheduler: &ProgramScheduler) -> BTreeMap<&'static str, u64> {
        scheduler.state.lock().next_turn_prefetch["rank-0"]
            .outcomes
            .clone()
    }

    fn skips(scheduler: &ProgramScheduler) -> BTreeMap<&'static str, u64> {
        scheduler.state.lock().next_turn_prefetch["rank-0"]
            .skips
            .clone()
    }

    fn phase(scheduler: &ProgramScheduler, program: &ProgramRef) -> Option<WaitPhase> {
        scheduler.state.lock().decisions[program]
            .next_turn_wait
            .as_ref()
            .map(|wait| wait.phase.clone())
    }

    fn hinted() -> serde_json::Value {
        json!({"next_turn_hint": {
            "version": "1", "expected_return_seconds": 5.0, "provenance": "test"
        }})
    }

    fn hinted_program(name: &str) -> serde_json::Value {
        json!({"program_id": name,
               "next_turn_hint": {"version": "1", "expected_return_seconds": 5.0}})
    }

    /// A planned wait, and the instant its plan was made.
    async fn planned(scheduler: &ProgramScheduler) -> (ProgramDispatch, Instant) {
        let dispatch = finished_turn(scheduler, hinted()).await;
        observe_restoring(scheduler, 0.5);
        let planned_at = evaluate_after(scheduler, 2.0);
        (dispatch, planned_at)
    }

    /// Move a plan back in time, so its load finished a second ago.
    fn load_finished(scheduler: &ProgramScheduler, program: &ProgramRef) {
        let mut state = scheduler.state.lock();
        let wait = state
            .decisions
            .get_mut(program)
            .unwrap()
            .next_turn_wait
            .as_mut()
            .unwrap();
        let WaitPhase::Planned(plan) = &mut wait.phase else {
            panic!("not planned");
        };
        let shift =
            plan.finish_at.saturating_duration_since(Instant::now()) + Duration::from_secs(1);
        plan.start_at -= shift;
        plan.finish_at -= shift;
        plan.expires_at -= shift;
    }

    async fn next_turn(scheduler: &ProgramScheduler) -> ProgramDispatch {
        scheduler
            .acquire(identity(json!({})), TOKENS, &[target()], None)
            .await
            .unwrap()
    }

    /// A planned wait whose load has finished, and the request that then ended it, still running.
    async fn arrived_in_time(scheduler: &ProgramScheduler) -> (ProgramDispatch, ProgramDispatch) {
        let (dispatch, _) = planned(scheduler).await;
        load_finished(scheduler, dispatch.program());
        let turn = next_turn(scheduler).await;
        (dispatch, turn)
    }

    #[tokio::test]
    async fn off_mode_keeps_no_state() {
        let scheduler = ProgramScheduler::new(ProgramSchedulerConfig::default());
        let dispatch = finished_turn(&scheduler, hinted()).await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 2.0);
        let state = scheduler.state.lock();
        assert!(state.decisions[dispatch.program()].next_turn_wait.is_none());
        assert!(state.next_turn_prefetch.is_empty());
    }

    #[tokio::test]
    async fn a_new_request_through_acquire_ends_the_wait() {
        let scheduler = profile_scheduler();
        let dispatch = finished_turn(&scheduler, hinted()).await;
        let _next = scheduler
            .acquire(identity(json!({})), TOKENS, &[target()], None)
            .await
            .unwrap();
        assert_eq!(phase(&scheduler, dispatch.program()), None);
        assert_eq!(skips(&scheduler).get("short_wait"), Some(&1));
    }

    #[tokio::test]
    async fn a_failed_turn_or_a_final_turn_opens_no_wait() {
        let scheduler = profile_scheduler();
        let failed = scheduler
            .acquire(identity(hinted()), TOKENS, &[target()], None)
            .await
            .unwrap();
        scheduler.complete(&failed, false, false, usage(None), None);
        assert_eq!(phase(&scheduler, failed.program()), None);

        let last = scheduler
            .acquire(
                identity(json!({"program_id": "last", "expected_resume": false})),
                TOKENS,
                &[target()],
                None,
            )
            .await
            .unwrap();
        scheduler.complete(&last, true, true, usage(None), None);
        let state = scheduler.state.lock();
        assert!(!state.decisions.contains_key(last.program()));
        assert!(state.next_turn_prefetch.is_empty());
    }

    #[tokio::test]
    async fn a_turn_ending_in_a_capacity_pause_opens_a_wait_only_if_it_succeeded() {
        for (success, opens) in [(true, true), (false, false)] {
            let scheduler = profile_scheduler();
            let dispatch = scheduler
                .acquire(identity(hinted()), TOKENS, &[target()], None)
                .await
                .unwrap();
            scheduler
                .state
                .lock()
                .decisions
                .get_mut(dispatch.program())
                .unwrap()
                .pause_when_idle = true;
            scheduler.complete(&dispatch, success, false, usage(None), None);
            assert_eq!(phase(&scheduler, dispatch.program()).is_some(), opens);
        }
    }

    #[tokio::test]
    async fn a_prefix_below_the_floor_opens_no_wait() {
        let scheduler = profile_scheduler();
        let finish = |tokens: usize| {
            let scheduler = &scheduler;
            async move {
                let dispatch = scheduler
                    .acquire(identity(json!({})), tokens, &[target()], None)
                    .await
                    .unwrap();
                scheduler.complete(
                    &dispatch,
                    true,
                    false,
                    ProgramUsageObservation {
                        prompt_tokens: Some(tokens),
                        completion_tokens: Some(10),
                        cached_prompt_tokens: None,
                    },
                    None,
                );
                dispatch
            }
        };
        let small = finish(1023).await;
        assert_eq!(phase(&scheduler, small.program()), None);
        assert_eq!(skips(&scheduler).get("prefix_too_small"), Some(&1));
        let enough = finish(1024).await;
        assert!(phase(&scheduler, enough.program()).is_some());
    }

    #[tokio::test]
    async fn a_short_wait_is_not_evaluated_and_a_longer_unticked_one_says_so() {
        for (seconds, reason) in [(0.5, "short_wait"), (1.5, "not_evaluated")] {
            let scheduler = profile_scheduler();
            let dispatch = finished_turn(&scheduler, hinted()).await;
            observe_restoring(&scheduler, 0.5);
            evaluate_after(&scheduler, 0.5);
            {
                let mut state = scheduler.state.lock();
                scheduler.next_turn_arrived(
                    &mut state,
                    dispatch.program(),
                    Instant::now() + Duration::from_secs_f64(seconds),
                );
            }
            assert_eq!(skips(&scheduler).get(reason), Some(&1), "{reason}");
            assert_eq!(skips(&scheduler).len(), 1);
        }
    }

    #[tokio::test]
    async fn an_in_time_arrival_settles_on_its_own_completion() {
        let scheduler = profile_scheduler();
        let (dispatch, turn) = arrived_in_time(&scheduler).await;
        assert!(matches!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::AwaitingTurn(..))
        ));
        assert!(outcomes(&scheduler).is_empty(), "settled by the completion");
        scheduler.complete(&turn, true, false, usage(Some(TOKENS)), None);
        assert_eq!(outcomes(&scheduler).get("in_time"), Some(&1));
    }

    #[tokio::test]
    async fn only_the_request_that_ended_the_wait_settles_its_plan() {
        let scheduler = profile_scheduler();
        let (dispatch, turn) = arrived_in_time(&scheduler).await;
        {
            // The window closes while that request runs, and a later one finishes first.
            let mut state = scheduler.state.lock();
            let wait = state
                .decisions
                .get_mut(dispatch.program())
                .unwrap()
                .next_turn_wait
                .as_mut()
                .unwrap();
            let WaitPhase::AwaitingTurn(plan, _) = &mut wait.phase else {
                panic!("not awaiting");
            };
            plan.expires_at = Instant::now() - Duration::from_millis(1);
        }
        let later = next_turn(&scheduler).await;
        scheduler.complete(&later, true, false, usage(Some(TOKENS)), None);
        assert!(outcomes(&scheduler).is_empty());
        assert!(matches!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::AwaitingTurn(..))
        ));
        scheduler.complete(&turn, true, false, usage(Some(TOKENS)), None);
        assert_eq!(outcomes(&scheduler).get("in_time"), Some(&1));
        assert_eq!(outcomes(&scheduler).len(), 1);
    }

    #[tokio::test]
    async fn the_outcome_follows_what_the_next_turn_reported() {
        for (cached, outcome, tokens) in [
            (Some(5_000), "in_time", 5_000),
            (Some(1024), "in_time", 1024),
            (Some(TOKENS + 500), "in_time", TOKENS as u64),
            (Some(1023), "no_prefix", TOKENS as u64),
            (None, "unverified", TOKENS as u64),
        ] {
            let scheduler = profile_scheduler();
            let (_, turn) = arrived_in_time(&scheduler).await;
            scheduler.complete(&turn, true, false, usage(cached), None);
            let state = scheduler.state.lock();
            let ledger = &state.next_turn_prefetch["rank-0"];
            assert_eq!(ledger.outcomes.get(outcome), Some(&1), "{cached:?}");
            assert_eq!(ledger.outcome_tokens.get(outcome), Some(&tokens));
        }
    }

    #[tokio::test]
    async fn a_failed_next_turn_leaves_the_load_unused() {
        let scheduler = profile_scheduler();
        let (_, turn) = arrived_in_time(&scheduler).await;
        scheduler.complete(
            &turn,
            false,
            false,
            ProgramUsageObservation::default(),
            None,
        );
        assert_eq!(outcomes(&scheduler).get("unused"), Some(&1));
    }

    #[tokio::test]
    async fn a_next_turn_on_another_target_leaves_the_load_unused() {
        let scheduler = profile_scheduler();
        let (dispatch, turn) = arrived_in_time(&scheduler).await;
        {
            let mut state = scheduler.state.lock();
            let wait = state
                .decisions
                .get_mut(dispatch.program())
                .unwrap()
                .next_turn_wait
                .as_mut()
                .unwrap();
            wait.target_id = "rank-1".to_string();
        }
        scheduler.complete(&turn, true, false, usage(Some(TOKENS)), None);
        let state = scheduler.state.lock();
        assert_eq!(
            state.next_turn_prefetch["rank-1"].outcomes.get("unused"),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn a_cancelled_next_turn_settles_the_plan_at_once() {
        let scheduler = profile_scheduler();
        let (dispatch, planned_at) = planned(&scheduler).await;
        let handle = {
            let mut state = scheduler.state.lock();
            let handle = state.runtime.retain_request(
                &identity(json!({})),
                TOKENS,
                None,
                planned_at + Duration::from_secs(3),
            );
            scheduler.next_turn_arrived(
                &mut state,
                dispatch.program(),
                planned_at + Duration::from_secs(3),
            );
            handle
        };
        assert!(outcomes(&scheduler).is_empty());
        scheduler.cancel_retained_request(&handle);
        assert_eq!(outcomes(&scheduler).get("unused"), Some(&1));
        assert_eq!(phase(&scheduler, dispatch.program()), None);
    }

    #[tokio::test]
    async fn only_cancelling_the_request_that_ended_the_wait_settles_its_plan() {
        let scheduler = profile_scheduler();
        let (dispatch, planned_at) = planned(&scheduler).await;
        let identity = identity(json!({}));
        let (turn_at, later_at) = (
            planned_at + Duration::from_secs(3),
            planned_at + Duration::from_secs(4),
        );
        let (turn, later) = {
            let mut state = scheduler.state.lock();
            let turn = state
                .runtime
                .retain_request(&identity, TOKENS, None, turn_at);
            scheduler.next_turn_arrived(&mut state, dispatch.program(), turn_at);
            let later = state
                .runtime
                .retain_request(&identity, TOKENS, None, later_at);
            scheduler.next_turn_arrived(&mut state, dispatch.program(), later_at);
            (turn, later)
        };
        scheduler.cancel_retained_request(&later);
        assert!(outcomes(&scheduler).is_empty());
        assert!(matches!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::AwaitingTurn(..))
        ));
        scheduler.cancel_retained_request(&turn);
        assert_eq!(outcomes(&scheduler).get("unused"), Some(&1));
        assert_eq!(phase(&scheduler, dispatch.program()), None);
    }

    #[tokio::test]
    async fn the_useful_window_includes_both_of_its_ends() {
        for (seconds, expired) in [(1.0, false), (11.0, false), (11.001, true)] {
            let scheduler = profile_scheduler();
            let (dispatch, planned_at) = planned(&scheduler).await;
            {
                let mut state = scheduler.state.lock();
                scheduler.next_turn_arrived(
                    &mut state,
                    dispatch.program(),
                    planned_at + Duration::from_secs_f64(seconds),
                );
            }
            if expired {
                assert_eq!(outcomes(&scheduler).get("expired"), Some(&1));
            } else {
                assert!(phase(&scheduler, dispatch.program()).is_some());
            }
        }
    }

    #[tokio::test]
    async fn an_arrival_before_the_load_finishes_is_late_and_gives_its_budget_back() {
        let scheduler = profile_scheduler();
        let (dispatch, planned_at) = planned(&scheduler).await;
        let mut state = scheduler.state.lock();
        assert_eq!(
            state.next_turn_prefetch["rank-0"].speculative_tokens,
            TOKENS
        );
        scheduler.next_turn_arrived(
            &mut state,
            dispatch.program(),
            planned_at + Duration::from_millis(500),
        );
        assert!(state.decisions[dispatch.program()].next_turn_wait.is_none());
        let ledger = &state.next_turn_prefetch["rank-0"];
        assert_eq!(ledger.outcomes.get("late"), Some(&1));
        assert_eq!(ledger.speculative_tokens, 0);
        assert_eq!(ledger.queue_delay_seconds(planned_at), 0.0);
    }

    #[tokio::test]
    async fn a_finished_load_stays_charged_until_its_plan_settles() {
        let scheduler = profile_scheduler();
        let (dispatch, planned_at) = planned(&scheduler).await;
        let charged = |scheduler: &ProgramScheduler| {
            scheduler.state.lock().next_turn_prefetch["rank-0"].speculative_tokens
        };
        {
            let mut state = scheduler.state.lock();
            // Well past the 1 s load, and still inside the useful window.
            scheduler.evaluate_next_turn_waits(&mut state, planned_at + Duration::from_secs(5));
        }
        assert_eq!(charged(&scheduler), TOKENS);
        load_finished(&scheduler, dispatch.program());
        let turn = next_turn(&scheduler).await;
        assert_eq!(charged(&scheduler), TOKENS, "the turn has not finished yet");
        scheduler.complete(&turn, true, false, usage(Some(TOKENS)), None);
        assert_eq!(charged(&scheduler), 0);
    }

    #[tokio::test]
    async fn an_expired_plan_gives_its_budget_back() {
        let scheduler = profile_scheduler();
        let (_, planned_at) = planned(&scheduler).await;
        let mut state = scheduler.state.lock();
        scheduler.evaluate_next_turn_waits(&mut state, planned_at + Duration::from_secs(12));
        assert_eq!(state.next_turn_prefetch["rank-0"].speculative_tokens, 0);
    }

    #[tokio::test]
    async fn a_cancelled_plan_counts_only_what_its_load_had_moved() {
        // The load runs for 1 s from the plan, so half of it is done at 0.5 s.
        for (seconds, moved) in [(0.0, 0_u64), (0.5, TOKENS as u64 / 2)] {
            let scheduler = profile_scheduler();
            let (dispatch, planned_at) = planned(&scheduler).await;
            let mut state = scheduler.state.lock();
            scheduler.release_next_turn_wait(
                &mut state,
                dispatch.program(),
                planned_at + Duration::from_secs_f64(seconds),
            );
            let ledger = &state.next_turn_prefetch["rank-0"];
            assert_eq!(ledger.outcomes.get("cancelled"), Some(&1));
            assert_eq!(ledger.outcome_tokens.get("cancelled"), Some(&moved));
            assert_eq!(ledger.speculative_tokens, 0);
        }
    }

    #[test]
    fn a_queued_load_has_moved_nothing_until_it_starts() {
        let now = Instant::now();
        let plan = PrefetchPlan {
            source: ForecastSource::Profile,
            tokens: TOKENS,
            start_at: now + Duration::from_secs(2),
            finish_at: now + Duration::from_secs(4),
            expires_at: now + Duration::from_secs(14),
        };
        assert_eq!(plan.moved_tokens(now + Duration::from_secs(1)), 0);
        assert_eq!(plan.moved_tokens(now + Duration::from_secs(3)), TOKENS / 2);
        assert_eq!(plan.moved_tokens(now + Duration::from_secs(9)), TOKENS);
        let instant = PrefetchPlan {
            start_at: now,
            finish_at: now,
            ..plan
        };
        assert_eq!(instant.moved_tokens(now), TOKENS);
    }

    #[tokio::test]
    async fn a_plan_cancelled_behind_another_load_has_moved_nothing() {
        let scheduler = profile_scheduler();
        finished_turn(&scheduler, hinted_program("first")).await;
        let second = finished_turn(&scheduler, hinted_program("second")).await;
        observe_restoring(&scheduler, 0.5);
        let planned_at = evaluate_after(&scheduler, 2.0);
        let mut state = scheduler.state.lock();
        // The first load takes 1 s, so the second has not started at 0.5 s.
        scheduler.release_next_turn_wait(
            &mut state,
            second.program(),
            planned_at + Duration::from_millis(500),
        );
        let ledger = &state.next_turn_prefetch["rank-0"];
        assert_eq!(ledger.outcomes.get("cancelled"), Some(&1));
        assert_eq!(ledger.outcome_tokens.get("cancelled"), Some(&0));
    }

    #[tokio::test]
    async fn a_plan_past_its_window_expires_on_the_tick_and_ignores_a_later_return() {
        let scheduler = profile_scheduler();
        let (dispatch, planned_at) = planned(&scheduler).await;
        {
            let mut state = scheduler.state.lock();
            scheduler.evaluate_next_turn_waits(&mut state, planned_at + Duration::from_secs(12));
            assert_eq!(
                state.next_turn_prefetch["rank-0"].outcomes.get("expired"),
                Some(&1)
            );
            scheduler.next_turn_arrived(
                &mut state,
                dispatch.program(),
                planned_at + Duration::from_secs(20),
            );
        }
        assert_eq!(outcomes(&scheduler).len(), 1);
        assert_eq!(outcomes(&scheduler).get("expired"), Some(&1));
    }

    #[tokio::test]
    async fn a_second_wait_queues_behind_the_first_on_the_same_target() {
        let scheduler = profile_scheduler();
        let first = finished_turn(&scheduler, hinted_program("first")).await;
        let second = finished_turn(&scheduler, hinted_program("second")).await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 2.0);
        let finish = |program: &ProgramRef| match phase(&scheduler, program) {
            Some(WaitPhase::Planned(plan)) => plan.finish_at,
            other => panic!("not planned: {other:?}"),
        };
        assert!(finish(first.program()) < finish(second.program()));
        let gap = finish(second.program()).saturating_duration_since(finish(first.program()));
        assert_eq!(
            gap,
            Duration::from_secs(1),
            "the load stream serves one transfer at a time, oldest wait first"
        );
    }

    /// Every level of the return-time profile, seeded with returns at `seconds`.
    async fn profile_returning_at(scheduler: &ProgramScheduler, seconds: f64) {
        let dispatch = finished_turn(scheduler, json!({"program_id": "seed"})).await;
        let keys = phase_keys(scheduler, dispatch.program());
        let mut state = scheduler.state.lock();
        for _ in 0..4 {
            state.prediction_profiles.observe_return(&keys, seconds);
        }
    }

    fn phase_keys(scheduler: &ProgramScheduler, program: &ProgramRef) -> Vec<ReturnTimeKey> {
        scheduler.state.lock().decisions[program]
            .next_turn_wait
            .as_ref()
            .unwrap()
            .keys
            .clone()
    }

    fn plan_source(scheduler: &ProgramScheduler, program: &ProgramRef) -> Option<ForecastSource> {
        match phase(scheduler, program) {
            Some(WaitPhase::Planned(plan)) => Some(plan.source),
            _ => None,
        }
    }

    #[tokio::test]
    async fn the_profile_forecasts_when_no_hint_is_fresh() {
        let scheduler = profile_scheduler();
        profile_returning_at(&scheduler, 6.0).await;
        let unhinted = finished_turn(&scheduler, json!({"program_id": "unhinted"})).await;
        let stale = finished_turn(
            &scheduler,
            json!({"program_id": "stale", "next_turn_hint": {
                "version": "1", "expected_return_seconds": 6.0, "valid_for_seconds": 1.0
            }}),
        )
        .await;
        let past = finished_turn(
            &scheduler,
            json!({"program_id": "past", "next_turn_hint": {
                "version": "1", "expected_return_seconds": 1.5
            }}),
        )
        .await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 2.0);
        for program in [&unhinted, &stale, &past] {
            assert_eq!(
                plan_source(&scheduler, program.program()),
                Some(ForecastSource::Profile)
            );
        }
    }

    #[tokio::test]
    async fn a_fresh_hint_outranks_the_profile_both_ways() {
        let scheduler = profile_scheduler();
        profile_returning_at(&scheduler, 6.0).await;
        let agrees = finished_turn(
            &scheduler,
            json!({"program_id": "agrees", "next_turn_hint": {
                "version": "1", "expected_return_seconds": 6.0
            }}),
        )
        .await;
        let refuses = finished_turn(
            &scheduler,
            json!({"program_id": "refuses", "next_turn_hint": {
                "version": "1", "expected_return_seconds": 100.0
            }}),
        )
        .await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 2.0);
        assert_eq!(
            plan_source(&scheduler, agrees.program()),
            Some(ForecastSource::Hint)
        );
        assert_eq!(
            phase(&scheduler, refuses.program()),
            Some(WaitPhase::Undecided(PrefetchSkip::LowProbability)),
            "the profile would plan, but the hint says the return is not due"
        );
    }

    #[tokio::test]
    async fn no_forecast_leaves_the_wait_undecided_and_counts_the_reason_once() {
        let scheduler = profile_scheduler();
        let dispatch = finished_turn(&scheduler, json!({})).await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 2.0);
        evaluate_after(&scheduler, 2.5);
        {
            let mut state = scheduler.state.lock();
            scheduler.next_turn_arrived(
                &mut state,
                dispatch.program(),
                Instant::now() + Duration::from_secs(5),
            );
        }
        assert_eq!(skips(&scheduler).len(), 1);
        assert_eq!(skips(&scheduler).get("prediction_unavailable"), Some(&1));
    }

    #[tokio::test]
    async fn a_turn_with_parallel_requests_opens_its_wait_only_when_all_finish() {
        let scheduler = profile_scheduler();
        let first = scheduler
            .acquire(identity(json!({})), TOKENS, &[target()], None)
            .await
            .unwrap();
        let second = scheduler
            .acquire(identity(json!({})), TOKENS, &[target()], None)
            .await
            .unwrap();
        scheduler.complete(&first, true, false, usage(None), None);
        assert_eq!(phase(&scheduler, first.program()), None);
        scheduler.complete(&second, true, false, usage(None), None);
        assert!(phase(&scheduler, first.program()).is_some());
    }

    #[tokio::test]
    async fn a_removed_target_closes_its_waits_and_its_ledger() {
        let scheduler = profile_scheduler();
        planned(&scheduler).await;
        finished_turn(&scheduler, json!({"program_id": "undecided"})).await;
        let other = ProgramTarget {
            id: "rank-1".into(),
            base_url: "http://worker-1".into(),
            dp_rank: Some(0),
        };
        scheduler.sync_targets("model", &[other]);
        let state = scheduler.state.lock();
        assert!(!state.next_turn_prefetch.contains_key("rank-0"));
        assert!(state
            .decisions
            .values()
            .all(|decision| decision.next_turn_wait.is_none()));
    }

    #[tokio::test]
    async fn release_settles_a_wait_by_where_its_load_had_got_to() {
        for (seconds, outcome) in [(0.5, "cancelled"), (5.0, "unused"), (12.0, "expired")] {
            let scheduler = profile_scheduler();
            let (dispatch, planned_at) = planned(&scheduler).await;
            let mut state = scheduler.state.lock();
            scheduler.release_next_turn_wait(
                &mut state,
                dispatch.program(),
                planned_at + Duration::from_secs_f64(seconds),
            );
            assert_eq!(
                state.next_turn_prefetch["rank-0"].outcomes.get(outcome),
                Some(&1),
                "{outcome}"
            );
            assert!(state.decisions[dispatch.program()].next_turn_wait.is_none());
        }
    }

    #[tokio::test]
    async fn release_program_settles_the_wait() {
        let scheduler = profile_scheduler();
        let (dispatch, _) = planned(&scheduler).await;
        let mut state = scheduler.state.lock();
        scheduler.release_program(&mut state, dispatch.program(), Instant::now());
        assert!(!state.decisions.contains_key(dispatch.program()));
        assert_eq!(
            state.next_turn_prefetch["rank-0"]
                .outcomes
                .values()
                .sum::<u64>(),
            1
        );
    }

    #[tokio::test]
    async fn release_of_an_undecided_wait_keeps_its_reason() {
        let scheduler = profile_scheduler();
        let dispatch = finished_turn(&scheduler, json!({})).await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 2.0);
        let mut state = scheduler.state.lock();
        scheduler.release_next_turn_wait(&mut state, dispatch.program(), Instant::now());
        assert_eq!(
            state.next_turn_prefetch["rank-0"]
                .skips
                .get("prediction_unavailable"),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn the_tick_evaluates_waits() {
        let scheduler = profile_scheduler();
        let dispatch = finished_turn(&scheduler, hinted()).await;
        {
            let mut state = scheduler.state.lock();
            let wait = state
                .decisions
                .get_mut(dispatch.program())
                .unwrap()
                .next_turn_wait
                .as_mut()
                .unwrap();
            wait.started_at -= Duration::from_secs(2);
        }
        scheduler.tick();
        assert_eq!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::Undecided(PrefetchSkip::ObservationStale))
        );
    }

    #[tokio::test]
    async fn stale_observations_and_old_restores_refuse_a_plan() {
        let scheduler = profile_scheduler();
        let dispatch = finished_turn(&scheduler, hinted()).await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 3.5);
        assert_eq!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::Undecided(PrefetchSkip::ObservationStale)),
            "older than three metric intervals"
        );
        let shift_restore = |change: fn(Instant) -> Instant| {
            let mut state = scheduler.state.lock();
            let observation = state.observations.get_mut("rank-0").unwrap();
            observation.last_restore_at = observation.last_restore_at.map(change);
        };
        shift_restore(|restored_at| restored_at - Duration::from_secs(61));
        evaluate_after(&scheduler, 1.5);
        assert_eq!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::Undecided(PrefetchSkip::NoRestoreEvidence))
        );
        shift_restore(|restored_at| restored_at + Duration::from_secs(5));
        evaluate_after(&scheduler, 1.5);
        assert!(matches!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::Planned(_))
        ));
    }

    fn capacity_scheduler(mode: NextTurnPrefetchMode) -> ProgramScheduler {
        let mut config = ProgramSchedulerConfig::default();
        config.next_turn_prefetch = self::config();
        config.next_turn_prefetch.mode = mode;
        config.progress_ttl.token_capacity = Some(TOKENS + 200);
        ProgramScheduler::new(config)
    }

    #[tokio::test]
    async fn only_a_paused_program_is_held_to_logical_capacity() {
        // Until the TTL window matures a finished Program pauses at once.
        for (kv_cache_usage, planned) in [(0.0, true), (0.5, false)] {
            let scheduler = capacity_scheduler(NextTurnPrefetchMode::Shadow);
            let dispatch = finished_turn(&scheduler, hinted()).await;
            observe_restoring(&scheduler, kv_cache_usage);
            evaluate_after(&scheduler, 2.0);
            assert_eq!(
                scheduler
                    .state
                    .lock()
                    .runtime
                    .state(dispatch.program())
                    .map(|state| state.0),
                Some(ProgramState::Paused)
            );
            let phase = phase(&scheduler, dispatch.program()).unwrap();
            if planned {
                assert!(matches!(phase, WaitPhase::Planned(_)), "{phase:?}");
            } else {
                assert_eq!(phase, WaitPhase::Undecided(PrefetchSkip::CapacityPressure));
            }
        }
        // An explicit TTL keeps the Program active: its prefix is already accounted.
        let scheduler = capacity_scheduler(NextTurnPrefetchMode::Shadow);
        let mut context = hinted();
        context["kv_retention_ttl_seconds"] = json!(60.0);
        let dispatch = finished_turn(&scheduler, context).await;
        observe_restoring(&scheduler, 0.5);
        evaluate_after(&scheduler, 2.0);
        assert_eq!(
            scheduler
                .state
                .lock()
                .runtime
                .state(dispatch.program())
                .map(|state| state.0),
            Some(ProgramState::Active)
        );
        assert!(matches!(
            phase(&scheduler, dispatch.program()),
            Some(WaitPhase::Planned(_))
        ));
    }

    #[tokio::test]
    async fn restore_evidence_needs_the_counter_to_move_and_survives_an_engine_restart() {
        let scheduler = profile_scheduler();
        finished_turn(&scheduler, hinted()).await;
        let restored_at = || scheduler.state.lock().observations["rank-0"].last_restore_at;
        observe(&scheduler, 0.5, 100.0);
        assert!(
            restored_at().is_none(),
            "a first sample has no earlier value"
        );
        observe(&scheduler, 0.5, 100.0);
        assert!(restored_at().is_none(), "the counter did not move");
        observe(&scheduler, 0.5, 50.0);
        assert!(restored_at().is_none(), "a restarted engine only rebases");
        observe(&scheduler, 0.5, 60.0);
        assert!(restored_at().is_some());
    }

    /// Everything admission and capacity repair read, apart from the shadow's own state.
    fn scheduling_facts(scheduler: &ProgramScheduler, programs: &[&ProgramRef]) -> String {
        let state = scheduler.state.lock();
        let now = Instant::now();
        let queued: Vec<_> = state
            .rank_queues
            .values()
            .map(|queue| queue.len())
            .collect();
        let programs: Vec<_> = programs
            .iter()
            .map(|program| {
                let decision = state.decisions.get(*program);
                (
                    state.runtime.state(program),
                    state.runtime.placement(program).map(str::to_string),
                    decision.map(|decision| decision.ttl_deadline.is_some()),
                    decision.and_then(|decision| decision.last_target.clone()),
                    decision.map(|decision| decision.estimated_context_tokens),
                )
            })
            .collect();
        format!(
            "{:.0} {queued:?} {:?} {programs:?}",
            scheduler.target_usage(&state, "rank-0", now),
            state.runtime.retained_request_count(),
        )
    }

    #[tokio::test]
    async fn shadow_changes_no_decision_under_contention() {
        let off = capacity_scheduler(NextTurnPrefetchMode::Off);
        let shadow = capacity_scheduler(NextTurnPrefetchMode::Shadow);
        let context = |name: &str| {
            let mut context = hinted();
            context["program_id"] = json!(name);
            context["kv_retention_ttl_seconds"] = json!(60.0);
            context
        };
        let mut rounds = Vec::new();
        for scheduler in [&off, &shadow] {
            // A holds the only capacity through its tool wait, while a plan exists.
            let first = finished_turn(scheduler, context("a")).await;
            observe_restoring(scheduler, 0.0);
            evaluate_after(scheduler, 2.0);
            scheduler.tick();
            let during_wait = scheduling_facts(scheduler, &[first.program()]);
            // B cannot fit beside A while A keeps its seat.
            let blocked = tokio::time::timeout(
                Duration::from_millis(50),
                scheduler.acquire(identity(context("b")), TOKENS, &[target()], None),
            )
            .await;
            assert!(blocked.is_err(), "B must wait for capacity");
            let after_block = scheduling_facts(scheduler, &[first.program()]);
            // A returns and finishes: the same turn order, whatever the shadow scored.
            let next = scheduler
                .acquire(identity(context("a")), TOKENS, &[target()], None)
                .await
                .unwrap();
            scheduler.complete(&next, true, false, usage(Some(TOKENS)), None);
            scheduler.tick();
            let after_turn = scheduling_facts(scheduler, &[first.program()]);
            rounds.push((during_wait, after_block, after_turn));
        }
        assert_eq!(rounds[0], rounds[1]);
        assert_eq!(
            shadow.state.lock().next_turn_prefetch["rank-0"]
                .plans
                .get("hint"),
            Some(&1),
            "the shadow really planned"
        );
    }

    #[tokio::test]
    async fn diagnostics_show_the_ledger_only_when_shadow_ran() {
        let off = ProgramScheduler::new(ProgramSchedulerConfig::default());
        finished_turn(&off, hinted()).await;
        let snapshot = serde_json::to_value(off.diagnostics()).unwrap();
        assert_eq!(snapshot["next_turn_prefetch_mode"], "off");
        assert!(snapshot["ranks"][0].get("next_turn_prefetch").is_none());

        let shadow = profile_scheduler();
        planned(&shadow).await;
        let snapshot = serde_json::to_value(shadow.diagnostics()).unwrap();
        assert_eq!(snapshot["next_turn_prefetch_mode"], "shadow");
        let ledger = &snapshot["ranks"][0]["next_turn_prefetch"];
        assert_eq!(ledger["plans"]["hint"], 1);
        assert_eq!(ledger["speculative_tokens"], TOKENS);
    }

    #[test]
    fn would_have_fit_spans_the_useful_window() {
        let scheduler = profile_scheduler();
        // Idle floor 1 s, load 1 s, validity 10 s.
        assert!(!scheduler.would_have_fit(TOKENS, 1.9));
        assert!(scheduler.would_have_fit(TOKENS, 2.0));
        assert!(scheduler.would_have_fit(TOKENS, 12.0));
        assert!(!scheduler.would_have_fit(TOKENS, 12.1));
    }

    #[test]
    fn window_store_is_empty_without_keys() {
        let store = PredictionProfileStore::empty(vec![1.0], vec![64], 16, 2);
        assert!(store
            .predict_return_window(&[], 0.0, 0.0, 1.0, 1, 1.0)
            .is_none());
    }

    fn returning_at_8s(name: &str) -> serde_json::Value {
        json!({"program_id": name,
               "next_turn_hint": {"version": "1", "expected_return_seconds": 8.0}})
    }

    /// Three waits planned on one tick: their 1 s loads finish 1, 2 and 3 s after it.
    async fn three_queued_loads(scheduler: &ProgramScheduler) -> (Vec<ProgramDispatch>, Instant) {
        let mut dispatches = Vec::new();
        for name in ["first", "second", "third"] {
            dispatches.push(finished_turn(scheduler, returning_at_8s(name)).await);
        }
        observe_restoring(scheduler, 0.5);
        (dispatches, evaluate_after(scheduler, 2.0))
    }

    fn plan_in(state: &ProgramSchedulerState, program: &ProgramRef) -> PrefetchPlan {
        match state.decisions[program]
            .next_turn_wait
            .as_ref()
            .map(|wait| &wait.phase)
        {
            Some(WaitPhase::Planned(plan)) => plan.clone(),
            other => panic!("not planned: {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_load_that_will_not_run_moves_the_loads_behind_it_up() {
        let scheduler = profile_scheduler();
        let (programs, planned_at) = three_queued_loads(&scheduler).await;
        let (second, third) = (programs[1].program(), programs[2].program());
        let mut state = scheduler.state.lock();
        assert_eq!(
            plan_in(&state, third).finish_at,
            planned_at + Duration::from_secs(3)
        );
        // The second load has not started at 0.5 s, so the third takes its place.
        scheduler.release_next_turn_wait(
            &mut state,
            second,
            planned_at + Duration::from_millis(500),
        );
        assert_eq!(
            plan_in(&state, third).finish_at,
            planned_at + Duration::from_secs(2)
        );
        scheduler.next_turn_arrived(&mut state, third, planned_at + Duration::from_millis(2500));
        assert!(matches!(
            state.decisions[third]
                .next_turn_wait
                .as_ref()
                .map(|wait| &wait.phase),
            Some(WaitPhase::AwaitingTurn(..))
        ));
        assert_eq!(
            state.next_turn_prefetch["rank-0"].outcomes.get("late"),
            None
        );
        assert_eq!(
            state.next_turn_prefetch["rank-0"].queue_delay_seconds(planned_at),
            2.0
        );
    }

    #[tokio::test]
    async fn a_late_load_gives_the_loads_behind_it_only_the_time_it_had_left() {
        let scheduler = profile_scheduler();
        let (programs, planned_at) = three_queued_loads(&scheduler).await;
        let mut state = scheduler.state.lock();
        // The first load is three quarters done when its turn arrives.
        scheduler.next_turn_arrived(
            &mut state,
            programs[0].program(),
            planned_at + Duration::from_millis(750),
        );
        assert_eq!(
            state.next_turn_prefetch["rank-0"].outcomes.get("late"),
            Some(&1)
        );
        let second = plan_in(&state, programs[1].program());
        assert_eq!(
            (second.start_at, second.finish_at),
            (
                planned_at + Duration::from_millis(750),
                planned_at + Duration::from_millis(1750)
            )
        );
        let third = plan_in(&state, programs[2].program());
        assert_eq!(third.finish_at, planned_at + Duration::from_millis(2750));
        assert_eq!(
            third.expires_at,
            third.finish_at + Duration::from_secs(10),
            "its window moves too"
        );
    }

    #[test]
    fn a_moved_load_starts_no_sooner_than_its_own_plan_was_made() {
        let t = Instant::now();
        let program = |name: &str| ProgramRef::new("model".into(), name.into(), 1);
        let plan = |start_at: Instant| PrefetchPlan {
            source: ForecastSource::Profile,
            tokens: TOKENS,
            start_at,
            finish_at: start_at + Duration::from_secs(1),
            expires_at: start_at + Duration::from_secs(11),
        };
        let mut ledger = NextTurnPrefetchLedger::default();
        ledger.commit(t, &program("a"), &plan(t));
        ledger.commit(
            t + Duration::from_millis(750),
            &program("b"),
            &plan(t + Duration::from_secs(1)),
        );
        assert!(ledger.release_load(&program("unknown"), t).is_empty());
        // The stream is free from 0.25 s, but the second plan was made at 0.75 s.
        assert_eq!(
            ledger.release_load(&program("a"), t + Duration::from_millis(250)),
            vec![(program("b"), Duration::from_millis(250))]
        );
        assert_eq!(ledger.queue_delay_seconds(t), 1.75);
        assert!(
            ledger
                .release_load(&program("b"), t + Duration::from_secs(1))
                .is_empty(),
            "nothing behind it"
        );
        assert_eq!(ledger.queue_delay_seconds(t), 0.0);
    }
}
