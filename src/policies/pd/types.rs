//! Immutable stage inputs, load snapshots and recommendations shared with PD admission.
use crate::core::Worker;
use crate::policies::{
    AffinityKey, AffinityRecord, PdWorkPrices, PrefillWorkEstimate, RequestHeaders,
    ValidatedWorkPrices,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PdEndpoint {
    pub http_url: String,
    pub dp_rank: Option<usize>,
}

impl PdEndpoint {
    pub(crate) fn for_worker(worker: &dyn Worker) -> Self {
        Self {
            http_url: worker.base_url().to_owned(),
            dp_rank: worker.dp_rank(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Stage {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PdLoadSample {
    pub prefill_requests: usize,
    pub active_prefill_tokens: u64,
    pub unknown_prefill_requests: usize,
    pub pending_decode_requests: usize,
    pub active_decode_requests: usize,
    /// Logical context tokens of Decode requests admitted but not yet
    /// dispatched. Their KV is not on the engine yet, so they are waiting work.
    pub pending_decode_tokens: u64,
    /// Logical context tokens of Decode requests already dispatched. Each
    /// sequence contributes its full prompt length; shared physical prefix
    /// blocks still count once per request: sequences share one copy.
    pub active_decode_tokens: u64,
    /// Admitted Decode requests whose prompt length the Router does not know.
    /// Any candidate holding one is compared by request counts.
    pub unknown_decode_requests: usize,
}

impl PdLoadSample {
    pub(crate) fn occupancy(self, stage: Stage) -> usize {
        match stage {
            Stage::Prefill => self.prefill_requests,
            Stage::Decode => self.pending_decode_requests + self.active_decode_requests,
        }
    }

    /// The unit Decode ranks by, when it can be known.
    ///
    /// `None` as soon as one admitted request on this endpoint has an unknown
    /// prompt length: mixing a known token figure with an unknown one would
    /// compare different units across the pool, and request counts are the
    /// documented fallback.
    pub(crate) fn decode_tokens(self) -> Option<u64> {
        (self.unknown_decode_requests == 0).then(|| {
            self.pending_decode_tokens
                .saturating_add(self.active_decode_tokens)
        })
    }
}

pub(crate) type Loads = HashMap<(Stage, PdEndpoint), PdLoadSample>;

/// Backend-effective tokens prepared outside the admission transaction.
#[derive(Debug, Clone)]
pub struct PdPrefillInput {
    pub prompts: Option<Arc<[Vec<u32>]>>,
    pub(crate) image: Option<crate::kv_index::request_hash::ImagePrefix>,
    pub allow_credit: bool,
    pub fallback_reason: Option<&'static str>,
}

#[derive(Clone, Copy)]
pub(crate) enum StagePricing<'a> {
    Prepared(&'a PdWorkPrices),
    Validated(&'a ValidatedWorkPrices<'a>),
}

/// Token inputs are keyed by the P base URL: each backend renders its own
/// effective input, so different tokenizer/template configurations never mix.
#[derive(Clone)]
pub struct PdRequestContext<'a> {
    pub worker_pairs: Option<&'a HashSet<(PdEndpoint, PdEndpoint)>>,
    pub prefill_inputs: Option<&'a HashMap<String, PdPrefillInput>>,
    pub model_id: Option<&'a str>,
    pub legacy_policy_input: Option<&'a str>,
    pub(crate) prefill_cache: Option<&'a crate::policies::PreparedCacheSelection>,
    pub(crate) decode_cache: Option<&'a crate::policies::PreparedCacheSelection>,
    pub headers: Option<&'a RequestHeaders>,
    /// Stable session identity for the optional kv_aware decode binding.
    /// Resolved at request entry, only when the affinity switch is on;
    /// `None` selects per request on both stages.
    pub affinity_key: Option<AffinityKey>,
    /// Borrowed LoRA name from the request; used for Decode binding only with an eligible affinity key.
    pub lora_name: Option<&'a str>,
    /// Request cache salt scoping the binding digest.
    pub cache_salt: Option<&'a str>,
    /// Prefill work per candidate, validated before admission ranks the workers.
    pub(crate) prefill_pricing: Option<StagePricing<'a>>,
    /// Marginal Decode context per candidate, validated at its own commit.
    pub(crate) decode_pricing: Option<StagePricing<'a>>,
    pub route: &'a str,
    /// True when this admission is a health probe: it takes a pair but starts
    /// no sequence, so it must not reserve Decode occupancy.
    pub probe: bool,
}

impl<'a> StagePricing<'a> {
    /// The caller keeps the guard slot outside the selection callback so it
    /// outlives both the callback and the ledger write.
    pub(crate) fn validate<'b>(
        self,
        workers: &[Arc<dyn Worker>],
        validated: &'b mut Option<ValidatedWorkPrices<'a>>,
    ) -> StagePricing<'b>
    where
        'a: 'b,
    {
        if let Self::Prepared(prices) = self {
            StagePricing::Validated(validated.insert(prices.validate(workers)))
        } else {
            self
        }
    }
}

impl PdRequestContext<'_> {
    pub(crate) fn prefill_work(&self, worker_url: &str) -> Option<PrefillWorkEstimate> {
        match self.prefill_pricing {
            Some(StagePricing::Validated(prices)) => prices.prefill_work(worker_url),
            _ => None,
        }
    }

    pub(crate) fn marginal_decode_context(&self) -> Option<&HashMap<String, u64>> {
        match self.decode_pricing {
            Some(StagePricing::Validated(prices)) => Some(prices.tokens()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StageDecision {
    pub fallback_reason: Option<&'static str>,
    pub credited_prefix_tokens: Option<u64>,
    pub worker_index: usize,
    pub endpoint: PdEndpoint,
    pub score: Option<f64>,
    /// Immutable token work charged to this attempt after KV credit.
    pub new_prefill_tokens: Option<u64>,
    /// How session affinity shaped this stage's choice: bind / sticky_hit /
    /// no_session / capacity_bypass, set only while the switch is on.
    pub affinity_outcome: Option<&'static str>,
    /// Held binding reference for this stage's decision; the admission
    /// attempt releases it on cleanup.
    pub affinity: Option<Arc<AffinityRecord>>,
}

#[derive(Debug, Clone)]
pub struct PdSelection {
    pub prefill: StageDecision,
    pub decode: StageDecision,
}
