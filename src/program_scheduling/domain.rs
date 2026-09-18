//! Protocol-neutral Program runtime values shared with Router adapters.

use std::hash::{Hash, Hasher};
use std::time::Instant;

use super::ProgramRequestHints;

/// Whether a Program currently owns logical admission capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramState {
    /// Requests may be forwarded on the current placement.
    Active,
    /// Requests remain in the RequestPool until a resume decision.
    Paused,
}

/// Whether a Program currently needs inference or is between model requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramStatus {
    /// At least one request is in flight or waiting for admission.
    Reasoning,
    /// The Program is between model requests, such as during a tool call.
    Acting,
}

/// Identity of one exact live Program generation.
///
/// A later generation with the same external Program ID must not be mutated by
/// a delayed completion or cancellation from an older generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProgramRef {
    model_pool: String,
    program_id: String,
    generation: u64,
}

impl ProgramRef {
    /// Construct an exact Program reference owned by the runtime.
    pub(crate) fn new(model_pool: String, program_id: String, generation: u64) -> Self {
        Self {
            model_pool,
            program_id,
            generation,
        }
    }

    /// Model pool containing the Program.
    pub fn model_pool(&self) -> &str {
        &self.model_pool
    }

    /// External Program identifier.
    pub fn program_id(&self) -> &str {
        &self.program_id
    }

    /// Monotonic generation assigned when this Program is materialized.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Stable privacy-preserving identifier suitable for diagnostics.
    pub fn redacted_id(&self) -> String {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
}

/// One concrete backend or internal DP rank available to Program scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramTarget {
    /// Stable target identifier, including a DP suffix when applicable.
    pub id: String,
    /// Base HTTP URL used for backend metrics and request forwarding.
    pub base_url: String,
    /// Internal data-parallel rank represented by this target.
    pub dp_rank: Option<usize>,
}

/// Immutable proof that one retained request was admitted to a target.
#[derive(Debug, Clone)]
pub struct ProgramDispatch {
    program: ProgramRef,
    /// Target selected by the accepted admission or resume transition.
    pub target_id: String,
    placement_epoch: u64,
    request_arrived_at: Instant,
    dispatched_at: Instant,
    estimated_context_tokens: usize,
    placement_start_request: bool,
    routing_text: Option<String>,
    request_hints: ProgramRequestHints,
}

impl ProgramDispatch {
    /// Construct a dispatch after admission has atomically committed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        program: ProgramRef,
        target_id: String,
        placement_epoch: u64,
        request_arrived_at: Instant,
        dispatched_at: Instant,
        estimated_context_tokens: usize,
        placement_start_request: bool,
        routing_text: Option<String>,
        request_hints: ProgramRequestHints,
    ) -> Self {
        Self {
            program,
            target_id,
            placement_epoch,
            request_arrived_at,
            dispatched_at,
            estimated_context_tokens,
            placement_start_request,
            routing_text,
            request_hints,
        }
    }

    /// Exact Program generation associated with this request attempt.
    pub fn program(&self) -> &ProgramRef {
        &self.program
    }

    /// Placement epoch used to reject stale completion bookkeeping.
    pub fn placement_epoch(&self) -> u64 {
        self.placement_epoch
    }

    /// Router arrival timestamp used for end-to-end latency accounting.
    pub fn request_arrived_at(&self) -> Instant {
        self.request_arrived_at
    }

    /// Admission timestamp used for RequestPool queue accounting.
    pub fn dispatched_at(&self) -> Instant {
        self.dispatched_at
    }

    /// Arrival-time context estimate for completion fallback.
    pub fn estimated_context_tokens(&self) -> usize {
        self.estimated_context_tokens
    }

    /// Whether this is the first request after admission or resume.
    pub fn placement_start_request(&self) -> bool {
        self.placement_start_request
    }

    /// Request text retained for a deferred Cache-aware binding commit.
    pub fn routing_text(&self) -> Option<&str> {
        self.routing_text.as_deref()
    }

    /// Request-scoped metadata that affected admission and completion TTL.
    pub fn request_hints(&self) -> &ProgramRequestHints {
        &self.request_hints
    }

    /// Stable privacy-preserving Program identifier for logs and metrics.
    pub fn redacted_program_id(&self) -> String {
        self.program.redacted_id()
    }
}

/// Backend token usage collected across streaming and final response paths.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProgramUsageObservation {
    /// Total prompt tokens reported for this request.
    pub prompt_tokens: Option<usize>,
    /// Total completion tokens reported for this request.
    pub completion_tokens: Option<usize>,
    /// Prompt tokens served from the backend prefix cache.
    pub cached_prompt_tokens: Option<usize>,
}

impl ProgramUsageObservation {
    /// Merge a later observation without allowing partial chunks to regress.
    pub fn merge(&mut self, other: Self) {
        self.prompt_tokens = self.prompt_tokens.max(other.prompt_tokens);
        self.completion_tokens = self.completion_tokens.max(other.completion_tokens);
        self.cached_prompt_tokens = self.cached_prompt_tokens.max(other.cached_prompt_tokens);
    }
}

impl From<Option<usize>> for ProgramUsageObservation {
    fn from(prompt_tokens: Option<usize>) -> Self {
        Self {
            prompt_tokens,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_merge_is_monotonic_per_field() {
        let mut usage = ProgramUsageObservation {
            prompt_tokens: Some(100),
            completion_tokens: Some(5),
            cached_prompt_tokens: None,
        };
        usage.merge(ProgramUsageObservation {
            prompt_tokens: Some(90),
            completion_tokens: Some(8),
            cached_prompt_tokens: Some(64),
        });
        assert_eq!(usage.prompt_tokens, Some(100));
        assert_eq!(usage.completion_tokens, Some(8));
        assert_eq!(usage.cached_prompt_tokens, Some(64));
    }

    #[test]
    fn generation_participates_in_exact_reference_identity() {
        let first = ProgramRef::new("model".into(), "program".into(), 1);
        let second = ProgramRef::new("model".into(), "program".into(), 2);
        assert_ne!(first, second);
        assert_ne!(first.redacted_id(), second.redacted_id());
    }
}
