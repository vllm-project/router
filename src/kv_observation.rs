//! Transport-independent prediction and exactly-once accounting for KV routing.
use crate::kv_index::RoutableTarget;
use crate::metrics::RouterMetrics;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Shared budget for JSON observation and SSE framing/reconciliation.
pub(crate) const MAX_USAGE_BYTES: usize = 1024 * 1024;

/// Multi-prompt usage cannot carry a comparable aggregate count under the
/// current backend usage contract. Unknown/untrusted predictions use `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvRoutePrediction {
    Single {
        target: RoutableTarget,
        cached_tokens: u64,
    },
    MultiPrompt {
        target: RoutableTarget,
    },
}

impl KvRoutePrediction {
    pub fn target(&self) -> &RoutableTarget {
        match self {
            Self::Single { target, .. } | Self::MultiPrompt { target } => target,
        }
    }
    pub fn cached_tokens(&self) -> Option<u64> {
        match self {
            Self::Single { cached_tokens, .. } => Some(*cached_tokens),
            Self::MultiPrompt { .. } => None,
        }
    }
}

pub(crate) struct KvRouteAttempt {
    prediction: KvRoutePrediction,
    finished: AtomicBool,
}

impl KvRouteAttempt {
    pub(crate) fn new(prediction: KvRoutePrediction) -> Arc<Self> {
        Arc::new(Self {
            prediction,
            finished: AtomicBool::new(false),
        })
    }

    /// Both colocated and P/D admission use the same transport eligibility.
    pub(crate) fn for_http(
        prediction: Option<KvRoutePrediction>,
        worker_url: &str,
    ) -> Option<Arc<Self>> {
        prediction
            .filter(|_| !crate::backend::is_grpc_url(worker_url))
            .map(Self::new)
    }

    pub(crate) fn record_once(&self, observation: CacheObservation) {
        if self
            .finished
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            record_observation(&self.prediction, observation);
        }
    }
    pub(crate) fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }
}

impl Drop for KvRouteAttempt {
    fn drop(&mut self) {
        self.record_once(CacheObservation::Unavailable(
            UnavailableReason::AttemptCancelled,
        ));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnavailableReason {
    AttemptCancelled,
    BackendError,
    RequestRejected,
    RequestError,
    ResponseError,
    MissingUsage,
    MultiPromptUsageNotAggregate,
    MalformedJson,
    OversizedJson,
    OversizedSseFrame,
    IncompleteSseFrame,
    InvalidUsage,
    MalformedUsage,
    StreamError,
    PdUsageNotComparable,
}

impl UnavailableReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::AttemptCancelled => "attempt_cancelled",
            Self::BackendError => "backend_error",
            Self::RequestRejected => "request_rejected",
            Self::RequestError => "request_error",
            Self::ResponseError => "response_error",
            Self::MissingUsage => "missing_usage",
            Self::MultiPromptUsageNotAggregate => "multi_prompt_usage_not_aggregate",
            Self::MalformedJson => "malformed_json",
            Self::OversizedJson => "oversized_json",
            Self::OversizedSseFrame => "oversized_sse_frame",
            Self::IncompleteSseFrame => "incomplete_sse_frame",
            Self::InvalidUsage => "invalid_usage",
            Self::MalformedUsage => "malformed_usage",
            Self::StreamError => "stream_error",
            Self::PdUsageNotComparable => "pd_usage_not_comparable",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheObservation {
    Observed(u64),
    Unavailable(UnavailableReason),
}

// All exported reason labels come from this closed set, including success.
enum OutcomeReason {
    None,
    Unavailable(UnavailableReason),
}
impl OutcomeReason {
    fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Unavailable(reason) => reason.as_str(),
        }
    }
}

fn record_observation(prediction: &KvRoutePrediction, observation: CacheObservation) {
    let observation = if matches!(prediction, KvRoutePrediction::MultiPrompt { .. })
        && matches!(observation, CacheObservation::Observed(_))
    {
        CacheObservation::Unavailable(UnavailableReason::MultiPromptUsageNotAggregate)
    } else {
        observation
    };
    let (status, reason, actual_cached_tokens) = match observation {
        CacheObservation::Observed(tokens) => ("observed", OutcomeReason::None, Some(tokens)),
        CacheObservation::Unavailable(reason) => {
            ("unavailable", OutcomeReason::Unavailable(reason), None)
        }
    };
    let reason = reason.as_str();
    RouterMetrics::record_kv_route_outcome(
        status,
        reason,
        prediction.cached_tokens(),
        actual_cached_tokens,
    );
    tracing::debug!(status, reason, instance_id = %prediction.target().instance_id,
        dp_rank = prediction.target().dp_rank, predicted_cached_tokens = ?prediction.cached_tokens(),
        actual_cached_tokens, "KV route prediction outcome");
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
pub(crate) mod test_helpers {
    use super::*;
    pub(crate) fn prediction() -> KvRoutePrediction {
        KvRoutePrediction::Single {
            target: RoutableTarget {
                instance_id: "worker-a".into(),
                dp_rank: 0,
            },
            cached_tokens: 32,
        }
    }
    pub(crate) fn render(f: impl FnOnce()) -> String {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::with_local_recorder(&recorder, f);
        handle.render()
    }
    pub(crate) fn has_outcome(text: &str, status: &str, reason: &str) -> bool {
        text.lines().any(|line| {
            line.starts_with("vllm_router_kv_route_outcomes_total{")
                && line.contains(&format!("status=\"{status}\""))
                && line.contains(&format!("reason=\"{reason}\""))
                && line.ends_with(" 1")
        })
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::test_helpers::*;
    use super::*;

    #[test]
    fn only_paired_single_prompt_observations_enter_accuracy_histograms() {
        for multi in [false, true] {
            for actual in [Some(0), Some(32), None] {
                let metrics = render(|| {
                    let prediction = if multi {
                        KvRoutePrediction::MultiPrompt {
                            target: prediction().target().clone(),
                        }
                    } else {
                        KvRoutePrediction::Single {
                            target: prediction().target().clone(),
                            cached_tokens: 0,
                        }
                    };
                    let attempt = KvRouteAttempt::new(prediction);
                    if let Some(actual) = actual {
                        attempt.record_once(CacheObservation::Observed(actual));
                    }
                });
                let reason = if actual.is_none() {
                    "attempt_cancelled"
                } else if multi {
                    "multi_prompt_usage_not_aggregate"
                } else {
                    "none"
                };
                assert!(
                    has_outcome(
                        &metrics,
                        if !multi && actual.is_some() {
                            "observed"
                        } else {
                            "unavailable"
                        },
                        reason
                    ),
                    "{metrics}"
                );
                assert_eq!(
                    metrics.contains("vllm_router_kv_predicted_cached_tokens_count 1"),
                    !multi && actual.is_some(),
                    "{metrics}"
                );
            }
        }
    }

    #[test]
    fn accuracy_pairs_preserve_error_direction() {
        let metrics = render(|| {
            for actual in [16, 32, 48] {
                KvRouteAttempt::new(prediction()).record_once(CacheObservation::Observed(actual));
            }
        });
        for kind in ["predicted", "actual"] {
            assert!(
                metrics.contains(&format!("vllm_router_kv_{kind}_cached_tokens_count 3")),
                "{metrics}"
            );
        }
        for (direction, error) in [("underpredicted", 16), ("exact", 0), ("overpredicted", 16)] {
            assert!(metrics.contains(&format!("vllm_router_kv_cached_token_absolute_error_sum{{direction=\"{direction}\"}} {error}")), "{metrics}");
        }
    }

    #[test]
    fn shared_attempt_finalizes_once_while_retry_owns_a_new_receipt() {
        let metrics = render(|| {
            let attempt = KvRouteAttempt::new(prediction());
            let clone = attempt.clone();
            attempt.record_once(CacheObservation::Unavailable(
                UnavailableReason::RequestError,
            ));
            clone.record_once(CacheObservation::Observed(32));
            assert!(attempt.is_finished());
            KvRouteAttempt::new(prediction()).record_once(CacheObservation::Observed(32));
        });
        assert!(
            has_outcome(&metrics, "unavailable", "request_error"),
            "{metrics}"
        );
        assert!(has_outcome(&metrics, "observed", "none"), "{metrics}");
        assert_eq!(
            metrics
                .lines()
                .filter(|line| line.starts_with("vllm_router_kv_route_outcomes_total{"))
                .count(),
            2
        );
    }

    #[test]
    fn shared_constructor_excludes_grpc_for_both_admission_paths() {
        for (url, eligible) in [
            ("grpc://worker:8000", false),
            ("http://worker:8000", true),
            ("https://worker:8000", true),
        ] {
            assert_eq!(
                KvRouteAttempt::for_http(Some(prediction()), url).is_some(),
                eligible
            );
            assert!(KvRouteAttempt::for_http(None, url).is_none());
        }
    }
}
