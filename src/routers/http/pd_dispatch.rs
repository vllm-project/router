//! P/D scheduling and legacy observation boundaries, separate from error rendering.
use super::pd_request_error::{
    capture_upstream_error, InternalErrorReason, PdPhase, PdRequestError,
};
use crate::metrics::RouterMetrics;
use std::future::Future;

/// A failure known at headers. Resolve the body only after scheduling, load
/// guards, and legacy metrics have reached their failure boundary.
pub(super) enum PdDispatchError {
    Classified(PdRequestError),
    Http {
        phase: PdPhase,
        response: Box<reqwest::Response>,
    },
}
impl std::fmt::Debug for PdDispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Classified(error) => error.fmt(f),
            Self::Http { phase, response } => f
                .debug_struct("PendingUpstreamError")
                .field("phase", phase)
                .field("status", &response.status().as_u16())
                .finish(),
        }
    }
}
impl From<PdRequestError> for PdDispatchError {
    fn from(error: PdRequestError) -> Self {
        Self::Classified(error)
    }
}
impl PdDispatchError {
    pub(super) fn http(phase: PdPhase, response: reqwest::Response) -> Self {
        Self::Http {
            phase,
            response: Box::new(response),
        }
    }
    pub(super) async fn resolve(self) -> PdRequestError {
        match self {
            Self::Classified(error) => error,
            Self::Http { phase, response } => capture_upstream_error(phase, *response).await,
        }
    }
}

// Exhaustive on internal reasons: adding a reason must decide whether its
// original exit contributes to the legacy counters.
pub(super) fn records_legacy_request(error: &PdRequestError) -> bool {
    match error {
        PdRequestError::Internal(reason) => match reason {
            InternalErrorReason::MoriIoTransferModeMissing
            | InternalErrorReason::SerializePrefillRequest
            | InternalErrorReason::SerializeDecodeRequest
            | InternalErrorReason::SerializeRequest => false,
            InternalErrorReason::SerializeResponse
            | InternalErrorReason::BuildResponse
            | InternalErrorReason::RequestBuilder { .. } => true,
        },
        _ => true,
    }
}

/// Poll both legs, but Prefill's complete validation gates the response. If D
/// completes first its response is held (backpressure, no output buffering).
/// Dropping this future drops both sends; P failure immediately drops pending D.
pub(super) async fn prefill_first<P, D, T>(
    prefill: P,
    decode: D,
) -> Result<(T, reqwest::Response), PdDispatchError>
where
    P: Future<Output = Result<T, PdDispatchError>>,
    D: Future<Output = Result<reqwest::Response, reqwest::Error>>,
{
    tokio::pin!(prefill, decode);
    tokio::select! {
        biased;
        p = &mut prefill => {
            let p = p?;
            let d = decode.await.map_err(|e| PdRequestError::from_reqwest(PdPhase::Decode, e))?;
            Ok((p, d))
        }
        d = &mut decode => {
            let p = prefill.await?;
            Ok((p, d.map_err(|e| PdRequestError::from_reqwest(PdPhase::Decode, e))?))
        }
    }
}

/// Record at the legacy response-header/error boundary, at most once.
/// Dropping an unfinished request does not add a terminal observation.
pub(super) struct RequestMetrics<'a> {
    route: &'a str,
    start: std::time::Instant,
    recorded: bool,
    count_errors: bool,
}
impl<'a> RequestMetrics<'a> {
    pub(super) fn new(route: &'a str) -> Self {
        Self {
            route,
            start: std::time::Instant::now(),
            recorded: false,
            count_errors: true,
        }
    }
    pub(super) fn exclude_legacy_error(&mut self) {
        self.count_errors = false;
    }
    pub(super) fn record_error(&mut self) -> bool {
        self.count_errors && self.record()
    }
    /// Returns true only for the first observation. Response-body failures
    /// must not count the request or its worker error a second time.
    pub(super) fn record(&mut self) -> bool {
        if self.recorded {
            return false;
        }
        self.recorded = true;
        crate::metrics::RouterMetrics::record_pd_request(self.route);
        crate::metrics::RouterMetrics::record_pd_request_duration(self.route, self.start.elapsed());
        true
    }
}

impl RequestMetrics<'_> {
    pub(super) fn record_failure(&mut self, error: &PdDispatchError, prefill: &str, decode: &str) {
        let phase = match error {
            PdDispatchError::Http { phase, .. } => Some(*phase),
            PdDispatchError::Classified(error) => {
                if !records_legacy_request(error) {
                    return;
                }
                error.phase()
            }
        };
        if self.record_error() {
            match phase {
                Some(PdPhase::Prefill) => RouterMetrics::record_pd_prefill_error(prefill),
                Some(PdPhase::Decode) => RouterMetrics::record_pd_decode_error(decode),
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    #[tokio::test]
    async fn failed_prefill_and_future_drop_release_pending_legs() {
        use futures_util::FutureExt;
        struct Guard(Arc<AtomicUsize>);
        impl Drop for Guard {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let drops = Arc::new(AtomicUsize::new(0));
        let guard = Guard(drops.clone());
        let d = futures_util::future::pending::<Result<reqwest::Response, reqwest::Error>>().map(
            move |result| {
                drop(guard);
                result
            },
        );
        let result = prefill_first(
            async { Err::<(), _>(PdRequestError::protocol(PdPhase::Prefill).into()) },
            d,
        )
        .await;
        assert!(matches!(
            result,
            Err(PdDispatchError::Classified(PdRequestError::Protocol {
                phase: PdPhase::Prefill
            }))
        ));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let p_guard = Guard(drops.clone());
        let d_guard = Guard(drops.clone());
        let p = futures_util::future::pending::<Result<(), PdDispatchError>>().map(move |r| {
            drop(p_guard);
            r
        });
        let d = futures_util::future::pending::<Result<reqwest::Response, reqwest::Error>>().map(
            move |r| {
                drop(d_guard);
                r
            },
        );
        let mut request = Box::pin(prefill_first(p, d));
        assert!(futures_util::poll!(&mut request).is_pending());
        drop(request);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }
}
