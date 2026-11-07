//! Opt-in per-worker admission capacity (RFC #435).
//!
//! `--max-inflight-requests-per-worker` bounds how many router-dispatched
//! requests one regular worker holds at once. A permit is reserved with a
//! compare-and-swap before dispatch, so concurrent arrivals cannot overshoot
//! the limit, and it is released exactly once when it is dropped. The router
//! stores the permit in the request's `RequestLoadGuard`, so the existing
//! body-lifetime ownership (EOF, stream error, client disconnect, retry
//! abandonment) also returns the permit.
//!
//! The counters are owned by this router process. They never depend on worker
//! telemetry, so a stale or missing signal cannot keep a worker blocked, but
//! they also cannot see other router replicas or the engine's own queue.
//! P/D stages enforce the same limit against `PdLoadManager` occupancy, which
//! is already the accounting of record there.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::core::Worker;
use crate::metrics::RouterMetrics;

/// Seconds advertised in `Retry-After`. A lower bound: any finishing request
/// frees a slot, but the router cannot know when that happens.
pub(crate) const ADMISSION_RETRY_AFTER_SECS: u64 = 1;

/// Marks a response the router generated because capacity was exhausted.
/// Such responses are not worker failures and must not be retried locally:
/// retrying would turn a fast rejection into hidden, unbounded waiting.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AdmissionRejected;

pub(crate) fn is_admission_rejection(response: &Response) -> bool {
    response.extensions().get::<AdmissionRejected>().is_some()
}

/// The 429 returned when every eligible worker (or P/D stage) is full.
pub(crate) fn capacity_rejection(router: &'static str, stage: &'static str) -> Response {
    RouterMetrics::record_admission_rejection(router, stage);
    let message = match stage {
        "prefill" => "All eligible prefill workers are at their in-flight capacity",
        "decode" => "All eligible decode workers are at their in-flight capacity",
        _ => "All eligible workers are at their in-flight capacity",
    };
    let mut response = (StatusCode::TOO_MANY_REQUESTS, message).into_response();
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from(ADMISSION_RETRY_AFTER_SECS),
    );
    response.extensions_mut().insert(AdmissionRejected);
    response
}

/// The selected worker filled between the capacity snapshot and the
/// reservation. Unlike [`capacity_rejection`] this stays retryable: the next
/// attempt takes a fresh snapshot and may find another worker with room.
pub(crate) fn lost_reservation() -> Response {
    let mut response = (
        StatusCode::TOO_MANY_REQUESTS,
        "Selected worker reached its in-flight capacity",
    )
        .into_response();
    response.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from(ADMISSION_RETRY_AFTER_SECS),
    );
    response
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AtCapacity;

#[derive(Debug, Default)]
pub(crate) struct WorkerAdmission {
    limit: Option<usize>,
    slots: DashMap<String, Arc<AtomicUsize>>,
}

impl WorkerAdmission {
    pub(crate) fn new(limit: Option<usize>) -> Self {
        Self {
            limit,
            slots: DashMap::new(),
        }
    }

    /// Permits currently held on `worker_url`.
    pub(crate) fn in_flight(&self, worker_url: &str) -> usize {
        self.slots
            .get(worker_url)
            .map_or(0, |slot| slot.load(Ordering::Acquire))
    }

    /// URLs of `workers` with no free permit in this snapshot. Empty when the
    /// feature is disabled. A snapshot only steers selection; the reservation
    /// in [`Self::try_acquire`] is what enforces the limit.
    pub(crate) fn saturated(&self, workers: &[Arc<dyn Worker>]) -> Vec<String> {
        let Some(limit) = self.limit else {
            return Vec::new();
        };
        workers
            .iter()
            .filter(|worker| self.in_flight(worker.url()) >= limit)
            .map(|worker| worker.url().to_string())
            .collect()
    }

    fn slot(&self, worker_url: &str) -> Arc<AtomicUsize> {
        if let Some(slot) = self.slots.get(worker_url) {
            return slot.clone();
        }
        self.slots
            .entry(worker_url.to_string())
            .or_default()
            .clone()
    }

    /// Reserve one permit on `worker_url`. `Ok(None)` when disabled, so the
    /// disabled path allocates and counts nothing.
    pub(crate) fn try_acquire(
        &self,
        worker_url: &str,
    ) -> Result<Option<WorkerAdmissionPermit>, AtCapacity> {
        let Some(limit) = self.limit else {
            return Ok(None);
        };
        let slot = self.slot(worker_url);
        let mut current = slot.load(Ordering::Acquire);
        loop {
            if current >= limit {
                return Err(AtCapacity);
            }
            match slot.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => current = observed,
            }
        }
        RouterMetrics::set_worker_admitted_requests(worker_url, current + 1);
        Ok(Some(WorkerAdmissionPermit {
            slot,
            worker_url: worker_url.to_string(),
        }))
    }

    /// Count a request whose target another admission controller (Program
    /// scheduling) already chose. It is not refused here, which would create
    /// a second, competing control plane, but its occupancy still reduces the
    /// headroom seen by native requests.
    pub(crate) fn acquire_unbounded(&self, worker_url: &str) -> Option<WorkerAdmissionPermit> {
        self.limit?;
        let slot = self.slot(worker_url);
        let held = slot.fetch_add(1, Ordering::AcqRel) + 1;
        RouterMetrics::set_worker_admitted_requests(worker_url, held);
        Some(WorkerAdmissionPermit {
            slot,
            worker_url: worker_url.to_string(),
        })
    }

    /// Drop the counter of a removed worker. Live permits keep their own
    /// handle and still release into it. A worker re-added under the same URL
    /// starts from zero, so it can briefly hold up to twice the limit while
    /// requests admitted before the removal drain; it is never locked out.
    pub(crate) fn forget(&self, worker_url: &str) {
        self.slots.remove(worker_url);
    }
}

/// One admitted request on one worker. Dropping it releases the slot.
#[derive(Debug)]
pub(crate) struct WorkerAdmissionPermit {
    slot: Arc<AtomicUsize>,
    worker_url: String,
}

impl Drop for WorkerAdmissionPermit {
    fn drop(&mut self) {
        let held = self.slot.fetch_sub(1, Ordering::AcqRel) - 1;
        RouterMetrics::set_worker_admitted_requests(&self.worker_url, held);
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn disabled_admission_counts_nothing() {
        let admission = WorkerAdmission::new(None);
        assert!(admission.try_acquire("http://w").unwrap().is_none());
        assert!(admission.acquire_unbounded("http://w").is_none());
        assert_eq!(admission.in_flight("http://w"), 0);
        assert!(admission.slots.is_empty());
    }

    #[test]
    fn permits_stop_at_the_limit_and_release_on_drop() {
        let admission = WorkerAdmission::new(Some(2));
        let first = admission.try_acquire("http://w").unwrap().unwrap();
        let second = admission.try_acquire("http://w").unwrap().unwrap();
        assert_eq!(admission.try_acquire("http://w").unwrap_err(), AtCapacity);
        assert_eq!(admission.in_flight("http://w"), 2);
        // Other workers are independent.
        assert!(admission.try_acquire("http://other").unwrap().is_some());
        drop(first);
        assert_eq!(admission.in_flight("http://w"), 1);
        let third = admission.try_acquire("http://w").unwrap().unwrap();
        drop((second, third));
        assert_eq!(admission.in_flight("http://w"), 0);
    }

    #[test]
    fn concurrent_reservations_never_exceed_the_limit() {
        const LIMIT: usize = 8;
        const THREADS: usize = 64;
        let admission = Arc::new(WorkerAdmission::new(Some(LIMIT)));
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let admission = admission.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    admission.try_acquire("http://w").ok().flatten()
                })
            })
            .collect();
        let permits: Vec<_> = handles
            .into_iter()
            .filter_map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(permits.len(), LIMIT);
        assert_eq!(admission.in_flight("http://w"), LIMIT);
        drop(permits);
        assert_eq!(admission.in_flight("http://w"), 0);
    }

    #[test]
    fn unbounded_permits_count_against_native_headroom() {
        let admission = WorkerAdmission::new(Some(1));
        let program = admission.acquire_unbounded("http://w").unwrap();
        let extra = admission.acquire_unbounded("http://w").unwrap();
        assert_eq!(admission.in_flight("http://w"), 2);
        assert!(admission.try_acquire("http://w").is_err());
        drop((program, extra));
        assert!(admission.try_acquire("http://w").unwrap().is_some());
    }

    #[test]
    fn forgotten_worker_releases_live_permits_without_lockout() {
        let admission = WorkerAdmission::new(Some(1));
        let live = admission.try_acquire("http://w").unwrap().unwrap();
        admission.forget("http://w");
        // Re-added under the same URL: never blocked by the removed counter.
        let readded = admission.try_acquire("http://w").unwrap().unwrap();
        drop(live);
        assert_eq!(admission.in_flight("http://w"), 1);
        drop(readded);
        assert_eq!(admission.in_flight("http://w"), 0);
    }

    #[test]
    fn capacity_rejection_is_marked_non_retryable_and_hints_retry_after() {
        let response = capacity_rejection("regular", "worker");
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        assert!(is_admission_rejection(&response));
        assert!(!is_admission_rejection(&lost_reservation()));
    }
}
