//! Retained request storage and per-Program FIFO wakeups.
//!
//! RequestPool owns request blocking data only. It does not select targets,
//! estimate capacity, run retries, or decide Program state transitions.

use super::ProgramRef;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Notify;

/// Stable handle for one retained request attempt.
#[derive(Debug, Clone)]
pub struct ProgramRequestHandle {
    program: ProgramRef,
    request_id: u64,
    notifier: Arc<Notify>,
}

impl ProgramRequestHandle {
    /// Exact Program generation that owns this request.
    pub fn program(&self) -> &ProgramRef {
        &self.program
    }

    /// Wait until the scheduler changes the front request's eligibility.
    pub async fn notified(&self) {
        self.notifier.notified().await;
    }
}

/// Request data retained while admission is blocked.
#[derive(Debug, Clone)]
pub(crate) struct PendingProgramRequest {
    pub(crate) id: u64,
    pub(crate) arrived_at: Instant,
    pub(crate) estimated_context_tokens: usize,
    pub(crate) routing_text: Option<String>,
    notifier: Arc<Notify>,
}

/// Program-scoped FIFO request storage.
#[derive(Debug, Default)]
pub(crate) struct RequestPool {
    requests: HashMap<ProgramRef, VecDeque<PendingProgramRequest>>,
    next_request_id: u64,
    retained_request_count: usize,
}

impl RequestPool {
    /// Insert a request at the tail of its exact Program generation.
    pub(crate) fn retain(
        &mut self,
        program: ProgramRef,
        arrived_at: Instant,
        estimated_context_tokens: usize,
        routing_text: Option<String>,
    ) -> ProgramRequestHandle {
        self.next_request_id = self.next_request_id.wrapping_add(1);
        let request_id = self.next_request_id;
        let notifier = Arc::new(Notify::new());
        self.requests
            .entry(program.clone())
            .or_default()
            .push_back(PendingProgramRequest {
                id: request_id,
                arrived_at,
                estimated_context_tokens,
                routing_text,
                notifier: Arc::clone(&notifier),
            });
        self.retained_request_count = self.retained_request_count.saturating_add(1);
        ProgramRequestHandle {
            program,
            request_id,
            notifier,
        }
    }

    /// Return the front request only when `handle` still names that request.
    #[cfg(test)]
    pub(crate) fn front(&self, handle: &ProgramRequestHandle) -> Option<&PendingProgramRequest> {
        self.requests
            .get(handle.program())
            .and_then(VecDeque::front)
            .filter(|request| request.id == handle.request_id)
    }

    /// Remove and return the front request named by `handle`.
    pub(crate) fn take_front(
        &mut self,
        handle: &ProgramRequestHandle,
    ) -> Option<PendingProgramRequest> {
        let queue = self.requests.get_mut(handle.program())?;
        if queue
            .front()
            .is_none_or(|front| front.id != handle.request_id)
        {
            return None;
        }
        let request = queue.pop_front();
        self.retained_request_count = self.retained_request_count.saturating_sub(1);
        if queue.is_empty() {
            self.requests.remove(handle.program());
        }
        request
    }

    /// Cancel one request regardless of its position in the Program FIFO.
    ///
    /// Returns whether the removed request was the front request so the caller
    /// can update its Program admission queue and wake the replacement.
    pub(crate) fn cancel(&mut self, handle: &ProgramRequestHandle) -> Option<bool> {
        let queue = self.requests.get_mut(handle.program())?;
        let position = queue
            .iter()
            .position(|request| request.id == handle.request_id)?;
        let was_front = position == 0;
        queue.remove(position);
        self.retained_request_count = self.retained_request_count.saturating_sub(1);
        if queue.is_empty() {
            self.requests.remove(handle.program());
        }
        Some(was_front)
    }

    /// Wake the front retained request for one Program generation.
    pub(crate) fn notify_front(&self, program: &ProgramRef) {
        if let Some(request) = self.requests.get(program).and_then(VecDeque::front) {
            request.notifier.notify_one();
        }
    }

    /// Number of retained requests across all Programs.
    pub(crate) fn len(&self) -> usize {
        self.retained_request_count
    }

    /// Whether no request is currently retained.
    pub(crate) fn is_empty(&self) -> bool {
        self.retained_request_count == 0
    }

    /// Number of retained requests for an exact Program generation.
    pub(crate) fn program_len(&self, program: &ProgramRef) -> usize {
        self.requests.get(program).map_or(0, VecDeque::len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(generation: u64) -> ProgramRef {
        ProgramRef::new("model".into(), "program".into(), generation)
    }

    #[test]
    fn retains_fifo_order_per_exact_generation() {
        let mut pool = RequestPool::default();
        let program = program(1);
        let first = pool.retain(program.clone(), Instant::now(), 100, None);
        let second = pool.retain(program.clone(), Instant::now(), 200, Some("text".into()));
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.program_len(&program), 2);
        assert!(pool.front(&second).is_none());
        assert_eq!(pool.front(&first).unwrap().estimated_context_tokens, 100);
        assert_eq!(
            pool.take_front(&first).unwrap().estimated_context_tokens,
            100
        );
        assert_eq!(
            pool.front(&second).unwrap().routing_text.as_deref(),
            Some("text")
        );
        assert_eq!(
            pool.take_front(&second).unwrap().estimated_context_tokens,
            200
        );
        assert!(pool.is_empty());
    }

    #[test]
    fn cancellation_reports_front_without_touching_new_generation() {
        let mut pool = RequestPool::default();
        let old = program(1);
        let replacement = program(2);
        let old_front = pool.retain(old.clone(), Instant::now(), 100, None);
        let old_tail = pool.retain(old.clone(), Instant::now(), 110, None);
        let new_front = pool.retain(replacement.clone(), Instant::now(), 120, None);
        assert_eq!(pool.cancel(&old_tail), Some(false));
        assert_eq!(pool.cancel(&old_front), Some(true));
        assert_eq!(pool.program_len(&old), 0);
        assert_eq!(pool.program_len(&replacement), 1);
        assert_eq!(
            pool.front(&new_front).unwrap().estimated_context_tokens,
            120
        );
    }

    #[tokio::test]
    async fn notification_wakes_only_the_program_front() {
        let mut pool = RequestPool::default();
        let program = program(1);
        let front = pool.retain(program.clone(), Instant::now(), 100, None);
        let tail = pool.retain(program.clone(), Instant::now(), 110, None);
        pool.notify_front(&program);
        tokio::time::timeout(std::time::Duration::from_millis(10), front.notified())
            .await
            .expect("front request should be notified");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(1), tail.notified())
                .await
                .is_err()
        );
    }
}
