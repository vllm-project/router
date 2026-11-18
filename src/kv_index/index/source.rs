//! Per-source stream progress and visibility.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Default)]
pub(crate) struct StreamControl {
    visible_generation: AtomicU64,
}

impl StreamControl {
    pub(crate) fn generation(&self) -> u64 {
        self.visible_generation.load(Ordering::Acquire)
    }
}

pub(crate) struct StreamState {
    control: Arc<StreamControl>,
    pub(crate) reset_generation: u64,
    pub(crate) next_seq: Option<u64>,
    pub(crate) lease_id: u64,
    pub(crate) phase: StreamPhase,
    pub(crate) rejection_reported: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamPhase {
    Retired,
    Replaying,
    Live,
}

fn next_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let generation = NEXT.fetch_add(1, Ordering::Relaxed);
    assert_ne!(generation, u64::MAX, "KV stream generation exhausted");
    generation
}

impl StreamState {
    pub(crate) fn new(control: Arc<StreamControl>) -> Self {
        control.visible_generation.store(0, Ordering::Release);
        Self {
            control,
            reset_generation: 0,
            next_seq: None,
            lease_id: 0,
            phase: StreamPhase::Retired,
            rejection_reported: false,
        }
    }

    pub(crate) fn has_control(&self, control: &Arc<StreamControl>) -> bool {
        Arc::ptr_eq(&self.control, control)
    }

    pub(crate) fn visible(&self) -> bool {
        self.reset_generation != 0 && self.control.generation() == self.reset_generation
    }

    pub(crate) fn publish(&self) {
        self.control
            .visible_generation
            .store(self.reset_generation, Ordering::Release);
    }

    pub(crate) fn hide(&self) {
        self.control.visible_generation.store(0, Ordering::Release);
    }

    pub(crate) fn bump_generation(&mut self) {
        let visible = self.visible();
        self.reset_generation = next_generation();
        if visible {
            self.publish();
        }
    }

    pub(crate) fn reset(&mut self, next_seq: Option<u64>, visible: bool) {
        self.bump_generation();
        self.next_seq = next_seq;
        self.phase = if visible {
            StreamPhase::Live
        } else if next_seq.is_some() {
            StreamPhase::Replaying
        } else {
            StreamPhase::Retired
        };
        if visible {
            self.publish();
        } else {
            self.control.visible_generation.store(0, Ordering::Release);
        }
    }

    pub(crate) fn begin_lease(&mut self) -> u64 {
        self.lease_id = next_generation();
        self.rejection_reported = false;
        self.lease_id
    }
}
