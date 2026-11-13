//! Terminal Engine outcome shared by transport adapters and Program settlement.
//!
//! A transport producer can know more than the client-visible response: the
//! Engine's prompt/completion counts (the SSE `usage` chunk is emitted only for
//! `stream_options.include_usage: true`) and whether the stream ended in an
//! error frame. Program settlement reads this observer so exact-count
//! reconciliation and TTL accounting do not depend on what the client asked to
//! see.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

/// Terminal Engine outcome that the client-visible response may hide.
#[derive(Debug, Default)]
pub struct EngineOutcomeObserver {
    usage: Mutex<Option<(Option<usize>, usize)>>,
    failed: AtomicBool,
}

impl EngineOutcomeObserver {
    /// Record the Engine's own counts. `prompt_tokens` stays `None` when the
    /// Engine did not report `prompt_info`: the Router's prepared count must
    /// not masquerade as Engine usage, or reconciliation would compare it with
    /// itself and never detect tokenizer drift.
    pub fn record_usage(&self, prompt_tokens: Option<usize>, completion_tokens: usize) {
        *self.usage.lock() = Some((prompt_tokens, completion_tokens));
    }

    pub fn record_error(&self) {
        self.failed.store(true, Ordering::Release);
    }

    pub fn usage(&self) -> Option<(Option<usize>, usize)> {
        *self.usage.lock()
    }

    pub fn failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
}
