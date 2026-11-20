//! Local subscription fencing shared by the supervisor and subscriber.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// All ranks of a worker share `active`; owners of one canonical publisher and
/// exact topic share `strict_rank`. Neither identifies a remote engine lifetime.
/// Lock order is supervisor workers -> strict rank -> indexer. Keep the final
/// rank decision and index mutation together, never holding this lock over await.
#[derive(Clone)]
pub(super) struct SourceLease {
    pub(super) strict_rank: Arc<Mutex<bool>>,
    pub(super) active: Arc<AtomicBool>,
}

impl SourceLease {
    pub(super) fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub(super) fn with_active<T>(&self, operation: impl FnOnce(bool) -> T) -> Option<T> {
        let strict = self.strict_rank.lock();
        self.is_active().then(|| operation(*strict))
    }
}
