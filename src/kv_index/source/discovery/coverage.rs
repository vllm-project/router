use metrics::{gauge, Gauge};

use super::lifecycle::LayoutWorkerProvider;

/// One installed worker owns its contribution, even if a retired routing
/// provider or subscriber is still referenced elsewhere.
pub(super) struct IndexCoverage {
    role: &'static str,
    ranks: Gauge,
    total: usize,
    visible: Gauge,
    unavailable: Gauge,
    last_visible: usize,
}

impl IndexCoverage {
    pub(super) fn new(role: &'static str, provider: &LayoutWorkerProvider) -> Self {
        let total = provider.indexes.len();
        let mut coverage = Self {
            role,
            ranks: gauge!("vllm_router_kv_index_ranks", "role" => role),
            total,
            visible: gauge!("vllm_router_kv_index_sources", "state" => "visible"),
            unavailable: gauge!("vllm_router_kv_index_sources", "state" => "unavailable"),
            last_visible: 0,
        };
        coverage.ranks.increment(total as f64);
        coverage.unavailable.increment(total as f64);
        coverage.refresh(provider);
        coverage
    }

    pub(super) fn set_role(&mut self, role: &'static str) {
        if self.role != role {
            self.ranks.decrement(self.total as f64);
            self.ranks = gauge!("vllm_router_kv_index_ranks", "role" => role);
            self.ranks.increment(self.total as f64);
            self.role = role;
        }
    }

    pub(super) fn refresh(&mut self, provider: &LayoutWorkerProvider) {
        let visible = provider.visible_rank_count();
        if visible != self.last_visible {
            let delta = visible as f64 - self.last_visible as f64;
            self.visible.increment(delta);
            self.unavailable.decrement(delta);
            self.last_visible = visible;
        }
    }
}

impl Drop for IndexCoverage {
    fn drop(&mut self) {
        self.ranks.decrement(self.total as f64);
        self.visible.decrement(self.last_visible as f64);
        self.unavailable
            .decrement((self.total - self.last_visible) as f64);
    }
}
