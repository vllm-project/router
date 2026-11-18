//! Adapt full-attention prefix queries to the existing policy interface.

use crate::kv_index::{KvBlockIndexer, MatchQuery, TierMatch};

impl crate::policies::TieredMatchProvider for KvBlockIndexer {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        self.find_matches(query)
    }
}
