//! Adapt index lookups to the routing policy.

use crate::kv_index::layout::align_down;
use crate::kv_index::SourceBinding;
use crate::kv_index::{KvBlockIndexer, Locality, MatchQuery, TierMatch};

impl crate::policies::TieredMatchProvider for KvBlockIndexer {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        self.find_matches(query)
    }
}

impl crate::policies::TieredMatchProvider for SourceBinding {
    fn matches(
        &self,
        tokens: &[u32],
        block_size: u32,
        lora_name: Option<&str>,
        dp_rank: u32,
    ) -> Vec<TierMatch> {
        // Events, ownership and history are bound to one rank. A hit computed
        // from another rank's stream is never evidence for this one.
        if block_size == 0 || dp_rank != self.target.dp_rank {
            return Vec::new();
        }
        // The policy's page size can be coarser than the hash granularity: the
        // engine restores the aligned prefix, so round down instead of dropping
        // a recoverable hit.
        let (tokens, credit) = self.credit_parts(lora_name, tokens, usize::MAX);
        let tokens = align_down(tokens, block_size);
        if tokens == 0 {
            return Vec::new();
        }
        let Ok(blocks) = u32::try_from(tokens / u64::from(block_size)) else {
            return Vec::new();
        };
        // The engine restores at the tier that completes the boundary; the policy
        // weighs that tier by its transfer credit.
        let Some(credit) = credit else {
            return Vec::new();
        };
        vec![TierMatch {
            target: self.target.clone(),
            tier: credit,
            locality: Locality::Local,
            segments: vec![(0, blocks)],
        }]
    }
}
