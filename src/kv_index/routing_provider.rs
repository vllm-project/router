//! Adapt index lookups to the routing policy.

use crate::kv_hash::SequenceHash;
use crate::kv_index::layout::align_down;
use crate::kv_index::request_hash::BlockKeyGenerator;
use crate::kv_index::storage::PageEvidence;
use crate::kv_index::SourceBinding;
use crate::kv_index::{KvBlockIndexer, Locality, MatchQuery, TierMatch};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use foldhash::fast::RandomState;

impl crate::policies::TieredMatchProvider for KvBlockIndexer {
    fn find_tiered_matches(&self, query: &MatchQuery) -> Vec<TierMatch> {
        self.find_matches(query)
    }
}

impl crate::policies::TieredMatchProvider for SourceBinding {
    fn target_scoped(&self) -> bool {
        true
    }

    fn matches(
        &self,
        tokens: &[u32],
        block_size: u32,
        lora_name: Option<&str>,
        dp_rank: u32,
    ) -> Vec<TierMatch> {
        self.matches_with_cache(
            tokens,
            block_size,
            lora_name,
            dp_rank,
            &mut RequestHashCache::default(),
        )
    }

    fn matches_with_cache<'a>(
        &self,
        tokens: &'a [u32],
        block_size: u32,
        lora_name: Option<&str>,
        dp_rank: u32,
        cache: &mut RequestHashCache<'a>,
    ) -> Vec<TierMatch> {
        // Events, ownership and history are bound to one rank. A hit computed
        // from another rank's stream is never evidence for this one.
        if block_size == 0 || dp_rank != self.target.dp_rank {
            return Vec::new();
        }
        if self.groups.len() == 1
            && self.groups[0].kind == crate::kv_index::GroupRule::FullAttention
            && self.replay_tokens == 0
            && self.tail_drop_group.is_none()
        {
            let Some((mut claims, version, generation)) =
                self.page_snapshot(lora_name, tokens, cache)
            else {
                return Vec::new();
            };
            let hits = crate::kv_index::layout::full_attention_matches(
                &self.target,
                &self.groups[0],
                claims.pop().expect("one group in the snapshot"),
                self.keys.block_size(),
                block_size,
                self.reusable_prompt_cap(tokens.len()) as u64,
            );
            if !self.snapshot_is_current(&version, generation) {
                return Vec::new();
            }
            return hits;
        }
        // The policy's page size can be coarser than the hash granularity: the
        // engine restores the aligned prefix, so round down instead of dropping
        // a recoverable hit.
        let (tokens, credit) = self.credit_parts(lora_name, tokens, usize::MAX, cache);
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

/// Identity of an immutable prompt slice within a selection.
#[derive(Clone, Copy)]
pub(crate) struct PromptKey<'a>(pub(crate) &'a [u32]);

impl PartialEq for PromptKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0)
    }
}

impl Eq for PromptKey<'_> {}

impl Hash for PromptKey<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
        self.0.len().hash(state);
    }
}

/// Prepared engine hashes shared across candidates in one routing selection.
#[derive(Default)]
pub struct RequestHashCache<'a> {
    seen: HashMap<PromptKey<'a>, &'a [u32], RandomState>,
    prompts: HashMap<&'a [u32], &'a [u32], RandomState>,
    chains: HashMap<PromptKey<'a>, Vec<PreparedChain>, RandomState>,
}

struct PreparedChain {
    generator: BlockKeyGenerator,
    adapter: Option<(String, String)>,
    keys: Vec<SequenceHash>,
    domains: Vec<(Arc<KvBlockIndexer>, HashMap<u32, PageEvidence>)>,
}

impl<'a> RequestHashCache<'a> {
    pub(crate) fn canonical_prompt(&mut self, tokens: &'a [u32]) -> &'a [u32] {
        self.seen
            .entry(PromptKey(tokens))
            .or_insert_with(|| self.prompts.entry(tokens).or_insert(tokens))
    }

    pub(in crate::kv_index) fn pages(
        &mut self,
        tokens: &'a [u32],
        generator: &BlockKeyGenerator,
        adapter: Option<(&str, &str)>,
        domain_index: &Arc<KvBlockIndexer>,
        groups: impl Iterator<Item = u32>,
    ) -> (&[SequenceHash], &HashMap<u32, PageEvidence>) {
        let tokens = self.canonical_prompt(tokens);
        let chains = self.chains.entry(PromptKey(tokens)).or_default();
        let index = chains.iter().position(|chain| {
            chain.generator == *generator
                && chain
                    .adapter
                    .as_ref()
                    .map(|(name, path)| (name.as_str(), path.as_str()))
                    == adapter
        });
        let index = index.unwrap_or_else(|| {
            let hashes = match adapter {
                Some((name, path)) => generator.generate_block_keys_for_adapter(tokens, name, path),
                None => generator.generate_block_keys(tokens),
            };
            chains.push(PreparedChain {
                generator: generator.clone(),
                adapter: adapter.map(|(name, path)| (name.to_owned(), path.to_owned())),
                keys: hashes,
                domains: Vec::new(),
            });
            chains.len() - 1
        });
        let chain = &mut chains[index];
        let domain = chain
            .domains
            .iter()
            .position(|(candidate, _)| Arc::ptr_eq(candidate, domain_index));
        let domain = domain.unwrap_or_else(|| {
            chain.domains.push((domain_index.clone(), HashMap::new()));
            chain.domains.len() - 1
        });
        let evidence = &mut chain.domains[domain].1;
        let missing: Vec<_> = groups
            .filter(|group| !evidence.contains_key(group))
            .collect();
        // A single source has no postings to share with another candidate.
        // The source snapshot reads all of its groups at one revision.
        if !missing.is_empty() && domain_index.has_multiple_sources() {
            evidence.extend(domain_index.collect_hash_groups(
                missing.into_iter(),
                adapter.map(|(name, _)| name),
                &chain.keys,
            ));
        }
        (&chain.keys, evidence)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    #[test]
    fn request_chains_share_only_identical_tokens_and_contracts() {
        let tokens: Vec<_> = (0..65).collect();
        let copy = tokens.clone();
        let full_domain = Arc::new(KvBlockIndexer::new());
        let hybrid_domain = Arc::new(KvBlockIndexer::new());
        let mut cache = RequestHashCache::default();
        let generator = BlockKeyGenerator::new(16, 42);
        let first = cache
            .pages(&tokens, &generator, None, &full_domain, [0].into_iter())
            .0
            .as_ptr();
        let second = cache
            .pages(&copy, &generator, None, &hybrid_domain, [0, 1].into_iter())
            .0
            .as_ptr();
        assert_eq!(first, second);
        let other = BlockKeyGenerator::new(16, 43);
        cache.pages(&tokens, &other, None, &hybrid_domain, [0, 1].into_iter());
        assert_eq!(cache.chains.values().map(Vec::len).sum::<usize>(), 2);
        let lookahead = generator.clone().with_lookahead(1);
        assert_ne!(
            cache
                .pages(&tokens, &lookahead, None, &hybrid_domain, [0].into_iter())
                .0
                .as_ptr(),
            first
        );
        assert_eq!(cache.chains.values().map(Vec::len).sum::<usize>(), 3);
        let mut changed = tokens.clone();
        changed[32] += 1;
        let original = generator.generate_block_keys(&tokens);
        let sibling = generator.generate_block_keys(&changed);
        assert_eq!(original[..2], sibling[..2]);
        assert_ne!(original[2..], sibling[2..]);
    }
}
