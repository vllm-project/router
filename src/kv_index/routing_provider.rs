//! Adapt index lookups to the routing policy.

use crate::kv_index::layout::align_down;
use crate::kv_index::request_hash::{BlockKeyGenerator, ImagePrefix, SequenceHash};
use crate::kv_index::source::binding::{PlacementSnapshot, SourceLease};
use crate::kv_index::storage::source_transaction::StreamControl;
use crate::kv_index::storage::{GroupPages, PageQuery, SourceReadGuard};
use crate::kv_index::SourceBinding;
use crate::kv_index::{KvBlockIndexer, RoutableTarget, TierMatch};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use foldhash::fast::RandomState;
use smallvec::SmallVec;

/// Read-only placement provider for the routing policy.
///
/// Hashes prompts using the verified source contract and reports the prefix
/// that the cache layout can restore.
///
/// Regular Prefill reservations and P/D selection grant credit only
/// for hits with evidence from that lookup. Hits from custom providers
/// without evidence are scored as misses, and their workers remain eligible.
/// To retain credit, forward `matches_with_cache` and its cache to the
/// source provider that produced the hit.
pub trait TieredMatchProvider: Send + Sync {
    /// Match the prompt using the provider's complete restore rules.
    fn matches(
        &self,
        _tokens: &[u32],
        _block_size: u32,
        _lora_name: Option<&str>,
        _dp_rank: u32,
    ) -> Vec<TierMatch> {
        Vec::new()
    }

    #[doc(hidden)]
    fn matches_with_cache<'a>(
        &self,
        tokens: &'a [u32],
        block_size: u32,
        lora_name: Option<&str>,
        dp_rank: u32,
        cache: &mut RequestHashCache<'a>,
    ) -> Vec<TierMatch> {
        if cache.images.contains_key(&PromptKey(tokens)) {
            return Vec::new();
        }
        self.matches(tokens, block_size, lora_name, dp_rank)
    }

    /// Group restore evidence belongs to one target; dense lookups cover all targets.
    fn target_scoped(&self) -> bool {
        false
    }

    #[doc(hidden)]
    fn shared_prefix_lookup(&self, _rank: u32) -> Option<SharedPrefixLookup> {
        None
    }

    fn nixl_decode_credit(&self, _rank: u32, _prompt_len: usize, reused: u64) -> u64 {
        reused
    }
}

/// Cache domain and hash contract for a plain group-zero prefix lookup.
pub struct SharedPrefixLookup {
    index: Arc<KvBlockIndexer>,
    keys: BlockKeyGenerator,
}

impl SharedPrefixLookup {
    pub(crate) fn same_contract(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.index, &other.index) && self.keys == other.keys
    }

    pub(crate) fn confirms_miss<'a>(
        &self,
        tokens: &'a [u32],
        cache: &mut RequestHashCache<'a>,
    ) -> bool {
        let group = PageQuery {
            group: 0,
            stop_at_gap: true,
        };
        let (keys, evidence) = cache.prepared_pages(tokens, &self.keys, None, &self.index);
        if let Some(page) = evidence.get(&group) {
            return page.confirms_empty(&self.index);
        }
        let Some(page) = self.index.collect_prefix(group, keys) else {
            return true;
        };
        evidence.insert(group, page);
        false
    }
}

impl SourceBinding {
    fn plain_attention_match(&self) -> bool {
        self.groups.len() == 1
            && self.groups[0].kind == crate::kv_index::GroupRule::FullAttention
            && !self.partial_hash_hits
            && self.replay_tokens == 0
            && self.tail_drop_group.is_none()
    }
}

/// Keep the hit and its source snapshot together until the provider accepts it.
pub(in crate::kv_index) struct PlacementMatch {
    hits: Vec<TierMatch>,
    snapshot: PlacementSnapshot,
}

impl PlacementMatch {
    pub(in crate::kv_index) fn publish(
        self,
        binding: &SourceBinding,
        cache: &mut RequestHashCache<'_>,
    ) -> Vec<TierMatch> {
        if let Some(placement) = &mut cache.placement {
            placement.push(PlacementEvidence {
                snapshot: self.snapshot,
                control: binding.control.clone(),
                lease: binding.binding.clone(),
            });
        }
        self.hits
    }
}

/// Trust evidence for a hit, carried to the routing commit boundary.
pub(crate) struct PlacementEvidence {
    snapshot: PlacementSnapshot,
    control: Arc<StreamControl>,
    lease: Option<SourceLease>,
}

pub(crate) enum PlacementPinError {
    Busy,
    Revoked,
    Missing,
}

impl PlacementEvidence {
    pub(crate) fn target(&self) -> &RoutableTarget {
        self.snapshot.version.target()
    }

    pub(crate) fn is_revoked(&self) -> bool {
        !self.snapshot.is_current(&self.control, self.lease.as_ref())
    }

    /// Try the source guards without waiting under the policy's selection lock.
    pub(crate) fn try_pin_all<'a>(
        evidence: impl Iterator<Item = &'a Self>,
    ) -> Result<SmallVec<[SourceReadGuard<'a>; 1]>, PlacementPinError> {
        let mut pins = SmallVec::new();
        for proof in evidence {
            let guard = proof
                .snapshot
                .version
                .try_read()
                .ok_or(PlacementPinError::Busy)?;
            if proof.is_revoked() {
                return Err(PlacementPinError::Revoked);
            }
            pins.push(guard);
        }
        if pins.is_empty() {
            return Err(PlacementPinError::Missing);
        }
        Ok(pins)
    }
}

impl crate::policies::TieredMatchProvider for SourceBinding {
    fn shared_prefix_lookup(&self, rank: u32) -> Option<SharedPrefixLookup> {
        (rank == self.target.dp_rank
            && self.plain_attention_match()
            && self.groups[0].group_idx == 0
            && self.groups[0].block_size as usize == self.keys.block_size())
        .then(|| SharedPrefixLookup {
            index: self.index.clone(),
            keys: self.keys.clone(),
        })
    }

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
            &mut RequestHashCache::for_single_target(),
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
        self.lookup_placement(tokens, block_size, lora_name, dp_rank, cache)
            .map_or_else(Vec::new, |hit| hit.publish(self, cache))
    }
}

impl SourceBinding {
    pub(in crate::kv_index) fn lookup_placement<'a>(
        &self,
        tokens: &'a [u32],
        block_size: u32,
        lora_name: Option<&str>,
        dp_rank: u32,
        cache: &mut RequestHashCache<'a>,
    ) -> Option<PlacementMatch> {
        // Events, ownership and history are bound to one rank. A hit computed
        // from another rank's stream is never evidence for this one.
        let prompt_cap = self.reusable_prompt_cap(tokens.len());
        if block_size == 0 || dp_rank != self.target.dp_rank || prompt_cap < block_size as usize {
            return None;
        }
        if self.plain_attention_match() {
            let (mut claims, snapshot) =
                self.page_snapshot(lora_name, tokens, cache.stop_at_gap, cache)?;
            let group_claims = claims.pop()?;
            let hits = crate::kv_index::layout::full_attention_matches(
                &self.target,
                &self.groups[0],
                group_claims,
                self.keys.block_size(),
                block_size,
                prompt_cap as u64,
            );
            if hits.is_empty() || !self.snapshot_is_current(&snapshot) {
                return None;
            }
            return Some(PlacementMatch { hits, snapshot });
        }
        // The policy's page size can be coarser than the hash granularity: the
        // engine restores the aligned prefix, so round down instead of dropping
        // a recoverable hit.
        let (tokens, tier, snapshot) = self.credit_parts(lora_name, tokens, usize::MAX, cache)?;
        let tokens = align_down(tokens, block_size);
        if tokens == 0 {
            return None;
        }
        let blocks = u32::try_from(tokens / u64::from(block_size)).ok()?;
        // The engine restores at the tier that completes the boundary; the policy
        // weighs that tier by its transfer credit.
        Some(PlacementMatch {
            hits: vec![TierMatch {
                target: self.target.clone(),
                tier,
                segments: vec![(0, blocks)],
            }],
            snapshot,
        })
    }
}

/// Identity of an immutable prompt slice within a selection.
#[derive(Clone, Copy)]
pub(crate) struct PromptKey<'a>(pub(crate) &'a [u32]);

impl PartialEq for PromptKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        // Slice pointers compare both their address and length.
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
    pub(crate) placement: Option<Vec<PlacementEvidence>>,
    /// Lower bound used only for the initial trust check within this routing stage.
    pub(crate) first_trust_check: Option<std::time::Instant>,
    /// A single candidate reads its own source snapshot without collecting other ranks.
    pub(crate) single_target: bool,
    /// Built-in scoring only consumes a device prefix and its tier continuation.
    pub(crate) stop_at_gap: bool,
    /// NIXL Decode counts resident pages even when rebuilding the sliding window.
    pub(crate) count_replay_pages: bool,
    images: HashMap<PromptKey<'a>, (&'a ImagePrefix, Vec<PreparedChain>), RandomState>,
    seen: HashMap<PromptKey<'a>, usize, RandomState>,
    prompts: HashMap<&'a [u32], usize, RandomState>,
    prepared: Vec<(&'a [u32], Vec<PreparedChain>)>,
}

struct PreparedChain {
    generator: BlockKeyGenerator,
    adapter: Option<(String, String)>,
    keys: Vec<SequenceHash>,
    domains: Vec<(Arc<KvBlockIndexer>, GroupPages)>,
}

impl<'a> RequestHashCache<'a> {
    pub(crate) fn for_single_target() -> Self {
        Self {
            single_target: true,
            ..Self::default()
        }
    }

    pub(crate) fn register_image(&mut self, tokens: &'a [u32], image: &'a ImagePrefix) {
        self.images
            .entry(PromptKey(tokens))
            .or_insert((image, Vec::new()));
    }

    pub(crate) fn has_images(&self) -> bool {
        !self.images.is_empty()
    }

    fn prompt_slot(&mut self, tokens: &'a [u32]) -> usize {
        let Some(&(first, _)) = self.prepared.first() else {
            self.prepared.push((tokens, Vec::new()));
            return 0;
        };
        if std::ptr::eq(first, tokens) {
            return 0;
        }
        // Populate the content map only when this request has distinct prompts.
        *self.seen.entry(PromptKey(tokens)).or_insert_with(|| {
            if self.prompts.is_empty() {
                if first == tokens {
                    return 0;
                }
                self.prompts.insert(first, 0);
            }
            *self.prompts.entry(tokens).or_insert_with(|| {
                let slot = self.prepared.len();
                self.prepared.push((tokens, Vec::new()));
                slot
            })
        })
    }

    pub(crate) fn canonical_prompt(&mut self, tokens: &'a [u32]) -> &'a [u32] {
        // Equal token IDs can reference different images.
        if self.images.contains_key(&PromptKey(tokens)) {
            tokens
        } else {
            let slot = self.prompt_slot(tokens);
            self.prepared[slot].0
        }
    }

    pub(in crate::kv_index) fn pages(
        &mut self,
        tokens: &'a [u32],
        generator: &BlockKeyGenerator,
        adapter: Option<(&str, &str)>,
        domain_index: &Arc<KvBlockIndexer>,
        groups: impl Iterator<Item = impl Into<PageQuery>>,
    ) -> (&[SequenceHash], &GroupPages) {
        let single_target = self.single_target;
        let (keys, evidence) = self.prepared_pages(tokens, generator, adapter, domain_index);
        let mut missing = groups
            .map(Into::into)
            .filter(|group| !evidence.contains_key(group))
            .peekable();
        if missing.peek().is_some() && !single_target && domain_index.has_multiple_sources() {
            let added =
                domain_index.collect_hash_groups(missing, adapter.map(|(name, _)| name), keys);
            if evidence.is_empty() {
                *evidence = added;
            } else {
                evidence.extend(added);
            }
        }
        (keys, evidence)
    }

    fn prepared_pages(
        &mut self,
        tokens: &'a [u32],
        generator: &BlockKeyGenerator,
        adapter: Option<(&str, &str)>,
        domain_index: &Arc<KvBlockIndexer>,
    ) -> (&[SequenceHash], &mut GroupPages) {
        let prompt = PromptKey(tokens);
        let (tokens, image, chains) = if self.images.contains_key(&prompt) {
            let (image, chains) = self.images.get_mut(&prompt).unwrap();
            (tokens, Some(*image), chains)
        } else {
            let slot = self.prompt_slot(tokens);
            let (tokens, chains) = &mut self.prepared[slot];
            (*tokens, None, chains)
        };
        let index = chains.iter().position(|chain| {
            chain.generator == *generator
                && chain
                    .adapter
                    .as_ref()
                    .map(|(name, path)| (name.as_str(), path.as_str()))
                    == adapter
        });
        let index = index.unwrap_or_else(|| {
            let hashes = match (adapter, image) {
                (Some(_), Some(_)) => Vec::new(),
                (Some((name, path)), None) => {
                    generator.generate_block_keys_for_adapter(tokens, name, path)
                }
                (None, Some(image)) => generator.generate_block_keys_for_image(tokens, image),
                (None, None) => generator.generate_block_keys(tokens),
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
            chain
                .domains
                .push((domain_index.clone(), GroupPages::default()));
            chain.domains.len() - 1
        });
        (&chain.keys, &mut chain.domains[domain].1)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn image_prefix_matches_native_events_without_aliasing_equal_token_ids() {
        // Independent vLLM sha256_cbor goldens for image positions 60 and -4.
        check_image_prefix(
            "",
            crate::kv_index::source::test_support::TEST_IMAGE_BLOCK_HASHES,
        );
        check_image_prefix(
            "content:",
            [
                "6521a2f6cff15637a5767844512eb22531d85a1834d4ad81fa398c5d601895f7",
                "8e23a619de5fe86153b8ed771642bd6ad6e326a11ff920a45c889d9617411d76",
            ],
        );
    }

    fn check_image_prefix(namespace: &str, block_hashes: [&str; 2]) {
        use crate::kv_index::source::test_support::{
            event_batch, full_binding, image_store, TEST_IMAGE_HASH,
        };
        use crate::kv_index::wire::KVEvent;

        let hash = format!("{namespace}{TEST_IMAGE_HASH}");
        let same_image = ImagePrefix::new(&hash, 60, 8, 129).unwrap();
        let other_hash = format!("{namespace}{}", "a".repeat(64));
        let other_image = ImagePrefix::new(&other_hash, 60, 8, 129).unwrap();
        let tokens: Vec<u32> = (0..129).collect();
        let identical_ids = tokens.clone();
        let text_ids = tokens.clone();
        let store = image_store(&hash, block_hashes);
        let binding = full_binding(
            crate::kv_index::RoutableTarget {
                instance_id: "image-worker".into(),
                dp_rank: 0,
            },
            64,
        );
        binding.begin_empty(0);
        binding
            .apply_batch(0, &event_batch(vec![KVEvent::BlockStored(store)]))
            .unwrap();
        for (seq, identity) in ["uuid:client-image", &other_hash].into_iter().enumerate() {
            let invalid = image_store(identity, block_hashes);
            binding
                .apply_batch(
                    seq as u64 + 1,
                    &event_batch(vec![KVEvent::BlockStored(invalid)]),
                )
                .unwrap();
        }
        let mut cache = RequestHashCache::default();
        cache.register_image(&tokens, &same_image);
        cache.register_image(&identical_ids, &other_image);
        let hits = binding.matches_with_cache(&tokens, 64, None, 0, &mut cache);
        assert_eq!(hits[0].segments, [(0, 2)]);
        assert!(binding
            .matches_with_cache(&identical_ids, 64, None, 0, &mut cache)
            .is_empty());
        assert!(binding
            .matches_with_cache(&text_ids, 64, None, 0, &mut cache)
            .is_empty());
    }

    #[test]
    fn ranks_hash_with_their_own_units_when_scored_at_a_common_unit() {
        use crate::kv_index::source::test_support::{full_binding, warm_full};
        use crate::kv_index::RoutableTarget;

        let prompt: Vec<u32> = (0..49).collect();
        let mut cache = RequestHashCache::default();
        for (rank, unit) in [(0, 16), (1, 24)] {
            let target = RoutableTarget {
                instance_id: "worker".into(),
                dp_rank: rank,
            };
            let binding = full_binding(target.clone(), unit);
            binding.begin_empty(0);
            warm_full(&binding, 0, &prompt[..48]);
            let hits = binding.matches_with_cache(&prompt, 8, None, rank, &mut cache);
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].target, target);
            assert_eq!(hits[0].segments, [(0, 6)]);
        }
    }

    #[test]
    fn request_chains_share_only_identical_tokens_and_contracts() {
        let tokens: Vec<_> = (0..65).collect();
        assert!(PromptKey(&tokens[..64]) != PromptKey(&tokens));
        let copy = tokens.clone();
        let full_domain = Arc::new(KvBlockIndexer::new());
        let hybrid_domain = Arc::new(KvBlockIndexer::new());
        let mut cache = RequestHashCache::default();
        let generator = BlockKeyGenerator::new(16, 42);
        let first = cache
            .pages(&tokens, &generator, None, &full_domain, [0u32].into_iter())
            .0
            .as_ptr();
        let second = cache
            .pages(
                &copy,
                &generator,
                None,
                &hybrid_domain,
                [0u32, 1].into_iter(),
            )
            .0
            .as_ptr();
        assert_eq!(first, second);
        let other = BlockKeyGenerator::new(16, 43);
        assert_ne!(
            cache
                .pages(&tokens, &other, None, &hybrid_domain, [0u32, 1].into_iter())
                .0
                .as_ptr(),
            first
        );
        let lookahead = generator.clone().with_lookahead(1);
        assert_ne!(
            cache
                .pages(
                    &tokens,
                    &lookahead,
                    None,
                    &hybrid_domain,
                    [0u32].into_iter()
                )
                .0
                .as_ptr(),
            first
        );
        let mut changed = tokens.clone();
        changed[32] += 1;
        let changed_copy = changed.clone();
        assert!(std::ptr::eq(
            cache.canonical_prompt(&changed),
            changed.as_slice()
        ));
        assert!(std::ptr::eq(
            cache.canonical_prompt(&changed_copy),
            changed.as_slice()
        ));
        assert!(std::ptr::eq(
            cache.canonical_prompt(&copy),
            tokens.as_slice()
        ));
        assert_eq!(cache.canonical_prompt(&tokens[..64]).len(), 64);
        let original = generator.generate_block_keys(&tokens);
        let sibling = generator.generate_block_keys(&changed);
        assert_eq!(original[..2], sibling[..2]);
        assert_ne!(original[2..], sibling[2..]);
    }
}
