//! Walk prefix paths and aggregate per-target cache claims.

use super::super::*;
use crate::kv_index::types::push_block_segment;

/// A contiguous run of pages with one immutable claim list; `end` is exclusive.
pub(in crate::kv_index::index) struct ClaimSpan {
    pub(in crate::kv_index::index) end: usize,
    pub(in crate::kv_index::index) owners: Arc<Vec<OwnerTier>>,
}

impl GroupShard {
    /// Walk the request's `local_hashes` from the root, collecting the
    /// per-position `(owner, tier)` claims along the matched prefix; the
    /// collapse below builds maximal `[start, end)` segments from them. Each
    /// edge element's claims are the claims of the node that edge belongs to.
    pub(in crate::kv_index::index) fn collect_claims(&self, query: &MatchQuery) -> Vec<ClaimSpan> {
        let mut node_claims: Vec<ClaimSpan> = Vec::new();
        let mut qi = 0; // index into query.local_hashes
        let mut next = query
            .local_hashes
            .first()
            .and_then(|local| self.root.read().children.get(local).cloned());
        while let Some(child) = next {
            // Match and resolve the next child under the same read guard. A
            // split moves children behind the remainder edge; re-reading this
            // node's children after advancing by its old length misaligns depth.
            let c = child.read();
            let mut advanced = 0;
            while advanced < c.edge.len()
                && qi + advanced < query.local_hashes.len()
                && c.edge[advanced].1 == query.local_hashes[qi + advanced]
            {
                // Derived references are maintained in the same source or
                // global transaction as `reverse`, the authoritative facts.
                let claims = &c.claims[advanced];
                let end = qi + advanced + 1;
                if let Some(last) = node_claims
                    .last_mut()
                    .filter(|last| Arc::ptr_eq(&last.owners, claims))
                {
                    last.end = end;
                } else {
                    node_claims.push(ClaimSpan {
                        end,
                        owners: claims.clone(),
                    });
                }
                advanced += 1;
            }
            // Partial edge match: children sit after the full edge in the chain,
            // so descending on a partial match misaligns chain positions.
            if advanced == 0 || advanced < c.edge.len() {
                break;
            }
            qi += advanced;
            next = query
                .local_hashes
                .get(qi)
                .and_then(|local| c.children.get(local).cloned());
            drop(c);
        }

        node_claims
    }

    pub(in crate::kv_index::index) fn aggregate_claims(
        query: &MatchQuery,
        node_claims: &[ClaimSpan],
    ) -> Vec<TierMatch> {
        // Per (target, tier), run-length collect the maximal contiguous
        // [start, end) block intervals the target holds at that tier. The index
        // stores by owner but answers by target; locality is derived from the
        // owner domain. A `(target, tier)` may have several owners at one
        // span (e.g. two pools pushed the same block); each span is counted
        // once, regardless of its number of owners.
        // An owner usually appears on many blocks. Resolve it once per query,
        // while still merging different owners of the same target and tier.
        let mut owner_slots: HashMap<&OwnerTier, usize> = HashMap::new();
        let mut target_slots: HashMap<(RoutableTarget, StorageTier), usize> = HashMap::new();
        let mut matches: Vec<TierMatch> = Vec::new();
        let mut start = 0;
        for span in node_claims {
            for key @ (owner, tier) in span.owners.iter() {
                if !query.tiers_of_interest.contains(tier) {
                    continue;
                }
                let slot = *owner_slots.entry(key).or_insert_with(|| {
                    let target = owner.target();
                    let locality = owner.locality();
                    let slot = *target_slots
                        .entry((target.clone(), *tier))
                        .or_insert_with(|| {
                            let slot = matches.len();
                            matches.push(TierMatch {
                                target,
                                tier: *tier,
                                segments: Vec::new(),
                                locality,
                            });
                            slot
                        });
                    // Different owners can have different locality.
                    if locality == Locality::Local {
                        matches[slot].locality = Locality::Local;
                    }
                    slot
                });
                let entry = &mut matches[slot];
                // Spans are processed in order. Another owner for this same
                // target and tier may already have credited the entire span.
                if entry
                    .segments
                    .last()
                    .is_some_and(|(_, end)| *end == span.end as u32)
                {
                    continue;
                }
                push_block_segment(&mut entry.segments, start as u32, span.end as u32);
            }
            start = span.end;
        }
        matches
            .into_iter()
            .filter(|m| !m.segments.is_empty())
            .collect()
    }
}
