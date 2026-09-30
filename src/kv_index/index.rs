//! Bounded, generation-safe ownership index for real KV Events.

use super::BlockHash;
use parking_lot::RwLock;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone)]
struct OwnershipOrderEntry {
    worker: String,
    block: BlockHash,
}

/// A fully validated event. All events in a publisher batch are applied under
/// one write lock, so readers never observe a half-applied clear/remove batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OwnershipEvent {
    Store(Vec<BlockHash>),
    Remove(Vec<BlockHash>),
    Clear,
}

#[derive(Debug, Default)]
struct IndexState {
    generations: HashMap<String, u64>,
    active_workers: HashSet<String>,
    blocks_by_worker: HashMap<String, HashMap<BlockHash, u64>>,
    insertion_order: BTreeMap<u64, OwnershipOrderEntry>,
    next_ticket: u64,
    ownership_count: usize,
}

/// Maps exact full block hashes to worker ownership without allowing events
/// from a retired subscriber generation to reintroduce stale entries.
pub struct KVBlockIndex {
    state: RwLock<IndexState>,
    max_entries: usize,
}

impl std::fmt::Debug for KVBlockIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read();
        formatter
            .debug_struct("KVBlockIndex")
            .field("ownership_count", &state.ownership_count)
            .field("worker_count", &state.blocks_by_worker.len())
            .field("max_entries", &self.max_entries)
            .finish()
    }
}

impl KVBlockIndex {
    pub fn new(max_entries: usize) -> Self {
        assert!(max_entries > 0, "max_entries must be greater than zero");
        Self {
            state: RwLock::new(IndexState::default()),
            max_entries,
        }
    }

    /// Start a new subscriber/lifecycle generation and purge older ownership.
    pub fn begin_worker(&self, worker: &str) -> u64 {
        let mut state = self.state.write();
        clear_worker(&mut state, worker);
        let generation = next_generation(state.generations.get(worker).copied().unwrap_or(0));
        state.generations.insert(worker.to_string(), generation);
        state.active_workers.insert(worker.to_string());
        generation
    }

    /// Roll the current generation after a sequence discontinuity or health
    /// transition. A stale caller cannot roll a newer generation.
    pub fn roll_worker(&self, worker: &str, expected_generation: u64) -> Option<u64> {
        let mut state = self.state.write();
        if !is_current(&state, worker, expected_generation) {
            return None;
        }
        clear_worker(&mut state, worker);
        let generation = next_generation(expected_generation);
        state.generations.insert(worker.to_string(), generation);
        Some(generation)
    }

    /// Retire a worker. Late events carrying the prior generation are ignored.
    pub fn retire_worker(&self, worker: &str) {
        let mut state = self.state.write();
        clear_worker(&mut state, worker);
        let generation = next_generation(state.generations.get(worker).copied().unwrap_or(0));
        state.generations.insert(worker.to_string(), generation);
        state.active_workers.remove(worker);
    }

    pub fn store(&self, worker: &str, generation: u64, blocks: &[BlockHash]) -> bool {
        self.apply_batch(
            worker,
            generation,
            &[OwnershipEvent::Store(blocks.to_vec())],
        )
    }

    pub fn remove(&self, worker: &str, generation: u64, blocks: &[BlockHash]) -> bool {
        self.apply_batch(
            worker,
            generation,
            &[OwnershipEvent::Remove(blocks.to_vec())],
        )
    }

    pub fn clear(&self, worker: &str, generation: u64) -> bool {
        self.apply_batch(worker, generation, &[OwnershipEvent::Clear])
    }

    pub fn apply_batch(&self, worker: &str, generation: u64, events: &[OwnershipEvent]) -> bool {
        let mut state = self.state.write();
        if !is_current(&state, worker, generation) {
            return false;
        }
        for event in events {
            match event {
                OwnershipEvent::Store(blocks) => {
                    for &block in blocks {
                        store_block(&mut state, worker, block);
                        evict_to_limit(&mut state, self.max_entries);
                    }
                }
                OwnershipEvent::Remove(blocks) => {
                    for block in blocks {
                        remove_block(&mut state, worker, block);
                    }
                }
                OwnershipEvent::Clear => clear_worker(&mut state, worker),
            }
        }
        true
    }

    /// Count the longest contiguous prefix present on `worker`.
    pub fn prefix_score(&self, worker: &str, blocks: &[BlockHash]) -> usize {
        let state = self.state.read();
        let Some(owned) = state.blocks_by_worker.get(worker) else {
            return 0;
        };
        blocks
            .iter()
            .take_while(|block| owned.contains_key(*block))
            .count()
    }

    pub fn ownership_count(&self) -> usize {
        self.state.read().ownership_count
    }

    /// None means health/removal retired this worker. Only begin_worker may
    /// reactivate it; the subscriber must not reactivate a retired worker.
    pub(crate) fn current_generation(&self, worker: &str) -> Option<u64> {
        let state = self.state.read();
        state
            .active_workers
            .contains(worker)
            .then(|| state.generations[worker])
    }
}

fn is_current(state: &IndexState, worker: &str, generation: u64) -> bool {
    state.active_workers.contains(worker)
        && state.generations.get(worker).copied() == Some(generation)
}

fn clear_worker(state: &mut IndexState, worker: &str) {
    if let Some(blocks) = state.blocks_by_worker.remove(worker) {
        state.ownership_count -= blocks.len();
        for ticket in blocks.values() {
            state.insertion_order.remove(ticket);
        }
    }
}

fn store_block(state: &mut IndexState, worker: &str, block: BlockHash) {
    if state
        .blocks_by_worker
        .get(worker)
        .is_some_and(|blocks| blocks.contains_key(&block))
    {
        return;
    }
    let ticket = state.next_ticket;
    state.next_ticket = ticket.checked_add(1).expect("ownership ticket exhausted");
    state
        .blocks_by_worker
        .entry(worker.to_string())
        .or_default()
        .insert(block, ticket);
    state.insertion_order.insert(
        ticket,
        OwnershipOrderEntry {
            worker: worker.to_string(),
            block,
        },
    );
    state.ownership_count += 1;
}

fn remove_block(state: &mut IndexState, worker: &str, block: &BlockHash) {
    let ticket = state
        .blocks_by_worker
        .get_mut(worker)
        .and_then(|blocks| blocks.remove(block));
    if let Some(ticket) = ticket {
        state.insertion_order.remove(&ticket);
        state.ownership_count -= 1;
    }
    if state
        .blocks_by_worker
        .get(worker)
        .is_some_and(HashMap::is_empty)
    {
        state.blocks_by_worker.remove(worker);
    }
}

fn evict_to_limit(state: &mut IndexState, max_entries: usize) {
    while state.ownership_count > max_entries {
        let (_, entry) = state
            .insertion_order
            .pop_first()
            .expect("ownership order invariant");
        remove_block(state, &entry.worker, &entry.block);
    }
}

fn next_generation(current: u64) -> u64 {
    current.checked_add(1).expect("worker generation exhausted")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(value: u8) -> BlockHash {
        [value; 32]
    }

    #[test]
    fn stale_generation_cannot_resurrect_ownership() {
        let index = KVBlockIndex::new(16);
        let old = index.begin_worker("w0");
        assert!(index.store("w0", old, &[block(1)]));
        let current = index.roll_worker("w0", old).unwrap();

        assert_eq!(index.prefix_score("w0", &[block(1)]), 0);
        assert!(!index.store("w0", old, &[block(1)]));
        assert!(index.store("w0", current, &[block(2)]));
        assert_eq!(index.prefix_score("w0", &[block(2)]), 1);
    }

    #[test]
    fn clear_and_remove_are_worker_scoped() {
        let index = KVBlockIndex::new(16);
        let w0 = index.begin_worker("w0");
        let w1 = index.begin_worker("w1");
        index.store("w0", w0, &[block(1), block(2)]);
        index.store("w1", w1, &[block(1), block(2)]);
        index.remove("w0", w0, &[block(2)]);

        assert_eq!(index.prefix_score("w0", &[block(1), block(2)]), 1);
        assert_eq!(index.prefix_score("w1", &[block(1), block(2)]), 2);
        index.clear("w1", w1);
        assert_eq!(index.prefix_score("w1", &[block(1)]), 0);
    }

    #[test]
    fn bounded_index_evicts_oldest_ownership_only() {
        let index = KVBlockIndex::new(2);
        let generation = index.begin_worker("w0");
        index.store("w0", generation, &[block(1), block(2), block(3)]);
        assert_eq!(index.ownership_count(), 2);
        assert_eq!(index.prefix_score("w0", &[block(1)]), 0);
        assert_eq!(index.prefix_score("w0", &[block(2), block(3)]), 2);
    }

    #[test]
    fn removed_and_reinserted_block_has_a_fresh_fifo_position() {
        let index = KVBlockIndex::new(2);
        let generation = index.begin_worker("w0");
        index.store("w0", generation, &[block(1), block(2)]);
        index.remove("w0", generation, &[block(1)]);
        index.store("w0", generation, &[block(1), block(3)]);
        assert_eq!(index.prefix_score("w0", &[block(1), block(3)]), 2);
        assert_eq!(index.prefix_score("w0", &[block(2)]), 0);
    }

    #[test]
    fn repeated_remove_has_no_fifo_tombstones() {
        let index = KVBlockIndex::new(2);
        let generation = index.begin_worker("w0");
        for _ in 0..1000 {
            index.store("w0", generation, &[block(1)]);
            index.remove("w0", generation, &[block(1)]);
        }
        assert!(index.state.read().insertion_order.is_empty());
        assert_eq!(index.ownership_count(), 0);
    }

    #[test]
    fn retired_worker_cannot_be_reactivated_by_a_subscriber_roll() {
        let index = KVBlockIndex::new(2);
        let generation = index.begin_worker("w0");
        index.store("w0", generation, &[block(1)]);
        index.retire_worker("w0");
        assert_eq!(index.current_generation("w0"), None);
        assert_eq!(index.roll_worker("w0", generation), None);
        assert!(!index.store("w0", generation + 1, &[block(1)]));
        let fresh = index.begin_worker("w0");
        assert!(index.store("w0", fresh, &[block(2)]));
        assert!(!index.store("w0", generation, &[block(1)]));
    }

    #[test]
    fn batch_order_preserves_clear_then_store_and_stale_batches_are_atomic() {
        let index = KVBlockIndex::new(4);
        let generation = index.begin_worker("w0");
        let events = [
            OwnershipEvent::Store(vec![block(1)]),
            OwnershipEvent::Clear,
            OwnershipEvent::Store(vec![block(2)]),
        ];
        assert!(index.apply_batch("w0", generation, &events));
        assert_eq!(index.prefix_score("w0", &[block(1)]), 0);
        assert_eq!(index.prefix_score("w0", &[block(2)]), 1);
        index.begin_worker("w0");
        assert!(!index.apply_batch("w0", generation, &events));
        assert_eq!(index.ownership_count(), 0);
    }
}
