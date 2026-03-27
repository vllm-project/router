//! Background task that consumes [`KVEventBatch`] messages from the event
//! pool channel and applies them to the [`KVBlockIndex`].

use crate::kv_events::KVEventBatch;
use crate::kv_events::decoder::KVEvent;
use crate::kv_index::KVBlockIndex;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Run the index updater loop.
///
/// This function consumes from `event_rx` (produced by [`KVEventPool`]) and
/// applies every event to `index`.  It returns when the channel is closed
/// (i.e., all senders are dropped).
///
/// Intended to be spawned as a long-running tokio task:
///
/// ```ignore
/// tokio::spawn(run_kv_index_updater(event_rx, index.clone()));
/// ```
pub async fn run_kv_index_updater(
    mut event_rx: mpsc::UnboundedReceiver<KVEventBatch>,
    index: Arc<KVBlockIndex>,
) {
    let mut total_events: u64 = 0;
    let mut total_batches: u64 = 0;

    info!("KV index updater started");

    while let Some(batch) = event_rx.recv().await {
        total_batches += 1;
        let worker_id = &batch.worker_id;

        for event in &batch.events {
            total_events += 1;

            match event {
                KVEvent::BlockStored {
                    block_hashes,
                    parent_block_hash: _,
                    token_ids: _,
                    block_size: _,
                    lora_name: _,
                } => {
                    for &hash in block_hashes {
                        index.on_block_stored(hash, worker_id);
                    }
                    debug!(
                        "Indexed {} stored blocks from {} (total events: {})",
                        block_hashes.len(),
                        worker_id,
                        total_events
                    );
                }
                KVEvent::BlockRemoved { block_hashes } => {
                    for &hash in block_hashes {
                        index.on_block_removed(hash, worker_id);
                    }
                    debug!(
                        "Removed {} blocks from {} index (total events: {})",
                        block_hashes.len(),
                        worker_id,
                        total_events
                    );
                }
                KVEvent::AllBlocksCleared => {
                    index.on_all_blocks_cleared(worker_id);
                    info!(
                        "Cleared all blocks for worker {} (total events: {})",
                        worker_id, total_events
                    );
                }
            }
        }

        // Periodic progress log every 10000 batches.
        if total_batches % 10_000 == 0 {
            info!(
                "KV index updater progress: {} batches, {} events, {} index entries",
                total_batches,
                total_events,
                index.len()
            );
        }
    }

    warn!(
        "KV index updater stopped (channel closed). Processed {} batches, {} events.",
        total_batches, total_events
    );
}
