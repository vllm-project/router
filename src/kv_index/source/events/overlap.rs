//! Queued live frames and matching replay payloads for the replay-to-live handoff.

use std::collections::{hash_map::Entry, HashMap, VecDeque};

use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use super::transport::live_sequence;

pub(super) const MAX_PENDING_LIVE_FRAMES: usize = 4096;
const RECENT_FINGERPRINT_LIMIT: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub(super) enum HandoffError {
    #[error("malformed queued live frame")]
    InvalidLiveFrame,
    #[error("missing queued live sequence {seq} in replay")]
    MissingReplayPayload { seq: i64 },
    #[error("payload conflict at queued live sequence {seq}")]
    PayloadConflict { seq: i64 },
}

#[derive(Debug)]
pub(super) struct ConflictingReplayFingerprint;

#[derive(Default)]
pub(super) struct LiveEventBuffer {
    state: Mutex<LiveEventBufferState>,
}

#[derive(Default)]
struct LiveEventBufferState {
    frames: VecDeque<Vec<Vec<u8>>>,
    bytes: usize,
    recent_by_seq: HashMap<i64, [u8; 32]>,
    queued_by_seq: HashMap<i64, QueuedSequence>,
}

struct QueuedSequence {
    frame_count: usize,
    replay_fingerprint: Option<[u8; 32]>,
    had_replay_conflict: bool,
}

impl LiveEventBuffer {
    pub(super) fn push(&self, frames: Vec<Vec<u8>>, topic: &[u8]) -> bool {
        let mut state = self.state.lock();
        if let Some(seq) = valid_live_seq(&frames, topic) {
            let replay_fingerprint = state.recent_by_seq.get(&seq).copied();
            state
                .queued_by_seq
                .entry(seq)
                .or_insert(QueuedSequence {
                    frame_count: 0,
                    replay_fingerprint,
                    had_replay_conflict: false,
                })
                .frame_count += 1;
        }
        state.bytes += frame_bytes(&frames);
        state.frames.push_back(frames);
        state.frames.len() >= MAX_PENDING_LIVE_FRAMES || state.bytes >= super::MAX_KV_EVENT_BYTES
    }

    /// Check a queued copy before releasing its fingerprint reference.
    pub(super) fn pop(
        &mut self,
        topic: &[u8],
        applied_seq: Option<i64>,
    ) -> Option<(Vec<Vec<u8>>, bool)> {
        let state = self.state.get_mut();
        let frames = state.frames.pop_front()?;
        let mut is_duplicate = false;
        if let Some(seq) = valid_live_seq(&frames, topic) {
            is_duplicate = applied_seq.is_some_and(|applied_seq| {
                seq <= applied_seq && state.compare_payload(seq, &frames[2]) == Some(true)
            });
            release_frame(&mut state.queued_by_seq, seq);
        }
        state.bytes -= frame_bytes(&frames);
        Some((frames, is_duplicate))
    }

    pub(super) fn reset(&mut self) {
        *self.state.get_mut() = LiveEventBufferState::default();
    }

    /// Discard the queue and its fingerprints, returning the highest valid sequence lost.
    pub(super) fn discard_all(&mut self, topic: &[u8]) -> Option<i64> {
        let highest = self
            .state
            .get_mut()
            .frames
            .iter()
            .filter_map(|frames| valid_live_seq(frames, topic))
            .max();
        self.reset();
        highest
    }

    pub(super) fn record_replay_payload(
        &self,
        seq: i64,
        payload: &[u8],
    ) -> Result<(), ConflictingReplayFingerprint> {
        let fingerprint = Sha256::digest(payload).into();
        let mut state = self.state.lock();
        if state
            .fingerprint(seq)
            .is_some_and(|known| known != fingerprint)
        {
            return Err(ConflictingReplayFingerprint);
        }
        if let Some(queued) = state.queued_by_seq.get_mut(&seq) {
            queued.replay_fingerprint = Some(fingerprint);
        }
        state.recent_by_seq.insert(seq, fingerprint);
        if state.recent_by_seq.len() > RECENT_FINGERPRINT_LIMIT {
            let floor = seq.saturating_sub((RECENT_FINGERPRINT_LIMIT / 2) as i64);
            state.recent_by_seq.retain(|seq, _| *seq >= floor);
        }
        Ok(())
    }

    pub(super) fn is_replayed_duplicate(
        &self,
        frames: &[Vec<u8>],
        topic: &[u8],
        applied_seq: Option<i64>,
    ) -> bool {
        applied_seq.is_some_and(|applied_seq| {
            valid_live_seq(frames, topic).is_some_and(|seq| {
                seq <= applied_seq
                    && self.state.lock().compare_payload(seq, &frames[2]) == Some(true)
            })
        })
    }

    /// Keep queued frames and known conflicts when restarting a replay.
    pub(super) fn clear_fingerprints(&mut self) {
        let state = self.state.get_mut();
        if !state.recent_by_seq.is_empty() {
            state.recent_by_seq = HashMap::new();
            for queued in state.queued_by_seq.values_mut() {
                queued.replay_fingerprint = None;
            }
        }
    }

    /// Remove malformed frames before retrying. Discard frames covered by a Clear only
    /// after every overlapping frame has been checked.
    pub(super) fn validate_handoff(
        &mut self,
        topic: &[u8],
        applied_seq: i64,
        clear_seq: Option<i64>,
    ) -> Result<(), HandoffError> {
        let state = self.state.get_mut();
        let mut error = None;
        state.frames.retain(|frames| {
            let Some(seq) = valid_live_seq(frames, topic) else {
                error.get_or_insert(HandoffError::InvalidLiveFrame);
                state.bytes -= frame_bytes(frames);
                return false;
            };
            if seq <= applied_seq {
                let queued = state.queued_by_seq.get_mut(&seq);
                let fingerprint = queued
                    .as_ref()
                    .and_then(|entry| entry.replay_fingerprint)
                    .or_else(|| state.recent_by_seq.get(&seq).copied());
                match fingerprint {
                    Some(fingerprint)
                        if fingerprint == <[u8; 32]>::from(Sha256::digest(&frames[2])) => {}
                    Some(_) => {
                        if let Some(queued) = queued {
                            queued.had_replay_conflict = true;
                        }
                        error.get_or_insert(HandoffError::PayloadConflict { seq });
                    }
                    None if queued.is_some_and(|queued| queued.had_replay_conflict) => {
                        error.get_or_insert(HandoffError::PayloadConflict { seq });
                    }
                    None if clear_seq.is_none_or(|clear_seq| seq >= clear_seq) => {
                        error.get_or_insert(HandoffError::MissingReplayPayload { seq });
                    }
                    None => {}
                }
            }
            true
        });
        if let Some(error) = error {
            return Err(error);
        }
        if let Some(clear_seq) = clear_seq {
            state.frames.retain(|frames| {
                let Some(seq) = valid_live_seq(frames, topic).filter(|&seq| seq < clear_seq) else {
                    return true;
                };
                release_frame(&mut state.queued_by_seq, seq);
                state.bytes -= frame_bytes(frames);
                false
            });
        }
        Ok(())
    }
}

impl LiveEventBufferState {
    fn fingerprint(&self, seq: i64) -> Option<[u8; 32]> {
        self.queued_by_seq
            .get(&seq)
            .and_then(|queued| queued.replay_fingerprint)
            .or_else(|| self.recent_by_seq.get(&seq).copied())
    }

    fn compare_payload(&self, seq: i64, payload: &[u8]) -> Option<bool> {
        let fingerprint = self.fingerprint(seq)?;
        Some(fingerprint == <[u8; 32]>::from(Sha256::digest(payload)))
    }
}

fn release_frame(queued_by_seq: &mut HashMap<i64, QueuedSequence>, seq: i64) {
    if let Entry::Occupied(mut entry) = queued_by_seq.entry(seq) {
        entry.get_mut().frame_count -= 1;
        if entry.get().frame_count == 0 {
            entry.remove();
        }
    }
}

fn valid_live_seq(frames: &[Vec<u8>], topic: &[u8]) -> Option<i64> {
    live_sequence(frames, topic)
        .ok()
        .flatten()
        .filter(|seq| (0..i64::MAX).contains(seq))
}

fn frame_bytes(frames: &[Vec<u8>]) -> usize {
    frames.iter().map(Vec::len).sum()
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn replay_fingerprint_stays_until_last_queued_copy_leaves() {
        let mut live_events = LiveEventBuffer::default();
        let frame = |seq: i64, payload: &[u8]| {
            vec![
                b"topic".to_vec(),
                seq.to_be_bytes().to_vec(),
                payload.to_vec(),
            ]
        };
        live_events.push(frame(0, b"new replay"), b"topic");
        live_events.push(frame(0, b"new replay"), b"topic");
        live_events.record_replay_payload(0, b"old replay").unwrap();

        live_events.clear_fingerprints();
        assert!(!live_events.is_replayed_duplicate(&frame(0, b"old replay"), b"topic", Some(0)));
        live_events.record_replay_payload(0, b"new replay").unwrap();
        live_events.state.get_mut().recent_by_seq.remove(&0);

        assert!(live_events.pop(b"topic", Some(0)).unwrap().1);
        assert!(live_events.pop(b"topic", Some(0)).unwrap().1);
        assert!(!live_events.is_replayed_duplicate(&frame(0, b"new replay"), b"topic", Some(0)));

        let recent = frame(1, b"replay");
        live_events.push(recent.clone(), b"topic");
        live_events.record_replay_payload(1, b"replay").unwrap();
        assert!(live_events.pop(b"topic", Some(1)).unwrap().1);
        assert!(live_events.is_replayed_duplicate(&recent, b"topic", Some(1)));
        assert!(live_events.state.get_mut().queued_by_seq.is_empty());
    }
}
