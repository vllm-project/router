//! Queue live KV event frames during replay and compare payloads for sequences
//! that arrive on both streams before handing back to live ingestion.

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

impl HandoffError {
    fn priority(&self) -> u8 {
        match self {
            Self::InvalidLiveFrame => 0,
            Self::MissingReplayPayload { .. } => 1,
            Self::PayloadConflict { .. } => 2,
        }
    }

    fn prefer(most_specific: &mut Option<Self>, error: Self) {
        if most_specific
            .as_ref()
            .is_none_or(|previous| previous.priority() < error.priority())
        {
            *most_specific = Some(error);
        }
    }
}

#[derive(Debug)]
pub(super) struct ConflictingReplayFingerprint;

#[derive(Default)]
pub(super) struct LiveEventBuffer {
    state: Mutex<LiveEventBufferState>,
}

#[derive(Default)]
struct LiveEventBufferState {
    frames: VecDeque<QueuedLiveFrame>,
    // While set, the trigger is the first frame in the queue.
    gap_trigger_seq: Option<i64>,
    bytes: usize,
    recent_by_seq: HashMap<i64, [u8; 32]>,
    queued_by_seq: HashMap<i64, QueuedSequence>,
}

struct QueuedLiveFrame {
    frames: Vec<Vec<u8>>,
    payload_fingerprint: Option<[u8; 32]>,
}

impl QueuedLiveFrame {
    fn payload_fingerprint(&mut self) -> [u8; 32] {
        let payload = &self.frames[2];
        *self
            .payload_fingerprint
            .get_or_insert_with(|| Sha256::digest(payload).into())
    }
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
            state.track_queued_sequence(seq);
        }
        state.bytes += frame_bytes(&frames);
        state.frames.push_back(QueuedLiveFrame {
            frames,
            payload_fingerprint: None,
        });
        state.frames.len() >= MAX_PENDING_LIVE_FRAMES || state.bytes >= super::MAX_KV_EVENT_BYTES
    }

    pub(super) fn push_gap_trigger(&mut self, seq: i64, frames: Vec<Vec<u8>>) {
        let state = self.state.get_mut();
        debug_assert!(state.gap_trigger_seq.is_none());
        state.gap_trigger_seq = Some(seq);
        state.track_queued_sequence(seq);
        state.bytes += frame_bytes(&frames);
        state.frames.push_front(QueuedLiveFrame {
            frames,
            payload_fingerprint: None,
        });
    }

    /// An already disputed trigger must be matched by replay itself.
    pub(super) fn apply_gap_trigger<E>(
        &self,
        seq: i64,
        apply: impl FnOnce(&[u8]) -> Result<(), E>,
    ) -> Result<bool, E> {
        let mut state = self.state.lock();
        if state.gap_trigger_seq != Some(seq) || state.queued_by_seq[&seq].had_replay_conflict {
            return Ok(false);
        }
        let trigger = state.frames.front_mut().expect("queued gap trigger");
        apply(&trigger.frames[2])?;
        let fingerprint = trigger.payload_fingerprint();
        state.remember_replay_fingerprint(seq, fingerprint);
        Ok(true)
    }

    /// Check a queued copy before releasing its fingerprint reference.
    pub(super) fn pop(
        &mut self,
        topic: &[u8],
        applied_seq: Option<i64>,
    ) -> Option<(Vec<Vec<u8>>, bool)> {
        let state = self.state.get_mut();
        let mut queued = state.frames.pop_front()?;
        let mut is_duplicate = false;
        if let Some(seq) = valid_live_seq(&queued.frames, topic) {
            is_duplicate = applied_seq.is_some_and(|applied_seq| {
                seq <= applied_seq
                    && state
                        .fingerprint(seq)
                        .is_some_and(|replay| replay == queued.payload_fingerprint())
            });
            release_frame(&mut state.queued_by_seq, seq);
        }
        state.bytes -= frame_bytes(&queued.frames);
        Some((queued.frames, is_duplicate))
    }

    pub(super) fn reset(&mut self) {
        *self.state.get_mut() = LiveEventBufferState::default();
    }

    /// Free the replay buffer after overflow, retaining the gap trigger and its conflict status.
    pub(super) fn discard_buffered_frames(&mut self, topic: &[u8]) -> Option<i64> {
        let state = self.state.get_mut();
        let highest = state
            .frames
            .iter()
            .skip(usize::from(state.gap_trigger_seq.is_some()))
            .filter_map(|queued| valid_live_seq(&queued.frames, topic))
            .max();
        if let Some(seq) = state.gap_trigger_seq {
            state.frames.truncate(1);
            state.bytes = frame_bytes(&state.frames[0].frames);
            state
                .queued_by_seq
                .retain(|&queued_seq, _| queued_seq == seq);
            let trigger = state
                .queued_by_seq
                .get_mut(&seq)
                .expect("queued gap trigger");
            trigger.frame_count = 1;
            trigger.replay_fingerprint = None;
            state.recent_by_seq.clear();
        } else {
            *state = LiveEventBufferState::default();
        }
        highest
    }

    pub(super) fn record_replay_payload(
        &self,
        seq: i64,
        payload: &[u8],
    ) -> Result<(), ConflictingReplayFingerprint> {
        let fingerprint = Sha256::digest(payload).into();
        let mut state = self.state.lock();
        if state.gap_trigger_seq == Some(seq)
            && state
                .frames
                .front_mut()
                .expect("queued gap trigger")
                .payload_fingerprint()
                != fingerprint
        {
            state
                .queued_by_seq
                .get_mut(&seq)
                .expect("queued gap trigger")
                .had_replay_conflict = true;
            return Err(ConflictingReplayFingerprint);
        }
        if state
            .fingerprint(seq)
            .is_some_and(|known| known != fingerprint)
        {
            return Err(ConflictingReplayFingerprint);
        }
        state.remember_replay_fingerprint(seq, fingerprint);
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

    /// Clear replay fingerprints while keeping queued live frames and known conflicts.
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
        state.frames.retain_mut(|live_frame| {
            let Some(seq) = valid_live_seq(&live_frame.frames, topic) else {
                HandoffError::prefer(&mut error, HandoffError::InvalidLiveFrame);
                state.bytes -= frame_bytes(&live_frame.frames);
                return false;
            };
            if seq <= applied_seq {
                let queued = state.queued_by_seq.get_mut(&seq);
                let fingerprint = queued
                    .as_ref()
                    .and_then(|entry| entry.replay_fingerprint)
                    .or_else(|| state.recent_by_seq.get(&seq).copied());
                match fingerprint {
                    Some(fingerprint) if fingerprint == live_frame.payload_fingerprint() => {}
                    Some(_) => {
                        if let Some(queued) = queued {
                            queued.had_replay_conflict = true;
                        }
                        HandoffError::prefer(&mut error, HandoffError::PayloadConflict { seq });
                    }
                    None if queued.is_some_and(|queued| queued.had_replay_conflict) => {
                        HandoffError::prefer(&mut error, HandoffError::PayloadConflict { seq });
                    }
                    None if clear_seq.is_none_or(|clear_seq| seq >= clear_seq) => {
                        HandoffError::prefer(
                            &mut error,
                            HandoffError::MissingReplayPayload { seq },
                        );
                    }
                    None => {}
                }
            }
            true
        });
        if let Some(error) = error {
            return Err(error);
        }
        if let Some(seq) = state.gap_trigger_seq.take() {
            let trigger = state.frames.pop_front().expect("queued gap trigger");
            release_frame(&mut state.queued_by_seq, seq);
            state.bytes -= frame_bytes(&trigger.frames);
        }
        if let Some(clear_seq) = clear_seq {
            state.frames.retain(|queued| {
                let Some(seq) =
                    valid_live_seq(&queued.frames, topic).filter(|&seq| seq < clear_seq)
                else {
                    return true;
                };
                release_frame(&mut state.queued_by_seq, seq);
                state.bytes -= frame_bytes(&queued.frames);
                false
            });
        }
        Ok(())
    }
}

impl LiveEventBufferState {
    fn track_queued_sequence(&mut self, seq: i64) {
        let replay_fingerprint = self.recent_by_seq.get(&seq).copied();
        self.queued_by_seq
            .entry(seq)
            .or_insert(QueuedSequence {
                frame_count: 0,
                replay_fingerprint,
                had_replay_conflict: false,
            })
            .frame_count += 1;
    }

    fn remember_replay_fingerprint(&mut self, seq: i64, fingerprint: [u8; 32]) {
        if let Some(queued) = self.queued_by_seq.get_mut(&seq) {
            queued.replay_fingerprint = Some(fingerprint);
        }
        self.recent_by_seq.insert(seq, fingerprint);
        if self.recent_by_seq.len() > RECENT_FINGERPRINT_LIMIT {
            let floor = seq.saturating_sub((RECENT_FINGERPRINT_LIMIT / 2) as i64);
            self.recent_by_seq.retain(|seq, _| *seq >= floor);
        }
    }

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

    fn frame(seq: i64, payload: &[u8]) -> Vec<Vec<u8>> {
        vec![
            b"topic".to_vec(),
            seq.to_be_bytes().to_vec(),
            payload.to_vec(),
        ]
    }

    #[test]
    fn handoff_reports_the_most_specific_error() {
        let mut live_events = LiveEventBuffer::default();
        live_events.push(Vec::new(), b"topic");
        live_events.push(frame(1, b"live"), b"topic");

        assert!(matches!(
            live_events.validate_handoff(b"topic", 1, None),
            Err(HandoffError::MissingReplayPayload { seq: 1 })
        ));

        live_events.push(frame(2, b"live"), b"topic");
        live_events
            .record_replay_payload(2, b"conflicting")
            .unwrap();
        assert!(matches!(
            live_events.validate_handoff(b"topic", 2, None),
            Err(HandoffError::PayloadConflict { seq: 2 })
        ));
    }

    #[test]
    fn replay_fingerprint_stays_until_last_queued_copy_leaves() {
        let mut live_events = LiveEventBuffer::default();
        live_events.push(frame(0, b"new replay"), b"topic");
        live_events.push(frame(0, b"new replay"), b"topic");
        live_events.record_replay_payload(0, b"old replay").unwrap();

        live_events.clear_fingerprints();
        assert!(!live_events.is_replayed_duplicate(&frame(0, b"old replay"), b"topic", Some(0)));
        live_events.record_replay_payload(0, b"new replay").unwrap();
        live_events.state.get_mut().recent_by_seq.remove(&0);
        live_events.validate_handoff(b"topic", 0, None).unwrap();
        assert!(live_events
            .state
            .get_mut()
            .frames
            .iter()
            .all(|queued| queued.payload_fingerprint.is_some()));

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
