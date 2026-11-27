//! Cost-model inputs shared by `kv_aware` scoring and `peer_fetch` planning.
//!
//! Kept apart so the planner does not import the policy that calls it.

use crate::kv_index::{RoutableTarget, StorageTier, TierMatch};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadSample {
    pub active_prefill_tokens: u64,
    pub active_requests: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferCredits {
    pub device: f64,
    pub host_pinned: f64,
    pub external: f64,
    pub disk: f64,
}

impl Default for TransferCredits {
    fn default() -> Self {
        Self {
            device: 1.0,
            host_pinned: 0.8,
            external: 0.6,
            disk: 0.3,
        }
    }
}

impl TransferCredits {
    fn for_tier(self, tier: StorageTier) -> f64 {
        match tier {
            StorageTier::Device => self.device,
            StorageTier::HostPinned => self.host_pinned,
            StorageTier::External => self.external,
            StorageTier::Disk => self.disk,
        }
    }

    pub(super) fn valid(self) -> bool {
        self.device == 1.0
            && [self.host_pinned, self.external, self.disk]
                .into_iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostParams {
    pub router_temperature: f64,
    pub transfer_credits: TransferCredits,
    pub block_size: u32,
}

impl Default for CostParams {
    fn default() -> Self {
        Self {
            router_temperature: 0.0,
            transfer_credits: TransferCredits::default(),
            block_size: 16,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CandidateView<'a> {
    pub target: &'a RoutableTarget,
    pub matches: &'a [TierMatch],
    pub load: LoadSample,
    pub input_tokens: usize,
    /// Sum of independently matched batch prompts, when this is a batch.
    pub prefill_tokens: Option<u64>,
}

/// Device prefix plus one external continuation, matching vLLM's two-segment
/// reuse rule. The return value is in tokens and can be fractional because a
/// transfer penalty is a credit ratio.
///
/// `extra_device_prefix` is a device prefix `[0, n)` counted on top of the
/// candidate's own matches without materializing a match for it (0 = none).
pub(super) fn credited_tokens(
    candidate: &CandidateView<'_>,
    params: &CostParams,
    extra_device_prefix: u32,
) -> f64 {
    let device_blocks = crate::kv_index::layout::full_attention_prefix(
        candidate
            .matches
            .iter()
            .filter(|hit| hit.tier == StorageTier::Device)
            .flat_map(|hit| hit.segments.iter().copied()),
    )
    .max(extra_device_prefix)
    // vLLM must recompute the last prompt token to obtain logits. Local
    // GPU prefix reuse is block-aligned, so never credit its final block
    // when that block ends at the prompt boundary.
    .min(
        candidate
            .input_tokens
            .saturating_sub(1)
            .checked_div(params.block_size as usize)
            .unwrap_or(0)
            .min(u32::MAX as usize) as u32,
    );

    let mut best_continuation = 0.0_f64;
    for hit in candidate
        .matches
        .iter()
        .filter(|hit| hit.tier != StorageTier::Device)
    {
        for &(start, end) in &hit.segments {
            if start <= device_blocks && device_blocks < end {
                let blocks = end - device_blocks;
                let credit = params.transfer_credits.for_tier(hit.tier);
                best_continuation = best_continuation.max(f64::from(blocks) * credit);
            }
        }
    }

    let credited_blocks = f64::from(device_blocks) + best_continuation;
    // External reuse may have a different alignment, but it cannot eliminate
    // the last-token logits work either.
    (credited_blocks * f64::from(params.block_size))
        .min(candidate.input_tokens.saturating_sub(1) as f64)
}

pub(super) fn request_prefill_tokens(candidate: &CandidateView<'_>, params: &CostParams) -> u64 {
    request_prefill_tokens_with_device_prefix(candidate, params, 0)
}

/// [`request_prefill_tokens`] as if the candidate also held device blocks
/// `[0, extra_device_prefix)`.
pub(super) fn request_prefill_tokens_with_device_prefix(
    candidate: &CandidateView<'_>,
    params: &CostParams,
    extra_device_prefix: u32,
) -> u64 {
    if let Some(tokens) = candidate.prefill_tokens {
        return tokens;
    }
    // Round remaining work up: a fractional transfer credit never erases a
    // whole token of compute that was not fully credited.
    ((candidate.input_tokens as f64 - credited_tokens(candidate, params, extra_device_prefix))
        .max(0.0))
    .ceil() as u64
}
