//! Request-level policy resolution and selection for colocated workers.
use super::{
    kv_aware_ops, kv_aware_policy, AffinityKey, KvAwarePolicy, KvSelectionReceipt,
    LoadBalancingPolicy, QualifiedKvInput, RequestHeaders,
};
use crate::core::Worker;
use std::{borrow::Cow, sync::Arc};

pub(crate) struct SelectedWorker {
    pub(crate) worker: Arc<dyn Worker>,
    pub(crate) kv_receipt: Option<KvSelectionReceipt>,
}

/// Resolve the KV implementation for fixed-target accounting or a session binding.
pub(crate) fn binding_kv_policy<'a>(
    native: &'a dyn LoadBalancingPolicy,
    dedicated: Option<&'a KvAwarePolicy>,
    default: Option<&'a dyn LoadBalancingPolicy>,
) -> Option<&'a KvAwarePolicy> {
    kv_aware_policy(native)
        .or(dedicated)
        .or_else(|| default.and_then(kv_aware_policy))
}

/// The Program binding option is independent of native routing. Only allocate
/// a dedicated index policy if the native stage does not already provide one.
pub(crate) fn dedicated_binding_kv(
    enabled: bool,
    native: &dyn LoadBalancingPolicy,
) -> Option<Arc<KvAwarePolicy>> {
    (enabled && kv_aware_ops(native).is_none()).then(|| Arc::new(KvAwarePolicy::default()))
}

pub(crate) struct RegularRequest<'a> {
    pub model_id: Option<&'a str>,
    pub input: QualifiedKvInput<'a>,
    pub lora_name: Option<&'a str>,
    pub cache_salt: Option<&'a str>,
    pub affinity_key: Option<&'a AffinityKey>,
    pub excluded_workers: &'a [String],
}

pub(crate) fn select_regular<'a>(
    policy: &dyn LoadBalancingPolicy,
    available: &[Arc<dyn Worker>],
    request: RegularRequest<'_>,
    legacy_input: impl FnOnce() -> (Option<Cow<'a, str>>, Option<RequestHeaders>),
) -> Option<SelectedWorker> {
    if available.is_empty() {
        return None;
    }

    if let Some(kv_aware) = kv_aware_ops(policy) {
        // The policy sees the full list: a retry exclusion must not
        // rewrite a still-valid binding.
        let receipt = kv_aware.select_worker_with_qualified_inputs(
            available,
            request.input.tokens,
            request.lora_name,
            request.cache_salt,
            request.affinity_key,
            request.excluded_workers,
            request.input.eligibility,
        )?;
        return Some(SelectedWorker {
            worker: available.get(receipt.worker_index)?.clone(),
            kv_receipt: Some(receipt),
        });
    }

    let idx = if let Some(cache) = policy
        .as_any()
        .downcast_ref::<crate::policies::CacheAwarePolicy>()
    {
        cache.select_worker_with_inputs(
            available,
            request.input.tokens,
            request.model_id,
            request.lora_name,
            request.cache_salt,
            "regular",
        )?
    } else {
        let (text, headers) = legacy_input();
        policy.select_worker_with_headers(available, text.as_deref(), headers.as_ref())?
    };
    Some(SelectedWorker {
        worker: available.get(idx)?.clone(),
        kv_receipt: None,
    })
}
