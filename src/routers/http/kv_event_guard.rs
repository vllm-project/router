//! Request admission for worker pools with incremental KV event subscribers.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

/// Full reports re-announce existing blocks and cannot be counted as physical
/// copies. Reject them before they contaminate a monitored event stream.
pub(crate) fn reject_full_report(
    monitored: bool,
    vllm_xargs: Option<&serde_json::Value>,
) -> Option<Response> {
    (monitored && is_full_report(vllm_xargs)).then(|| {
        (
            StatusCode::BAD_REQUEST,
            "kv_cache_report_mode=full is unsupported with KV-aware event monitoring; use incremental",
        )
            .into_response()
    })
}

pub(crate) fn is_full_report(vllm_xargs: Option<&serde_json::Value>) -> bool {
    vllm_xargs
        .and_then(|args| args.get("kv_cache_report_mode"))
        .and_then(serde_json::Value::as_str)
        == Some("full")
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn full_report_is_rejected_only_for_monitored_streams() {
        for (monitored, mode, rejected) in [
            (true, json!({"kv_cache_report_mode": "full"}), true),
            (true, json!({"kv_cache_report_mode": "incremental"}), false),
            (true, json!({}), false),
            (false, json!({"kv_cache_report_mode": "full"}), false),
        ] {
            assert_eq!(
                reject_full_report(monitored, Some(&mode)).is_some(),
                rejected
            );
        }
        assert!(reject_full_report(true, None).is_none());
    }
}
