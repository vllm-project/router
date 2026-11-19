//! Request-level Completion preparation and forwarding, separate from attempts.

use axum::{
    body::Bytes,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
};
use reqwest::{Client, RequestBuilder};

use super::completion_output::CompletionOutputObserver;
use crate::protocols::spec::CompletionRequest;

pub(super) struct CompletionRoute {
    pub(super) cache_read_fallback: Option<&'static str>,
    pub(super) raw_body: Option<Bytes>,
    pub(super) echo: bool,
}

impl CompletionRoute {
    pub(super) fn forward_request<T: serde::Serialize>(
        &self,
        client: &Client,
        url: &str,
        headers: Option<&HeaderMap>,
        typed_req: &T,
    ) -> RequestBuilder {
        if let Some(raw) = &self.raw_body {
            client
                .post(url)
                .header(
                    CONTENT_TYPE,
                    headers
                        .and_then(|headers| headers.get(CONTENT_TYPE))
                        .cloned()
                        .unwrap_or_else(|| HeaderValue::from_static("application/json")),
                )
                .body(raw.clone())
        } else {
            client.post(url).json(typed_req)
        }
    }

    pub(super) fn output_observer(&self, status: StatusCode) -> CompletionOutputObserver {
        // Echo cannot distinguish prompt from generated text. Each response gets
        // its own observer; unsuccessful responses remain in prefill as well.
        CompletionOutputObserver::new(self.echo || !status.is_success())
    }
}

pub(super) fn prepare_completion_route(
    request: &CompletionRequest,
    raw_body: Option<Bytes>,
) -> CompletionRoute {
    let cache_read_fallback = crate::backend::preprocess::completion_cache_read_fallback(request);
    CompletionRoute {
        cache_read_fallback,
        raw_body,
        echo: request.echo,
    }
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::routers::reject_full_report;
    use serde_json::json;

    #[test]
    fn return_token_ids_preserves_cache_credit_only_for_boolean_values() {
        for (value, expected) in [
            (json!(true), true),
            (json!(false), true),
            (json!(null), true),
            (json!("true"), false),
        ] {
            let request: CompletionRequest = serde_json::from_value(json!({
                "prompt": [1, 2, 3], "stream": true, "return_token_ids": value
            }))
            .unwrap();
            assert_eq!(
                prepare_completion_route(&request, None)
                    .cache_read_fallback
                    .is_none(),
                expected
            );
        }
        let unknown: CompletionRequest = serde_json::from_value(json!({
            "prompt": [1, 2, 3], "future_identity_extension": true
        }))
        .unwrap();
        assert!(prepare_completion_route(&unknown, None)
            .cache_read_fallback
            .is_some());
    }

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

    #[test]
    fn session_and_report_mode_preserve_cache_credit_only_for_known_values() {
        for (extension, expected) in [
            (json!({"echo": true, "logprobs": 0}), false),
            (
                json!({"echo": true, "logprobs": 0, "return_token_ids": true}),
                false,
            ),
            (json!({"echo": false, "logprobs": 0}), true),
            (json!({"session_id": "shared"}), true),
            (json!({"session_id": null}), true),
            (json!({"session_id": 7}), false),
            (
                json!({"vllm_xargs": {"kv_cache_report_mode": "incremental"}}),
                true,
            ),
            (
                json!({"vllm_xargs": {"kv_cache_report_mode": "full"}}),
                false,
            ),
            (json!({"vllm_xargs": {}}), true),
            (json!({"vllm_xargs": null}), true),
            (
                json!({"vllm_xargs": {"kv_cache_report_mode": "unknown"}}),
                false,
            ),
            (json!({"vllm_xargs": {"kv_cache_report_mode": true}}), false),
            (json!({"vllm_xargs": {"other": 1}}), false),
            (
                json!({"vllm_xargs": {"kv_cache_report_mode": "full", "cache_salt": "x"}}),
                false,
            ),
            (json!({"vllm_xargs": "full"}), false),
        ] {
            let mut body = json!({"prompt": [1, 2, 3]});
            body.as_object_mut()
                .unwrap()
                .extend(extension.as_object().unwrap().clone());
            let request: CompletionRequest = serde_json::from_value(body).unwrap();
            assert_eq!(
                prepare_completion_route(&request, None)
                    .cache_read_fallback
                    .is_none(),
                expected,
                "{extension}"
            );
        }
    }
}
