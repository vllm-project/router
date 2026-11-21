//! Request-level Completion preparation and forwarding, separate from attempts.

use std::sync::Arc;

use axum::{
    body::Bytes,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
};
use reqwest::{Client, RequestBuilder};

use super::completion_output::CompletionOutputObserver;
use crate::core::Worker;
use crate::protocols::spec::CompletionRequest;

pub(super) struct CompletionRoute {
    pub(super) worker_credit: bool,
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
    // Worker preparation models these modifiers. Unknown identity extensions
    // still execute normally but cannot acquire cache credit.
    let worker_credit = request.lora_path.is_none()
        && request
            .other
            .iter()
            .all(|(field, value)| match field.as_str() {
                "add_special_tokens" => value.is_boolean(),
                "priority" => value.is_i64() || value.is_u64(),
                "truncate_prompt_tokens" => true,
                "vllm_xargs" => {
                    value.is_null()
                        || value.as_object().is_some_and(|args| {
                            args.iter().all(|(key, value)| {
                                key == "kv_cache_report_mode"
                                    && value.as_str() == Some("incremental")
                            })
                        })
                }
                _ => false,
            });
    CompletionRoute {
        worker_credit,
        raw_body,
        echo: request.echo,
    }
}

pub(super) fn placement_eligibility(
    completion: &CompletionRoute,
    workers: &[Arc<dyn Worker>],
) -> Option<Vec<bool>> {
    (!completion.worker_credit).then(|| vec![false; workers.len()])
}

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn incremental_report_retains_prefix_credit() {
        for (args, credit) in [
            (json!({"kv_cache_report_mode": "incremental"}), true),
            (json!({"kv_cache_report_mode": "full"}), false),
            (
                json!({"kv_cache_report_mode": "incremental", "unknown": true}),
                false,
            ),
        ] {
            let request = serde_json::from_value(json!({
                "prompt": [1, 2, 3],
                "vllm_xargs": args,
            }))
            .unwrap();
            assert_eq!(
                prepare_completion_route(&request, None).worker_credit,
                credit
            );
        }
    }
}
