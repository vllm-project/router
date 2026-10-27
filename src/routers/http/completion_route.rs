//! Request-level Completion preparation and forwarding, separate from attempts.

use std::sync::Arc;

use axum::{
    body::Bytes,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
};
use reqwest::{Client, RequestBuilder};

use super::completion_output::CompletionOutputObserver;
use crate::backend::completion_input::{CompletionInput, CompletionPreparation};
use crate::core::Worker;
use crate::kv_index::KvIndexSupervisor;
use crate::policies::{kv_aware_ops, LoadBalancingPolicy};
use crate::protocols::spec::CompletionRequest;

pub(super) struct CompletionRoute<'a> {
    pub(super) tokens: Option<Vec<u32>>,
    pub(super) input_active: bool,
    pub(super) raw_body: Option<&'a Bytes>,
    pub(super) echo: bool,
}

impl CompletionRoute<'_> {
    pub(super) fn forward_request<T: serde::Serialize>(
        &self,
        client: &Client,
        url: &str,
        headers: Option<&HeaderMap>,
        typed_req: &T,
    ) -> RequestBuilder {
        if let Some(raw) = self.raw_body {
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

pub(super) async fn prepare_completion_route<'a>(
    input: Option<&CompletionInput>,
    preparation: &CompletionPreparation,
    policy: &dyn LoadBalancingPolicy,
    request: Arc<CompletionRequest>,
    raw_body: Option<&'a Bytes>,
) -> Result<CompletionRoute<'a>, (StatusCode, String)> {
    let echo = request.echo;
    let input = input.filter(|input| {
        kv_aware_ops(policy).is_some()
            && request
                .model
                .as_deref()
                .is_none_or(|model| model == input.model)
    });
    let tokens = match input {
        Some(input) => preparation.prepare(request, input.clone()).await?,
        _ => None,
    };
    // Fallback drops only input credit, not the original body or output contract.
    Ok(CompletionRoute {
        tokens,
        input_active: input.is_some(),
        raw_body,
        echo,
    })
}

pub(super) fn placement_eligibility(
    input: Option<&CompletionInput>,
    supervisor: Option<&KvIndexSupervisor>,
    workers: &[Arc<dyn Worker>],
) -> Option<Vec<bool>> {
    let input = input?;
    // Qualification is per target, not an all-workers grant. Unknown sources
    // retain ordinary routing and the full known input cost.
    let urls: Vec<_> = workers.iter().map(|worker| worker.base_url()).collect();
    let mut eligible = supervisor.map_or_else(
        || vec![false; workers.len()],
        |supervisor| supervisor.dense_model_eligibility(&urls, &input.cache_model),
    );
    for (eligible, worker) in eligible.iter_mut().zip(workers) {
        *eligible &= worker.dp_rank().unwrap_or(0) == 0;
    }
    Some(eligible)
}
