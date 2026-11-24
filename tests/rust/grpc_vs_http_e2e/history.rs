use super::*;

#[path = "../common/grpc_kv_fixture.rs"]
mod kv_fixture;
use kv_fixture::{KvPublisher, TOPIC};

/// Startup uses real static source configuration and the normal gRPC health
/// probe. No fixture-only policy registration stands in for discovery.
async fn history_app(
    state: MockVllmRs,
) -> (
    Vec<MockVllmRsServer>,
    Arc<vllm_router_rs::server::AppContext>,
    AxumRouter,
    Vec<KvPublisher>,
) {
    let mut workers = Vec::new();
    let mut publishers = Vec::new();
    let mut sources = serde_json::Map::new();
    for _ in 0..2 {
        let worker = MockVllmRsServer::spawn_with_state(MockVllmRs {
            captured: Default::default(),
            kv_block_size: 16,
            ..state.clone()
        })
        .await;
        let publisher = KvPublisher::bind();
        sources.insert(
            worker.grpc_url.clone(),
            json!({
                "endpoints": [publisher.endpoint],
                "replay_endpoints": [publisher.replay_endpoint],
                "topic": TOPIC,
                "block_size": 16, "model": "test-model",
                "hybrid": publisher.layout("test-model", 16)
            }),
        );
        workers.push(worker);
        publishers.push(publisher);
    }
    let urls: Vec<_> = workers
        .iter()
        .map(|worker| worker.grpc_url.clone())
        .collect();
    let mut config = test_config(urls.clone());
    config.kv_aware_completion_history = true;
    config.policy = serde_json::from_value(json!({
        "type": "kv_aware", "router_temperature": 0.0,
        "static_kv_event_sources": sources
    }))
    .unwrap();
    let ctx = create_test_context(config.clone());
    let router = Router::new(urls, &ctx).await.unwrap();
    for publisher in &publishers {
        // Prove the source contract without warming the request used below.
        publisher.restore(&[7; 16], 16).await;
    }
    router.pin_test_token_ids(vec![42; 33]);
    let app = create_test_app(Arc::new(router), reqwest::Client::new(), &config);
    (workers, ctx, app, publishers)
}

async fn history_request(app: AxumRouter, stream: bool) -> axum::response::Response {
    let mut body = json!({"model":"test-model", "prompt":vec![42;33]});
    body["stream"] = stream.into();
    body["max_tokens"] = 8.into();
    app.oneshot(common::http::post("/v1/completions", body.to_string()))
        .await
        .unwrap()
}

fn history_reason_count(
    handle: &metrics_exporter_prometheus::PrometheusHandle,
    reasons: &[&str],
) -> u64 {
    handle
        .render()
        .lines()
        .filter(|line| {
            line.starts_with("vllm_router_history_decisions_total{")
                && reasons
                    .iter()
                    .any(|reason| line.contains(&format!("reason=\"{reason}\"")))
        })
        .map(|line| {
            line.split_whitespace()
                .last()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        })
        .sum()
}

/// Real frontend preparation -> GenerateStream -> valid output -> history.
/// The pinned tokenizer is a CPU fixture, not model/Worker compatibility proof.
#[tokio::test]
async fn grpc_history_observes_valid_output_for_completion() {
    use common::mock_vllm_rs::HoldGate;
    use vllm_router_rs::kv_index::RoutableTarget;
    use vllm_router_rs::policies::KvAwarePolicy;
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    for stream in [false, true] {
        let gate = HoldGate::hold_all();
        let (workers, ctx, app, _publishers) = history_app(MockVllmRs {
            output_gate: Some(gate.clone()),
            ..Default::default()
        })
        .await;
        let policy = ctx.policy_registry.get_default_policy();
        let slots = policy
            .as_any()
            .downcast_ref::<KvAwarePolicy>()
            .unwrap()
            .slot_manager()
            .unwrap();
        let first = tokio::spawn(history_request(app.clone(), stream));
        let selected = tokio::time::timeout(std::time::Duration::from_secs(3), async {
            loop {
                if let Some(index) = workers
                    .iter()
                    .position(|worker| !worker.captured().is_empty())
                {
                    break index;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        let target = RoutableTarget {
            instance_id: workers[selected].grpc_url.clone().into(),
            dp_rank: 0,
        };
        let before = slots.sample(&target);
        assert_eq!(
            (before.active_prefill_tokens, before.active_requests),
            (33, 1),
            "stream={stream}"
        );
        let eligible = history_reason_count(&handle, &["eligible_ordinary", "override"]);
        gate.release();
        let response = first.await.unwrap();
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let response = body_text(response).await;
        assert!(response.contains(if stream { "[DONE]" } else { "text_completion" }));
        assert_eq!(slots.sample(&target), Default::default());
        assert_eq!(
            ctx.worker_registry
                .get_by_url(&workers[selected].grpc_url)
                .unwrap()
                .load(),
            0
        );
        assert_eq!(workers[selected].captured()[0].token_ids, vec![42; 33]);
        assert!(!workers[selected].captured()[0].had_text_prompt);
        let second = history_request(app, stream).await;
        assert_eq!(second.status(), axum::http::StatusCode::OK);
        body_text(second).await;
        assert_eq!(
            workers[selected].captured().len(),
            2,
            "same-cost history choice stream={stream}"
        );
        assert_eq!(workers[1 - selected].captured().len(), 0);
        assert_eq!(
            history_reason_count(&handle, &["eligible_ordinary", "override"]),
            eligible + 1
        );
        assert_eq!(slots.sample(&target), Default::default());
    }
}

#[tokio::test]
async fn grpc_history_does_not_observe_open_error_abort_or_cancelled_stream() {
    use common::mock_vllm_rs::HoldGate;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use vllm_router_rs::kv_index::RoutableTarget;
    use vllm_router_rs::policies::KvAwarePolicy;
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let handle = recorder.handle();
    let _metrics = metrics::set_default_local_recorder(&recorder);
    for case in ["error", "abort", "eof", "cancel"] {
        let gate = HoldGate::hold_all();
        let state = MockVllmRs {
            stream_failures_remaining: Arc::new(AtomicUsize::new(usize::from(case == "error"))),
            aborted_finishes_remaining: Arc::new(AtomicUsize::new(usize::from(case == "abort"))),
            frames: (case == "eof").then(Vec::new),
            output_gate: (case == "cancel").then(|| gate.clone()),
            ..Default::default()
        };
        let (workers, ctx, app, _publishers) = history_app(state).await;
        let policy = ctx.policy_registry.get_default_policy();
        let slots = policy
            .as_any()
            .downcast_ref::<KvAwarePolicy>()
            .unwrap()
            .slot_manager()
            .unwrap();
        let eligible = history_reason_count(&handle, &["eligible_ordinary", "override"]);
        let response = history_request(app.clone(), case != "abort").await;
        if case == "cancel" {
            assert_eq!(response.status(), axum::http::StatusCode::OK);
            // Completion emits no role-only frame. The real RPC has begun,
            // but its output gate is still held when the client disconnects.
            let selected = workers
                .iter()
                .position(|worker| !worker.captured().is_empty())
                .expect("GenerateStream must be dispatched before cancellation");
            assert_eq!(workers[selected].captured()[0].token_ids, vec![42; 33]);
            let target = RoutableTarget {
                instance_id: workers[selected].grpc_url.clone().into(),
                dp_rank: 0,
            };
            let before = slots.sample(&target);
            assert_eq!(
                (before.active_prefill_tokens, before.active_requests),
                (33, 1)
            );
            assert_eq!(
                ctx.worker_registry
                    .get_by_url(&workers[selected].grpc_url)
                    .unwrap()
                    .load(),
                1
            );
            drop(response);
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while workers[selected].state.cancelled.load(Ordering::Relaxed) == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            gate.release();
        } else {
            let text = body_text(response).await;
            assert!(
                text.contains(if case == "abort" { "abort" } else { "error" }),
                "{case}: {text}"
            );
        }
        for worker in &workers {
            let target = RoutableTarget {
                instance_id: worker.grpc_url.clone().into(),
                dp_rank: 0,
            };
            tokio::time::timeout(std::time::Duration::from_secs(3), async {
                while slots.sample(&target) != Default::default() {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            assert_eq!(
                ctx.worker_registry
                    .get_by_url(&worker.grpc_url)
                    .unwrap()
                    .load(),
                0
            );
        }
        let no_overlap = history_reason_count(&handle, &["no_overlap"]);
        let probe = history_request(app, false).await;
        body_text(probe).await;
        assert_eq!(
            history_reason_count(&handle, &["no_overlap"]),
            no_overlap + 1,
            "{case}"
        );
        assert_eq!(
            history_reason_count(&handle, &["eligible_ordinary", "override"]),
            eligible,
            "{case}"
        );
    }
}
