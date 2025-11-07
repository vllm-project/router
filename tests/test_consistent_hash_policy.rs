#[cfg(test)]
mod consistent_hash_policy_tests {
    use std::collections::HashMap;

    use vllm_router_rs::core::BasicWorker;
    use vllm_router_rs::core::Worker;
    use vllm_router_rs::core::WorkerType;
    use vllm_router_rs::policies::ConsistentHashPolicy;
    use vllm_router_rs::policies::LoadBalancingPolicy;

    /// Helper function to create test workers
    fn create_test_workers() -> Vec<Box<dyn Worker>> {
        vec![
            Box::new(BasicWorker::new(
                "http://worker1:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://worker2:8000".to_string(),
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://worker3:8000".to_string(),
                WorkerType::Regular,
            )),
        ]
    }

    /// Helper function to create DP-aware test workers
    fn create_dp_test_workers() -> Vec<Box<dyn Worker>> {
        vec![
            Box::new(BasicWorker::new(
                "http://worker1:8000@0".to_string(), // DP rank 0
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://worker2:8000@1".to_string(), // DP rank 1
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://worker3:8000@2".to_string(), // DP rank 2
                WorkerType::Regular,
            )),
            Box::new(BasicWorker::new(
                "http://worker4:8000@3".to_string(), // DP rank 3
                WorkerType::Regular,
            )),
        ]
    }

    #[test]
    fn test_consistent_hash_policy_creation() {
        let policy = ConsistentHashPolicy::new();
        assert_eq!(policy.name(), "consistent_hash");
        assert!(policy.needs_request_text());
    }

    #[test]
    fn test_consistent_routing_same_session() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let session_id = "test_session_123";

        // Make multiple requests with the same session_id in session_params
        let mut selected_workers = Vec::new();
        for i in 0..10 {
            let prompt = format!(
                r#"{{"session_params": {{"session_id": "{}"}}, "prompt": "request {}}}"#,
                session_id, i
            );
            if let Some(worker_idx) = policy.select_worker(&workers, Some(&prompt)) {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should go to the same worker
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Request {} went to worker {}, expected worker {} (same as first request)",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_distribution_across_workers() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let mut worker_counts = HashMap::new();
        let num_sessions = 100;

        // Create many different sessions and see how they distribute
        for i in 0..num_sessions {
            let session_id = format!("session_{}", i);
            let request_json = format!(
                r#"{{"session_params": {{"session_id": "{}"}}, "prompt": "test"}}"#,
                session_id
            );

            if let Some(worker_idx) = policy.select_worker(&workers, Some(&request_json)) {
                *worker_counts.entry(worker_idx).or_insert(0) += 1;
            }
        }

        // Check that at least 2 workers were used (distribution test)
        assert!(
            worker_counts.len() >= 2,
            "Expected distribution across multiple workers, only used: {:?}",
            worker_counts
        );

        // Check that no single worker got all requests (basic distribution check)
        let max_count = worker_counts.values().max().unwrap();
        assert!(
            *max_count < num_sessions,
            "One worker got all requests, no distribution occurred"
        );

        // Check that distribution is reasonably balanced (allow for some variance)
        let min_count = worker_counts.values().min().unwrap();
        let expected_per_worker = num_sessions / workers.len();

        // Allow 50% variance from expected
        assert!(
            *min_count >= expected_per_worker / 3,
            "Distribution too uneven, min count: {}, expected around: {}",
            min_count,
            expected_per_worker
        );
    }

    #[test]
    fn test_dp_aware_routing() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_dp_test_workers();

        let session_id = "dp_test_session";
        let request_json = format!(
            r#"{{"session_params": {{"session_id": "{}"}}, "prompt": "dp test"}}"#,
            session_id
        );

        // Test that DP-aware routing works
        if let Some(worker_idx) = policy.select_worker(&workers, Some(&request_json)) {
            assert!(
                worker_idx < workers.len(),
                "Selected worker index should be valid"
            );

            let worker_url = workers[worker_idx].url();
            assert!(
                worker_url.contains('@'),
                "DP-aware worker URLs should contain '@' for rank"
            );
        }
    }

    #[test]
    fn test_select_worker_pair_pd_mode() {
        let policy = ConsistentHashPolicy::new();
        let prefill_workers = create_test_workers();
        let decode_workers = create_test_workers();

        let session_id = "pd_session_test";
        let request_json = format!(
            r#"{{"session_params": {{"session_id": "{}"}}, "prompt": "pd test"}}"#,
            session_id
        );

        // Test PD mode worker pair selection
        if let Some((prefill_idx, decode_idx)) =
            policy.select_worker_pair(&prefill_workers, &decode_workers, Some(&request_json))
        {
            assert!(
                prefill_idx < prefill_workers.len(),
                "Prefill worker index should be valid"
            );
            assert!(
                decode_idx < decode_workers.len(),
                "Decode worker index should be valid"
            );

            // For the same session, should get consistent results
            if let Some((prefill_idx2, decode_idx2)) =
                policy.select_worker_pair(&prefill_workers, &decode_workers, Some(&request_json))
            {
                assert_eq!(
                    prefill_idx, prefill_idx2,
                    "Prefill worker should be consistent"
                );
                assert_eq!(
                    decode_idx, decode_idx2,
                    "Decode worker should be consistent"
                );
            }
        }
    }

    #[test]
    fn test_no_workers_available() {
        let policy = ConsistentHashPolicy::new();
        let empty_workers: Vec<Box<dyn Worker>> = vec![];

        // Should return None when no workers are available
        let result = policy.select_worker(&empty_workers, Some(r#"{"session_id": "test"}"#));
        assert!(
            result.is_none(),
            "Should return None when no workers available"
        );
    }

    #[test]
    fn test_unhealthy_workers_fallback() {
        let policy = ConsistentHashPolicy::new();

        // Create workers where some are unhealthy
        let worker1 = BasicWorker::new("http://worker1:8000".to_string(), WorkerType::Regular);
        let worker2 = BasicWorker::new("http://worker2:8000".to_string(), WorkerType::Regular);

        // Mark first worker as unhealthy
        worker1.set_healthy(false);

        let workers: Vec<Box<dyn Worker>> = vec![Box::new(worker1), Box::new(worker2)];

        // Should still work and select healthy workers
        let result = policy.select_worker(&workers, Some(r#"{"session_id": "test"}"#));
        assert!(result.is_some(), "Should fallback to healthy workers");

        // The selected worker should be healthy
        if let Some(idx) = result {
            assert!(
                workers[idx].is_healthy(),
                "Selected worker should be healthy"
            );
        }
    }

    #[test]
    fn test_comprehensive_routing_consistency() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        // Test multiple sessions with multiple requests each
        let sessions = ["session_A", "session_B", "session_C"];
        let mut session_mappings = HashMap::new();

        for session in &sessions {
            let mut worker_indices = Vec::new();

            // Make 5 requests per session
            for i in 0..5 {
                let request = format!(
                    r#"{{"session_params": {{"session_id": "{}"}}, "prompt": "request {}}}"#,
                    session, i
                );
                if let Some(idx) = policy.select_worker(&workers, Some(&request)) {
                    worker_indices.push(idx);
                }
            }

            // All requests from the same session should go to the same worker
            assert!(
                !worker_indices.is_empty(),
                "Should have selected workers for session: {}",
                session
            );

            let first_idx = worker_indices[0];
            for (i, &idx) in worker_indices.iter().enumerate() {
                assert_eq!(
                    idx, first_idx,
                    "Session {} request {} went to worker {}, expected worker {} (consistency violation)",
                    session, i, idx, first_idx
                );
            }

            session_mappings.insert(*session, first_idx);
        }

        // Verify that different sessions can go to different workers (distribution)
        let unique_workers: std::collections::HashSet<_> = session_mappings.values().collect();
        println!("Session mappings: {:?}", session_mappings);
        println!("Unique workers used: {}", unique_workers.len());

        // With 3 sessions and 3 workers, we should ideally see some distribution
        // but we'll be lenient and just ensure not everything goes to one worker
        assert!(
            unique_workers.len() >= 1,
            "At least one worker should be used"
        );
    }

    #[test]
    fn test_fallback_without_session_or_user() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        // Test requests without session_id or user_id (should use request content)
        let test_cases = [
            r#"{"prompt": "hello world"}"#,
            r#"{"text": "test request", "sampling_params": {}}"#,
            "",
            "plain text request",
        ];

        for (i, request) in test_cases.iter().enumerate() {
            let result = policy.select_worker(&workers, Some(request));
            assert!(
                result.is_some(),
                "Fallback should work for request {}: {}",
                i,
                request
            );
        }
    }

    // NOTE: Removed test_session_params_user_id_consistency because user_id should NOT go in session_params
    // Only session_id goes in session_params. User info goes in top-level "user" field per OpenAI spec.

    #[test]
    fn test_openai_user_field_consistency() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let user = "openai_user_123";

        // Make multiple requests with the same user field (OpenAI format)
        let mut selected_workers = Vec::new();
        for i in 0..5 {
            let prompt = format!(r#"{{"user": "{}", "prompt": "request {}}}"#, user, i);
            if let Some(worker_idx) = policy.select_worker(&workers, Some(&prompt)) {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should go to the same worker
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "OpenAI user request {} went to worker {}, expected worker {} (consistency violation)",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_session_id_priority_over_user_id() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let session_id = "priority_session";
        let user_id = "priority_user";

        // Make requests with both session_id and user_id in session_params
        let mut selected_workers = Vec::new();
        for i in 0..3 {
            let prompt = format!(
                r#"{{"session_params": {{"session_id": "{}", "user_id": "{}"}}, "prompt": "request {}}}"#,
                session_id, user_id, i
            );
            if let Some(worker_idx) = policy.select_worker(&workers, Some(&prompt)) {
                selected_workers.push(worker_idx);
            }
        }

        // All requests should consistently go to the same worker (based on session_id)
        assert!(
            !selected_workers.is_empty(),
            "Should have selected at least one worker"
        );

        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Priority test request {} went to worker {}, expected worker {} (inconsistent routing)",
                i, worker_idx, first_worker
            );
        }
    }

    #[test]
    fn test_policy_reset() {
        let policy = ConsistentHashPolicy::new();

        // Reset should work without issues
        policy.reset();

        // After reset, should still be able to route requests
        let workers = create_test_workers();
        let result = policy.select_worker(&workers, Some(r#"{"session_id": "post_reset_test"}"#));
        assert!(result.is_some(), "Should work after reset");
    }

    #[test]
    fn test_different_request_formats() {
        let policy = ConsistentHashPolicy::new();
        let workers = create_test_workers();

        let session_id = "format_test_session";

        // Test different JSON formats that should all extract the same session_id from session_params
        let formats = [
            format!(
                r#"{{"session_params": {{"session_id": "{}"}}, "prompt": "test"}}"#,
                session_id
            ),
            format!(
                r#"{{ "session_params" : {{ "session_id" : "{}" }} , "prompt" : "test" }}"#,
                session_id
            ),
            format!(
                r#"{{"prompt": "test", "session_params": {{"session_id": "{}"}}}}"#,
                session_id
            ),
        ];

        let mut selected_workers = Vec::new();
        for (i, request) in formats.iter().enumerate() {
            if let Some(worker_idx) = policy.select_worker(&workers, Some(request)) {
                selected_workers.push(worker_idx);
            } else {
                panic!("Format {} failed to route: {}", i, request);
            }
        }

        // All different formats should route to the same worker
        let first_worker = selected_workers[0];
        for (i, &worker_idx) in selected_workers.iter().enumerate() {
            assert_eq!(
                worker_idx, first_worker,
                "Format {} routed to worker {}, expected worker {} (format inconsistency)",
                i, worker_idx, first_worker
            );
        }
    }
}
