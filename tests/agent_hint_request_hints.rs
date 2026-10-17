use std::time::Duration;
use vllm_router_rs::program_scheduling::{ProgramIdentity, ProgramSchedulingEnableKey};

#[test]
fn optional_agent_hint_fields_reach_request_scoped_scheduling() {
    let body = serde_json::json!({"agent_hint": {
        "session_id": "task:child", "parent_session_id": "task:root",
        "agent_role": "researcher", "step_id": 7,
        "request_id": "request-7", "priority": 3, "deadline_seconds": 2.5,
        "expected_output_tokens": 512, "kv_retention_ttl_seconds": 4.0
    }});
    let identity = ProgramIdentity::from_request(None, Some(&body), Some("model-a"))
        .unwrap()
        .unwrap();
    assert_eq!(identity.program_id(), "task:child");
    assert_eq!(identity.parent_program_id(), Some("task:root"));
    assert_eq!(identity.agent_role(), Some("researcher"));
    assert_eq!(identity.step_id(), Some(7));
    let hints = identity.request_hints();
    assert_eq!(hints.request_id.as_deref(), Some("request-7"));
    assert_eq!(hints.priority, 3);
    assert_eq!(hints.deadline, Some(Duration::from_secs_f64(2.5)));
    assert_eq!(hints.expected_output_tokens, Some(512));
    assert_eq!(hints.kv_retention_ttl, Some(Duration::from_secs(4)));
}

#[test]
fn malformed_optional_hints_are_rejected_instead_of_silently_ignored() {
    for (name, value) in [
        ("priority", serde_json::json!("urgent")),
        ("deadline", serde_json::json!(-1)),
        ("expected_output_length", serde_json::json!(-1)),
        ("kv_retention_ttl", serde_json::json!("forever")),
        ("step_id", serde_json::json!(-1)),
    ] {
        let mut hint = serde_json::json!({
            "session_id": "p", "blocks_parent": false
        });
        hint[name] = value;
        let body = serde_json::json!({"agent_hint": hint});
        assert!(
            ProgramIdentity::from_request(None, Some(&body), None).is_err(),
            "{name}"
        );
    }
}

#[test]
fn hint_aliases_match_the_canonical_context_contract() {
    let fields = serde_json::json!({
        "request_id":"r", "priority": -2, "deadline": 3,
        "expected_output_length": 64, "kv_retention_ttl": 1
    });
    let mut canonical = fields.clone();
    canonical["program_id"] = serde_json::json!("p");
    let mut hint = fields;
    hint["session_id"] = serde_json::json!("p");
    hint["blocks_parent"] = serde_json::json!(false);
    let a = ProgramIdentity::from_request(
        None,
        Some(&serde_json::json!({
            "vllm_xargs":{"agentic_context": canonical}
        })),
        None,
    )
    .unwrap()
    .unwrap();
    let b = ProgramIdentity::from_request(
        None,
        Some(&serde_json::json!({
            "agent_hint":hint
        })),
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(a.request_hints(), b.request_hints());
}

#[test]
fn ordinary_requests_and_explicit_opt_in_gate_are_preserved() {
    assert!(ProgramIdentity::from_request(
        None,
        Some(&serde_json::json!({
            "model":"m","messages":[]
        })),
        None
    )
    .unwrap()
    .is_none());
    let body = serde_json::json!({"agent_hint":{
        "session_id":"p","blocks_parent":false,"priority":"invalid"
    }});
    assert!(ProgramIdentity::from_request_with_enable_key(
        None,
        Some(&body),
        None,
        ProgramSchedulingEnableKey::AgenticContext
    )
    .unwrap()
    .is_none());
}

#[test]
fn queued_hints_transfer_to_their_own_dispatch_without_bleeding_into_next_request() {
    use std::time::Instant;
    use vllm_router_rs::program_scheduling::ProgramRuntime;
    let hinted = ProgramIdentity::from_request(
        None,
        Some(&serde_json::json!({
            "agent_hint":{"session_id":"p","blocks_parent":false,
                "priority":8,"deadline":2,"expected_output_tokens":128,"kv_retention_ttl":3}
        })),
        Some("m"),
    )
    .unwrap()
    .unwrap();
    let plain = ProgramIdentity::from_request(
        None,
        Some(&serde_json::json!({
            "agent_hint":{"session_id":"p","blocks_parent":false}
        })),
        Some("m"),
    )
    .unwrap()
    .unwrap();
    let now = Instant::now();
    let mut runtime = ProgramRuntime::default();
    let first = runtime.retain_request(&hinted, 64, None, now);
    let second = runtime.retain_request(&plain, 64, None, now);
    let a = runtime.admit_front(&first, "rank-0".into(), now).unwrap();
    let b = runtime.admit_front(&second, "rank-0".into(), now).unwrap();
    assert_eq!(a.request_hints().priority, 8);
    assert_eq!(a.request_hints().expected_output_tokens, Some(128));
    assert_eq!(
        a.request_hints().kv_retention_ttl,
        Some(Duration::from_secs(3))
    );
    assert_eq!(b.request_hints(), plain.request_hints());
    assert_eq!(b.request_hints().priority, 0);
    assert_eq!(b.request_hints().expected_output_tokens, None);
}
