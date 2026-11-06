//! Current-request output-length shadow prediction.

use super::prediction_profile::{OutputLengthKey, OutputPrediction};
use super::ProgramIdentity;

pub(crate) fn keys_for_request(identity: &ProgramIdentity) -> Vec<OutputLengthKey> {
    let profile_role = identity.spawn_reason().or_else(|| identity.agent_role());
    let exact = OutputLengthKey {
        workflow_profile_id: identity.workflow_profile_id().map(str::to_string),
        profile_version: identity.profile_version().map(str::to_string),
        model_pool: Some(identity.model_pool().to_string()),
        profile_role: profile_role.map(str::to_string),
    };
    let mut keys = vec![exact.clone()];

    let model_role = OutputLengthKey {
        workflow_profile_id: None,
        profile_version: None,
        model_pool: exact.model_pool.clone(),
        profile_role: exact.profile_role.clone(),
    };
    push_unique(&mut keys, model_role);

    push_unique(
        &mut keys,
        OutputLengthKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: exact.model_pool,
            profile_role: None,
        },
    );
    push_unique(
        &mut keys,
        OutputLengthKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: None,
            profile_role: None,
        },
    );
    keys
}

fn push_unique(keys: &mut Vec<OutputLengthKey>, key: OutputLengthKey) {
    if !keys.contains(&key) {
        keys.push(key);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PendingOutputPrediction {
    pub keys: Vec<OutputLengthKey>,
    pub prediction: Option<OutputPrediction>,
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use serde_json::json;

    fn identity(agent_role: &str, spawn_reason: Option<&str>) -> ProgramIdentity {
        let request = json!({
            "vllm_xargs": {"agentic_context": {
                "program_id": "workflow:agent",
                "task_id": "workflow",
                "expected_resume": true,
                "agent_role": agent_role,
                "spawn_reason": spawn_reason,
                "workflow_profile_id": "autogen-swebench",
                "profile_version": "1"
            }}
        });
        ProgramIdentity::from_request(None, Some(&request), Some("qwen"))
            .unwrap()
            .unwrap()
    }

    #[test]
    fn workflow_stage_is_used_as_profile_role_with_layered_backoff() {
        for role in ["planner", "explorer", "engineer", "reviewer", "verifier"] {
            let keys = keys_for_request(&identity("lead", Some(role)));

            assert_eq!(keys.len(), 4);
            assert_eq!(keys[0].profile_role.as_deref(), Some(role));
            assert_eq!(keys[1].workflow_profile_id, None);
            assert_eq!(keys[1].profile_role.as_deref(), Some(role));
            assert_eq!(keys[2].profile_role, None);
            assert_eq!(keys[3].model_pool, None);
        }
    }

    #[test]
    fn agent_role_is_the_fallback_when_workflow_stage_is_absent() {
        let keys = keys_for_request(&identity("subagent", None));

        assert_eq!(keys[0].profile_role.as_deref(), Some("subagent"));
    }
}
