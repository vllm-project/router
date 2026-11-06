//! Identity-conditioned return-time shadow prediction.

use super::prediction_profile::{ReturnPrediction, ReturnTimeKey};
use super::scheduler_state::ProgramDecisionState;
use crate::config::types::PredictionProfileConfig;

pub(crate) fn keys_for_decision(
    decision: &ProgramDecisionState,
    model_pool: &str,
    config: &PredictionProfileConfig,
) -> Vec<ReturnTimeKey> {
    let exact = ReturnTimeKey {
        workflow_profile_id: decision.workflow_profile_id.clone(),
        profile_version: decision.profile_version.clone(),
        model_pool: Some(model_pool.to_string()),
        agent_role: decision.agent_role.clone(),
        spawn_reason: decision.spawn_reason.clone(),
        blocks_parent: Some(decision.blocks_parent),
        expected_resume: Some(decision.expected_resume),
        step_bucket: Some(decision.step_id / config.step_bucket_size),
    };
    let mut keys = vec![exact.clone()];

    let mut without_step = exact.clone();
    without_step.step_bucket = None;
    push_unique(&mut keys, without_step);

    let role_state = ReturnTimeKey {
        workflow_profile_id: exact.workflow_profile_id.clone(),
        profile_version: exact.profile_version.clone(),
        model_pool: exact.model_pool.clone(),
        agent_role: exact.agent_role.clone(),
        spawn_reason: None,
        blocks_parent: None,
        expected_resume: exact.expected_resume,
        step_bucket: None,
    };
    push_unique(&mut keys, role_state);

    let model_role = ReturnTimeKey {
        workflow_profile_id: None,
        profile_version: None,
        model_pool: exact.model_pool.clone(),
        agent_role: exact.agent_role.clone(),
        spawn_reason: None,
        blocks_parent: None,
        expected_resume: None,
        step_bucket: None,
    };
    push_unique(&mut keys, model_role);

    push_unique(
        &mut keys,
        ReturnTimeKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: exact.model_pool,
            agent_role: None,
            spawn_reason: None,
            blocks_parent: None,
            expected_resume: None,
            step_bucket: None,
        },
    );
    push_unique(
        &mut keys,
        ReturnTimeKey {
            workflow_profile_id: None,
            profile_version: None,
            model_pool: None,
            agent_role: None,
            spawn_reason: None,
            blocks_parent: None,
            expected_resume: None,
            step_bucket: None,
        },
    );
    keys
}

fn push_unique(keys: &mut Vec<ReturnTimeKey>, key: ReturnTimeKey) {
    if !keys.contains(&key) {
        keys.push(key);
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PendingReturnPrediction {
    pub keys: Vec<ReturnTimeKey>,
    pub prediction: Option<ReturnPrediction>,
}
