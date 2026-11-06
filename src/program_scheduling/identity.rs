//! Canonical Program identity parsing at the Router boundary.
//!
//! The parser preserves the existing AgentInfer Router precedence: canonical
//! `vllm_xargs.agentic_context`, then `agent_hint`, then supported framework
//! headers. Request and trace identifiers never become Program identifiers.

use super::ScheduleError;
use crate::policies::{hash_key, RequestHeaders};
use http::HeaderMap;
use serde_json::{Map as JsonMap, Value as JsonValue};
use std::time::Duration;

use crate::request_identity::{
    canonical_context, context_string, framework_identity_headers, identity_from_parts,
    normalized_agent_component, read_nonempty_header, scoped_program_id, IdentityError,
    IdentitySource, StableIdentity, CLAUDE_SESSION_HEADER, CODEX_SESSION_HEADER,
    GENERIC_SESSION_HEADER, MAX_IDENTITY_COMPONENT_BYTES,
};

const MAX_PLACEMENT_HASH_KEY_BYTES: usize = 512;

impl From<IdentityError> for ScheduleError {
    fn from(error: IdentityError) -> Self {
        Self::InvalidIdentity(error.0)
    }
}

/// Metadata source that opts one request into Program scheduling.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramSchedulingEnableKey {
    /// Require the canonical `vllm_xargs.agentic_context` body contract.
    #[default]
    #[serde(rename = "vllm_xargs.agentic_context")]
    AgenticContext,
    /// Also accept supported `agent_hint` and framework-header identities.
    Auto,
}

/// Scheduling hints whose lifetime is exactly one retained request.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProgramRequestHints {
    /// Optional upstream correlation ID; never participates in identity.
    pub request_id: Option<String>,
    /// Larger values are resumed before smaller values after force-resume.
    pub priority: i64,
    /// Optional upper bound on time spent waiting in the Router RequestPool.
    pub deadline: Option<Duration>,
    /// Optional replacement for the default per-request decode capacity.
    pub expected_output_tokens: Option<usize>,
    /// Optional replacement for the TTL computed after this request completes.
    pub kv_retention_ttl: Option<Duration>,
}

/// Stable scheduling identity for one Program generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramIdentity {
    model_pool: String,
    program_id: String,
    /// Exact key consumed by the standalone Consistent Hash policy.
    placement_hash_key: String,
    /// Stable task/session group used by Program-scoped scheduling.
    placement_key: String,
    expected_resume: bool,
    task_id: Option<String>,
    session_id: Option<String>,
    agent_id: Option<String>,
    parent_program_id: Option<String>,
    root_program_id: Option<String>,
    blocks_parent: bool,
    agent_role: Option<String>,
    spawn_reason: Option<String>,
    workflow_profile_id: Option<String>,
    profile_version: Option<String>,
    step_id: Option<u64>,
    request_hints: ProgramRequestHints,
}

impl ProgramIdentity {
    /// Cheap conservative gate for typed requests before serializing the body.
    /// The full parser still validates any metadata that passes this gate.
    pub(crate) fn may_be_enabled(
        headers: Option<&HeaderMap>,
        other: &JsonMap<String, JsonValue>,
        enable_key: ProgramSchedulingEnableKey,
    ) -> bool {
        if other.contains_key("vllm_xargs") {
            return true;
        }
        enable_key == ProgramSchedulingEnableKey::Auto
            && (other.contains_key("agent_hint")
                || headers.is_some_and(|headers| {
                    [
                        CLAUDE_SESSION_HEADER,
                        CODEX_SESSION_HEADER,
                        GENERIC_SESSION_HEADER,
                    ]
                    .iter()
                    .any(|name| headers.contains_key(*name))
                }))
    }

    /// Reserve at least the wire output limit, regardless of a smaller hint.
    pub(crate) fn reserve_pd_output(&mut self, wire_limit: usize) {
        self.request_hints.expected_output_tokens = Some(
            self.request_hints
                .expected_output_tokens
                .unwrap_or(0)
                .max(wire_limit),
        );
    }

    pub(crate) fn for_rebinding(
        reference: &super::ProgramRef,
        placement_hash_key: String,
        placement_key: String,
        expected_resume: bool,
    ) -> Self {
        Self {
            model_pool: reference.model_pool().to_string(),
            program_id: reference.program_id().to_string(),
            placement_hash_key,
            placement_key,
            expected_resume,
            task_id: None,
            session_id: None,
            agent_id: None,
            parent_program_id: None,
            root_program_id: None,
            blocks_parent: false,
            agent_role: None,
            spawn_reason: None,
            workflow_profile_id: None,
            profile_version: None,
            step_id: None,
            request_hints: ProgramRequestHints::default(),
        }
    }

    /// Resolve the existing Program identity contract at the Router boundary.
    ///
    /// `headers` contains the framework HTTP headers, `request` is the
    /// serialized generation request, and `model_pool` scopes identical
    /// Program IDs belonging to different served models.
    pub fn from_request(
        headers: Option<&HeaderMap>,
        request: Option<&JsonValue>,
        model_pool: Option<&str>,
    ) -> Result<Option<Self>, ScheduleError> {
        Self::from_request_with_enable_key(
            headers,
            request,
            model_pool,
            ProgramSchedulingEnableKey::Auto,
        )
    }

    /// Resolve identity only when the configured metadata source enables it.
    pub fn from_request_with_enable_key(
        headers: Option<&HeaderMap>,
        request: Option<&JsonValue>,
        model_pool: Option<&str>,
        enable_key: ProgramSchedulingEnableKey,
    ) -> Result<Option<Self>, ScheduleError> {
        let request_object = request.and_then(JsonValue::as_object);
        let canonical = canonical_context(request_object)?;
        if enable_key == ProgramSchedulingEnableKey::AgenticContext && canonical.is_none() {
            return Ok(None);
        }
        let Some(stable) = identity_from_parts(headers, request_object, canonical.as_ref())? else {
            if let Some(context) = &canonical {
                context_string(context, &["agent_role"])?;
            }
            return Ok(None);
        };
        let source = stable.source;
        let mut identity = Self::from_stable(stable, model_pool)?;
        if let Some(context) = &canonical {
            apply_canonical_scheduling(&mut identity, context, headers)?;
        } else if let Some(hint) = request_object
            .and_then(|request| request.get("agent_hint"))
            .and_then(JsonValue::as_object)
        {
            identity.parent_program_id = context_string(hint, &["parent_session_id"])?;
            identity.blocks_parent = context_bool(hint, &["blocks_parent"])?.unwrap_or(true);
            identity.expected_resume =
                context_bool(hint, &["expected_resume"])?.unwrap_or(!identity.blocks_parent);
        } else {
            let (_, _, parent_actor_id) = framework_identity_headers(headers)?;
            identity.expected_resume = identity.agent_id.as_deref().is_none_or(|id| id == "lead");
            identity.agent_role = Some(
                if identity.expected_resume {
                    "lead"
                } else {
                    "subagent"
                }
                .to_string(),
            );
            identity.parent_program_id =
                normalized_agent_component(identity.session_id.as_deref(), parent_actor_id)?
                    .map(|parent| scoped_program_id(&identity.placement_key, Some(&parent)))
                    .or_else(|| {
                        (!identity.expected_resume)
                            .then(|| scoped_program_id(&identity.placement_key, Some("lead")))
                    });
            identity.blocks_parent = !identity.expected_resume;
        }
        if identity.blocks_parent && identity.parent_program_id.is_none() {
            return Err(ScheduleError::InvalidIdentity(match source {
                IdentitySource::AgentHint => {
                    "blocking agent_hint requires parent_session_id".to_string()
                }
                _ => "blocks_parent requires parent_program_id".to_string(),
            }));
        }
        let identity =
            identity.with_placement_hash_key(consistent_hash_request_key(headers, request))?;
        identity.validate_persisted_strings()?;
        Ok(Some(identity))
    }

    fn from_stable(
        stable: StableIdentity,
        model_pool: Option<&str>,
    ) -> Result<Self, ScheduleError> {
        let model_pool = model_pool.unwrap_or("default").to_string();
        if model_pool.len() > MAX_IDENTITY_COMPONENT_BYTES {
            return Err(ScheduleError::InvalidIdentity(format!(
                "model pool exceeds {MAX_IDENTITY_COMPONENT_BYTES} bytes"
            )));
        }
        let placement_key = stable.placement_key().to_string();
        Ok(Self {
            model_pool,
            placement_hash_key: placement_key.clone(),
            placement_key,
            program_id: stable.program_id,
            task_id: stable.task_id,
            session_id: stable.session_id,
            agent_id: stable.agent_id,
            expected_resume: true,
            parent_program_id: None,
            root_program_id: None,
            blocks_parent: false,
            agent_role: None,
            spawn_reason: None,
            workflow_profile_id: None,
            profile_version: None,
            step_id: None,
            request_hints: ProgramRequestHints::default(),
        })
    }

    fn with_placement_hash_key(
        mut self,
        placement_hash_key: String,
    ) -> Result<Self, ScheduleError> {
        if placement_hash_key.len() > MAX_PLACEMENT_HASH_KEY_BYTES {
            return Err(ScheduleError::InvalidIdentity(format!(
                "placement hash key exceeds {MAX_PLACEMENT_HASH_KEY_BYTES} bytes"
            )));
        }
        self.placement_hash_key = placement_hash_key;
        Ok(self)
    }

    /// Bound every string retained beyond request parsing.
    fn validate_persisted_strings(&self) -> Result<(), ScheduleError> {
        let fields = [
            ("model_pool", Some(self.model_pool.as_str())),
            ("program_id", Some(self.program_id.as_str())),
            ("placement_key", Some(self.placement_key.as_str())),
            ("task_id", self.task_id.as_deref()),
            ("session_id", self.session_id.as_deref()),
            ("agent_id", self.agent_id.as_deref()),
            ("parent_program_id", self.parent_program_id.as_deref()),
            ("root_program_id", self.root_program_id.as_deref()),
            ("agent_role", self.agent_role.as_deref()),
            ("spawn_reason", self.spawn_reason.as_deref()),
            ("workflow_profile_id", self.workflow_profile_id.as_deref()),
            ("profile_version", self.profile_version.as_deref()),
            ("request_id", self.request_hints.request_id.as_deref()),
        ];
        for (name, value) in fields {
            if value.is_some_and(|value| value.len() > MAX_IDENTITY_COMPONENT_BYTES) {
                return Err(ScheduleError::InvalidIdentity(format!(
                    "{name} exceeds {MAX_IDENTITY_COMPONENT_BYTES} bytes"
                )));
            }
        }
        Ok(())
    }

    /// Model pool that scopes this Program ID.
    pub fn model_pool(&self) -> &str {
        &self.model_pool
    }

    /// Stable Program ID across requests in one live generation.
    pub fn program_id(&self) -> &str {
        &self.program_id
    }

    /// Stable key used by Program initial-binding policies.
    pub fn placement_hash_key(&self) -> &str {
        &self.placement_hash_key
    }

    /// Task/session group used by Program-scoped scheduling.
    pub fn placement_key(&self) -> &str {
        &self.placement_key
    }

    /// Whether upstream expects another model request after this round.
    pub fn expected_resume(&self) -> bool {
        self.expected_resume
    }

    pub fn task_id(&self) -> Option<&str> {
        self.task_id.as_deref()
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn agent_id(&self) -> Option<&str> {
        self.agent_id.as_deref()
    }

    pub fn parent_program_id(&self) -> Option<&str> {
        self.parent_program_id.as_deref()
    }

    pub fn root_program_id(&self) -> Option<&str> {
        self.root_program_id.as_deref()
    }

    pub fn blocks_parent(&self) -> bool {
        self.blocks_parent
    }

    pub fn agent_role(&self) -> Option<&str> {
        self.agent_role.as_deref()
    }

    pub fn spawn_reason(&self) -> Option<&str> {
        self.spawn_reason.as_deref()
    }

    pub fn workflow_profile_id(&self) -> Option<&str> {
        self.workflow_profile_id.as_deref()
    }

    pub fn profile_version(&self) -> Option<&str> {
        self.profile_version.as_deref()
    }

    pub fn step_id(&self) -> Option<u64> {
        self.step_id
    }

    pub fn request_hints(&self) -> &ProgramRequestHints {
        &self.request_hints
    }
}

fn consistent_hash_request_key(headers: Option<&HeaderMap>, request: Option<&JsonValue>) -> String {
    if let Some(headers) = headers {
        for name in [
            CLAUDE_SESSION_HEADER,
            CODEX_SESSION_HEADER,
            GENERIC_SESSION_HEADER,
        ] {
            if let Ok(Some(value)) = read_nonempty_header(headers, name) {
                return format!("header:{name}:{value}");
            }
        }
    }
    let request_headers = headers.map(|headers| {
        headers
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_lowercase(), value.to_string()))
            })
            .collect::<RequestHeaders>()
    });
    if let Some(key) = request_headers
        .as_ref()
        .and_then(hash_key::extract_hash_key_from_headers)
    {
        return key;
    }
    let request_text = request.and_then(|request| serde_json::to_string(request).ok());
    hash_key::extract_hash_key(request_text.as_deref(), None)
}

fn apply_canonical_scheduling(
    identity: &mut ProgramIdentity,
    context: &JsonMap<String, JsonValue>,
    headers: Option<&HeaderMap>,
) -> Result<(), ScheduleError> {
    let (_, _, header_parent_agent_id) = framework_identity_headers(headers)?;
    let task_id = &identity.task_id;
    let agent_id = &identity.agent_id;
    let program_id = &identity.program_id;
    let agent_role = context_string(context, &["agent_role"])?.or_else(|| {
        agent_id.as_ref().map(|agent_id| {
            if agent_id == "lead" {
                "lead".to_string()
            } else {
                "subagent".to_string()
            }
        })
    });
    let parent_program_id = context_string(context, &["parent_program_id", "parent_id"])?;
    let parent_agent_id = normalized_agent_component(
        task_id.as_deref(),
        context_string(context, &["parent_agent_id"])?.or(header_parent_agent_id),
    )?;
    let parent_program_id = parent_program_id
        .or_else(|| {
            parent_agent_id.as_deref().and_then(|parent| {
                related_program_id(program_id, agent_id.as_deref(), parent, task_id.as_deref())
            })
        })
        .or_else(|| {
            (agent_role.as_deref() == Some("subagent"))
                .then(|| {
                    related_program_id(program_id, agent_id.as_deref(), "lead", task_id.as_deref())
                })
                .flatten()
        });
    let root_program_id = context_string(context, &["root_program_id", "root_id"])?;
    let blocks_parent = context_bool(context, &["blocks_parent", "blocking_parent"])?
        .unwrap_or(agent_role.as_deref() == Some("subagent"));
    if parent_program_id.as_deref() == Some(program_id.as_str()) {
        return Err(ScheduleError::InvalidIdentity(
            "parent_program_id must differ from program_id".to_string(),
        ));
    }
    identity.expected_resume = context_bool(context, &["expected_resume"])?
        .unwrap_or_else(|| !(blocks_parent && agent_role.as_deref() == Some("subagent")));
    identity.parent_program_id = parent_program_id;
    identity.root_program_id = root_program_id;
    identity.blocks_parent = blocks_parent;
    identity.agent_role = agent_role;
    identity.spawn_reason =
        context_string(context, &["spawn_reason", "subagent_type", "agent_type"])?;
    identity.workflow_profile_id =
        context_string(context, &["workflow_profile_id", "workflow_type"])?;
    identity.profile_version = context_string(context, &["profile_version"])?;
    identity.step_id = context_u64(context, &["step_id"])?;
    identity.request_hints = ProgramRequestHints {
        request_id: context_string(context, &["request_id"])?,
        priority: context_i64(context, &["priority"])?.unwrap_or_default(),
        deadline: context_duration(context, &["deadline", "deadline_seconds"])?,
        expected_output_tokens: context_usize(
            context,
            &["expected_output_length", "expected_output_tokens"],
        )?,
        kv_retention_ttl: context_duration(
            context,
            &["kv_retention_ttl", "kv_retention_ttl_seconds"],
        )?,
    };
    Ok(())
}

fn related_program_id(
    program_id: &str,
    agent_id: Option<&str>,
    related_agent_id: &str,
    task_id: Option<&str>,
) -> Option<String> {
    if let Some(agent_id) = agent_id {
        let suffix = format!(":{agent_id}");
        if let Some(prefix) = program_id.strip_suffix(&suffix) {
            return Some(format!("{prefix}:{related_agent_id}"));
        }
    }
    task_id.map(|task_id| scoped_program_id(task_id, Some(related_agent_id)))
}

fn context_bool(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<bool>, ScheduleError> {
    for key in keys {
        let Some(value) = context.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        return value
            .as_bool()
            .map(Some)
            .ok_or_else(|| ScheduleError::InvalidIdentity(format!("{key} must be a boolean")));
    }
    Ok(None)
}

fn context_u64(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<u64>, ScheduleError> {
    for key in keys {
        let Some(value) = context.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        return value.as_u64().map(Some).ok_or_else(|| {
            ScheduleError::InvalidIdentity(format!("{key} must be a non-negative integer"))
        });
    }
    Ok(None)
}

fn context_i64(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<i64>, ScheduleError> {
    for key in keys {
        let Some(value) = context.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        return value
            .as_i64()
            .map(Some)
            .ok_or_else(|| ScheduleError::InvalidIdentity(format!("{key} must be an integer")));
    }
    Ok(None)
}

fn context_usize(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<usize>, ScheduleError> {
    let value = context_u64(context, keys)?;
    value
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| ScheduleError::InvalidIdentity(format!("{} is too large", keys[0])))
        })
        .transpose()
}

fn context_duration(
    context: &JsonMap<String, JsonValue>,
    keys: &[&str],
) -> Result<Option<Duration>, ScheduleError> {
    for key in keys {
        let Some(value) = context.get(*key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let seconds = value.as_f64().ok_or_else(|| {
            ScheduleError::InvalidIdentity(format!("{key} must be seconds as a number"))
        })?;
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(ScheduleError::InvalidIdentity(format!(
                "{key} must be finite and non-negative"
            )));
        }
        return Duration::try_from_secs_f64(seconds)
            .map(Some)
            .map_err(|_| ScheduleError::InvalidIdentity(format!("{key} overflows a duration")));
    }
    Ok(None)
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use super::*;
    use crate::request_identity::{
        resolve_stable_identity, CLAUDE_AGENT_HEADER, CODEX_THREAD_HEADER,
        MAX_AGENTIC_CONTEXT_BYTES,
    };

    #[test]
    fn parses_existing_program_identity_contracts() {
        let canonical = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "program_id": "task-a:researcher",
                "expected_resume": false
            }}
        });
        let identity = ProgramIdentity::from_request(None, Some(&canonical), Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.model_pool(), "model");
        assert_eq!(identity.program_id(), "task-a:researcher");
        assert!(!identity.expected_resume());

        let mut codex_headers = HeaderMap::new();
        codex_headers.insert(CODEX_SESSION_HEADER, "session-a".parse().unwrap());
        codex_headers.insert(CODEX_THREAD_HEADER, "worker".parse().unwrap());
        let identity = ProgramIdentity::from_request(Some(&codex_headers), None, Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "session-a:worker");
        assert_eq!(identity.parent_program_id(), Some("session-a:lead"));
        assert_eq!(identity.placement_hash_key(), "header:session-id:session-a");
        assert!(!identity.expected_resume());

        let mut claude_headers = HeaderMap::new();
        claude_headers.insert(CLAUDE_SESSION_HEADER, "session-c".parse().unwrap());
        claude_headers.insert(CLAUDE_AGENT_HEADER, "researcher".parse().unwrap());
        let identity = ProgramIdentity::from_request(Some(&claude_headers), None, Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "session-c:researcher");
        assert_eq!(
            identity.placement_hash_key(),
            "header:x-claude-code-session-id:session-c"
        );
        assert_eq!(identity.placement_key(), "session-c");
        assert_eq!(identity.parent_program_id(), Some("session-c:lead"));
        assert!(!identity.expected_resume());

        let mut generic_headers = HeaderMap::new();
        generic_headers.insert(GENERIC_SESSION_HEADER, "session-d".parse().unwrap());
        let identity = ProgramIdentity::from_request(Some(&generic_headers), None, Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "session-d:lead");
        assert_eq!(
            identity.placement_hash_key(),
            "header:x-session-id:session-d"
        );
        assert!(identity.expected_resume());
    }

    #[test]
    fn body_contract_precedence_and_defaults_match_current_behavior() {
        let encoded_context = serde_json::json!({
            "vllm_xargs": {"agentic_context": serde_json::json!({
                "program_id": "encoded-program",
                "task_id": null,
                "expected_resume": true
            }).to_string()}
        });
        let identity = ProgramIdentity::from_request(None, Some(&encoded_context), Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "encoded-program");

        let inferred_blocking_subagent = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "task_id": "task-b", "agent_id": "researcher"
            }}
        });
        let identity =
            ProgramIdentity::from_request(None, Some(&inferred_blocking_subagent), Some("model"))
                .unwrap()
                .unwrap();
        assert_eq!(identity.program_id(), "task-b:researcher");
        assert_eq!(identity.placement_key(), "task-b");
        assert!(!identity.expected_resume());

        let explicit_resume = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "task_id": "task-b",
                "agent_id": "researcher",
                "expected_resume": true
            }}
        });
        let identity = ProgramIdentity::from_request(None, Some(&explicit_resume), Some("model"))
            .unwrap()
            .unwrap();
        assert!(identity.expected_resume());

        let agent_hint = serde_json::json!({
            "agent_hint": {
                "session_id": "child-session",
                "parent_session_id": "parent-session",
                "blocks_parent": true
            }
        });
        let identity = ProgramIdentity::from_request(None, Some(&agent_hint), Some("model"))
            .unwrap()
            .unwrap();
        assert_eq!(identity.program_id(), "child-session");
        assert!(!identity.expected_resume());
    }

    #[test]
    fn request_and_trace_ids_do_not_create_program_identity() {
        let mut headers = HeaderMap::new();
        headers.insert("x-request-id", "request-1".parse().unwrap());
        headers.insert("x-trace-id", "trace-1".parse().unwrap());
        assert!(
            ProgramIdentity::from_request(Some(&headers), None, Some("model"))
                .unwrap()
                .is_none()
        );

        let unsupported = serde_json::json!({"session_params": {"session_id": "task-a"}});
        assert!(
            ProgramIdentity::from_request(None, Some(&unsupported), Some("model"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn malformed_recognized_contract_is_rejected() {
        let malformed = serde_json::json!({
            "vllm_xargs": {"agentic_context": {"task_id": [], "agent_id": "worker"}}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&malformed), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));

        let blocking_root = serde_json::json!({
            "agent_hint": {"session_id": "root", "blocks_parent": true}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&blocking_root), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));
    }

    #[test]
    fn canonical_contract_carries_relationships_and_request_hints() {
        let request = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "program_id": "workflow:child",
                "task_id": "workflow",
                "session_id": "session-a",
                "agent_id": "child",
                "parent_agent_id": "lead",
                "root_program_id": "workflow:root",
                "blocks_parent": false,
                "expected_resume": true,
                "agent_role": "subagent",
                "spawn_reason": "research",
                "workflow_profile_id": "autogen-swebench",
                "profile_version": "2026-10-13",
                "request_id": "request-a",
                "step_id": 7,
                "priority": 9,
                "deadline_seconds": 12.5,
                "expected_output_length": 2048,
                "kv_retention_ttl_seconds": 4.5
            }}
        });
        let identity = ProgramIdentity::from_request_with_enable_key(
            None,
            Some(&request),
            Some("model"),
            ProgramSchedulingEnableKey::AgenticContext,
        )
        .unwrap()
        .unwrap();
        assert_eq!(identity.parent_program_id(), Some("workflow:lead"));
        assert_eq!(identity.root_program_id(), Some("workflow:root"));
        assert_eq!(identity.step_id(), Some(7));
        assert_eq!(identity.workflow_profile_id(), Some("autogen-swebench"));
        assert_eq!(identity.profile_version(), Some("2026-10-13"));
        assert_eq!(
            identity.request_hints().request_id.as_deref(),
            Some("request-a")
        );
        assert_eq!(identity.request_hints().priority, 9);
        assert_eq!(
            identity.request_hints().deadline,
            Some(Duration::from_secs_f64(12.5))
        );
        assert_eq!(identity.request_hints().expected_output_tokens, Some(2048));
        assert_eq!(
            identity.request_hints().kv_retention_ttl,
            Some(Duration::from_secs_f64(4.5))
        );
    }

    #[test]
    fn every_persisted_identity_string_is_bounded() {
        for field in [
            "task_id",
            "session_id",
            "agent_id",
            "parent_program_id",
            "root_program_id",
            "agent_role",
            "spawn_reason",
            "workflow_profile_id",
            "profile_version",
            "request_id",
        ] {
            let mut context = serde_json::json!({
                "program_id": "program",
                "task_id": null,
                "blocks_parent": false,
                "expected_resume": true
            });
            context[field] = JsonValue::String("x".repeat(MAX_IDENTITY_COMPONENT_BYTES + 1));
            let request = serde_json::json!({
                "vllm_xargs": {"agentic_context": context}
            });
            assert!(matches!(
                ProgramIdentity::from_request(None, Some(&request), Some("model")),
                Err(ScheduleError::InvalidIdentity(_))
            ));
        }

        let accepted = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "program_id": "program",
                "task_id": null,
                "spawn_reason": "x".repeat(MAX_IDENTITY_COMPONENT_BYTES),
                "expected_resume": true
            }}
        });
        assert!(ProgramIdentity::from_request(None, Some(&accepted), Some("model")).is_ok());

        let multibyte = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "program_id": "program",
                "task_id": null,
                "spawn_reason": "é".repeat(MAX_IDENTITY_COMPONENT_BYTES / 2 + 1),
                "expected_resume": true
            }}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&multibyte), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));
    }

    #[test]
    fn canonical_context_total_size_is_bounded_for_string_and_object_forms() {
        let accepted_context = serde_json::json!({
            "program_id": "program",
            "task_id": null,
            "expected_resume": true,
            "ignored": "x".repeat(1024)
        });
        let accepted_object = serde_json::json!({
            "vllm_xargs": {"agentic_context": accepted_context.clone()}
        });
        assert!(ProgramIdentity::from_request(None, Some(&accepted_object), Some("model")).is_ok());

        let accepted_string = serde_json::json!({
            "vllm_xargs": {"agentic_context": serde_json::to_string(&accepted_context).unwrap()}
        });
        assert!(ProgramIdentity::from_request(None, Some(&accepted_string), Some("model")).is_ok());

        let oversized_context = serde_json::json!({
            "program_id": "program",
            "task_id": null,
            "expected_resume": true,
            "ignored": "x".repeat(MAX_AGENTIC_CONTEXT_BYTES)
        });
        let oversized_object = serde_json::json!({
            "vllm_xargs": {"agentic_context": oversized_context.clone()}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&oversized_object), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));

        let oversized_string = serde_json::json!({
            "vllm_xargs": {"agentic_context": serde_json::to_string(&oversized_context).unwrap()}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&oversized_string), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));
    }

    #[test]
    fn persisted_string_bounds_cover_headers_hints_and_derived_ids() {
        let mut headers = HeaderMap::new();
        headers.insert(
            GENERIC_SESSION_HEADER,
            "x".repeat(MAX_IDENTITY_COMPONENT_BYTES + 1)
                .parse()
                .unwrap(),
        );
        assert!(matches!(
            ProgramIdentity::from_request(Some(&headers), None, Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));

        let hint = serde_json::json!({
            "agent_hint": {
                "session_id": "child",
                "parent_session_id": "x".repeat(MAX_IDENTITY_COMPONENT_BYTES + 1),
                "blocks_parent": true
            }
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&hint), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));

        let derived = serde_json::json!({
            "vllm_xargs": {"agentic_context": {
                "task_id": "x".repeat(MAX_IDENTITY_COMPONENT_BYTES - 2),
                "agent_id": "worker"
            }}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&derived), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));
    }

    #[test]
    fn oversized_hint_durations_degrade_instead_of_panicking() {
        let request = serde_json::json!({
            "vllm_xargs": {"agentic_context": {"program_id": "p", "deadline": 1e300}}
        });
        assert!(matches!(
            ProgramIdentity::from_request(None, Some(&request), Some("model")),
            Err(ScheduleError::InvalidIdentity(_))
        ));
        assert!(resolve_stable_identity(None, Some(&request))
            .unwrap()
            .is_some());
    }

    #[test]
    fn enable_key_defaults_to_canonical_and_auto_enables_compatibility_inputs() {
        let mut headers = HeaderMap::new();
        headers.insert(CLAUDE_SESSION_HEADER, "session-a".parse().unwrap());
        assert!(ProgramIdentity::from_request_with_enable_key(
            Some(&headers),
            None,
            Some("model"),
            ProgramSchedulingEnableKey::AgenticContext,
        )
        .unwrap()
        .is_none());
        assert!(ProgramIdentity::from_request_with_enable_key(
            Some(&headers),
            None,
            Some("model"),
            ProgramSchedulingEnableKey::Auto,
        )
        .unwrap()
        .is_some());
    }

    #[test]
    fn typed_program_gate_keeps_plain_requests_out_and_all_identity_sources_in() {
        let mut other = JsonMap::new();
        let mut headers = HeaderMap::new();
        assert!(!ProgramIdentity::may_be_enabled(
            None,
            &other,
            ProgramSchedulingEnableKey::AgenticContext,
        ));
        assert!(!ProgramIdentity::may_be_enabled(
            None,
            &other,
            ProgramSchedulingEnableKey::Auto,
        ));

        headers.insert(CLAUDE_AGENT_HEADER, "agent-a".parse().unwrap());
        assert!(!ProgramIdentity::may_be_enabled(
            Some(&headers),
            &other,
            ProgramSchedulingEnableKey::Auto,
        ));
        headers.insert(CLAUDE_SESSION_HEADER, "session-a".parse().unwrap());
        assert!(ProgramIdentity::may_be_enabled(
            Some(&headers),
            &other,
            ProgramSchedulingEnableKey::Auto,
        ));
        assert!(!ProgramIdentity::may_be_enabled(
            Some(&headers),
            &other,
            ProgramSchedulingEnableKey::AgenticContext,
        ));

        other.insert("agent_hint".into(), JsonValue::Null);
        assert!(ProgramIdentity::may_be_enabled(
            None,
            &other,
            ProgramSchedulingEnableKey::Auto,
        ));
        assert!(!ProgramIdentity::may_be_enabled(
            None,
            &other,
            ProgramSchedulingEnableKey::AgenticContext,
        ));

        other.insert("vllm_xargs".into(), JsonValue::String("invalid".into()));
        assert!(ProgramIdentity::may_be_enabled(
            None,
            &other,
            ProgramSchedulingEnableKey::AgenticContext,
        ));
    }
}
