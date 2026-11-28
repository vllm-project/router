//! One user-facing policy with optional placement-role overrides.
use crate::config::{ConfigError, ConfigResult, PolicyConfig};
use crate::program_scheduling::ProgramBindingStrategy;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug, Serialize)]
pub struct RoutingPolicy {
    pub default: PolicyConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program: Option<ProgramBindingStrategy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill: Option<PolicyConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_decode: Option<PolicyConfig>,
}

impl From<PolicyConfig> for RoutingPolicy {
    fn from(default: PolicyConfig) -> Self {
        Self {
            default,
            program: None,
            prefill: None,
            native_decode: None,
        }
    }
}

impl RoutingPolicy {
    pub fn prefill(&self) -> &PolicyConfig {
        self.prefill.as_ref().unwrap_or(&self.default)
    }

    pub fn native_decode(&self) -> &PolicyConfig {
        self.native_decode.as_ref().unwrap_or(&self.default)
    }

    pub fn program(&self) -> ProgramBindingStrategy {
        self.program.unwrap_or_else(|| match self.default {
            PolicyConfig::Random => ProgramBindingStrategy::Random,
            PolicyConfig::RoundRobin => ProgramBindingStrategy::ProgramRoundRobin,
            PolicyConfig::CacheAware { .. } => ProgramBindingStrategy::CacheAware,
            PolicyConfig::KvAware { .. } => ProgramBindingStrategy::KvAware,
            PolicyConfig::PowerOfTwo { .. } => ProgramBindingStrategy::PowerOfTwo,
            PolicyConfig::ConsistentHash { .. } => ProgramBindingStrategy::ConsistentHash,
            PolicyConfig::StickyLeastLoaded => ProgramBindingStrategy::StickyLeastLoaded,
            PolicyConfig::RendezvousHash => ProgramBindingStrategy::RendezvousHash,
        })
    }
}

/// A launcher's tuning flags apply equally to all roles using a named policy.
pub struct PolicyTuning {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
    pub eviction_interval_secs: u64,
    pub max_tree_size: usize,
    pub router_temperature: f64,
    pub kv_penalty_host: f64,
    pub kv_penalty_external: f64,
    pub kv_penalty_disk: f64,
}

impl Default for PolicyTuning {
    fn default() -> Self {
        Self {
            cache_threshold: 0.3,
            balance_abs_threshold: 64,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 120,
            max_tree_size: 1 << 25,
            router_temperature: 0.0,
            kv_penalty_host: 0.8,
            kv_penalty_external: 0.6,
            kv_penalty_disk: 0.3,
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Algorithm {
    Named(String),
    Config(PolicyConfig),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Roles {
    default: Algorithm,
    program: Option<ProgramBindingStrategy>,
    prefill: Option<Algorithm>,
    native_decode: Option<Algorithm>,
}

enum Input {
    Roles(Roles),
    Single(Algorithm),
}

impl PolicyTuning {
    fn algorithm(&self, algorithm: Algorithm) -> ConfigResult<PolicyConfig> {
        let name = match algorithm {
            Algorithm::Named(name) => name,
            Algorithm::Config(config) => return Ok(config),
        };
        Ok(match name.as_str() {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "consistent_hash" => PolicyConfig::ConsistentHash { virtual_nodes: 160 },
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_check_interval_secs: 5,
            },
            "sticky_least_loaded" => PolicyConfig::StickyLeastLoaded,
            "rendezvous_hash" => PolicyConfig::RendezvousHash,
            "cache_aware" => PolicyConfig::CacheAware {
                cache_threshold: self.cache_threshold,
                balance_abs_threshold: self.balance_abs_threshold,
                balance_rel_threshold: self.balance_rel_threshold,
                eviction_interval_secs: self.eviction_interval_secs,
                max_tree_size: self.max_tree_size,
            },
            "kv_aware" => PolicyConfig::KvAware {
                router_temperature: self.router_temperature,
                kv_penalty_host: self.kv_penalty_host,
                kv_penalty_external: self.kv_penalty_external,
                kv_penalty_disk: self.kv_penalty_disk,
                static_kv_event_sources: Box::default(),
            },
            _ => {
                return Err(ConfigError::ValidationFailed {
                    reason: format!("unknown request routing policy '{name}'"),
                })
            }
        })
    }

    fn resolve(&self, input: Input) -> ConfigResult<RoutingPolicy> {
        match input {
            Input::Single(algorithm) => self.algorithm(algorithm).map(Into::into),
            Input::Roles(roles) => Ok(RoutingPolicy {
                default: self.algorithm(roles.default)?,
                program: roles.program,
                prefill: roles.prefill.map(|a| self.algorithm(a)).transpose()?,
                native_decode: roles.native_decode.map(|a| self.algorithm(a)).transpose()?,
            }),
        }
    }

    pub fn parse(&self, text: &str) -> ConfigResult<RoutingPolicy> {
        let text = text.trim();
        let input = if text.starts_with('{') {
            // CLI objects use the role schema directly, retaining field-specific errors.
            Input::Roles(
                serde_json::from_str(text).map_err(|e| ConfigError::ValidationFailed {
                    reason: format!("invalid --policy: {e}"),
                })?,
            )
        } else {
            Input::Single(Algorithm::Named(text.to_owned()))
        };
        self.resolve(input)
    }
}

impl<'de> Deserialize<'de> for RoutingPolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        let input = match value {
            serde_json::Value::String(name) => Input::Single(Algorithm::Named(name)),
            serde_json::Value::Object(ref fields) if fields.contains_key("type") => {
                if ["default", "program", "prefill", "native_decode"]
                    .iter()
                    .any(|key| fields.contains_key(*key))
                {
                    return Err(serde::de::Error::custom(
                        "a tagged algorithm cannot include policy roles",
                    ));
                }
                Input::Single(serde_json::from_value(value).map_err(serde::de::Error::custom)?)
            }
            _ => Input::Roles(serde_json::from_value(value).map_err(serde::de::Error::custom)?),
        };
        PolicyTuning::default()
            .resolve(input)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scalar_is_inherited_by_all_roles() {
        for name in [
            "random",
            "round_robin",
            "consistent_hash",
            "cache_aware",
            "kv_aware",
            "power_of_two",
            "sticky_least_loaded",
            "rendezvous_hash",
        ] {
            let p = PolicyTuning::default().parse(name).unwrap();
            assert_eq!(p.default.name(), name);
            assert_eq!(p.prefill().name(), name);
            assert_eq!(p.native_decode().name(), name);
            assert_eq!(serde_json::to_value(p.program()).unwrap(), json!(name));
        }
    }

    #[test]
    fn role_overrides_are_independent_and_survive_serialization() {
        for (input, expected) in [
            (
                json!({"default":"kv_aware", "native_decode":"round_robin"}),
                ["kv_aware", "kv_aware", "round_robin"],
            ),
            (
                json!({"default":"round_robin","program":"least_program_count","prefill":"kv_aware"}),
                ["least_program_count", "kv_aware", "round_robin"],
            ),
            (
                json!({"default":"random","program":"consistent_hash","prefill":"kv_aware","native_decode":"power_of_two"}),
                ["consistent_hash", "kv_aware", "power_of_two"],
            ),
        ] {
            let cli = PolicyTuning::default().parse(&input.to_string()).unwrap();
            let stored = serde_json::to_value(&cli).unwrap();
            for p in [
                cli,
                serde_json::from_value(input).unwrap(),
                serde_json::from_value(stored).unwrap(),
            ] {
                assert_eq!(
                    serde_json::to_value(p.program()).unwrap(),
                    json!(expected[0])
                );
                assert_eq!(p.prefill().name(), expected[1]);
                assert_eq!(p.native_decode().name(), expected[2]);
            }
        }
    }

    #[test]
    fn invalid_or_removed_role_inputs_fail_early() {
        for text in [
            r#"{"prefill":"random"}"#,
            r#"{"type":"random","default":"round_robin","program":"typo"}"#,
            r#"{"default":"kv_aware","decode":"random"}"#,
            r#"{"default":"random","prefill":"least_program_count"}"#,
            r#"{"default":"random","program":"typo"}"#,
        ] {
            assert!(PolicyTuning::default().parse(text).is_err(), "{text}");
            assert!(
                serde_json::from_str::<RoutingPolicy>(text).is_err(),
                "{text}"
            );
        }
        let err = crate::config::ProgramSchedulingConfig::resolve(
            true,
            Some(r#"{"binding_strategy":"kv_aware"}"#),
        )
        .unwrap_err();
        assert!(err.to_string().contains("policy.program"));
    }
}
