use super::{ConfigError, ConfigResult, RouterConfig};
use std::path::Path;

impl RouterConfig {
    /// Load router configuration from a JSON or YAML file.
    ///
    /// Format selection follows the file extension (`.yaml`/`.yml` -> YAML,
    /// `.json` -> JSON). Extensionless or unrecognized filenames try JSON first
    /// and fall back to YAML so Kubernetes ConfigMap mounts work transparently.
    pub fn from_file(path: impl AsRef<Path>) -> ConfigResult<Self> {
        let path = path.as_ref();
        let content =
            std::fs::read_to_string(path).map_err(|error| ConfigError::ValidationFailed {
                reason: format!("Failed to read config file '{}': {error}", path.display()),
            })?;
        if content.trim().is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: format!("Config file '{}' is empty", path.display()),
            });
        }

        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());

        match ext.as_deref() {
            Some("yaml" | "yml") => {
                Self::from_yaml_str(&content).map_err(|err| Self::with_file_context(path, err))
            }
            Some("json") => {
                Self::from_json_str(&content).map_err(|err| Self::with_file_context(path, err))
            }
            _ => Self::from_json_str(&content)
                .or_else(|_| Self::from_yaml_str(&content))
                .map_err(|err| Self::with_file_context(path, err)),
        }
    }

    /// Parse router configuration from a YAML string.
    pub fn from_yaml_str(content: &str) -> ConfigResult<Self> {
        if content.trim().is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "YAML config is empty".to_string(),
            });
        }
        serde_yaml::from_str(content).map_err(|error| ConfigError::ValidationFailed {
            reason: format!("Failed to parse YAML config: {error}"),
        })
    }

    /// Parse router configuration from a JSON string.
    pub fn from_json_str(content: &str) -> ConfigResult<Self> {
        if content.trim().is_empty() {
            return Err(ConfigError::ValidationFailed {
                reason: "JSON config is empty".to_string(),
            });
        }
        serde_json::from_str(content).map_err(|error| ConfigError::ValidationFailed {
            reason: format!("Failed to parse JSON config: {error}"),
        })
    }

    fn with_file_context(path: &Path, err: ConfigError) -> ConfigError {
        match err {
            ConfigError::ValidationFailed { reason } => ConfigError::ValidationFailed {
                reason: format!("Config file '{}': {reason}", path.display()),
            },
            other => other,
        }
    }
}
