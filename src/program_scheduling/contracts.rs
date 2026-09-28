//! Protocol-neutral errors shared by Program scheduling entry points.

/// Errors returned before a Program request can enter native Router forwarding.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ScheduleError {
    /// A recognized Program identity contract contains malformed data.
    #[error("invalid Program identity: {0}")]
    InvalidIdentity(String),
    /// No healthy DP target is available for the request.
    #[error("no healthy DP targets are available")]
    NoTargets,
    /// A retained request exceeded the configured admission timeout.
    #[error("Program request exceeded its admission queue timeout")]
    QueueTimeout,
}
