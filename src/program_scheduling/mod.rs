//! Program-scoped scheduling contracts and implementation modules.
//!
//! This module is opt-in. Merely linking it into the Router does not alter
//! request routing, worker selection, or request forwarding.

mod binding;
mod contracts;
mod domain;
mod factors;
mod identity;
mod observations;
mod policy;
mod request_pool;
mod runtime;
mod scheduler;
mod scheduler_admission;
mod scheduler_lifecycle;
mod scheduler_state;

pub use binding::{
    ProgramBindingCandidate, ProgramBindingPolicy, ProgramBindingStrategy, ProgramBindings,
};
pub use contracts::ScheduleError;
pub use domain::{
    ProgramDispatch, ProgramRef, ProgramState, ProgramStatus, ProgramTarget,
    ProgramUsageObservation,
};
pub use factors::{ContinuitySample, ProgressTtlFactors, RequestSample};
pub use identity::ProgramIdentity;
pub use observations::{
    BackendObservation, BackendObservationFailure, BackendObservationProvider,
    VllmMetricsObservationProvider,
};
pub use policy::{
    BatchGainEstimate, BatchGainInputs, DecodeThroughputModel, PrefillCostModel, ProgressTtlConfig,
    ProgressTtlPolicyMath,
};
pub use request_pool::ProgramRequestHandle;
pub use runtime::ProgramRuntime;
pub use scheduler::{BackendObservationEpoch, ProgramScheduler};
pub use scheduler_state::ProgramSchedulerConfig;
