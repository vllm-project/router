//! Program-scoped scheduling contracts and implementation modules.
//!
//! This module is opt-in. Merely linking it into the Router does not alter
//! request routing, worker selection, or request forwarding.

mod binding;
mod contracts;
mod domain;
mod identity;
mod observations;
mod request_pool;
mod runtime;

pub use binding::{
    ProgramBindingCandidate, ProgramBindingPolicy, ProgramBindingStrategy, ProgramBindings,
};
pub use contracts::ScheduleError;
pub use domain::{
    ProgramDispatch, ProgramRef, ProgramState, ProgramStatus, ProgramTarget,
    ProgramUsageObservation,
};
pub use identity::ProgramIdentity;
pub use observations::{
    BackendObservation, BackendObservationFailure, BackendObservationProvider,
    VllmMetricsObservationProvider,
};
pub use request_pool::ProgramRequestHandle;
pub use runtime::ProgramRuntime;
