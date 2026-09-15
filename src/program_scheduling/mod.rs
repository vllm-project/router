//! Program-scoped scheduling contracts and implementation modules.
//!
//! This module is opt-in. Merely linking it into the Router does not alter
//! request routing, worker selection, or request forwarding.

mod contracts;
mod identity;

pub use contracts::ScheduleError;
pub use identity::ProgramIdentity;
