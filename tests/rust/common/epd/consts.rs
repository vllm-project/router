//! Constants that `rig.rs`, `mock.rs` and `client.rs` share. This file
//! imports none of them, so each of them can import it without a cycle.

use std::time::Duration;

pub const CHAT_ROUTE: &str = "/v1/chat/completions";
/// How long a check waits for something that must happen.
pub const WAIT_DURATION: Duration = Duration::from_secs(5);
/// How long a check waits for something that must *not* happen (`QUIET` in the
/// test file). Such a check can miss a late event, but it cannot fail while the
/// behavior is as described.
pub const QUIET_DURATION: Duration = Duration::from_millis(300);
/// Image 0 in `two_images()`. A new router sends it to E0.
pub const IMAGE_0: &str = "data:image/png;base64,AAAA";
/// Image 1 in `two_images()`. A new router sends it to E1.
pub const IMAGE_1: &str = "data:image/png;base64,BBBB";
