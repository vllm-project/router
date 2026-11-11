#![cfg(all(feature = "cpu", feature = "ready"))]

//! E/P/D failure, cancellation and cleanup ownership: timing tests.

#[path = "common/epd/rig.rs"]
mod epd;

use axum::http::StatusCode;
use epd::*;
use serde_json::Value;

/// The EC consumer sends 2xx headers, then its body stalls or is invalid.
mod consumer_2xx_with_stalled_or_invalid_body {
    use super::*;

    /// E+PD hands off at PD's 2xx headers, so a stalled or invalid body comes
    /// after the handoff.
    #[tokio::test]
    async fn regular_e_pd() {
        // The body completes, but it is not JSON.
        {
            let mut rig = Rig::new(Path::CombinedPd).await;
            rig.consumer.set_mode(Mode::StallBodyInvalidJson);
            let response = answer(rig.send(two_images())).await;
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "[client] the 2xx headers come through at once"
            );
            let mut body = response.into_body().into_data_stream();
            assert_eq!(next_chunk(&mut body).await, "{", "[client] first part");
            // Disarmed at the headers: nothing to cancel while the body stalls.
            rig.expect_no_cancel().await;
            rig.consumer.release();
            let (rest, broken) = rest_of_body(body).await;
            assert!(!broken, "[client] the body ends normally");
            assert!(
                serde_json::from_str::<Value>(&format!("{{{rest}")).is_err(),
                "[client] the invalid body comes through as it is: {rest:?}"
            );
            rig.expect_no_cancel().await;
        }
        // The client goes away while the body stalls.
        {
            let mut rig = Rig::new(Path::CombinedPd).await;
            rig.consumer.set_mode(Mode::StallBody);
            let response = answer(rig.send(two_images())).await;
            let mut body = response.into_body().into_data_stream();
            next_chunk(&mut body).await;
            drop(body); // [local drop]: the response body
            rig.consumer.expect_aborted().await;
            rig.expect_no_cancel().await;
        }
    }

    /// Sequential E/P/D reads P's whole body, parses it and checks for KV
    /// parameters before it disarms. Each failure drops an armed guard, and D
    /// is never called.
    #[tokio::test]
    async fn sequential_e_p_d() {
        for (mode, error) in [
            (
                Mode::StallBodyInvalidJson,
                "Failed to parse prefill response as JSON",
            ),
            (Mode::StallBodyWithoutKv, "refusing metadata-only decode"),
            (Mode::BreakBody, "Failed to read response body"),
        ] {
            let mut rig = Rig::new(Path::SplitPdSequential).await;
            rig.consumer.set_mode(mode);
            let mut client = rig.send(two_images());
            rig.consumer.wait_requests(1).await;
            // While P's body stalls, the guard is armed and D waits.
            expect_waiting(&mut client).await;
            rig.expect_no_cancel().await;
            rig.decode.expect_no_request();
            rig.consumer.release();
            let response = answer(client).await;
            assert_eq!(
                response.status(),
                StatusCode::INTERNAL_SERVER_ERROR,
                "[client] {mode:?}"
            );
            let text = text(response).await;
            assert!(text.contains(error), "[client] {text}");
            rig.expect_cancel().await;
            rig.decode.expect_no_request();
        }
        // The client goes away while P's body stalls.
        let mut rig = Rig::new(Path::SplitPdSequential).await;
        rig.consumer.set_mode(Mode::StallBody);
        let mut client = rig.send(two_images());
        rig.consumer.wait_requests(1).await;
        expect_waiting(&mut client).await;
        disconnect(client).await;
        rig.consumer.expect_aborted().await;
        rig.expect_cancel().await;
        rig.decode.expect_no_request();
    }

    /// The concurrent path joins P's whole body with D's headers, parses P's
    /// body with `.ok()` and disarms even when that gives nothing. It has no
    /// KV check, and D is already running.
    #[tokio::test]
    async fn concurrent_nixl() {
        for mode in [Mode::StallBodyInvalidJson, Mode::StallBodyWithoutKv] {
            let mut rig = Rig::new(Path::SplitPdConcurrent).await;
            rig.consumer.set_mode(mode);
            let mut client = rig.send(two_images());
            rig.consumer.wait_requests(1).await;
            // [dispatch] D got its request at the same time as P, and answered.
            rig.decode.expect_completed().await;
            expect_waiting(&mut client).await;
            rig.expect_no_cancel().await;
            rig.consumer.release();
            let response = answer(client).await;
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "[client] D's reply, with P in {mode:?}"
            );
            let reply: Value = serde_json::from_str(&text(response).await).unwrap();
            assert_eq!(
                reply["choices"][0]["message"]["content"], "A",
                "[client] D's reply"
            );
            rig.expect_no_cancel().await;
        }
        // The client goes away while P's body stalls. D has sent its headers
        // and holds its body.
        let mut rig = Rig::new(Path::SplitPdConcurrent).await;
        rig.consumer.set_mode(Mode::StallBody);
        rig.decode.set_mode(Mode::StallBody);
        let mut client = rig.send(two_images());
        rig.consumer.wait_requests(1).await;
        rig.decode.wait_requests(1).await;
        expect_waiting(&mut client).await;
        disconnect(client).await;
        rig.consumer.expect_aborted().await;
        // D's cleanup is a separate observation: its connection closes too.
        rig.decode.expect_aborted().await;
        rig.expect_cancel().await;
    }
}
