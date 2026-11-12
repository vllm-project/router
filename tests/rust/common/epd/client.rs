//! The client side of the E/P/D rig: the request body and checks on the router's answer.

use super::{QUIET_DURATION, WAIT_DURATION};
use axum::body::BodyDataStream;
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{json, Value};
use tokio::task::JoinHandle;

/// A chat request with two images. A new router sends image 0 to E0 and
/// image 1 to E1 (round-robin per item), and E0's batch comes first.
pub fn two_images() -> Value {
    json!({
        "model": "test",
        "stream": false,
        "max_tokens": 8,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
                {"type": "text", "text": "compare"},
            ],
        }],
    })
}

/// `[client]`: the router's answer.
pub async fn answer(client: JoinHandle<Response>) -> Response {
    tokio::time::timeout(WAIT_DURATION, client)
        .await
        .expect("[client] no answer")
        .expect("[client] the router task failed")
}

/// `[client]`: the router has not answered during `QUIET`.
pub async fn expect_waiting(client: &mut JoinHandle<Response>) {
    assert!(
        tokio::time::timeout(QUIET_DURATION, client).await.is_err(),
        "[client] expected the router to be still waiting"
    );
}

/// `[local drop]`: drop the router's request future while it runs, as hyper
/// does when the client goes away.
pub async fn disconnect(client: JoinHandle<Response>) {
    client.abort();
    match client.await {
        Err(error) => assert!(error.is_cancelled(), "[local drop] {error}"),
        Ok(response) => panic!(
            "[local drop] the router answered {} before the drop",
            response.status()
        ),
    }
}

/// `[client]`: the next chunk of a response body.
pub async fn next_chunk(body: &mut BodyDataStream) -> String {
    let chunk = tokio::time::timeout(WAIT_DURATION, body.next())
        .await
        .expect("[client] no body chunk")
        .expect("[client] the body ended")
        .expect("[client] body error");
    String::from_utf8(chunk.to_vec()).unwrap()
}

/// `[client]`: the rest of a response body, and whether it ended with an
/// error.
pub async fn rest_of_body(mut body: BodyDataStream) -> (String, bool) {
    let mut text = String::new();
    loop {
        match tokio::time::timeout(WAIT_DURATION, body.next())
            .await
            .expect("[client] the body did not end")
        {
            Some(Ok(chunk)) => text.push_str(&String::from_utf8_lossy(&chunk)),
            Some(Err(_)) => return (text, true),
            None => return (text, false),
        }
    }
}

/// `[client]`: a whole response body that ends without an error.
pub async fn text(response: Response) -> String {
    let (text, broken) = rest_of_body(response.into_body().into_data_stream()).await;
    assert!(!broken, "[client] the body ended with an error: {text}");
    text
}
