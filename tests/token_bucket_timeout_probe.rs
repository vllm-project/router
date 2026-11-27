//! Local regression probe for the explicit acquisition timeout budget.

use std::time::{Duration, Instant};
use vllm_router_rs::core::token_bucket::TokenBucket;

#[tokio::test]
async fn concurrent_waiters_can_use_the_full_explicit_budget() {
    let bucket = TokenBucket::new(1, 1);
    bucket.try_acquire(1.0).await.unwrap();
    let started = Instant::now();
    let budget = Duration::from_secs(4);
    let (first, second) = tokio::join!(
        bucket.acquire_timeout(1.0, budget),
        bucket.acquire_timeout(1.0, budget),
    );
    eprintln!(
        "elapsed={:?}; first={first:?}; second={second:?}",
        started.elapsed()
    );
    assert!(
        first.is_ok() && second.is_ok(),
        "both tokens refill within the four-second budget"
    );
}

#[tokio::test]
async fn unavailable_tokens_still_obey_short_timeout() {
    let bucket = TokenBucket::new(1, 1);
    bucket.try_acquire(1.0).await.unwrap();
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        bucket.acquire_timeout(1.0, Duration::from_millis(30)),
    )
    .await
    .expect("the explicit timeout must finish without waiting for a refill");
    assert!(result.is_err());
    assert!(bucket.try_acquire(1.0).await.is_err());
}

#[tokio::test]
async fn returned_tokens_wake_waiting_acquisition() {
    let bucket = TokenBucket::new(1, 1);
    bucket.try_acquire(1.0).await.unwrap();
    let (result, ()) = tokio::join!(
        bucket.acquire_timeout(1.0, Duration::from_millis(500)),
        async {
            tokio::time::sleep(Duration::from_millis(30)).await;
            bucket.return_tokens(1.0).await;
        },
    );
    assert!(result.is_ok());
}

#[tokio::test]
async fn dropping_waiter_does_not_consume_future_tokens() {
    let bucket = TokenBucket::new(1, 1);
    bucket.try_acquire(1.0).await.unwrap();
    {
        let waiter = bucket.acquire_timeout(1.0, Duration::from_secs(4));
        tokio::pin!(waiter);
        tokio::select! {
            result = &mut waiter => panic!("acquisition finished before cancellation: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(30)) => {},
        }
    }
    bucket.return_tokens(1.0).await;
    assert!(bucket.try_acquire(1.0).await.is_ok());
}
