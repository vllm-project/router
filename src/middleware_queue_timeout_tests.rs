use super::*;

async fn queue_results(queue_timeout: Duration, age: Duration) -> [Result<(), StatusCode>; 2] {
    let now = Instant::now();
    queue_results_with_timestamp(queue_timeout, age, now, now.checked_sub(age)).await
}

async fn queue_results_with_timestamp(
    queue_timeout: Duration,
    age: Duration,
    now: Instant,
    backdated: Option<Instant>,
) -> [Result<(), StatusCode>; 2] {
    let queued_at = match backdated {
        Some(queued_at) => queued_at,
        None => {
            // Preserve the requested age even if the clock cannot be backdated.
            // Falling back to `now` without waiting would change the test case.
            tokio::time::sleep(age).await;
            now
        }
    };
    // Start token replenishment only after any fallback aging delay.
    let bucket = Arc::new(TokenBucket::new(1, 1));
    bucket.try_acquire(1.0).await.unwrap();
    let (tx, rx) = mpsc::channel(2);
    let (first_tx, first_rx) = oneshot::channel();
    let (second_tx, second_rx) = oneshot::channel();
    for permit_tx in [first_tx, second_tx] {
        tx.send(QueuedRequest {
            queued_at,
            permit_tx,
        })
        .await
        .unwrap();
    }
    drop(tx);
    let processor = QueueProcessor::new(bucket, rx, queue_timeout);
    tokio::time::timeout(Duration::from_secs(6), async {
        let ((), first, second) = tokio::join!(processor.run(), first_rx, second_rx);
        [first.unwrap(), second.unwrap()]
    })
    .await
    .expect("queue permits must resolve within the test deadline")
}

#[tokio::test]
async fn unrepresentable_backdate_preserves_age_without_refilling_bucket() {
    assert_eq!(
        queue_results_with_timestamp(
            Duration::from_secs(4),
            Duration::from_millis(3970),
            Instant::now(),
            None,
        )
        .await,
        [Err(StatusCode::REQUEST_TIMEOUT); 2],
    );
}

#[tokio::test]
async fn queued_waiters_can_use_the_configured_budget() {
    assert_eq!(
        queue_results(Duration::from_secs(4), Duration::ZERO).await,
        [Ok(()), Ok(())],
    );
}

#[tokio::test]
async fn queue_short_timeout_still_rejects_waiters() {
    assert_eq!(
        queue_results(Duration::from_millis(30), Duration::ZERO).await,
        [Err(StatusCode::REQUEST_TIMEOUT); 2],
    );
}

#[tokio::test]
async fn already_expired_queue_entries_are_rejected() {
    assert_eq!(
        queue_results(Duration::from_secs(4), Duration::from_secs(5)).await,
        [Err(StatusCode::REQUEST_TIMEOUT); 2],
    );
}

#[tokio::test]
async fn time_already_spent_in_queue_is_not_granted_again() {
    assert_eq!(
        queue_results(Duration::from_secs(4), Duration::from_millis(3970)).await,
        [Err(StatusCode::REQUEST_TIMEOUT); 2],
    );
}
