//! Startup snapshot for the opt-in, static HTTP Completion input contract.
//!
//! The operator binds an alias to immutable local frontend assets and an
//! identically configured Worker cohort. This is not remote attestation.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use axum::http::StatusCode;
use serde_json::Value;
use tokio::sync::{Semaphore, TryAcquireError};
use vllm_tokenizer::DynTokenizer;

use crate::protocols::spec::CompletionRequest;

use super::preprocess::FrontendHandle;
use super::{validate_completion_tokenizer_definition, EngineFrontend};

/// Shared CPU budget, with a bounded number of outstanding preparations per Router.
#[derive(Debug)]
pub(crate) struct CompletionPreparation {
    cpu: Arc<Semaphore>,
    admission: Arc<Semaphore>,
    request_timeout: Duration,
    queue_timeout: Duration,
}

struct AbortPreparation(tokio::task::AbortHandle);

impl Drop for AbortPreparation {
    fn drop(&mut self) {
        // A queued blocking task can be cancelled. An encoder already running
        // cannot be interrupted; its owned permits survive until actual exit.
        self.0.abort();
    }
}

impl CompletionPreparation {
    #[cfg(all(test, feature = "cpu", feature = "ready"))]
    pub(crate) fn available_admission_permits(&self) -> usize {
        self.admission.available_permits()
    }

    pub fn new(max_requests: usize, request_timeout: Duration, queue_timeout: Duration) -> Self {
        static CPU: OnceLock<Arc<Semaphore>> = OnceLock::new();
        Self {
            cpu: CPU
                .get_or_init(|| {
                    Arc::new(Semaphore::new(
                        std::thread::available_parallelism().map_or(1, usize::from),
                    ))
                })
                .clone(),
            admission: Arc::new(Semaphore::new(max_requests)),
            request_timeout,
            queue_timeout,
        }
    }

    pub async fn prepare(
        &self,
        request: Arc<CompletionRequest>,
        input: CompletionInput,
    ) -> Result<Option<Vec<u32>>, (StatusCode, String)> {
        let deadline = tokio::time::Instant::now() + self.request_timeout;
        let admission = self.admission.clone().try_acquire_owned().map_err(|_| {
            (
                StatusCode::TOO_MANY_REQUESTS,
                "Completion preparation capacity exhausted".into(),
            )
        })?;
        let queue_deadline = deadline.min(tokio::time::Instant::now() + self.queue_timeout);
        let cpu = match self.cpu.clone().try_acquire_owned() {
            Ok(cpu) => cpu,
            Err(TryAcquireError::NoPermits) => {
                let cpu = tokio::time::timeout_at(queue_deadline, self.cpu.clone().acquire_owned())
                    .await
                    .map_err(|_| {
                        (
                            StatusCode::REQUEST_TIMEOUT,
                            "Completion preparation queue timed out".into(),
                        )
                    })?
                    .map_err(|_| {
                        (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "Completion preparation unavailable".into(),
                        )
                    })?;
                // Timeout polls a ready permit before its timer. A delayed
                // waiter must still respect the queue deadline before encoding.
                if tokio::time::Instant::now() >= queue_deadline {
                    return Err((
                        StatusCode::REQUEST_TIMEOUT,
                        "Completion preparation queue timed out".into(),
                    ));
                }
                cpu
            }
            Err(TryAcquireError::Closed) => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Completion preparation unavailable".into(),
                ));
            }
        };
        if tokio::time::Instant::now() >= deadline {
            return Err((
                StatusCode::REQUEST_TIMEOUT,
                "Completion preparation timed out".into(),
            ));
        }
        let job = tokio::task::spawn_blocking(move || {
            let (_admission, _cpu) = (admission, cpu);
            super::prepare_completion_classified(
                &request,
                input.tokenizer.as_ref(),
                input.vocab_size,
            )
        });
        let _abort = AbortPreparation(job.abort_handle());
        let prepared = tokio::time::timeout_at(deadline, job)
            .await
            .map_err(|_| {
                (
                    StatusCode::REQUEST_TIMEOUT,
                    "Completion preparation timed out".into(),
                )
            })?
            .map_err(|_| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "Completion preparation job failed".into(),
                )
            })?;
        if tokio::time::Instant::now() >= deadline {
            return Err((
                StatusCode::REQUEST_TIMEOUT,
                "Completion preparation timed out".into(),
            ));
        }
        completion_preparation_result(prepared)
    }
}

fn completion_preparation_result(
    prepared: Result<Vec<u32>, super::CompletionPreparationError>,
) -> Result<Option<Vec<u32>>, (StatusCode, String)> {
    match prepared {
        Ok(tokens) => {
            tracing::debug!(token_count = tokens.len(), "Completion input prepared once");
            Ok(Some(tokens))
        }
        Err(super::CompletionPreparationError::Unsupported(reason)) => {
            let reason = match reason.as_str() {
                "completion preparation does not support lora_path" => "lora_path",
                "completion preparation does not support session_params" => "session_params",
                "completion preparation does not support suffix" => "suffix",
                "completion preparation does not support batched prompts" => "batched_prompts",
                _ => "extension_field",
            };
            tracing::debug!(reason, "Completion input uses ordinary fallback");
            Ok(None)
        }
        Err(super::CompletionPreparationError::InvalidRequest(error)) => {
            Err((StatusCode::BAD_REQUEST, error))
        }
        Err(super::CompletionPreparationError::ServiceFailure(error)) => {
            Err((StatusCode::SERVICE_UNAVAILABLE, error))
        }
    }
}

#[derive(Clone)]
pub(crate) struct CompletionInput {
    /// Exact served alias; distinct from local asset paths and event identity.
    pub model: String,
    pub cache_model: String,
    pub vocab_size: u32,
    pub tokenizer: DynTokenizer,
}

impl std::fmt::Debug for CompletionInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionInput")
            .field("model", &self.model)
            .field("cache_model", &self.cache_model)
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl CompletionInput {
    pub async fn from_env(frontend: &EngineFrontend) -> Result<Option<Self>> {
        let Some(model) = std::env::var("VLLM_ROUTER_COMPLETION_MODEL_ALIAS").ok() else {
            return Ok(None);
        };
        let path = PathBuf::from(std::env::var("VLLM_ROUTER_MODEL").context(
            "Completion requires VLLM_ROUTER_MODEL as an immutable local model directory",
        )?);
        let cache_model =
            std::env::var("VLLM_ROUTER_COMPLETION_CACHE_MODEL").unwrap_or_else(|_| model.clone());
        Self::load(
            &model,
            &cache_model,
            &path,
            frontend.completion_frontend(&model),
        )
        .await
        .map(Some)
    }

    pub(super) async fn load(
        model: &str,
        cache_model: &str,
        path: &Path,
        frontend: impl Future<Output = Result<FrontendHandle>>,
    ) -> Result<Self> {
        if model.trim().is_empty() || model != model.trim() {
            bail!("VLLM_ROUTER_COMPLETION_MODEL_ALIAS must be a nonempty exact served-model alias");
        }
        if cache_model.trim().is_empty() || cache_model != cache_model.trim() {
            bail!(
                "VLLM_ROUTER_COMPLETION_CACHE_MODEL must be a nonempty exact cache-model identity"
            );
        }
        let path = path.canonicalize().context("Completion model directory")?;
        if !path.is_dir() {
            bail!("Completion model assets must be a local directory");
        }
        let config: Value =
            serde_json::from_slice(&tokio::fs::read(path.join("config.json")).await?)
                .context("Completion model config")?;
        let vocab_size = config
            .get("vocab_size")
            .and_then(Value::as_u64)
            .and_then(|size| u32::try_from(size).ok())
            .filter(|size| *size != 0)
            .context("Completion config.json requires a nonzero u32 vocab_size")?;
        // Do not poll the generic resolver until the local-only Completion
        // contract is valid: its normal gRPC path may resolve remote assets.
        let handle = frontend.await?;
        // Resolve through the existing alias-keyed cache. A cache hit must
        // validate its loaded source, not a freshly recomputed environment path.
        let source = Path::new(
            handle
                .asset_source()
                .context("Completion requires a real tokenizer backend")?,
        )
        .canonicalize()
        .context("Completion frontend asset directory")?;
        if !source.is_dir() {
            bail!("Completion frontend assets must be a local directory");
        }
        let definition: Value =
            serde_json::from_slice(&tokio::fs::read(source.join("tokenizer.json")).await?)
                .context("Completion tokenizer definition")?;
        validate_completion_tokenizer_definition(&definition).map_err(anyhow::Error::msg)?;
        let tokenizer = handle
            .tokenizer()
            .context("Completion requires a real tokenizer backend")?;
        Ok(Self {
            model: model.to_owned(),
            cache_model: cache_model.to_owned(),
            vocab_size,
            tokenizer,
        })
    }
}

#[cfg(all(test, feature = "cpu", feature = "ready"))]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    struct ControlledTokenizer {
        started: tokio::sync::mpsc::UnboundedSender<()>,
        release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
        encodes: Arc<AtomicUsize>,
    }

    impl vllm_tokenizer::Tokenizer for ControlledTokenizer {
        fn encode(&self, text: &str, _: bool) -> vllm_tokenizer::Result<Vec<u32>> {
            self.encodes.fetch_add(1, Ordering::Relaxed);
            self.started.send(()).unwrap();
            self.release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            assert_ne!(text, "panic", "controlled tokenizer panic");
            Ok(vec![1, 3, 4])
        }

        fn encode_ordinary(&self, _: &str) -> vllm_tokenizer::Result<Vec<u32>> {
            unreachable!()
        }
        fn decode(&self, _: &[u32], _: bool) -> vllm_tokenizer::Result<String> {
            unreachable!()
        }
        fn token_to_id(&self, _: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _: u32) -> Option<String> {
            None
        }
    }

    fn controlled_input() -> (
        CompletionInput,
        tokio::sync::mpsc::UnboundedReceiver<()>,
        std::sync::mpsc::Sender<()>,
        Arc<AtomicUsize>,
    ) {
        let (started, events) = tokio::sync::mpsc::unbounded_channel();
        let (release, released) = std::sync::mpsc::channel();
        let encodes = Arc::new(AtomicUsize::new(0));
        (
            CompletionInput {
                model: "public".into(),
                cache_model: "engine".into(),
                vocab_size: 16,
                tokenizer: Arc::new(ControlledTokenizer {
                    started,
                    release: std::sync::Mutex::new(released),
                    encodes: encodes.clone(),
                }),
            },
            events,
            release,
            encodes,
        )
    }

    fn preparation(capacity: usize) -> Arc<CompletionPreparation> {
        Arc::new(CompletionPreparation {
            cpu: Arc::new(Semaphore::new(1)),
            admission: Arc::new(Semaphore::new(capacity)),
            request_timeout: Duration::from_secs(5),
            queue_timeout: Duration::from_secs(5),
        })
    }

    fn request(prompt: serde_json::Value) -> Arc<CompletionRequest> {
        Arc::new(
            serde_json::from_value(serde_json::json!({"model":"public","prompt":prompt})).unwrap(),
        )
    }

    async fn permits(preparation: &CompletionPreparation, cpu: usize, admission: usize) {
        tokio::time::timeout(Duration::from_secs(5), async {
            while preparation.cpu.available_permits() != cpu
                || preparation.admission.available_permits() != admission
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("preparation permits reached expected state");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_cpu_queue_and_running_cancellation_keep_real_budget() {
        let prep = preparation(2);
        let (input, mut started, release, encodes) = controlled_input();
        let first = tokio::spawn({
            let (prep, input) = (prep.clone(), input.clone());
            async move {
                prep.prepare(request(serde_json::json!("first")), input)
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(5), started.recv())
            .await
            .unwrap()
            .unwrap();
        // This current-thread executor remains available while encode is blocked.
        assert_eq!(tokio::spawn(async { 7 }).await.unwrap(), 7);
        let queued = tokio::spawn({
            let (prep, input) = (prep.clone(), input.clone());
            async move {
                prep.prepare(request(serde_json::json!("queued")), input)
                    .await
            }
        });
        permits(&prep, 0, 0).await;
        assert_eq!(
            prep.prepare(request(serde_json::json!([1])), input.clone())
                .await
                .unwrap_err()
                .0,
            StatusCode::TOO_MANY_REQUESTS
        );
        queued.abort();
        assert!(queued.await.unwrap_err().is_cancelled());
        permits(&prep, 0, 1).await;
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        permits(&prep, 0, 1).await;
        assert_eq!(encodes.load(Ordering::Relaxed), 1);
        release.send(()).unwrap();
        permits(&prep, 1, 2).await;
        assert_eq!(
            prep.prepare(request(serde_json::json!([1, 3])), input)
                .await
                .unwrap(),
            Some(vec![1, 3])
        );
        assert_eq!(
            encodes.load(Ordering::Relaxed),
            1,
            "IDs and cancelled queued requests never encode"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_cpu_timeout_and_panic_release_only_after_job_exit() {
        let mut prep = preparation(1);
        Arc::get_mut(&mut prep).unwrap().request_timeout = Duration::from_millis(200);
        let (input, mut started, release, _) = controlled_input();
        let task = tokio::spawn({
            let (prep, input) = (prep.clone(), input.clone());
            async move {
                prep.prepare(request(serde_json::json!("first")), input)
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(5), started.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            task.await.unwrap().unwrap_err().1,
            "Completion preparation timed out"
        );
        permits(&prep, 0, 0).await;
        release.send(()).unwrap();
        permits(&prep, 1, 1).await;
        release.send(()).unwrap();
        let error = prep
            .prepare(request(serde_json::json!("panic")), input)
            .await
            .unwrap_err();
        assert_eq!(error.1, "Completion preparation job failed");
        permits(&prep, 1, 1).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_cpu_expired_queue_cannot_start_when_permit_becomes_ready() {
        let mut prep = preparation(1);
        Arc::get_mut(&mut prep).unwrap().queue_timeout = Duration::from_millis(20);
        let held = prep.cpu.clone().acquire_owned().await.unwrap();
        let (input, _started, release, encodes) = controlled_input();
        // Keep a regressing implementation's blocking job finite, too.
        release.send(()).unwrap();
        let mut queued = Box::pin(prep.prepare(request(serde_json::json!("queued")), input));
        assert!(futures::poll!(&mut queued).is_pending());
        assert_eq!(prep.admission.available_permits(), 0);
        // Delay this single-thread runtime beyond the queue deadline, then make
        // the semaphore ready before the timeout future is polled again.
        std::thread::sleep(Duration::from_millis(50));
        drop(held);
        let result = queued.await;
        assert_eq!(
            result,
            Err((
                StatusCode::REQUEST_TIMEOUT,
                "Completion preparation queue timed out".into(),
            ))
        );
        permits(&prep, 1, 1).await;
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_cpu_zero_queue_timeout_allows_only_immediate_capacity() {
        let mut prep = preparation(1);
        Arc::get_mut(&mut prep).unwrap().queue_timeout = Duration::ZERO;
        let (input, _started, _release, encodes) = controlled_input();
        assert_eq!(
            prep.prepare(request(serde_json::json!([1])), input.clone())
                .await
                .unwrap(),
            Some(vec![1])
        );
        let held = prep.cpu.clone().acquire_owned().await.unwrap();
        assert_eq!(
            prep.prepare(request(serde_json::json!([1])), input.clone())
                .await,
            Err((
                StatusCode::REQUEST_TIMEOUT,
                "Completion preparation queue timed out".into(),
            ))
        );
        drop(held);
        permits(&prep, 1, 1).await;
        prep.cpu.close();
        assert_eq!(
            prep.prepare(request(serde_json::json!([1])), input).await,
            Err((
                StatusCode::SERVICE_UNAVAILABLE,
                "Completion preparation unavailable".into(),
            ))
        );
        assert_eq!(prep.admission.available_permits(), 1);
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completion_cpu_queue_deadline_and_helper_errors_do_not_fallback() {
        let mut prep = preparation(1);
        Arc::get_mut(&mut prep).unwrap().queue_timeout = Duration::from_millis(200);
        let held = prep.cpu.clone().acquire_owned().await.unwrap();
        let (input, _started, _release, encodes) = controlled_input();
        let error = prep
            .prepare(request(serde_json::json!([1])), input.clone())
            .await
            .unwrap_err();
        assert_eq!(error.1, "Completion preparation queue timed out");
        assert_eq!(prep.admission.available_permits(), 1);
        drop(held);
        let error = prep
            .prepare(request(serde_json::json!([-1])), input.clone())
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        let error = prep
            .prepare(request(serde_json::json!([16])), input.clone())
            .await
            .unwrap_err();
        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        let mut invalid = input;
        invalid.vocab_size = 0;
        assert_eq!(
            prep.prepare(request(serde_json::json!([1])), invalid)
                .await
                .unwrap_err()
                .0,
            StatusCode::SERVICE_UNAVAILABLE
        );
        permits(&prep, 1, 1).await;
        assert_eq!(encodes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn completion_cpu_cancelled_blocking_pool_queue_never_encodes() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap()
            .block_on(async {
                let (started, occupied) = tokio::sync::oneshot::channel();
                let (release_pool, released) = std::sync::mpsc::channel();
                let blocker = tokio::task::spawn_blocking(move || {
                    started.send(()).unwrap();
                    released.recv_timeout(Duration::from_secs(10)).unwrap();
                });
                occupied.await.unwrap();
                let prep = preparation(1);
                let (input, _events, _release, encodes) = controlled_input();
                let task = tokio::spawn({
                    let prep = prep.clone();
                    async move {
                        prep.prepare(request(serde_json::json!("queued in blocking pool")), input)
                            .await
                    }
                });
                permits(&prep, 0, 0).await;
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
                release_pool.send(()).unwrap();
                blocker.await.unwrap();
                // A later job proves the occupied pool has processed the cancelled job.
                tokio::task::spawn_blocking(|| ()).await.unwrap();
                permits(&prep, 1, 1).await;
                assert_eq!(encodes.load(Ordering::Relaxed), 0);
            });
    }

    #[tokio::test]
    async fn completion_invalid_local_contract_never_polls_frontend_resolver() {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), r#"{"vocab_size":0}"#).unwrap();
        let missing = directory.path().join("not-a-local-model");
        let attempts = AtomicUsize::new(0);
        for (model, cache_model, path, expected) in [
            (" public", "engine", directory.path(), "MODEL_ALIAS"),
            ("public", " engine", directory.path(), "CACHE_MODEL"),
            ("public", "engine", missing.as_path(), "model directory"),
            ("public", "engine", directory.path(), "vocab_size"),
        ] {
            let error = CompletionInput::load(model, cache_model, path, async {
                attempts.fetch_add(1, Ordering::Relaxed);
                Err(anyhow::anyhow!("frontend resolver must not be polled"))
            })
            .await
            .unwrap_err();
            assert!(error.to_string().contains(expected), "{error:#}");
            assert_eq!(attempts.load(Ordering::Relaxed), 0);
        }
    }
}
