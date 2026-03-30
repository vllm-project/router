//! Factory for creating router instances

use super::{
    http::{
        openai_router::OpenAIRouter, pd_router::PDRouter, router::Router,
        vllm_pd_router::VllmPDRouter,
    },
    RouterTrait,
};
use crate::config::{ConnectionMode, KVEventsConfig, PolicyConfig, RoutingMode};
use crate::kv_events::pool::KVEventPoolConfig;
use crate::kv_events::KVEventPool;
use crate::kv_index::{run_kv_index_updater, KVBlockIndex};
use crate::policies::kv_aware::{KvAwareConfig, KvAwarePolicy};
use crate::policies::{LoadBalancingPolicy, PolicyFactory};
use crate::server::AppContext;
use crate::tokenizer::factory::create_tokenizer_async;
use crate::tokenizer::traits::Encoder;
use std::sync::Arc;
use std::time::Duration;

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        // Check connection mode and route to appropriate implementation
        match ctx.router_config.connection_mode {
            ConnectionMode::Grpc => {
                // Route to gRPC implementation based on routing mode
                match &ctx.router_config.mode {
                    RoutingMode::Regular { worker_urls } => {
                        Self::create_grpc_router(worker_urls, &ctx.router_config.policy, ctx).await
                    }
                    RoutingMode::PrefillDecode {
                        prefill_urls,
                        decode_urls,
                        prefill_policy,
                        decode_policy,
                    } => {
                        Self::create_grpc_pd_router(
                            prefill_urls,
                            decode_urls,
                            prefill_policy.as_ref(),
                            decode_policy.as_ref(),
                            &ctx.router_config.policy,
                            ctx,
                        )
                        .await
                    }
                    RoutingMode::VllmPrefillDecode {
                        prefill_urls: _,
                        decode_urls: _,
                        prefill_policy: _,
                        decode_policy: _,
                        discovery_address: _,
                        kv_events: _,
                    } => Err("vLLM PD mode requires HTTP connection_mode".to_string()),
                    RoutingMode::OpenAI { .. } => {
                        Err("OpenAI mode requires HTTP connection_mode".to_string())
                    }
                }
            }
            ConnectionMode::Http => {
                // Route to HTTP implementation based on routing mode
                match &ctx.router_config.mode {
                    RoutingMode::Regular { worker_urls } => {
                        Self::create_regular_router(worker_urls, ctx).await
                    }
                    RoutingMode::PrefillDecode {
                        prefill_urls,
                        decode_urls,
                        prefill_policy,
                        decode_policy,
                    } => {
                        tracing::info!(
                            "Creating regular PDRouter with prefill_urls: {:?}, decode_urls: {:?}",
                            prefill_urls,
                            decode_urls
                        );
                        Self::create_pd_router(
                            prefill_urls,
                            decode_urls,
                            prefill_policy.as_ref(),
                            decode_policy.as_ref(),
                            &ctx.router_config.policy,
                            ctx,
                        )
                        .await
                    }
                    RoutingMode::VllmPrefillDecode {
                        prefill_urls,
                        decode_urls,
                        prefill_policy,
                        decode_policy,
                        discovery_address,
                        kv_events,
                    } => {
                        tracing::info!(
                            "Creating VllmPDRouter with prefill_urls: {:?}, decode_urls: {:?}, \
                             discovery: {:?}, kv_events: {:?}",
                            prefill_urls,
                            decode_urls,
                            discovery_address,
                            kv_events.is_some(),
                        );
                        Self::create_vllm_pd_router(
                            prefill_urls,
                            decode_urls,
                            discovery_address.clone(),
                            prefill_policy.as_ref(),
                            decode_policy.as_ref(),
                            &ctx.router_config.policy,
                            kv_events.as_ref(),
                            ctx,
                        )
                        .await
                    }
                    RoutingMode::OpenAI { worker_urls, .. } => {
                        Self::create_openai_router(worker_urls.clone(), ctx).await
                    }
                }
            }
        }
    }

    /// Create a regular router
    pub async fn create_regular_router(
        worker_urls: &[String],
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create regular router with context
        let router = Router::new(worker_urls.to_vec(), ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a PD router with injected policy
    pub async fn create_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Initialize policies in PolicyRegistry - use specific policies if provided, otherwise fall back to main policy
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        // Set the prefill and decode policies in the registry
        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        // Create PD router with context (policies are in PolicyRegistry)
        let router = PDRouter::new(prefill_urls.to_vec(), decode_urls.to_vec(), ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a vLLM PD router with service discovery and/or static URLs.
    ///
    /// When `kv_events_config` is `Some` (automatically set when any policy is
    /// `kv_aware`), this method bootstraps the full KV event infrastructure:
    ///
    /// 1. **KVBlockIndex** – Shared, thread-safe global index of block→worker.
    /// 2. **KVEventPool** – ZMQ SUB connections to every known vLLM worker.
    /// 3. **Index updater task** – Async task consuming events and updating the index.
    /// 4. **Tokenizer** – HuggingFace tokenizer for request text → token IDs.
    /// 5. **KvAwarePolicy** – The real policy replacing the registry placeholder.
    #[allow(clippy::too_many_arguments)]
    pub async fn create_vllm_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        discovery_address: Option<String>,
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        kv_events_config: Option<&KVEventsConfig>,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // ── KV infrastructure bootstrap ────────────────────────────────────
        let kv_infra = if let Some(kv_cfg) = kv_events_config {
            tracing::info!(
                "Bootstrapping KV event infrastructure (topic={}, port={}, max_entries={})",
                kv_cfg.topic_filter,
                kv_cfg.default_port,
                kv_cfg.index_max_entries,
            );

            // 1. Global KV block index (shared between updater + policies).
            let block_index = Arc::new(KVBlockIndex::new(kv_cfg.index_max_entries));

            // 2. Event pool: manages per-worker ZMQ SUB connections.
            let pool_config = KVEventPoolConfig {
                topic_filter: kv_cfg.topic_filter.clone(),
                default_kv_events_port: kv_cfg.default_port,
            };
            let (mut event_pool, event_rx) = KVEventPool::new(pool_config);

            // Subscribe to all statically configured workers.
            for (url, _bootstrap_port) in prefill_urls {
                event_pool.subscribe_worker_by_http(url.clone(), url);
            }
            for url in decode_urls {
                event_pool.subscribe_worker_by_http(url.clone(), url);
            }
            tracing::info!(
                "KV event pool: subscribed to {} initial workers",
                event_pool.worker_count(),
            );

            // 3. Background updater: consumes events and mutates the index.
            let index_for_updater = Arc::clone(&block_index);
            tokio::spawn(async move {
                run_kv_index_updater(event_rx, index_for_updater).await;
            });

            // 4. Tokenizer: required for text → token IDs → block keys.
            let model_path = ctx
                .router_config
                .model_path
                .as_ref()
                .or(ctx.router_config.tokenizer_path.as_ref())
                .ok_or_else(|| {
                    "kv_aware policy requires --model-path or --tokenizer-path".to_string()
                })?;

            tracing::info!("Loading tokenizer from: {}", model_path);
            let tokenizer: Arc<dyn Encoder> = create_tokenizer_async(model_path)
                .await
                .map_err(|e| format!("Failed to load tokenizer for kv_aware: {}", e))?
                as Arc<dyn Encoder>;

            Some((block_index, event_pool, tokenizer, kv_cfg.clone()))
        } else {
            None
        };

        // ── Policy construction ────────────────────────────────────────────
        // For KvAware, we must use the bootstrapped KV infrastructure (index +
        // tokenizer).  For all other policies, delegate to PolicyFactory.
        let make_real_policy =
            |cfg: &PolicyConfig| -> Result<Arc<dyn LoadBalancingPolicy>, String> {
                match cfg {
                    PolicyConfig::KvAware {
                        block_size,
                        hash_seed,
                        enable_speculative,
                        speculative_ttl_ms,
                    } => {
                        let (ref block_index, _, ref tok, _) =
                            kv_infra.as_ref().ok_or_else(|| {
                                "KvAware policy requires KV event infrastructure (--model-path or \
                                 --tokenizer-path must be set and kv_events config must be present)"
                                    .to_string()
                            })?;
                        let kv_config = KvAwareConfig {
                            block_size: *block_size,
                            hash_seed: *hash_seed,
                            enable_speculative: *enable_speculative,
                            speculative_ttl: Duration::from_millis(*speculative_ttl_ms),
                        };
                        tracing::info!(
                            "Creating KvAwarePolicy \
                             (block_size={}, hash_seed={}, speculative={})",
                            block_size,
                            hash_seed,
                            enable_speculative,
                        );
                        Ok(Arc::new(KvAwarePolicy::new(
                            kv_config,
                            Arc::clone(block_index),
                            Arc::clone(tok),
                        )))
                    }
                    _ => Ok(PolicyFactory::create_from_config(cfg)),
                }
            };

        let effective_prefill = prefill_policy_config.unwrap_or(main_policy_config);
        let effective_decode = decode_policy_config.unwrap_or(main_policy_config);

        let prefill_policy = make_real_policy(effective_prefill)?;
        let decode_policy = make_real_policy(effective_decode)?;

        // Install policies into the shared registry.
        ctx.policy_registry.set_prefill_policy(prefill_policy);
        ctx.policy_registry.set_decode_policy(decode_policy);

        // ── Router construction ────────────────────────────────────────────
        if discovery_address.is_some() {
            tracing::info!(
                "Creating VllmPDRouter with service discovery at: {:?}",
                discovery_address
            );
        }
        if !prefill_urls.is_empty() || !decode_urls.is_empty() {
            tracing::info!(
                "Creating VllmPDRouter with static URLs - prefill: {:?}, decode: {:?}",
                prefill_urls,
                decode_urls
            );
        }

        // Extract the KVEventPool so the router can manage dynamic subscriptions
        // when new workers are discovered via service discovery.
        let kv_event_pool = kv_infra.map(|(_, pool, _, cfg)| (pool, cfg));

        let router = VllmPDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            discovery_address,
            kv_event_pool,
            ctx,
        )
        .await?;
        tracing::info!("VllmPDRouter instance created successfully");

        Ok(Box::new(router))
    }

    /// Create a gRPC router with injected policy
    pub async fn create_grpc_router(
        worker_urls: &[String],
        policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        use super::grpc::router::GrpcRouter;

        // Create policy
        let policy = PolicyFactory::create_from_config(policy_config);

        // Create gRPC router with context
        let router = GrpcRouter::new(worker_urls.to_vec(), policy, ctx).await?;

        Ok(Box::new(router))
    }

    /// Create a gRPC PD router with tokenizer and worker configuration
    pub async fn create_grpc_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        prefill_policy_config: Option<&PolicyConfig>,
        decode_policy_config: Option<&PolicyConfig>,
        main_policy_config: &PolicyConfig,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        use super::grpc::pd_router::GrpcPDRouter;

        // Create policies - use specific policies if provided, otherwise fall back to main policy
        let prefill_policy =
            PolicyFactory::create_from_config(prefill_policy_config.unwrap_or(main_policy_config));
        let decode_policy =
            PolicyFactory::create_from_config(decode_policy_config.unwrap_or(main_policy_config));

        // Create gRPC PD router with context
        let router = GrpcPDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            prefill_policy,
            decode_policy,
            ctx,
        )
        .await?;

        Ok(Box::new(router))
    }

    /// Create an OpenAI router
    async fn create_openai_router(
        worker_urls: Vec<String>,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Use the first worker URL as the OpenAI-compatible base
        let base_url = worker_urls
            .first()
            .cloned()
            .ok_or_else(|| "OpenAI mode requires at least one worker URL".to_string())?;

        let router =
            OpenAIRouter::new(base_url, Some(ctx.router_config.circuit_breaker.clone())).await?;

        Ok(Box::new(router))
    }
}
