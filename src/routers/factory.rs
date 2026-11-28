//! Factory for creating router instances

use super::{
    http::{openai_router::OpenAIRouter, router::Router, vllm_pd_router::VllmPDRouter},
    RouterTrait,
};
use crate::config::RoutingMode;
use crate::server::AppContext;
use std::sync::Arc;

/// Factory for creating router instances based on configuration
pub struct RouterFactory;

impl RouterFactory {
    /// Create a router instance from application context
    pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
        match &ctx.router_config.mode {
            RoutingMode::Regular { worker_urls } => {
                Self::create_regular_router(worker_urls, ctx).await
            }
            RoutingMode::VllmPrefillDecode {
                prefill_urls,
                decode_urls,
                discovery_address,
            } => {
                tracing::info!("Creating VllmPDRouter with prefill_urls: {:?}, decode_urls: {:?}, discovery: {:?}",
                              prefill_urls, decode_urls, discovery_address);
                Self::create_vllm_pd_router(
                    prefill_urls,
                    decode_urls,
                    discovery_address.clone(),
                    ctx,
                )
                .await
            }
            RoutingMode::OpenAI { worker_urls, .. } => {
                Self::create_openai_router(worker_urls.clone(), ctx).await
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

    /// Create a vLLM PD router with service discovery and/or static URLs
    pub async fn create_vllm_pd_router(
        prefill_urls: &[(String, Option<u16>)],
        decode_urls: &[String],
        discovery_address: Option<String>,
        ctx: &Arc<AppContext>,
    ) -> Result<Box<dyn RouterTrait>, String> {
        // Create vLLM PD router with both static URLs and service discovery support
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

        let router = VllmPDRouter::new(
            prefill_urls.to_vec(),
            decode_urls.to_vec(),
            discovery_address,
            ctx,
        )
        .await?;
        tracing::info!("VllmPDRouter instance created successfully");

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

#[cfg(all(test, feature = "cpu", feature = "ready"))]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::config::{PolicyConfig, RouterConfig, RoutingMode};
    use crate::policies::kv_aware_ops;

    /// The PD path builds its prefill/decode policies through PolicyFactory,
    /// bypassing the registry default policy. If the factory-built policies
    /// are not wired to the registry's TTL'd affinity table, the CLI
    /// `--kv-aware-session-affinity-ttl-secs` flag silently no-ops.
    #[tokio::test]
    async fn pd_router_policies_inherit_registry_affinity_ttl() {
        let config = RouterConfig {
            policy: PolicyConfig::KvAware {
                router_temperature: 0.0,
                kv_penalty_host: 0.8,
                kv_penalty_external: 0.6,
                kv_penalty_disk: 0.3,
                static_kv_event_sources: Default::default(),
            }
            .into(),
            kv_aware_session_affinity_ttl_secs: Some(1800),
            // Worker lists stay empty on purpose: the prefill/decode policies
            // are wired before worker startup, and an empty set keeps this test
            // off the network (a populated set would block on the 600s health
            // wait).
            mode: RoutingMode::VllmPrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                discovery_address: None,
            },
            ..RouterConfig::default()
        };

        let ctx = Arc::new(
            AppContext::new(config, reqwest::Client::new(), 8, None, vec![])
                .expect("test app context"),
        );
        RouterFactory::create_router(&ctx)
            .await
            .expect("PD policies inherit the main policy and registry TTL");

        for name in ["prefill", "decode"] {
            let policy = match name {
                "prefill" => ctx.policy_registry.get_prefill_policy(),
                _ => ctx.policy_registry.get_decode_policy(),
            };
            assert_eq!(policy.name(), "kv_aware", "{name} policy");
            assert!(
                kv_aware_ops(policy.as_ref())
                    .expect("kv_aware policy")
                    .session_affinity_enabled(),
                "{name} policy must inherit the registry affinity TTL"
            );
        }
    }
}
