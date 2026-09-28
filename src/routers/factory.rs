//! Factory for creating router instances

use super::{
    http::{openai_router::OpenAIRouter, router::Router, vllm_pd_router::VllmPDRouter},
    RouterTrait,
};
use crate::config::{PolicyConfig, RoutingMode};
use crate::policies::PolicyFactory;
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
                prefill_policy,
                decode_policy,
                discovery_address,
            } => {
                tracing::info!("Creating VllmPDRouter with prefill_urls: {:?}, decode_urls: {:?}, discovery: {:?}",
                              prefill_urls, decode_urls, discovery_address);
                Self::create_vllm_pd_router(
                    prefill_urls,
                    decode_urls,
                    discovery_address.clone(),
                    prefill_policy.as_ref(),
                    decode_policy.as_ref(),
                    &ctx.router_config.policy,
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
