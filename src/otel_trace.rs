//! OpenTelemetry tracing integration.

use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    time::Duration,
};

use anyhow::Result;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use opentelemetry::{
    global, propagation::TextMapCompositePropagator, trace::TracerProvider as _, KeyValue,
};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    propagation::{BaggagePropagator, TraceContextPropagator},
    runtime,
    trace::{BatchConfigBuilder, BatchSpanProcessor, Sampler, Tracer as SdkTracer, TracerProvider},
    Resource,
};
use tracing::{Metadata, Subscriber};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{
    layer::{Context, Filter},
    Layer,
};

use crate::config::TraceConfig;

/// Whether OpenTelemetry tracing is enabled.
///
/// This flag guards runtime code paths (make_span parent extraction,
/// inject_trace_context_http). Only set to true after both the OTel layer
/// and subscriber are installed via `activate_otel`.
static ENABLED: AtomicBool = AtomicBool::new(false);
static PROVIDER: Mutex<Option<TracerProvider>> = Mutex::new(None);
static EXCLUDED_PATHS: OnceLock<Box<[String]>> = OnceLock::new();

const OTEL_SPAN_TARGET: &str = "otel_trace";

/// Filter that only allows specific module targets to be exported to OTEL.
#[derive(Clone, Copy, Default)]
pub(crate) struct CustomOtelFilter;

impl CustomOtelFilter {
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    #[inline]
    fn is_allowed(target: &str) -> bool {
        target.starts_with(OTEL_SPAN_TARGET)
    }
}

impl<S> Filter<S> for CustomOtelFilter
where
    S: Subscriber,
{
    #[inline]
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        Self::is_allowed(meta.target())
    }

    #[inline]
    fn callsite_enabled(&self, meta: &'static Metadata<'static>) -> tracing::subscriber::Interest {
        if Self::is_allowed(meta.target()) {
            tracing::subscriber::Interest::always()
        } else {
            tracing::subscriber::Interest::never()
        }
    }
}

/// Prepared OTel state ready to be activated.
///
/// Built by `prepare_otel()`, this holds the provider and tracer without
/// mutating any global state. Call `layer()` to create the tracing layer,
/// then `activate_otel()` after the subscriber is installed.
pub struct PreparedOtel {
    provider: TracerProvider,
    tracer: SdkTracer,
    excluded_paths: Box<[String]>,
}

impl PreparedOtel {
    /// Build from pre-existing parts (useful for tests with in-memory exporters).
    pub fn from_parts(provider: TracerProvider, tracer: SdkTracer) -> Self {
        Self::from_parts_with_excluded_paths(provider, tracer, Vec::new())
    }

    /// Build from pre-existing parts with an explicit excluded-path list.
    pub fn from_parts_with_excluded_paths(
        provider: TracerProvider,
        tracer: SdkTracer,
        excluded_paths: Vec<String>,
    ) -> Self {
        Self {
            provider,
            tracer,
            excluded_paths: excluded_paths.into_boxed_slice(),
        }
    }

    /// Create the tracing-opentelemetry layer for this prepared state.
    pub fn layer<S>(&self) -> Box<dyn Layer<S> + Send + Sync + 'static>
    where
        S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
    {
        let layer = tracing_opentelemetry::layer()
            .with_tracer(self.tracer.clone())
            .with_filter(CustomOtelFilter::new());
        Box::new(layer)
    }
}

/// Cross-platform hostname with PID for unique instance identification.
fn service_instance_id() -> String {
    let hostname = ["HOSTNAME", "HOST", "COMPUTERNAME"]
        .iter()
        .find_map(|key| std::env::var(key).ok().filter(|value| !value.is_empty()));
    match hostname {
        Some(hostname) => format!("{hostname}-{}", std::process::id()),
        None => format!("pid-{}", std::process::id()),
    }
}

fn parent_based_sampler(sampling_ratio: f64) -> Sampler {
    if sampling_ratio <= 0.0 {
        Sampler::ParentBased(Box::new(Sampler::AlwaysOff))
    } else if sampling_ratio >= 1.0 {
        Sampler::ParentBased(Box::new(Sampler::AlwaysOn))
    } else {
        Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(sampling_ratio)))
    }
}

/// Build OTel provider and tracer without mutating global state.
pub fn prepare_otel(trace_config: &TraceConfig) -> Result<PreparedOtel> {
    anyhow::ensure!(
        (0.0..=1.0).contains(&trace_config.sampling_ratio),
        "OTel sampling_ratio must be within [0.0, 1.0], got {}",
        trace_config.sampling_ratio
    );

    let mut exporter_builder = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_protocol(opentelemetry_otlp::Protocol::Grpc);

    // Only set the endpoint explicitly when configured; otherwise the SDK
    // respects OTEL_EXPORTER_OTLP_ENDPOINT (defaulting to localhost:4317).
    if let Some(ep) = trace_config.otlp_traces_endpoint.as_deref() {
        let ep = if !ep.starts_with("http://") && !ep.starts_with("https://") {
            format!("http://{ep}")
        } else {
            ep.to_string()
        };
        exporter_builder = exporter_builder.with_endpoint(&ep);
    }

    let exporter = exporter_builder
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to create OTLP exporter: {e}"))?;

    let batch_config = BatchConfigBuilder::default()
        .with_scheduled_delay(Duration::from_millis(500))
        .with_max_export_batch_size(64)
        .build();

    let span_processor = BatchSpanProcessor::builder(exporter, runtime::Tokio)
        .with_batch_config(batch_config)
        .build();

    let resource_attrs = vec![
        KeyValue::new("service.name", "vllm-router"),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        KeyValue::new("service.instance.id", service_instance_id()),
    ];
    let resource = Resource::default().merge(&Resource::new(resource_attrs));

    let provider = TracerProvider::builder()
        .with_sampler(parent_based_sampler(trace_config.sampling_ratio))
        .with_span_processor(span_processor)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer("vllm-router");

    Ok(PreparedOtel {
        provider,
        tracer,
        excluded_paths: trace_config.excluded_paths.clone().into_boxed_slice(),
    })
}

/// Set global propagator and provider, flip ENABLED.
///
/// Must be called after the subscriber with the OTel layer is installed.
pub fn activate_otel(prepared: PreparedOtel) {
    let PreparedOtel {
        provider,
        excluded_paths,
        ..
    } = prepared;

    global::set_text_map_propagator(TextMapCompositePropagator::new(vec![
        Box::new(TraceContextPropagator::new()),
        Box::new(BaggagePropagator::new()),
    ]));

    let _ = EXCLUDED_PATHS.set(excluded_paths);
    let _ = global::set_tracer_provider(provider.clone());
    *PROVIDER.lock().unwrap() = Some(provider);
    ENABLED.store(true, Ordering::Release);
    tracing::info!("[tracing] OpenTelemetry tracing enabled");
}

/// Mark OpenTelemetry tracing as enabled (for tests that manage their own provider).
pub fn mark_otel_enabled() {
    ENABLED.store(true, Ordering::Release);
}

/// Check if OpenTelemetry tracing is enabled.
#[inline]
pub fn is_otel_enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

#[inline]
pub fn is_excluded_http_path(path: &str) -> bool {
    EXCLUDED_PATHS
        .get()
        .is_some_and(|paths| paths.iter().any(|excluded| excluded == path))
}

pub fn shutdown_otel() {
    if ENABLED.load(Ordering::Acquire) {
        tracing::info!("[tracing] OpenTelemetry shutting down");
        if let Some(provider) = PROVIDER.lock().unwrap().take() {
            let _ = provider.shutdown();
        }
        ENABLED.store(false, Ordering::Release);
    }
}

/// Extract the incoming W3C trace context, including level-2 trace flags.
///
/// The OpenTelemetry 0.27 propagator rejects otherwise valid v00 traceparent
/// headers when non-sampling flag bits are set (for example 03). Preserve
/// the sampled bit during extraction so the incoming trace identity remains
/// intact; the propagator continues to validate all remaining fields.
pub fn extract_trace_context_http(headers: &HeaderMap) -> opentelemetry::Context {
    let normalized = headers
        .get("traceparent")
        .and_then(|value| value.to_str().ok())
        .and_then(|traceparent| {
            // The v00 wire layout has exactly four fields and is 55 bytes long.
            if traceparent.len() != 55 || !traceparent.starts_with("00-") {
                return None;
            }
            let (prefix, flags) = traceparent.rsplit_once('-')?;
            // Do not turn invalid wire syntax into a valid remote parent.
            if flags.len() != 2
                || !flags
                    .bytes()
                    .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
            {
                return None;
            }
            let flags = u8::from_str_radix(flags, 16).ok()?;
            if flags & !1 == 0 {
                return None;
            }
            let mut normalized = headers.clone();
            normalized.insert(
                "traceparent",
                HeaderValue::from_str(&format!("{prefix}-{:02x}", flags & 1)).ok()?,
            );
            Some(normalized)
        });

    let input = normalized.as_ref().unwrap_or(headers);
    global::get_text_map_propagator(|propagator| {
        propagator.extract(&opentelemetry_http::HeaderExtractor(input))
    })
}

/// Inject W3C trace context headers into an HTTP request's HeaderMap.
///
/// When OTel is enabled, this injects the current span's trace context
/// (traceparent, tracestate) into outgoing requests, making the router's
/// span the parent of backend spans.
///
/// When OTel is disabled, this is a no-op.
#[inline]
pub fn inject_trace_context_http(headers: &mut HeaderMap) {
    if !is_otel_enabled() {
        return;
    }

    let context = tracing::Span::current().context();

    struct HeaderInjector<'a>(&'a mut HeaderMap);

    impl opentelemetry::propagation::Injector for HeaderInjector<'_> {
        #[inline]
        fn set(&mut self, key: &str, value: String) {
            if let Ok(header_name) = HeaderName::from_bytes(key.as_bytes()) {
                if let Ok(header_value) = HeaderValue::from_str(&value) {
                    self.0.insert(header_name, header_value);
                }
            }
        }
    }

    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(&context, &mut HeaderInjector(headers));
    });
}
