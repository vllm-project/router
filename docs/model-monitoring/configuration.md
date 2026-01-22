# Monitoring and Metrics

This guide covers metrics collection, monitoring, and observability for vLLM Router.

## Overview

vLLM Router exposes Prometheus metrics for monitoring performance, health, and behavior.

## Prometheus Metrics

### Default Configuration

Metrics are exposed at `http://127.0.0.1:29000/metrics` by default.

### Custom Configuration

```bash
vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 9000
```

### Configuration Options

- `--prometheus-host`: Host address for metrics endpoint (default: `127.0.0.1`)
- `--prometheus-port`: Port for metrics endpoint (default: `29000`)

## Available Metrics

### Request Metrics

- `vllm_router_requests_total`: Total number of requests
- `vllm_router_requests_duration_seconds`: Request duration histogram
- `vllm_router_requests_in_flight`: Current number of in-flight requests
- `vllm_router_requests_failed_total`: Total number of failed requests

### Worker Metrics

- `vllm_router_workers_total`: Total number of registered workers
- `vllm_router_workers_healthy`: Number of healthy workers
- `vllm_router_workers_unhealthy`: Number of unhealthy workers
- `vllm_router_worker_requests_total`: Requests per worker
- `vllm_router_worker_errors_total`: Errors per worker

### Circuit Breaker Metrics

- `vllm_router_circuit_breaker_state`: Current circuit breaker state (0=Closed, 1=Open, 2=HalfOpen)
- `vllm_router_circuit_breaker_transitions_total`: State transition count
- `vllm_router_circuit_breaker_failures_total`: Failure count per worker

### Retry Metrics

- `vllm_router_retries_total`: Total number of retry attempts
- `vllm_router_retry_backoff_seconds`: Retry backoff duration

### Load Balancing Metrics

- `vllm_router_policy_selections_total`: Worker selections per policy
- `vllm_router_cache_hits_total`: Cache-aware policy hits
- `vllm_router_cache_misses_total`: Cache-aware policy misses

## Accessing Metrics

### Local Access

```bash
curl http://localhost:29000/metrics
```

### Remote Access

For remote monitoring, bind to all interfaces:

```bash
vllm-router \
    --worker-urls http://worker1:8000 \
    --prometheus-host 0.0.0.0 \
    --prometheus-port 29000
```

Then access from remote:
```bash
curl http://<router-ip>:29000/metrics
```

## Prometheus Configuration

### prometheus.yml

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm-router'
    static_configs:
      - targets: ['localhost:29000']
        labels:
          service: 'vllm-router'
          environment: 'production'
```

### Kubernetes ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-router
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: vllm-router
  endpoints:
  - port: metrics
    interval: 15s
    path: /metrics
```

## Grafana Dashboards

### Key Panels

1. **Request Rate**
```promql
rate(vllm_router_requests_total[5m])
```

2. **Request Latency (p50, p95, p99)**
```promql
histogram_quantile(0.50, rate(vllm_router_requests_duration_seconds_bucket[5m]))
histogram_quantile(0.95, rate(vllm_router_requests_duration_seconds_bucket[5m]))
histogram_quantile(0.99, rate(vllm_router_requests_duration_seconds_bucket[5m]))
```

3. **Error Rate**
```promql
rate(vllm_router_requests_failed_total[5m])
```

4. **Worker Health**
```promql
vllm_router_workers_healthy
vllm_router_workers_unhealthy
```

5. **Circuit Breaker Status**
```promql
vllm_router_circuit_breaker_state
```

## Alerting

### Example Prometheus Alerts

```yaml
groups:
  - name: vllm-router
    rules:
      - alert: HighErrorRate
        expr: rate(vllm_router_requests_failed_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} requests/sec"

      - alert: AllWorkersUnhealthy
        expr: vllm_router_workers_healthy == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "All workers are unhealthy"
          description: "No healthy workers available"

      - alert: CircuitBreakerOpen
        expr: vllm_router_circuit_breaker_state == 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker is open"
          description: "Worker {{ $labels.worker }} circuit breaker is open"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(vllm_router_requests_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High request latency"
          description: "P95 latency is {{ $value }}s"
```

## Logging

### Log Levels

Configure log verbosity:

```bash
vllm-router \
    --worker-urls http://worker1:8000 \
    --log-level info
```

Available levels: `trace`, `debug`, `info`, `warn`, `error`

### Structured Logging

Logs are output in structured format for easy parsing:

```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "level": "INFO",
  "message": "Request routed to worker",
  "worker_url": "http://worker1:8000",
  "request_id": "req-123",
  "latency_ms": 45
}
```

## Health Checks

### Router Health Endpoint

Check router health:

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "workers": {
    "total": 3,
    "healthy": 3,
    "unhealthy": 0
  }
}
```

## Best Practices

1. **Monitor Key Metrics**: Focus on request rate, latency, and error rate
2. **Set Up Alerts**: Configure alerts for critical conditions
3. **Use Dashboards**: Create Grafana dashboards for visualization
4. **Log Aggregation**: Use tools like Loki or ELK for log analysis
5. **Distributed Tracing**: Consider adding OpenTelemetry for request tracing

## Troubleshooting

### Metrics Not Available

1. Check if metrics endpoint is accessible:
```bash
curl http://localhost:29000/metrics
```

2. Verify Prometheus configuration
3. Check firewall rules

### High Memory Usage

Monitor these metrics:
- `vllm_router_requests_in_flight`
- Worker memory usage
- Cache size (for cache-aware policy)

## Next Steps

- [Model Protection](../model-protection/README.md) - Retries and circuit breakers
- [Basic Routing](../model-routing/basic-routing.md) - Standard routing configuration
- [Kubernetes Deployment](../environment/kubernetes.md) - K8s monitoring setup
