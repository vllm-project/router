# Model Protection

This guide covers retry logic and circuit breakers for protecting vLLM workers and ensuring system reliability.

## Overview

vLLM Router provides enterprise-grade protection mechanisms:
- **Retries**: Automatic retry with exponential backoff and jitter
- **Circuit Breakers**: Prevent cascading failures and enable automatic recovery

## Retries

### Default Configuration

Retries are **enabled by default** with the following settings:
- Max retries: 3
- Initial backoff: 100ms
- Max backoff: 10 seconds
- Backoff multiplier: 2.0
- Jitter factor: 0.1

### Retry Behavior

The router automatically retries on these HTTP status codes:
- `408` - Request Timeout
- `429` - Too Many Requests
- `500` - Internal Server Error
- `502` - Bad Gateway
- `503` - Service Unavailable
- `504` - Gateway Timeout

### Custom Configuration

```bash
vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --retry-max-retries 5 \
    --retry-initial-backoff-ms 200 \
    --retry-max-backoff-ms 30000 \
    --retry-backoff-multiplier 2.5 \
    --retry-jitter-factor 0.2
```

### Configuration Options

- `--retry-max-retries`: Maximum number of retry attempts (default: 3)
- `--retry-initial-backoff-ms`: Initial backoff duration in milliseconds (default: 100)
- `--retry-max-backoff-ms`: Maximum backoff duration in milliseconds (default: 10000)
- `--retry-backoff-multiplier`: Multiplier for exponential backoff (default: 2.0)
- `--retry-jitter-factor`: Jitter factor to randomize backoff (default: 0.1)

### Backoff Calculation

The backoff duration is calculated as:
```
backoff = min(initial_backoff * (multiplier ^ attempt), max_backoff)
actual_backoff = backoff * (1 + random(-jitter, +jitter))
```

**Example with defaults:**
- Attempt 1: 100ms ± 10ms
- Attempt 2: 200ms ± 20ms
- Attempt 3: 400ms ± 40ms

## Circuit Breakers

### Overview

Circuit breakers protect workers from being overwhelmed and provide automatic recovery. They operate in three states:

```
┌─────────┐  N failures   ┌──────┐  timeout   ┌──────────┐
│ Closed  │──────────────>│ Open │───────────>│ HalfOpen │
└─────────┘               └──────┘            └──────────┘
     ^                                              │
     │                                              │
     └──────────────────────────────────────────────┘
              M successes
```

### State Machine

1. **Closed**: Normal operation, requests flow through
2. **Open**: Worker is failing, requests are rejected immediately
3. **HalfOpen**: Testing if worker has recovered

**Transitions:**
- `Closed` → `Open`: After N consecutive failures (failure-threshold)
- `Open` → `HalfOpen`: After timeout period (timeout-duration-secs)
- `HalfOpen` → `Closed`: After M consecutive successes (success-threshold)
- `HalfOpen` → `Open`: If any failure occurs

### Default Configuration

Circuit breakers are enabled by default:
- Failure threshold: 5 consecutive failures
- Success threshold: 2 consecutive successes
- Timeout duration: 30 seconds
- Window duration: 60 seconds

### Custom Configuration

```bash
vllm-router \
    --worker-urls http://localhost:8000 http://localhost:8001 \
    --cb-failure-threshold 10 \
    --cb-success-threshold 3 \
    --cb-timeout-duration-secs 60 \
    --cb-window-duration-secs 120
```

### Configuration Options

- `--cb-failure-threshold`: Number of failures before opening circuit (default: 5)
- `--cb-success-threshold`: Number of successes to close circuit (default: 2)
- `--cb-timeout-duration-secs`: Seconds before trying HalfOpen (default: 30)
- `--cb-window-duration-secs`: Time window for failure counting (default: 60)

## Combined Configuration

Use retries and circuit breakers together for maximum reliability:

```bash
vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 \
    --policy power_of_two \
    --retry-max-retries 3 \
    --retry-initial-backoff-ms 100 \
    --retry-max-backoff-ms 10000 \
    --cb-failure-threshold 5 \
    --cb-success-threshold 2 \
    --cb-timeout-duration-secs 30
```

## Best Practices

### Retry Configuration

1. **Set appropriate max retries**: Too many retries can increase latency
2. **Use jitter**: Prevents thundering herd problem
3. **Set reasonable backoff**: Balance between quick recovery and system load

### Circuit Breaker Configuration

1. **Tune failure threshold**: Based on expected error rate
2. **Adjust timeout duration**: Consider worker recovery time
3. **Monitor state transitions**: Track circuit breaker metrics

### Monitoring

Monitor these metrics (available at Prometheus endpoint):
- Retry attempts per request
- Circuit breaker state changes
- Request success/failure rates
- Latency percentiles

## Example Scenarios

### High-Traffic Production

```bash
vllm-router \
    --worker-urls http://worker1:8000 http://worker2:8000 http://worker3:8000 \
    --policy power_of_two \
    --retry-max-retries 2 \
    --retry-initial-backoff-ms 50 \
    --cb-failure-threshold 10 \
    --cb-timeout-duration-secs 60
```

### Development/Testing

```bash
vllm-router \
    --worker-urls http://localhost:8000 \
    --retry-max-retries 5 \
    --retry-initial-backoff-ms 200 \
    --cb-failure-threshold 3 \
    --cb-timeout-duration-secs 10
```

### Unstable Network

```bash
vllm-router \
    --worker-urls http://remote-worker:8000 \
    --retry-max-retries 5 \
    --retry-initial-backoff-ms 500 \
    --retry-max-backoff-ms 30000 \
    --cb-failure-threshold 8 \
    --cb-timeout-duration-secs 120
```

## Troubleshooting

### Too Many Retries

If you see excessive retries:
1. Check worker health and capacity
2. Reduce `--retry-max-retries`
3. Increase worker resources or add more workers

### Circuit Breaker Always Open

If circuit breakers stay open:
1. Check worker logs for errors
2. Increase `--cb-failure-threshold`
3. Increase `--cb-timeout-duration-secs`
4. Verify worker health endpoints

## Next Steps

- [Model Monitoring](../model-monitoring/configuration.md) - Metrics and observability
- [Basic Routing](../model-routing/basic-routing.md) - Standard routing configuration
- [Load Balancing](../model-routing/load-balancing.md) - Policy details
