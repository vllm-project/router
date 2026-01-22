# Kubernetes Deployment

This guide covers deploying vLLM Router in Kubernetes environments with automatic service discovery.

## Overview

vLLM Router supports Kubernetes-native service discovery, automatically detecting and managing vLLM worker pods based on label selectors.

## Features

- **Automatic Worker Discovery**: Detects pods based on Kubernetes labels
- **Health Monitoring**: Continuously monitors pod health and availability
- **Dynamic Updates**: Automatically adds/removes workers as pods scale
- **Namespace Support**: Watch pods in specific namespaces or cluster-wide
- **P/D Disaggregation**: Separate discovery for prefill and decode workers

## Basic Service Discovery

### Configuration

```bash
vllm-router \
    --service-discovery \
    --selector app=vllm-worker role=inference \
    --service-discovery-namespace default \
    --service-discovery-port 8000
```

### Command Line Options

- `--service-discovery`: Enable Kubernetes service discovery
- `--selector`: Label selectors for worker pods (format: `key1=value1 key2=value2`)
- `--service-discovery-namespace`: Kubernetes namespace to watch (omit for cluster-wide)
- `--service-discovery-port`: Port for worker URLs (default: 8000)

## Prefill-Decode Mode with Service Discovery

For P/D disaggregation, use separate selectors for prefill and decode workers:

```bash
vllm-router \
    --service-discovery \
    --vllm-pd-disaggregation \
    --prefill-selector app=vllm role=prefill \
    --decode-selector app=vllm role=decode \
    --service-discovery-namespace llm-production \
    --prefill-policy cache_aware \
    --decode-policy power_of_two
```

### Options for P/D Mode

- `--prefill-selector`: Label selectors for prefill pods
- `--decode-selector`: Label selectors for decode pods
- `--prefill-policy`: Load balancing policy for prefill workers
- `--decode-policy`: Load balancing policy for decode workers

## Deployment Example

### 1. Deploy vLLM Workers

**worker-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-worker
  namespace: default
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-worker
      role: inference
  template:
    metadata:
      labels:
        app: vllm-worker
        role: inference
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-3.1-8B-Instruct
          - --port
          - "8000"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
```

### 2. Deploy vLLM Router

**router-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-router
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-router
  template:
    metadata:
      labels:
        app: vllm-router
    spec:
      serviceAccountName: vllm-router
      containers:
      - name: router
        image: your-registry/vllm-router:latest
        args:
          - --service-discovery
          - --selector
          - app=vllm-worker
          - role=inference
          - --service-discovery-namespace
          - default
          - --policy
          - cache_aware
          - --host
          - 0.0.0.0
          - --port
          - "8080"
        ports:
        - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-router
  namespace: default
spec:
  selector:
    app: vllm-router
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### 3. Create RBAC Permissions

**rbac.yaml:**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: vllm-router
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: vllm-router
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: vllm-router
subjects:
- kind: ServiceAccount
  name: vllm-router
  namespace: default
roleRef:
  kind: ClusterRole
  name: vllm-router
  apiGroup: rbac.authorization.k8s.io
```

### 4. Deploy

```bash
kubectl apply -f rbac.yaml
kubectl apply -f worker-deployment.yaml
kubectl apply -f router-deployment.yaml
```

## Complete P/D Example

See `scripts/k8s/llama3.1/` for a complete working example with:
- Prefill and decode worker deployments
- Router configuration with service discovery
- Helmfile-based deployment
- HTTPRoute configuration for ingress

## Monitoring

### Check Router Logs

```bash
kubectl logs -f deployment/vllm-router -n default
```

### Check Discovered Workers

The router logs will show discovered workers:
```
INFO: Adding pod: vllm-worker-abc123 | type: Regular | url: http://10.0.1.5:8000
INFO: Adding pod: vllm-worker-def456 | type: Regular | url: http://10.0.2.8:8000
```

### Metrics

Access Prometheus metrics:
```bash
kubectl port-forward deployment/vllm-router 29000:29000
curl http://localhost:29000/metrics
```

## Troubleshooting

### No Workers Discovered

1. **Check RBAC permissions:**
```bash
kubectl auth can-i list pods --as=system:serviceaccount:default:vllm-router
```

2. **Verify label selectors:**
```bash
kubectl get pods -l app=vllm-worker,role=inference
```

3. **Check router logs:**
```bash
kubectl logs deployment/vllm-router
```

### Workers Not Healthy

Check pod status and readiness:
```bash
kubectl get pods -l app=vllm-worker
kubectl describe pod <pod-name>
```

## Advanced Configuration

### Multi-Namespace Discovery

For cluster-wide discovery, omit `--service-discovery-namespace`:

```bash
vllm-router \
    --service-discovery \
    --selector app=vllm-worker
```

### Custom Health Checks

The router automatically performs health checks on discovered workers. Configure timeouts:

```bash
vllm-router \
    --service-discovery \
    --selector app=vllm-worker \
    --worker-startup-timeout-secs 600 \
    --worker-startup-check-interval 30
```

## Next Steps

- [Basic Routing](../model-routing/basic-routing.md) - Standard routing configuration
- [P/D Disaggregation](../model-routing/pd-disaggregation.md) - Prefill-decode separation
- [Model Protection](../model-protection/README.md) - Retries and circuit breakers
