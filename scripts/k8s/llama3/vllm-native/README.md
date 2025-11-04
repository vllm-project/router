# Llama 3.1 8B – 16 GPU vLLM Data Parallel Deployment

This directory contains a Kubernetes + Helmfile deployment that follows the
[vLLM external load balancing topology](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html#external-load-balancing)
with **one rank-0 master** that exposes the public API and **15 worker ranks**
that serve requests over the data-parallel (DP) RPC mesh. Every pod owns a
single GPU so the full cluster delivers throughput from all 16 devices.

## Topology

```
Client traffic
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ Rank 0 pod (decode)                                           │
│  • 1× GPU, exposes HTTP :8000 and DP RPC :13345               │
│  • Runs the DP coordinator / router                          │
└──────────────┬───────────────────────────────────────────────┘
               │ gRPC (port 13345)
               ▼
┌──────────────────────────────────────────────────────────────┐
│ Ranks 1-15 pods (prefill workers)                             │
│  • 1× GPU per pod                                             │
│  • vLLM engine only, no public HTTP endpoint                  │
│  • Register with the coordinator via the decode service       │
└──────────────────────────────────────────────────────────────┘
```

* Rank 0 advertises `--data-parallel-size 16` and routes requests across all
  ranks once each worker registers.
* Workers are independent Helm releases so Kubernetes schedules up to 15 pods
  across the available GPU nodes.
* The shared PVC stores the model weights so redeployments skip the download.

## Prerequisites

* Kubernetes cluster with at least 16 visible NVIDIA GPUs.
* [`kubectl`](https://kubernetes.io/docs/tasks/tools/) and
  [`helmfile`](https://github.com/helmfile/helmfile#installation) installed.
* HuggingFace token with access to `meta-llama/Llama-3.1-8B-Instruct` stored as
  a Kubernetes secret named `llm-d-hf-token`.

```
kubectl create namespace llm-d-llama31-multinode
kubectl create secret generic llm-d-hf-token \
  --from-literal=HF_TOKEN=hf_xxxxxxxxxxxxx \
  -n llm-d-llama31-multinode
```

## Deployment

```bash
cd scripts/k8s/llama3/vllm-native
./deploy.sh [namespace] [gateway-provider]
```

Defaults are `namespace=llm-d-llama31-multinode` and
`gateway-provider=default` (Istio benchmarking config). The script:

1. Applies the Helmfile which creates
   * one `ms-<release>` chart instance for the master pod, and
   * fifteen `ms-<release>-worker-rank-XX` releases for the worker pods.
2. Ensures the decode service exists so workers can reach the coordinator.
3. Waits for the gateway, rank-0 pod, and all worker pods to become Ready.

### Verifying the rollout

```bash
# All 16 pods (1 decode + 15 prefill) should report Ready
kubectl get pods -n llm-d-llama31-multinode -o wide

# Rank 0 exposes the HTTP API
kubectl logs -n llm-d-llama31-multinode -l llm-d.ai/role=decode -c vllm | tail

# Workers only emit DP connection logs
kubectl logs -n llm-d-llama31-multinode -l llm-d.ai/role=prefill -c vllm | grep "Connected to data parallel coordinator"
```

To run ad-hoc tests, port-forward the decode service and issue OpenAI-compatible
requests:

```bash
kubectl port-forward -n llm-d-llama31-multinode \
  svc/ms-llama31-multinode-llm-d-modelservice-decode 8000:8000

curl http://localhost:8000/v1/models
```

## Benchmarking & evaluation

The existing helper scripts continue to work unchanged and always hit the rank-0
endpoint:

```bash
# Throughput benchmark (defaults: 100 prompts, concurrency 16)
./run-benchmark.sh [namespace]

# Eval harness wrapper
./run-eval.sh [task] [namespace] [concurrency]
```

Monitor GPU utilization to confirm that traffic reaches all DP ranks:

```bash
kubectl exec -n llm-d-llama31-multinode -l llm-d.ai/role=prefill -c vllm -- \
  nvidia-smi --query-gpu=index,utilization.gpu --format=csv
```

You should see activity on every GPU once load is applied.

## Cleanup

```bash
./cleanup.sh [namespace] [gateway-provider]
```

The script tears down the Helmfile releases (master + workers + infra), removes
any straggling Helm releases, and deletes the decode service. The namespace is
left intact so the PVC with the downloaded model can be reused. Delete the
namespace manually if you want to reclaim the cache.

```bash
kubectl delete namespace llm-d-llama31-multinode
```

## Reference

* vLLM external load balancing guide – rank-0 coordinator with external router.
* Helm charts: `llm-d-modelservice` v0.2.11 and `llm-d-infra` v1.3.3.

This configuration ensures all 16 GPUs participate in inference while keeping a
single public endpoint for clients.
