# P/D Disaggregation Integration Tests using vLLM

This directory contains integration tests for Prefill/Decode (P/D) disaggregation using vLLM's NixlConnector.

## Overview

These tests validate that the vLLM router correctly handles P/D disaggregated inference, where:
- **Prefill instances** handle the initial prompt processing and KV cache generation
- **Decode instances** handle token generation using the transferred KV cache
- **Router** manages request routing between prefill and decode instances

## Test Components

### 1. `run_accuracy_test.sh`
Main test script that:
- Launches vLLM prefill and decode instances in Docker containers
- Starts the vLLM router with P/D disaggregation enabled
- Runs accuracy validation tests
- Cleans up resources after completion

**Configuration variables:**
- `VLLM_DOCKER_IMAGE`: Docker image to use (default: `vllm/vllm-openai:latest`)
- `MODEL_NAMES`: Model to test (default: `meta-llama/Llama-3.2-1B-Instruct`)
- `NUM_PREFILL_INSTANCES`: Number of prefill instances (default: 1)
- `NUM_DECODE_INSTANCES`: Number of decode instances (default: 1)
- `PREFILLER_TP_SIZE`: Tensor parallelism for prefill (default: 1)
- `DECODER_TP_SIZE`: Tensor parallelism for decode (default: 1)
- `GPU_MEMORY_UTILIZATION`: GPU memory utilization (default: 0.6)
- `KV_BUFFER_DEVICE`: KV buffer device - `cuda` or `cpu` (default: `cuda`)

### 2. `test_pd_accuracy.py`
Python script that validates accuracy by:
- Testing non-streaming completions
- Testing streaming completions
- Validating response structure and content
- Checking router health

### 3. `tp_config_sweep_test.sh`
Wrapper script that runs tests with multiple TP configurations:
- TP=2 for both prefill and decode
- TP=1 for prefill, TP=2 for decode (asymmetric)
- Baseline single TP configuration
- Various block size configurations

### 4. ROCm MoRI runners

`run_moriio_accuracy_test.sh` runs TP1 prefill and TP1 decode on two GPUs of
one ROCm host with MoRI's xGMI backend. `run_moriio_rdma_accuracy_test.sh` runs
prefill on the coordinator and decode on a second ROCm host over GPU-memory
RDMA. Both runners cover MoRI read and write modes, Router discovery, sanity
requests, and a bounded GSM8K accuracy evaluation with `Qwen/Qwen3-0.6B`.

The two-node runner requires the coordinator's `vllm-router-rocm-worker` SSH
alias, the pinned ROCm vLLM image and model cache on both hosts, and access to
`/dev/kfd`, `/dev/dri`, and `/dev/infiniband`. For example:

```bash
MORIIO_READ_MODE=true bash ./run_moriio_accuracy_test.sh
MORIIO_READ_MODE=false bash ./run_moriio_rdma_accuracy_test.sh
```

## Running Tests

### Prerequisites
- Docker installed and running
- NVIDIA Docker runtime configured
- At least 4 GPUs available (for full test suite)
- vLLM router binary built and in PATH

### Quick Start

Run all tests with default configuration:
```bash
cd py_test/e2e/pd_disagg_vllm
./tp_config_sweep_test.sh
```

### Run Single Configuration

Run with specific settings:
```bash
GPU_MEMORY_UTILIZATION=0.6 \
PREFILLER_TP_SIZE=2 \
DECODER_TP_SIZE=2 \
./run_accuracy_test.sh
```

### Custom Docker Image

Use a specific vLLM Docker image:
```bash
VLLM_DOCKER_IMAGE=vllm/vllm-openai:v0.6.0 \
./run_accuracy_test.sh
```

### Test Different Models

Test with a different model:
```bash
MODEL_NAMES=Qwen/Qwen2.5-1.5B-Instruct \
GPU_MEMORY_UTILIZATION=0.6 \
./run_accuracy_test.sh
```

### Enable FlashInfer Backend

Test with FlashInfer attention backend:
```bash
TEST_FLASHINFER=1 ./tp_config_sweep_test.sh
```

## Test Flow

1. **Setup Phase**
   - Creates Docker network for container communication
   - Pulls vLLM Docker image
   - Cleans up any existing containers

2. **Launch Prefill Instances**
   - Starts prefill containers with NixlConnector enabled
   - Configures bootstrap ports for KV transfer
   - Waits for health checks to pass

3. **Launch Decode Instances**
   - Starts decode containers with NixlConnector enabled
   - Connects to prefill instances via KV transfer
   - Waits for health checks to pass

4. **Launch Router**
   - Starts router with P/D disaggregation mode
   - Registers prefill and decode instance URLs
   - Validates router startup

5. **Run Accuracy Tests**
   - Sends completion requests through router
   - Validates response accuracy and structure
   - Tests streaming functionality

6. **Cleanup**
   - Stops all Docker containers
   - Removes Docker network
   - Reports test results

## GPU Requirements

The tests automatically calculate GPU assignments:
- Prefill instances use GPUs 0 through (N_prefill * TP_prefill - 1)
- Decode instances use subsequent GPUs

Example for `PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2`:
- Prefill instance 0: GPUs 0-1
- Decode instance 0: GPUs 2-3

### Supported GPU Types

| GPU Type | Memory | Recommended Model | Notes |
|----------|--------|-------------------|-------|
| L4 | 24GB | Llama-3.2-1B-Instruct | Default for CI |
| A100 | 40GB/80GB | Meta-Llama-3.1-8B-Instruct | Larger models OK |
| H100 | 80GB | Meta-Llama-3.1-70B-Instruct (with TP) | Best performance |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_DOCKER_IMAGE` | vLLM Docker image | `vllm/vllm-openai:latest` |
| `MODEL_NAMES` | Model to test | `meta-llama/Llama-3.2-1B-Instruct` |
| `NUM_PREFILL_INSTANCES` | Number of prefill instances | `1` |
| `NUM_DECODE_INSTANCES` | Number of decode instances | `1` |
| `PREFILLER_TP_SIZE` | Prefill tensor parallelism | `1` |
| `DECODER_TP_SIZE` | Decode tensor parallelism | `1` |
| `GPU_MEMORY_UTILIZATION` | GPU memory utilization | `0.6` |
| `PREFILL_BLOCK_SIZE` | Prefill block size | `128` |
| `DECODE_BLOCK_SIZE` | Decode block size | `128` |
| `KV_BUFFER_DEVICE` | KV buffer device | `cuda` |
| `TEST_FLASHINFER` | Enable FlashInfer tests | (unset) |

## Troubleshooting

### Containers fail to start
Check Docker logs:
```bash
docker logs vllm_prefill_0
docker logs vllm_decode_0
```

### GPU out of memory
Reduce `GPU_MEMORY_UTILIZATION`:
```bash
GPU_MEMORY_UTILIZATION=0.4 ./run_accuracy_test.sh
```

Or use a smaller model:
```bash
MODEL_NAMES=facebook/opt-125m ./run_accuracy_test.sh
```

### Port conflicts
Change base ports:
```bash
PREFILL_BASE_PORT=9100 \
DECODE_BASE_PORT=9200 \
ROUTER_PORT=9300 \
./run_accuracy_test.sh
```

### Clean up manually
If tests fail and leave containers running:
```bash
docker rm -f $(docker ps -a -q --filter "name=vllm_")
docker network rm vllm_pd_test_network
```

## CI/CD Integration

These tests are integrated into the Buildkite pipeline. See `.buildkite/pipeline.yml` for the CI configuration.

The tests run on a 4-GPU (L4) queue with a 30-minute timeout:
- Uses Llama-3.2-1B-Instruct (fits well on L4 GPUs)
- Tests multiple TP configurations
- Validates both streaming and non-streaming requests

## References

- [vLLM P/D Disaggregation Documentation](https://docs.vllm.ai/en/latest/serving/disaggregated_prefill_decode.html)
- [NixlConnector KV Transfer](https://github.com/vllm-project/vllm/tree/main/vllm/distributed/kv_transfer)
- [vLLM Router Documentation](../../README.md)
