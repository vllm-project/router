# Prefill/Decode Disaggregation Setup

Router Configuration:
- Router IP: 10.0.1.77
- Router HTTP Port: 10001
- ZMQ Discovery Port: 30001

## Quick Start

1. **Start Router:**
   ```bash
   ./start_router.sh
   ```

2. **Configure P/D Servers:**
   Copy the template commands below and customize for your environment.

## Prefill Server Template

```bash
# Customize these variables for your environment
VENV_PATH="~/uv_env/vllm"
VLLM_PATH="/home/ubuntu/gitrepos/vllm"
GPU_ID="0"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PREFILL_PORT="20003"
KV_PREFILL_PORT="21001"

source ${VENV_PATH}/bin/activate && \
cd ${VLLM_PATH} && \
CUDA_VISIBLE_DEVICES=${GPU_ID} VLLM_USE_V1=1 vllm serve ${MODEL} \
    --enforce-eager \
    --host 0.0.0.0 \
    --port ${PREFILL_PORT} \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"'${KV_PREFILL_PORT}'","kv_connector_extra_config":{"proxy_ip":"10.0.1.77","proxy_port":"30001","http_port":"'${PREFILL_PORT}'","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' \
    > prefill.log 2>&1 &
```

## Decode Server Template

```bash
# Customize these variables for your environment
VENV_PATH="~/uv_env/vllm"
VLLM_PATH="/home/ubuntu/gitrepos/vllm"
GPU_ID="1"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DECODE_PORT="20005"
KV_DECODE_PORT="22001"

source ${VENV_PATH}/bin/activate && \
cd ${VLLM_PATH} && \
CUDA_VISIBLE_DEVICES=${GPU_ID} VLLM_USE_V1=1 vllm serve ${MODEL} \
    --enforce-eager \
    --host 0.0.0.0 \
    --port ${DECODE_PORT} \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"'${KV_DECODE_PORT}'","kv_connector_extra_config":{"proxy_ip":"10.0.1.77","proxy_port":"30001","http_port":"'${DECODE_PORT}'","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' \
    > decode.log 2>&1 &
```

## Important Notes

- **Start router first** before P/D servers
- **proxy_ip and proxy_port must match router address** (10.0.1.77:30001)
- Customize all variables marked with YOUR_* or ${VARIABLE} for your environment
- Ensure network connectivity from P/D servers to router
- Use different KV ports for each server instance

Generated with router at 10.0.1.77:30001
