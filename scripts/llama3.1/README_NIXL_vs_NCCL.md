# NIXL vs NCCL Connector Comparison for Llama 3.1 Testing

This document explains the differences between NIXL and NCCL (P2P) connectors for KV transfer in vLLM disaggregated serving.

## Git Commit References

- **NCCL Connector**: `b410562` - "NCCL Connector test for Llama"
- **NIXL Connector**: `38cd171` - "Switch to NIXL Connector for Llama 3.1 testing"

## Key Differences

### 1. KV Transfer Configuration

#### NCCL (P2P) Connector
```bash
--kv-transfer-config '{"kv_connector":"P2PNCCLConnector","kv_role":"kv_both","kv_connector_extra_config":{"http_port":8085}}'
```

#### NIXL Connector
```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"],"http_port":8085}}'
```

### 2. Environment Variables

#### NCCL Connector
- No special environment variables required
- Uses standard vLLM configuration

#### NIXL Connector
Required environment variables:
```bash
export VLLM_NIXL_SIDE_CHANNEL_HOST=$INTERNAL_IP  # Critical for NIXL handshake
export NVIDIA_GDRCOPY=enabled                     # For optimal performance
export UCX_TLS=all                                # UCX transport layer
export UCX_NET_DEVICES=all                        # UCX network devices
export VLLM_USE_V1=1                              # Required for NIXL
export VLLM_LOGGING_LEVEL=DEBUG                   # For debugging
```

### 3. Network Backend

#### NCCL Connector
- Uses NCCL's built-in P2P communication
- Direct GPU-to-GPU memory transfer via PCIe/NVLink
- Simpler setup, works on single-node or multi-node with GPUs

#### NIXL Connector
- Uses UCX (Unified Communication X) as backend
- Supports multiple backends: UCX, GDS (GPUDirect Storage)
- Designed for high-performance networking (InfiniBand, RoCE)
- Requires proper UCX and optionally gdrcopy installation

### 4. Performance Characteristics

#### NCCL Connector
- **Pros**:
  - Simpler setup
  - Well-tested for multi-GPU workloads
  - Good for PCIe/NVLink environments
- **Cons**:
  - May have higher latency over network
  - Limited to NCCL's networking capabilities

#### NIXL Connector
- **Pros**:
  - Optimized for high-speed networking (InfiniBand, RoCE)
  - Better performance with RDMA
  - Supports multiple backend protocols
- **Cons**:
  - More complex setup
  - Requires proper UCX/gdrcopy installation
  - More environment configuration needed

### 5. Troubleshooting

#### NCCL Connector Issues
- Check NCCL version compatibility
- Verify GPU P2P accessibility: `nvidia-smi topo -m`
- Check NCCL debug logs with `NCCL_DEBUG=INFO`

#### NIXL Connector Issues
- Verify UCX installation: `ucx_info -v`
- Install nixl if not already installed or getting "NIXL is not available" errors: `uv pip install nixl`
- Check gdrcopy: `dpkg -l | grep libgdrapi`
- Verify `VLLM_NIXL_SIDE_CHANNEL_HOST` is set to correct IP
- Check NIXL handshake logs for connection issues
- Ensure UCX backends are properly configured

## Switching Between Connectors

### To use NCCL Connector:
```bash
cd /home/congc/gitrepos/router
git checkout b410562
```

### To use NIXL Connector:
```bash
cd /home/congc/gitrepos/router
git checkout 38cd171
```

## Installation Prerequisites

### For NCCL Connector
- CUDA toolkit with NCCL support
- vLLM with P2P NCCL connector enabled

### For NIXL Connector
1. **UCX Installation** (required):
   ```bash
   # UCX is usually installed via package manager or from source
   # Check: ucx_info -v
   ```

2. **gdrcopy Installation** (optional but recommended):
   ```bash
   # For Ubuntu 20.04:
   sudo tools/install_gdrcopy.sh "ubuntu2004" "12.8" "x64"

   # For Ubuntu 22.04:
   sudo tools/install_gdrcopy.sh "ubuntu2204" "12.8" "x64"
   ```

3. **vLLM Docker Build**:
   ```bash
   docker build --build-arg INSTALL_KV_CONNECTORS=true -f docker/Dockerfile .
   ```

## Script Files

All test scripts are in: `/home/congc/gitrepos/router/scripts/llama3.1/`

- `start_prefill.sh` - Start prefill server
- `start_decode.sh` - Start decode server
- `start_router.sh` - Start router (port 8090)

## Testing Workflow

1. **Start servers**:
   ```bash
   # In separate terminals:
   cd /home/congc/gitrepos/router/scripts/llama3.1
   ./start_prefill.sh
   ./start_decode.sh
   ```

2. **Start router**:
   ```bash
   cd /home/congc/gitrepos/router
   cargo run --release -- \
     --vllm-pd-disaggregation \
     --prefill http://[::1]:8081 \
     --decode http://[::1]:8082 \
     --host 127.0.0.1 \
     --port 8090
   ```

3. **Send test request**:
   ```bash
   curl -X POST http://127.0.0.1:8090/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "meta-llama/Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 100
     }'
   ```

## Router Changes

The router code in `src/routers/http/vllm_pd_router.rs` includes IPv6 support improvements that work with both connectors:

- IPv6-aware address formatting using `join_host_port()` and `split_host_port()`
- Proper handling of IPv6 wildcard addresses (`::`)
- HTTP connection routing for IPv4/IPv6 mixed environments

These changes are in commit `b410562` and later.
