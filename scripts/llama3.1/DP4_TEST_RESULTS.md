# DP=4 Configuration Test Results

## Commit Information
- **Commit Hash**: `bbee91a`
- **Branch**: `feature/dp-aware-vllm-pd-router`
- **Date**: October 20, 2025
- **GitHub**: https://github.com/vllm-project/router/tree/feature/dp-aware-vllm-pd-router

## Configuration Summary

### Data Parallelism Setup
- **DP Size**: 4 (4 data parallel replicas)
- **Prefill Server**: GPUs 0-3, Port 8081
- **Decode Server**: GPUs 4-7, Port 8082
- **Connector**: NIXL with UCX/GDS backends

### Port Allocation (DP=4)

| Service | HTTP | NIXL Side Channels | NIXL HTTP |
|---------|------|-------------------|-----------|
| Prefill | 8081 | 8083-8086 (DP=4) | 8097 |
| Decode  | 8082 | 8093-8096 (DP=4) | 8098 |
| Router  | 8090 | N/A | N/A |

**Important Note**: When DP=4, the NIXL side channel base port automatically expands to use 4 consecutive ports (one per replica).

## Test Results (WITHOUT --dp-aware)

### Test Configuration
- **Router Flag**: `--dp-aware` **DISABLED** (baseline test)
- **Policy**: `consistent_hash`
- **Input**: 31 words (41 tokens) - exceeds 30 token threshold

### Request Details
```bash
curl -X POST http://127.0.0.1:8090/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Write a detailed explanation of how neural networks work...",
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### Results
✅ **SUCCESS** - All tests passed

| Metric | Status | Details |
|--------|--------|---------|
| Request Routing | ✅ PASS | Successfully routed through prefill → decode |
| Token Count | ✅ PASS | 41 prompt tokens (>30 as required) |
| Response | ✅ PASS | HTTP 200, valid completion generated |
| NIXL Connector | ✅ PASS | No connection or transfer errors |
| Port Conflicts | ✅ PASS | No port conflicts with DP=4 |
| DP Replicas | ✅ PASS | All 4 replicas initialized correctly |

### Response Sample
```json
{
  "id": "cmpl-___prefill_addr_127.0.0.1:8081___decode_addr_127.0.0.1:8082_...",
  "object": "text_completion",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "usage": {
    "prompt_tokens": 41,
    "total_tokens": 141,
    "completion_tokens": 100
  }
}
```

## Scripts Updated

### 1. `start_prefill.sh`
- Added `--data-parallel-size 4`
- Updated CUDA devices: `0,1,2,3`
- Updated NIXL side channel base port: `8083` (expands to 8083-8086)
- Updated NIXL HTTP port: `8097`
- Added informative echo statements

### 2. `start_decode.sh`
- Added `--data-parallel-size 4`
- Updated CUDA devices: `4,5,6,7`
- Updated NIXL side channel base port: `8093` (expands to 8093-8096)
- Updated NIXL HTTP port: `8098`
- Added informative echo statements

### 3. `start_router.sh`
- Added comments documenting DP=4 test plan
- Currently running WITHOUT `--dp-aware` flag (baseline)

## Environment
- **OS**: Linux (6.9.0)
- **CUDA Devices**: GPUs 0-7 available
- **vLLM**: V1 engine with NIXL connector
- **UCX**: Enabled (UCX_TLS=all, UCX_NET_DEVICES=all)

## Next Steps

### Phase 2: Enable --dp-aware
1. Update `start_router.sh` to add `--dp-aware` flag
2. Test request routing with DP-aware scheduling
3. Verify load balancing across DP replicas
4. Compare performance vs baseline (without --dp-aware)

### Prerequisites
Before running the commands, please update the path variables in the startup scripts:

1. **Identify your repository path:**
   - Find the path for the cloned `router` repository
   - Example: `/data/users/nlalit/gitrepos/router`

2. **Update script paths:**
   - Open each script (`start_prefill.sh`, `start_decode.sh`, `start_router.sh`)
   - Update the `cd` command with your actual router repository path
   - Example: Change `cd ~/gitrepos/router` to `cd /data/users/nlalit/gitrepos/router`

### Commands to Run
```bash
# Start prefill server
bash /home/congc/gitrepos/router/scripts/llama3.1/start_prefill.sh

# Start decode server
bash /home/congc/gitrepos/router/scripts/llama3.1/start_decode.sh

# Start router (without --dp-aware)
bash /home/congc/gitrepos/router/scripts/llama3.1/start_router.sh

# Test request
bash /tmp/test_router_clean.sh
```

## Troubleshooting

### Port Conflict Issues
If you encounter port conflicts with DP=4:
- Ensure base ports are spaced appropriately (need 4 consecutive ports)
- Current allocation avoids conflicts: 8083-8086 (prefill), 8093-8096 (decode)

### NIXL Connection Issues
- Verify UCX is properly installed and configured
- Check that all ports (base + DP replicas) are accessible
- Monitor logs for NIXL side channel initialization messages

## References
- Previous commit: `fcb6cdd` - Working NIXL connector configuration
- GitHub PR: https://github.com/vllm-project/router/pull/[PR_NUMBER]
- NIXL Documentation: `scripts/llama3.1/README_NIXL_vs_NCCL.md`
