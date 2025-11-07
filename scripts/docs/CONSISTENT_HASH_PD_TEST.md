# Consistent Hash Load Balancing Test for vLLM PD Disaggregation

## Overview

This document provides a comprehensive test plan for validating consistent hash load balancing in vLLM Prefill-Decode (PD) disaggregation mode. The test verifies that:

1. **Prefill workers** use consistent hash policy for session-aware routing
2. **Decode workers** use consistent hash policy for session-aware routing
3. **Same session requests** consistently route to the same prefill AND decode workers
4. **Different sessions** are distributed across available workers
5. **Service discovery** properly integrates with policy-based load balancing

## Test Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client        │    │  vLLM Router     │    │  vLLM Workers   │
│                 │───▶│  (Port 30000)    │───▶│                 │
│ Session-aware   │    │  --vllm-pd-      │    │ Prefill: 20001  │
│ requests        │    │  disaggregation  │    │ Prefill: 20002  │
└─────────────────┘    │  --prefill-      │    │ Decode:  20003  │
                       │  policy          │    │ Decode:  20004  │
                       │  consistent_hash │    └─────────────────┘
                       │  --decode-policy │
                       │  consistent_hash │
                       └──────────────────┘
```

## Prerequisites

1. **vLLM Environment Setup**
```bash
source ~/uv_env/vllm/bin/activate
cd /home/ubuntu/gitrepos/vllm
```

2. **vLLM Router Environment**
```bash
cd /home/ubuntu/gitrepos/vllm-router
cargo build --release
```

## Test Setup Commands

### Step 1: Start Prefill Workers (4 instances)

**Terminal 1 - Prefill Worker 1:**
```bash
source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=0 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20000 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20000","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'

source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=1 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20001 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21002","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20001","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'

source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=2 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20002 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21003","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20002","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'

source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=3 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20003 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21004","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'


```

### Step 2: Start Decode Workers (4 instances)

**Terminal 3 - Decode Worker 1:**
```bash
source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=4 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20004 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22005","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20004","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'

source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=5 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20005 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22006","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'

source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=6 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20006 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22007","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20006","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'

source ~/uv_env/vllm/bin/activate && \
cd /home/ubuntu/gitrepos/vllm && \
CUDA_VISIBLE_DEVICES=7 vllm serve \
  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 20007 \
  --host 0.0.0.0 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22008","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20007","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'


```

### Step 3: Wait for Workers to be Ready

```bash
# Test all workers are responding
curl -s http://localhost:20001/health && echo " - Prefill Worker 1 Ready"
curl -s http://localhost:20002/health && echo " - Prefill Worker 2 Ready"
curl -s http://localhost:20003/health && echo " - Decode Worker 1 Ready"
curl -s http://localhost:20004/health && echo " - Decode Worker 2 Ready"
```

### Step 4: Start vLLM Router with Consistent Hash Policies

**Terminal 5 - vLLM Router:**
```bash
cd /home/ubuntu/gitrepos/vllm-router && cargo run --release -- \
    --prefill-policy consistent_hash \
    --decode-policy consistent_hash \
    --vllm-pd-disaggregation \
    --vllm-discovery-address 0.0.0.0:30001
    --host 0.0.0.0 \
    --port 10001


# Method 2: vLLM service discovery mode (if implemented)
# ./target/release/vllm-router \
#   --vllm-pd-disaggregation \
#   --vllm-discovery-address 0.0.0.0:30001 \
#   --prefill-policy consistent_hash \
#   --decode-policy consistent_hash \
#   --host 0.0.0.0 \
#   --port 30000 \
#   --log-level info
```

### Step 5: Verify Router Health

```bash
curl -s http://localhost:30000/health && echo " - Router Ready"
curl -s http://localhost:30000/v1/models
```

## Test Script

Create a comprehensive test script to validate consistent hash behavior:

**File: `/home/ubuntu/gitrepos/vllm-router/test_consistent_hash_pd.py`**

```python
#!/usr/bin/env python3
"""
Consistent Hash Load Balancing Test for vLLM PD Disaggregation

This script tests that:
1. Same session_id requests consistently route to same prefill + decode workers
2. Different sessions distribute across available workers
3. Policy-based selection works in PD mode
"""

import asyncio
import aiohttp
import json
import time
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple

# Test configuration
ROUTER_URL = "http://localhost:30000"
SESSIONS = [
    "session_alpha_001",
    "session_beta_002",
    "session_gamma_003",
    "session_delta_004",
    "session_epsilon_005",
    "session_zeta_006"
]
REQUESTS_PER_SESSION = 20
CONCURRENT_SESSIONS = 3

class PDConsistentHashTester:
    def __init__(self):
        self.session_to_prefill = defaultdict(set)
        self.session_to_decode = defaultdict(set)
        self.request_count = 0
        self.success_count = 0
        self.start_time = time.time()

    async def send_request(self, session: aiohttp.ClientSession, session_id: str, request_num: int) -> Dict:
        """Send a single request with session_id"""

        # Vary prompts to test routing logic, not cache hits
        prompts = [
            f"What is the capital of France? (Request {request_num})",
            f"Explain quantum computing in simple terms. (Request {request_num})",
            f"Write a short poem about the ocean. (Request {request_num})",
            f"How does photosynthesis work? (Request {request_num})",
            f"Describe the process of making coffee. (Request {request_num})"
        ]

        payload = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompts[request_num % len(prompts)]
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1,
            "session_params": {
                "session_id": session_id
            },
            "user": session_id  # Also set user field for compatibility
        }

        try:
            async with session.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:

                self.request_count += 1

                if response.status == 200:
                    result = await response.json()
                    self.success_count += 1

                    # Extract routing information from headers (if available)
                    prefill_worker = response.headers.get('X-Prefill-Worker', 'unknown')
                    decode_worker = response.headers.get('X-Decode-Worker', 'unknown')

                    # Track which workers handle each session
                    self.session_to_prefill[session_id].add(prefill_worker)
                    self.session_to_decode[session_id].add(decode_worker)

                    return {
                        "session_id": session_id,
                        "request_num": request_num,
                        "success": True,
                        "prefill_worker": prefill_worker,
                        "decode_worker": decode_worker,
                        "response_length": len(result.get("choices", [{}])[0].get("message", {}).get("content", "")),
                        "latency": time.time() - time.time()  # Will be calculated properly
                    }
                else:
                    print(f"❌ Request failed: {response.status} - {await response.text()}")
                    return {
                        "session_id": session_id,
                        "request_num": request_num,
                        "success": False,
                        "error": f"HTTP {response.status}"
                    }

        except Exception as e:
            print(f"❌ Request exception for {session_id}: {str(e)}")
            return {
                "session_id": session_id,
                "request_num": request_num,
                "success": False,
                "error": str(e)
            }

    async def test_session_consistency(self, session_id: str) -> List[Dict]:
        """Test multiple requests for a single session"""
        print(f"🧪 Testing session: {session_id}")

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(REQUESTS_PER_SESSION):
                task = self.send_request(session, session_id, i)
                tasks.append(task)

                # Add small delay to avoid overwhelming
                if i % 5 == 0:
                    await asyncio.sleep(0.1)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, dict)]

    async def run_test(self):
        """Run the complete test suite"""
        print("🚀 Starting Consistent Hash PD Test")
        print(f"📊 Testing {len(SESSIONS)} sessions with {REQUESTS_PER_SESSION} requests each")
        print(f"🎯 Target: {len(SESSIONS) * REQUESTS_PER_SESSION} total requests")
        print("-" * 60)

        # Test sessions concurrently
        session_groups = [SESSIONS[i:i+CONCURRENT_SESSIONS] for i in range(0, len(SESSIONS), CONCURRENT_SESSIONS)]

        all_results = []
        for group in session_groups:
            group_tasks = [self.test_session_consistency(session_id) for session_id in group]
            group_results = await asyncio.gather(*group_tasks)

            for results in group_results:
                all_results.extend(results)

            # Brief pause between groups
            await asyncio.sleep(0.5)

        # Analyze results
        self.analyze_results(all_results)

    def analyze_results(self, results: List[Dict]):
        """Analyze test results for consistency and distribution"""
        print("\n" + "="*60)
        print("📈 TEST RESULTS ANALYSIS")
        print("="*60)

        # Basic statistics
        total_requests = len(results)
        successful_requests = len([r for r in results if r.get("success", False)])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        print(f"📊 Total Requests: {total_requests}")
        print(f"✅ Successful Requests: {successful_requests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Total Test Time: {time.time() - self.start_time:.1f}s")

        if successful_requests == 0:
            print("❌ No successful requests to analyze!")
            return

        # Session consistency analysis
        print(f"\n🎯 SESSION CONSISTENCY ANALYSIS")
        print("-" * 40)

        consistent_sessions = 0
        for session_id in SESSIONS:
            prefill_workers = self.session_to_prefill[session_id]
            decode_workers = self.session_to_decode[session_id]

            prefill_consistent = len(prefill_workers) <= 1
            decode_consistent = len(decode_workers) <= 1
            fully_consistent = prefill_consistent and decode_consistent

            if fully_consistent:
                consistent_sessions += 1
                status = "✅"
            else:
                status = "❌"

            print(f"{status} {session_id}:")
            print(f"    Prefill Workers: {list(prefill_workers) if prefill_workers else ['No data']}")
            print(f"    Decode Workers: {list(decode_workers) if decode_workers else ['No data']}")
            print(f"    Consistent: Prefill={prefill_consistent}, Decode={decode_consistent}")

        consistency_rate = (consistent_sessions / len(SESSIONS) * 100) if SESSIONS else 0
        print(f"\n📈 Session Consistency Rate: {consistency_rate:.1f}% ({consistent_sessions}/{len(SESSIONS)})")

        # Load distribution analysis
        print(f"\n⚖️  LOAD DISTRIBUTION ANALYSIS")
        print("-" * 40)

        all_prefill_workers = set()
        all_decode_workers = set()

        for workers in self.session_to_prefill.values():
            all_prefill_workers.update(workers)

        for workers in self.session_to_decode.values():
            all_decode_workers.update(workers)

        print(f"🎯 Prefill Workers Used: {list(all_prefill_workers)}")
        print(f"🎯 Decode Workers Used: {list(all_decode_workers)}")
        print(f"📊 Prefill Worker Count: {len(all_prefill_workers)}")
        print(f"📊 Decode Worker Count: {len(all_decode_workers)}")

        # Test verdict
        print(f"\n🏆 TEST VERDICT")
        print("-" * 40)

        if consistency_rate >= 95 and success_rate >= 95:
            print("✅ PASS: Consistent hash policy working correctly!")
            print("   ✓ High success rate")
            print("   ✓ Excellent session consistency")
            print("   ✓ Load distribution across workers")
        elif consistency_rate >= 80:
            print("⚠️  PARTIAL PASS: Some consistency issues detected")
            print(f"   - Session consistency: {consistency_rate:.1f}%")
            print(f"   - Success rate: {success_rate:.1f}%")
        else:
            print("❌ FAIL: Consistent hash policy not working properly!")
            print(f"   - Poor session consistency: {consistency_rate:.1f}%")
            print(f"   - Success rate: {success_rate:.1f}%")

async def main():
    """Main test execution"""
    print("🧪 vLLM PD Consistent Hash Load Balancing Test")
    print("=" * 60)

    # Verify router is accessible
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ROUTER_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print(f"❌ Router health check failed: {response.status}")
                    return
                print("✅ Router health check passed")
    except Exception as e:
        print(f"❌ Cannot connect to router at {ROUTER_URL}: {e}")
        return

    # Run the test
    tester = PDConsistentHashTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())
```

## Running the Test

### Step 6: Execute the Test Script

```bash
cd /home/ubuntu/gitrepos/vllm-router

# Make the script executable
chmod +x test_consistent_hash_pd.py

# Install required dependencies
pip install aiohttp

# Run the test
python test_consistent_hash_pd.py
```

## Expected Results

### Successful Test Output:
```
🧪 vLLM PD Consistent Hash Load Balancing Test
============================================================
✅ Router health check passed
🚀 Starting Consistent Hash PD Test
📊 Testing 6 sessions with 20 requests each
🎯 Target: 120 total requests

🧪 Testing session: session_alpha_001
🧪 Testing session: session_beta_002
🧪 Testing session: session_gamma_003
...

============================================================
📈 TEST RESULTS ANALYSIS
============================================================
📊 Total Requests: 120
✅ Successful Requests: 120
📈 Success Rate: 100.0%
⏱️  Total Test Time: 45.2s

🎯 SESSION CONSISTENCY ANALYSIS
----------------------------------------
✅ session_alpha_001:
    Prefill Workers: ['http://0.0.0.0:20001']
    Decode Workers: ['http://0.0.0.0:20003']
    Consistent: Prefill=True, Decode=True
✅ session_beta_002:
    Prefill Workers: ['http://0.0.0.0:20002']
    Decode Workers: ['http://0.0.0.0:20004']
    Consistent: Prefill=True, Decode=True
...

📈 Session Consistency Rate: 100.0% (6/6)

⚖️  LOAD DISTRIBUTION ANALYSIS
----------------------------------------
🎯 Prefill Workers Used: ['http://0.0.0.0:20001', 'http://0.0.0.0:20002']
🎯 Decode Workers Used: ['http://0.0.0.0:20003', 'http://0.0.0.0:20004']
📊 Prefill Worker Count: 2
📊 Decode Worker Count: 2

🏆 TEST VERDICT
----------------------------------------
✅ PASS: Consistent hash policy working correctly!
   ✓ High success rate
   ✓ Excellent session consistency
   ✓ Load distribution across workers
```

## Monitoring and Debugging

### Router Logs Analysis
Monitor the router logs for policy decisions:
```bash
# In the router terminal, look for lines like:
# INFO vLLM policy-based routing: prefill=0.0.0.0:20001(zmq_addr) [policy:consistent_hash], decode=0.0.0.0:20003(zmq_addr) [policy:consistent_hash]
```

### Individual Request Testing
Test individual requests manually:
```bash
# Test with session_id
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello world"}],
    "max_tokens": 10,
    "session_params": {"session_id": "test_session_001"}
  }'
```

### Performance Monitoring
```bash
# Monitor GPU usage across workers
watch -n 1 nvidia-smi

# Monitor router metrics (if enabled)
curl http://localhost:30000/metrics
```

## Cleanup

```bash
# Stop all processes (Ctrl+C in each terminal)
# Or use pkill if running in background:
pkill -f "vllm.entrypoints.openai.api_server"
pkill -f "vllm-router"
```

## Test Validation Criteria

The test **PASSES** if:
1. ✅ **Session Consistency Rate ≥ 95%**: Same session_id requests route to same workers
2. ✅ **Success Rate ≥ 95%**: Requests complete successfully
3. ✅ **Load Distribution**: Multiple workers are utilized for prefill and decode
4. ✅ **Policy Logging**: Router logs show consistent_hash policy usage
5. ✅ **No Routing Errors**: No policy selection failures

## Troubleshooting

**Issue**: Policy selection failures
- **Solution**: Check worker health and router logs

**Issue**: All requests go to same worker
- **Solution**: Verify session_id varies across requests and hash ring setup

**Issue**: Workers not responding
- **Solution**: Check GPU memory and worker startup logs

This comprehensive test validates that the consistent hash implementation works correctly in vLLM PD disaggregation mode, ensuring both prefill and decode workers are selected using session-aware consistent hashing.