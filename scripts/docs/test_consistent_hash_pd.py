#!/usr/bin/env python3
"""
Simplified Consistent Hash Load Balancing Test for vLLM PD Disaggregation

This script tests that:
1. Same session_id requests consistently route to same prefill + decode workers
2. Policy-based selection works in PD mode

Simplified version: 1 session, 20 requests
"""

import asyncio
import aiohttp
import json
import time
from collections import defaultdict
from typing import Dict, List

# Test configuration
ROUTER_URL = "http://localhost:30000"
SESSION_IDS = [
    "user_alice_7x9k2m",
    "session_bob_4z8a1n",
    "client_charlie_3j5f8p",
    "app_diana_9w2e6q",
    "browser_eve_1r4t7u",
    "mobile_frank_8h3s5v",
    "tablet_grace_6d1g9w",
    "desktop_henry_2c7y4x",
    "session_iris_5n8k1z",
    "user_jack_9m3j6a",
    "client_kate_4p7l2b",
    "app_liam_8q1n5c",
    "browser_mia_3s4r8d",
    "mobile_noah_7u9v1e",
    "tablet_olivia_2x6w3f",
    "desktop_peter_5z8y7g",
    "session_quinn_1a4b9h",
    "user_rachel_6c2d5i",
    "client_sam_9e7f1j",
    "app_tina_3g8h4k"
]
REQUESTS_PER_SESSION = 5

class MultiSessionPDConsistentHashTester:
    def __init__(self):
        self.session_to_prefill = {}
        self.session_to_decode = {}
        self.all_prefill_workers = set()
        self.all_decode_workers = set()
        self.request_count = 0
        self.success_count = 0
        self.start_time = time.time()
        self.session_results = {}

    async def send_request(self, session: aiohttp.ClientSession, session_id: str, request_num: int) -> Dict:
        """Send a single request with session_id"""

        # Vary prompts to test routing logic, not cache hits
        prompts = [
            f"What is the capital of France? (Request {request_num})",
            f"Explain quantum computing in simple terms. (Request {request_num})",
            f"Write a short poem about the ocean. (Request {request_num})",
            f"How does photosynthesis work? (Request {request_num})",
            f"Describe the process of making coffee. (Request {request_num})",
            f"What is machine learning? (Request {request_num})",
            f"Explain the water cycle. (Request {request_num})",
            f"How do computers work? (Request {request_num})",
            f"What is artificial intelligence? (Request {request_num})",
            f"Describe the solar system. (Request {request_num})"
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

        request_start_time = time.time()

        try:
            async with session.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:

                self.request_count += 1
                latency = time.time() - request_start_time

                if response.status == 200:
                    result = await response.json()
                    self.success_count += 1

                    # Extract routing information from response ID (vLLM PD embeds routing info)
                    response_id = result.get('id', '')
                    prefill_worker = 'unknown'
                    decode_worker = 'unknown'

                    # Parse routing info from response ID format: ___prefill_addr_X___decode_addr_Y_uuid
                    if '___prefill_addr_' in response_id and '___decode_addr_' in response_id:
                        import re
                        pattern = r'___prefill_addr_([^_]+)___decode_addr_([^_]+)_'
                        match = re.search(pattern, response_id)
                        if match:
                            prefill_worker = match.group(1)
                            decode_worker = match.group(2)

                    # Track which workers handle this session
                    if session_id not in self.session_to_prefill:
                        self.session_to_prefill[session_id] = set()
                        self.session_to_decode[session_id] = set()

                    self.session_to_prefill[session_id].add(prefill_worker)
                    self.session_to_decode[session_id].add(decode_worker)
                    self.all_prefill_workers.add(prefill_worker)
                    self.all_decode_workers.add(decode_worker)

                    # Extract response content
                    response_content = ""
                    if result.get("choices") and len(result["choices"]) > 0:
                        response_content = result["choices"][0].get("message", {}).get("content", "")

                    result_data = {
                        "session_id": session_id,
                        "request_num": request_num,
                        "success": True,
                        "prefill_worker": prefill_worker,
                        "decode_worker": decode_worker,
                        "response_length": len(response_content),
                        "latency": latency,
                        "content_preview": response_content[:100] + "..." if len(response_content) > 100 else response_content
                    }

                    print(f"✅ {session_id} Req{request_num}: Prefill={prefill_worker}, Decode={decode_worker}, Latency={latency:.2f}s")
                    return result_data

                else:
                    error_text = await response.text()
                    print(f"❌ {session_id} Req{request_num} failed: {response.status} - {error_text}")
                    return {
                        "session_id": session_id,
                        "request_num": request_num,
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "latency": latency
                    }

        except Exception as e:
            latency = time.time() - request_start_time
            print(f"❌ {session_id} Req{request_num} exception: {str(e)}")
            return {
                "session_id": session_id,
                "request_num": request_num,
                "success": False,
                "error": str(e),
                "latency": latency
            }

    async def run_test(self):
        """Run the complete test suite"""
        print("🚀 Starting Multi-Session Consistent Hash PD Test")
        print(f"📊 Testing {len(SESSION_IDS)} sessions with {REQUESTS_PER_SESSION} requests each")
        print(f"🎯 Total requests: {len(SESSION_IDS) * REQUESTS_PER_SESSION}")
        print("-" * 60)

        async with aiohttp.ClientSession() as session:
            # Test each session sequentially
            for session_id in SESSION_IDS:
                print(f"\n🧪 Testing session: {session_id}")
                session_results = []

                # Send requests for this session with 1 second gap
                for i in range(REQUESTS_PER_SESSION):
                    result = await self.send_request(session, session_id, i + 1)
                    session_results.append(result)

                    # 1 second delay between requests as requested
                    await asyncio.sleep(1.0)

                self.session_results[session_id] = session_results
                print(f"✅ Completed session {session_id}")

        # Analyze results
        self.analyze_results()

    def analyze_results(self):
        """Analyze test results for consistency and distribution"""
        print("\n" + "="*80)
        print("📈 MULTI-SESSION TEST RESULTS ANALYSIS")
        print("="*80)

        # Basic statistics
        all_results = []
        for session_results in self.session_results.values():
            all_results.extend(session_results)

        total_requests = len(all_results)
        successful_requests = len([r for r in all_results if r.get("success", False)])
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0

        # Calculate average latency for successful requests
        successful_results = [r for r in all_results if r.get("success", False)]
        avg_latency = sum(r.get("latency", 0) for r in successful_results) / len(successful_results) if successful_results else 0

        print(f"📊 Total Requests: {total_requests}")
        print(f"✅ Successful Requests: {successful_requests}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        print(f"⏱️  Average Latency: {avg_latency:.2f}s")
        print(f"⏱️  Total Test Time: {time.time() - self.start_time:.1f}s")

        if successful_requests == 0:
            print("❌ No successful requests to analyze!")
            return

        # Per-session consistency analysis
        print(f"\n🎯 PER-SESSION CONSISTENCY ANALYSIS")
        print("-" * 80)

        consistent_sessions = 0
        for session_id in SESSION_IDS:
            if session_id in self.session_to_prefill and session_id in self.session_to_decode:
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
                print(f"    Prefill: {list(prefill_workers)} ({'✅ Consistent' if prefill_consistent else '❌ Inconsistent'})")
                print(f"    Decode:  {list(decode_workers)} ({'✅ Consistent' if decode_consistent else '❌ Inconsistent'})")
            else:
                print(f"❌ {session_id}: No data collected")

        consistency_rate = (consistent_sessions / len(SESSION_IDS) * 100) if SESSION_IDS else 0
        print(f"\n📈 Session Consistency Rate: {consistency_rate:.1f}% ({consistent_sessions}/{len(SESSION_IDS)})")

        # Load distribution analysis
        print(f"\n⚖️  LOAD DISTRIBUTION ANALYSIS")
        print("-" * 80)

        print(f"🎯 All Prefill Workers Used: {sorted(list(self.all_prefill_workers))}")
        print(f"🎯 All Decode Workers Used: {sorted(list(self.all_decode_workers))}")
        print(f"📊 Unique Prefill Workers: {len(self.all_prefill_workers)}")
        print(f"📊 Unique Decode Workers: {len(self.all_decode_workers)}")

        # Worker distribution statistics
        print(f"\n📊 WORKER DISTRIBUTION STATISTICS")
        print("-" * 80)

        # Count sessions per worker
        prefill_distribution = {}
        decode_distribution = {}

        for session_id in SESSION_IDS:
            if session_id in self.session_to_prefill:
                prefill_worker = list(self.session_to_prefill[session_id])[0] if self.session_to_prefill[session_id] else "none"
                decode_worker = list(self.session_to_decode[session_id])[0] if self.session_to_decode[session_id] else "none"

                prefill_distribution[prefill_worker] = prefill_distribution.get(prefill_worker, 0) + 1
                decode_distribution[decode_worker] = decode_distribution.get(decode_worker, 0) + 1

        print("Prefill Worker Distribution:")
        for worker, count in sorted(prefill_distribution.items()):
            percentage = (count / len(SESSION_IDS) * 100) if SESSION_IDS else 0
            print(f"  {worker}: {count} sessions ({percentage:.1f}%)")

        print(f"\nDecode Worker Distribution:")
        for worker, count in sorted(decode_distribution.items()):
            percentage = (count / len(SESSION_IDS) * 100) if SESSION_IDS else 0
            print(f"  {worker}: {count} sessions ({percentage:.1f}%)")

        # Session distribution table (compact for 20 sessions)
        print(f"\n📋 SESSION-TO-WORKER MAPPING (Showing first 10)")
        print("-" * 80)
        print("Session ID           | Prefill Worker     | Decode Worker      | Reqs")
        print("-" * 80)

        # Show first 10 sessions in detail
        for i, session_id in enumerate(SESSION_IDS[:10]):
            if session_id in self.session_to_prefill:
                prefill = list(self.session_to_prefill[session_id])[0] if self.session_to_prefill[session_id] else "none"
                decode = list(self.session_to_decode[session_id])[0] if self.session_to_decode[session_id] else "none"
                req_count = len(self.session_results.get(session_id, []))
                print(f"{session_id:20} | {prefill:18} | {decode:18} | {req_count:4}")
            else:
                print(f"{session_id:20} | {'NO DATA':18} | {'NO DATA':18} | {0:4}")

        if len(SESSION_IDS) > 10:
            print(f"... and {len(SESSION_IDS) - 10} more sessions")

        # Summary of all sessions (compact)
        print(f"\n🔍 ALL SESSIONS SUMMARY")
        print("-" * 80)
        session_pairs = set()
        for session_id in SESSION_IDS:
            if session_id in self.session_to_prefill:
                prefill = list(self.session_to_prefill[session_id])[0] if self.session_to_prefill[session_id] else "none"
                decode = list(self.session_to_decode[session_id])[0] if self.session_to_decode[session_id] else "none"
                session_pairs.add((prefill, decode))

        print(f"Unique Prefill+Decode Combinations: {len(session_pairs)}")
        for i, (prefill, decode) in enumerate(sorted(session_pairs), 1):
            sessions_with_combo = [s for s in SESSION_IDS
                                 if s in self.session_to_prefill
                                 and list(self.session_to_prefill[s])[0] == prefill
                                 and list(self.session_to_decode[s])[0] == decode]
            print(f"  Combo {i}: {prefill} + {decode} → {len(sessions_with_combo)} sessions")

        # Test verdict
        print(f"\n🏆 FINAL TEST VERDICT")
        print("-" * 80)

        distribution_good = len(self.all_prefill_workers) > 1 and len(self.all_decode_workers) > 1

        if consistency_rate >= 95 and success_rate >= 95 and distribution_good:
            print("✅ EXCELLENT: Consistent hash policy working perfectly!")
            print("   ✓ All sessions maintain consistent worker routing")
            print("   ✓ High success rate across all requests")
            print("   ✓ Good load distribution across available workers")
            print("   ✓ Prefill and decode policies both working correctly")
        elif consistency_rate >= 95 and success_rate >= 95:
            print("✅ GOOD: Sessions are consistent but limited worker distribution")
            print(f"   ✓ Session consistency: {consistency_rate:.1f}%")
            print(f"   ✓ Success rate: {success_rate:.1f}%")
            print(f"   ⚠️  Worker distribution: {len(self.all_prefill_workers)} prefill, {len(self.all_decode_workers)} decode")
        elif consistency_rate >= 80:
            print("⚠️  PARTIAL: Some consistency issues detected")
            print(f"   - Session consistency: {consistency_rate:.1f}%")
            print(f"   - Success rate: {success_rate:.1f}%")
            print(f"   - Worker distribution: {len(self.all_prefill_workers)} prefill, {len(self.all_decode_workers)} decode")
        else:
            print("❌ FAIL: Consistent hash policy not working properly!")
            print(f"   - Poor session consistency: {consistency_rate:.1f}%")
            print(f"   - Success rate: {success_rate:.1f}%")
            print(f"   - Worker distribution: {len(self.all_prefill_workers)} prefill, {len(self.all_decode_workers)} decode")

async def main():
    """Main test execution"""
    print("🧪 Comprehensive 20-Session vLLM PD Consistent Hash Test")
    print("=" * 80)

    # Verify router is accessible
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{ROUTER_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    print(f"❌ Router health check failed: {response.status}")
                    print("Make sure the router is running on localhost:30000")
                    return
                print("✅ Router health check passed")
    except Exception as e:
        print(f"❌ Cannot connect to router at {ROUTER_URL}: {e}")
        print("Make sure the router is running on localhost:30000")
        return

    # Check if we can reach the chat completions endpoint
    try:
        async with aiohttp.ClientSession() as session:
            test_payload = {
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            async with session.post(
                f"{ROUTER_URL}/v1/chat/completions",
                json=test_payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    print("✅ Chat completions endpoint accessible")
                else:
                    print(f"⚠️  Chat completions endpoint returned {response.status}")
                    print("Proceeding with test anyway...")
    except Exception as e:
        print(f"⚠️  Could not test chat completions endpoint: {e}")
        print("Proceeding with test anyway...")

    # Run the test
    tester = MultiSessionPDConsistentHashTester()
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())