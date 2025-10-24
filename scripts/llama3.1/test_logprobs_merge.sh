#!/bin/bash
# Test script to compare logprobs responses from prefill, decode, and router

set -e

echo "======================================"
echo "Testing Logprobs Merge Implementation"
echo "======================================"
echo ""

# Test request payload
REQUEST='{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "The capital of France is", "max_tokens": 5, "temperature": 0.0, "logprobs": 2, "echo": true}'

echo "Request payload:"
echo "$REQUEST" | jq .
echo ""

# Test prefill server (port 8081)
echo "1. Testing PREFILL server (http://127.0.0.1:8081)..."
curl -s http://127.0.0.1:8081/v1/completions \
  -H "Content-Type: application/json" \
  -d "$REQUEST" > /tmp/prefill_response.json

if [ -s /tmp/prefill_response.json ]; then
    echo "   ✓ Response received"
    jq -r '.choices[0] | "   Text: \(.text // "N/A")\n   Prompt logprobs: \(.prompt_logprobs | length) items\n   Token logprobs: \(.logprobs.token_logprobs | length) items\n   Tokens: \(.logprobs.tokens | length) items"' /tmp/prefill_response.json
else
    echo "   ✗ No response or empty response"
fi
echo ""

# Test decode server (port 8082)
echo "2. Testing DECODE server (http://127.0.0.1:8082)..."
curl -s http://127.0.0.1:8082/v1/completions \
  -H "Content-Type: application/json" \
  -d "$REQUEST" > /tmp/decode_response.json

if [ -s /tmp/decode_response.json ]; then
    echo "   ✓ Response received"
    jq -r '.choices[0] | "   Text: \(.text // "N/A")\n   Prompt logprobs: \(.prompt_logprobs | length) items\n   Token logprobs: \(.logprobs.token_logprobs | length) items\n   Tokens: \(.logprobs.tokens | length) items"' /tmp/decode_response.json
else
    echo "   ✗ No response or empty response"
fi
echo ""

# Test router (port 8090)
echo "3. Testing ROUTER (http://127.0.0.1:8090)..."
curl -s http://127.0.0.1:8090/v1/completions \
  -H "Content-Type: application/json" \
  -d "$REQUEST" > /tmp/router_response.json

if [ -s /tmp/router_response.json ]; then
    echo "   ✓ Response received"
    jq -r '.choices[0] | "   Text: \(.text // "N/A")\n   Prompt logprobs: \(.prompt_logprobs | length) items\n   Token logprobs: \(.logprobs.token_logprobs | length) items\n   Tokens: \(.logprobs.tokens | length) items"' /tmp/router_response.json
else
    echo "   ✗ No response or empty response"
fi
echo ""

# Compare responses
echo "======================================"
echo "Comparison"
echo "======================================"
echo ""

# Compare text output
echo "Generated text comparison:"
echo "  Prefill: $(jq -r '.choices[0].text' /tmp/prefill_response.json)"
echo "  Decode:  $(jq -r '.choices[0].text' /tmp/decode_response.json)"
echo "  Router:  $(jq -r '.choices[0].text' /tmp/router_response.json)"
echo ""

# Compare array lengths
echo "Array lengths:"
printf "%-20s | %-10s | %-10s | %-10s\n" "Field" "Prefill" "Decode" "Router"
echo "-------------------------------------------------------------"
printf "%-20s | %-10s | %-10s | %-10s\n" \
  "prompt_logprobs" \
  "$(jq -r '.choices[0].prompt_logprobs | length' /tmp/prefill_response.json)" \
  "$(jq -r '.choices[0].prompt_logprobs | length' /tmp/decode_response.json)" \
  "$(jq -r '.choices[0].prompt_logprobs | length' /tmp/router_response.json)"

printf "%-20s | %-10s | %-10s | %-10s\n" \
  "token_logprobs" \
  "$(jq -r '.choices[0].logprobs.token_logprobs | length' /tmp/prefill_response.json)" \
  "$(jq -r '.choices[0].logprobs.token_logprobs | length' /tmp/decode_response.json)" \
  "$(jq -r '.choices[0].logprobs.token_logprobs | length' /tmp/router_response.json)"

printf "%-20s | %-10s | %-10s | %-10s\n" \
  "tokens" \
  "$(jq -r '.choices[0].logprobs.tokens | length' /tmp/prefill_response.json)" \
  "$(jq -r '.choices[0].logprobs.tokens | length' /tmp/decode_response.json)" \
  "$(jq -r '.choices[0].logprobs.tokens | length' /tmp/router_response.json)"

printf "%-20s | %-10s | %-10s | %-10s\n" \
  "text_offset" \
  "$(jq -r '.choices[0].logprobs.text_offset | length' /tmp/prefill_response.json)" \
  "$(jq -r '.choices[0].logprobs.text_offset | length' /tmp/decode_response.json)" \
  "$(jq -r '.choices[0].logprobs.text_offset | length' /tmp/router_response.json)"

printf "%-20s | %-10s | %-10s | %-10s\n" \
  "top_logprobs" \
  "$(jq -r '.choices[0].logprobs.top_logprobs | length' /tmp/prefill_response.json)" \
  "$(jq -r '.choices[0].logprobs.top_logprobs | length' /tmp/decode_response.json)" \
  "$(jq -r '.choices[0].logprobs.top_logprobs | length' /tmp/router_response.json)"
echo ""

# Expected: Router should match Decode (both should have merged logprobs)
echo "Expected behavior:"
echo "  - Decode and Router should have SAME text output"
echo "  - Decode and Router should have SAME array lengths"
echo "  - Router = merged prefill prompt logprobs + decode full logprobs"
echo ""

# Save detailed responses for inspection
echo "Full responses saved to:"
echo "  /tmp/prefill_response.json"
echo "  /tmp/decode_response.json"
echo "  /tmp/router_response.json"
echo ""

# Show first few tokens for detailed comparison
echo "First 3 tokens comparison (Router):"
jq '.choices[0].logprobs | {tokens: .tokens[0:3], token_logprobs: .token_logprobs[0:3], text_offset: .text_offset[0:3]}' /tmp/router_response.json
