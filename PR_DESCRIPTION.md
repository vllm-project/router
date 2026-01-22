# Fix: Add `add_generation_prompt` parameter to ChatCompletionRequest

## Problem Description

When sending requests through the router with `continue_final_message: true` set but without explicitly setting `add_generation_prompt`, the following error occurs:

```
continue_final_message and add_generation_prompt are not compatible. 
Use continue_final_message when you want the model to continue the final 
message, and add_generation_prompt when you want to add a header that will 
prompt it to start a new assistant message instead.
```

## Root Cause

The `ChatCompletionRequest` struct in the router was missing the `add_generation_prompt` parameter. When this parameter was not included in the user's request, the router would not forward it to the backend, causing the backend to use its default value of `true`. If `continue_final_message: true` was also set, these two parameters would conflict.

This issue does not occur when sending requests directly to the backend model, as all parameters can be controlled directly.

## Solution

Added the `add_generation_prompt` field to the `ChatCompletionRequest` struct with a default value of `true` (aligned with vLLM). This ensures:

1. The router can properly receive and forward the `add_generation_prompt` parameter
2. When users explicitly set `add_generation_prompt: false`, it can be used compatibly with `continue_final_message: true`
3. The default behavior is consistent with vLLM

## Changes

- Added `add_generation_prompt: bool` field to `ChatCompletionRequest` struct
- Set default value to `true` using `#[serde(default = "default_true")]` to align with vLLM behavior

## Testing

Verified the fix using the following curl command:

```bash
curl -X POST "http://127.0.0.1:30000/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "IQuestLoopCoder-Instruct",
           "messages": [
             {
               "role": "user",
               "content": "tell me a common saying"
             },
             {
               "role": "assistant",
               "content": "Here is a common saying about apple. An apple a day, keeps"
             }
           ],
           "add_generation_prompt": false,
           "continue_final_message": true,
           "echo": false
         }'
```

The request now works correctly without parameter conflict errors.

