# NIxL Connector Error Log

## Context

1. I followed the example and recipe in [https://docs.vllm.ai/en/latest/features/nixl_connector_usage.html#basic-usage-on-the-same-host](https://docs.vllm.ai/en/latest/features/nixl_connector_usage.html#basic-usage-on-the-same-host) and ran `meta-llama/Meta-Llama-3.1-8B-Instruct`
2. I installed NIXL / UCX and ran their tests and confirmed they are correct

## Error Output

```
(APIServer pid=2725876) INFO: Started server process [2725876]
(APIServer pid=2725876) INFO: Waiting for application startup.
(APIServer pid=2725876) INFO: Application startup complete.
(APIServer pid=2725876) 2249: UnsupportedFieldAttributeWarning: The 'deprecated' attribute with value 'max_tokens is deprecated in favor of the max_completion_tokens field' was provided to the `Field()` function, which has no effect in the context it was used. 'deprecated' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
(APIServer pid=2725876) warnings.warn(
(APIServer pid=2725876) INFO 10-19 13:07:49 [chat_utils.py:545] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] EngineCore encountered a fatal error.
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] Traceback (most recent call last):
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 790, in run_engine_core
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] engine_core.run_busy_loop()
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 817, in run_busy_loop
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] self._process_engine_step()
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 846, in _process_engine_step
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] outputs, model_executed = self.step_fn()
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] ^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 328, in step
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] engine_core_outputs = self.scheduler.update_from_output(
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] File "/home/congc/gitrepos/vllm/vllm/v1/core/sched/scheduler.py", line 923, in update_from_output
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] stats = self.connector.get_kv_connector_stats()
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] File "/home/congc/gitrepos/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py", line 244, in get_kv_connector_stats
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] assert self.connector_worker is not None
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) ERROR 10-19 13:07:53 [core.py:799] AssertionError
(APIServer pid=2725876) ERROR 10-19 13:07:53 [async_llm.py:521] AsyncLLM output_handler failed.
(APIServer pid=2725876) ERROR 10-19 13:07:53 [async_llm.py:521] Traceback (most recent call last):
(APIServer pid=2725876) ERROR 10-19 13:07:53 [async_llm.py:521] File "/home/congc/gitrepos/vllm/vllm/v1/engine/async_llm.py", line 475, in output_handler
(APIServer pid=2725876) ERROR 10-19 13:07:53 [async_llm.py:521] outputs = await engine_core.get_output_async()
(APIServer pid=2725876) ERROR 10-19 13:07:53 [async_llm.py:521] ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(APIServer pid=2725876) ERROR 10-19 13:07:53 [async_llm.py:521] raise self._format_exception(outputs) from None
(APIServer pid=2725876) ERROR 10-19 13:07:53 [async_llm.py:521] vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
(APIServer pid=2725876) INFO: 127.0.0.1:51160 - "POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
(EngineCore_DP0 pid=2731503) Process EngineCore_DP0:
(EngineCore_DP0 pid=2731503) Traceback (most recent call last):
(EngineCore_DP0 pid=2731503) File "/usr/lib64/python3.12/multiprocessing/process.py", line 314, in _bootstrap
(EngineCore_DP0 pid=2731503) self.run()
(EngineCore_DP0 pid=2731503) File "/usr/lib64/python3.12/multiprocessing/process.py", line 108, in run
(EngineCore_DP0 pid=2731503) self._target(*self._args, **self._kwargs)
(EngineCore_DP0 pid=2731503) File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 801, in run_engine_core
(EngineCore_DP0 pid=2731503) raise e
(EngineCore_DP0 pid=2731503) File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 790, in run_engine_core
(EngineCore_DP0 pid=2731503) engine_core.run_busy_loop()
(EngineCore_DP0 pid=2731503) File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 817, in run_busy_loop
(EngineCore_DP0 pid=2731503) self._process_engine_step()
(EngineCore_DP0 pid=2731503) File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 846, in _process_engine_step
(EngineCore_DP0 pid=2731503) outputs, model_executed = self.step_fn()
(EngineCore_DP0 pid=2731503) ^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) File "/home/congc/gitrepos/vllm/vllm/v1/engine/core.py", line 328, in step
(EngineCore_DP0 pid=2731503) engine_core_outputs = self.scheduler.update_from_output(
(EngineCore_DP0 pid=2731503) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) File "/home/congc/gitrepos/vllm/vllm/v1/core/sched/scheduler.py", line 923, in update_from_output
(EngineCore_DP0 pid=2731503) stats = self.connector.get_kv_connector_stats()
(EngineCore_DP0 pid=2731503) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) File "/home/congc/gitrepos/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py", line 244, in get_kv_connector_stats
(EngineCore_DP0 pid=2731503) assert self.connector_worker is not None
(EngineCore_DP0 pid=2731503) ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=2731503) AssertionError
[rank0]:[W1019 13:07:55.261767965 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
(APIServer pid=2725876) INFO: Shutting down
(APIServer pid=2725876) INFO: Waiting for application shutdown.
(APIServer pid=2725876) INFO: Application shutdown complete.
(APIServer pid=2725876) INFO: Finished server process [2725876]
```

## Root Cause

The error occurs in `/home/congc/gitrepos/vllm/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` at line 244:

```python
assert self.connector_worker is not None
```

This assertion fails, indicating that the `connector_worker` was not properly initialized before `get_kv_connector_stats()` was called. This happens during the scheduler's `update_from_output()` method when attempting to get KV connector statistics.

## Date

October 19, 2025 at 13:07:53
