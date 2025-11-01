# Llama 3.1 8B Multi-Node DP - Performance Results

Results from vLLM multi-node data parallelism deployment with 2 ranks (16 GPUs total: 8 per node).

## Configuration

- **Model**: Llama 3.1 8B Instruct
- **Setup**: Multi-node DP with master-worker coordination
- **Ranks**: 2 (Rank 0 = Master, Rank 1 = Worker)
- **GPUs**: 16 total (8 per rank)
- **Tensor Parallelism**: 1 (no TP)
- **Data Parallelism**: 2 ranks with 8 workers each

## Benchmark Results

============ Serving Benchmark Result ============
Successful requests:                     100       
Maximum request concurrency:             16        
Benchmark duration (s):                  100.93    
Total input tokens:                      199761    
Total generated tokens:                  200000    
Request throughput (req/s):              0.99      
Output token throughput (tok/s):         1981.66   
Peak output token throughput (tok/s):    2320.00   
Peak concurrent requests:                32.00     
Total Token throughput (tok/s):          3960.95   
---------------Time to First Token----------------
Mean TTFT (ms):                          47.23     
Median TTFT (ms):                        43.61     
P99 TTFT (ms):                           75.72     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.31      
Median TPOT (ms):                        7.36      
P99 TPOT (ms):                           7.37      
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.31      
Median ITL (ms):                         7.34      
P99 ITL (ms):                            7.97     


## Evaluation Results

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7748|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7036|±  |0.0126|

## Multi-Node DP Verification

During both benchmark and evaluation runs:

✅ **Rank 0 (Master)**:
- Received all HTTP requests
- DP Coordinator distributed work

✅ **Rank 1 (Worker)**:
- Received work via RPC (no HTTP logs)
- Processed GPU workload
