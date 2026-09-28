# Benchmark results
============ Serving Benchmark Result ============
Successful requests:                     100       
Maximum request concurrency:             16        
Benchmark duration (s):                  41.24     
Total input tokens:                      99744     
Total generated tokens:                  100000    
Request throughput (req/s):              2.43      
Output token throughput (tok/s):         2425.01   
Peak output token throughput (tok/s):    2742.00   
Peak concurrent requests:                32.00     
Total Token throughput (tok/s):          4843.81   
---------------Time to First Token----------------
Mean TTFT (ms):                          60.77     
Median TTFT (ms):                        59.30     
P99 TTFT (ms):                           78.48     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.87      
Median TPOT (ms):                        5.88      
P99 TPOT (ms):                           5.91      
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.87      
Median ITL (ms):                         5.85      
P99 ITL (ms):                            6.49 

# Eval Results
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7748|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7036|±  |0.0126|