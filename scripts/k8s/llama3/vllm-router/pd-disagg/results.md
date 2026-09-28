# Benchmark Results
============ Serving Benchmark Result ============
Successful requests:                     100       
Maximum request concurrency:             16        
Benchmark duration (s):                  42.11     
Total input tokens:                      99744     
Total generated tokens:                  100000    
Request throughput (req/s):              2.37      
Output token throughput (tok/s):         2374.72   
Peak output token throughput (tok/s):    2704.00   
Peak concurrent requests:                32.00     
Total Token throughput (tok/s):          4743.35   
---------------Time to First Token----------------
Mean TTFT (ms):                          64.51     
Median TTFT (ms):                        61.32     
P99 TTFT (ms):                           101.17    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.98      
Median TPOT (ms):                        5.98      
P99 TPOT (ms):                           6.03      
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.98      
Median ITL (ms):                         5.95      
P99 ITL (ms):                            6.70    

# Eval Results
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7748|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7036|±  |0.0126|