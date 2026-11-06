"""GPU helpers, split by job.

``process`` starts and stops real vLLM workers. ``monitor`` samples GPU
utilization. ``bench`` runs genai-bench and checks the recorded metrics.
"""
