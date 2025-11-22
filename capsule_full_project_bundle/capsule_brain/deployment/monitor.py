"""Prometheus metrics for Capsule Brain deployment.

This module defines counters, histograms and gauges that can be used
to instrument a running Capsule Brain.  Start an HTTP server via
``prometheus_client.start_http_server`` to expose the metrics.
"""

from prometheus_client import Counter, Histogram, Gauge

# Total number of inference calls
inference_count = Counter("capsule_brain_inferences_total", "Total inference calls")
# Distribution of inference latency
inference_latency = Histogram("capsule_brain_latency_seconds", "Inference latency in seconds")
# Current memory pressure reported by the PMM (0–1)
memory_pressure = Gauge("capsule_brain_memory_pressure", "Memory pressure (0-1)")
# Number of active agents/tasks
active_agents = Gauge("capsule_brain_active_tasks", "Number of active tasks")

# Number of inference calls per skill
skill_inference_count = Counter(
    "capsule_brain_skill_inferences_total", "Inference calls per skill", ["skill"]
)
