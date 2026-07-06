import logging
import numpy as np

logger = logging.getLogger("cheetah_latency")

class LatencyProfiler:
    def __init__(self, budget_ms: float = 50.0, max_samples: int = 500):
        """
        Tracks sub-millisecond execution times in the fast loop and computes
        rolling statistical distributions (p50, p95, p99).
        """
        self.budget_ms = budget_ms
        self.max_samples = max_samples
        self.samples = []

    def record_latency(self, latency_ms: float):
        """Appends a new latency measurement to the rolling sampler."""
        self.samples.append(latency_ms)
        if len(self.samples) > self.max_samples:
            self.samples.pop(0)
            
        # Logging check
        if latency_ms > self.budget_ms:
            logger.warning(
                f"Latency Alert: Execution took {latency_ms:.2f}ms, exceeding budget of {self.budget_ms}ms!"
            )

    def get_percentiles(self) -> dict:
        """Computes current p50, p95, and p99 from sampling history."""
        if not self.samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "samples_count": 0}
            
        arr = np.array(self.samples)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "samples_count": len(self.samples)
        }

    def check_budget_breach(self) -> bool:
        """Returns True if the p95 latency exceeds our budget."""
        metrics = self.get_percentiles()
        if metrics["samples_count"] < 10:
            # Need a minimum warm-up window before reporting breaches
            return False
        return metrics["p95"] > self.budget_ms
