# Realistic Cost Audit Report

This report analyzes historical bid/ask tick spread metrics from the MT5 broker connection during the OOS backtest window, comparing it to the base parameters of the cost stress tests.

## Spread Metrics Summary (in Price Points)
- **Sampled Ticks**: 100,000 ticks
- **Mean Spread**: 0.1238 (12.4 points)
- **Median Spread**: 0.1201 (12.0 points)
- **95th Percentile (p95) Spread**: 0.1813 (18.1 points)
- **Max Spread observed**: 0.3677 (36.8 points)

## Friction Comparison
- **Backtest Base Spread**: 0.1500 (15.0 points)
- **Broker Difference**: -17.44%
- **Materiality Threshold**: 20%
- **Status**: **PASS: Real spread is within safe threshold limits of backtest base parameter.**

## Conclusion
The sampled live broker spreads are highly aligned with the 1x cost base parameter of 0.15 points (15 points). The median spread of 12.0 points is below our 15 points parameter, meaning our backtest costs are appropriately conservative and did not underestimate transaction friction during normal trading hours.
