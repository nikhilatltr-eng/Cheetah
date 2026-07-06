# Purge & Embargo Validation Audit Report

This report audits the out-of-sample (OOS) test partition against the training partition to ensure zero overlap/leakage between training trade windows and testing trade windows.

## Audit Configuration
- **Training Period**: 2024-01-01 to 2025-12-31
- **Out-of-Sample (OOS) Test Period**: 2026-04-01 to 2026-06-30
- **Embargo Window**: 10 M1 bars (10 minutes)
- **Max Train Exit Time observed**: 2026-01-01 00:19:00
- **Embargo Cutoff Time**: 2026-01-01 00:29:00

## Audit Summary
- **Total OOS Samples Evaluated**: 131040
- **Count with Training Overlap**: 0
- **Count with Embargo Violations**: 0
- **Purge Audit Verdict**: **PASS**

## Analysis & Rationale
Because the validation design incorporates a 3-month chronological hold-out buffer (2026-01-01 to 2026-03-31, used for calibration/validation) between the training partition and the OOS test partition, no training samples can overlap with OOS entries. The 3-month gap exceeds the maximum triple-barrier holding window (20 M1 bars = 20 minutes) by several orders of magnitude.

Therefore, the split is mathematically guaranteed to be leakage-free and fully purged.
