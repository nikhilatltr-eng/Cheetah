# Out-of-Sample (OOS) Validation Report

## Chronological Split Metrics
| Partition | Date Range | Sample Count | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 2024-01-01 to 2025-12-31 | 1052640 | N/A | N/A | N/A | N/A |
| **Validation** | 2026-01-01 to 2026-03-31 | 129600 | 65.05% | 89.74% | 99.13% | 0.9420 |
| **OOS Test** | 2026-04-01 to 2026-06-30 | 131040 | 65.02% | 89.58% | 99.37% | 0.9422 |

## Target Realization
All splits are segmented chronologically without overlap or leakage. No lookahead features are fed into training arrays.
