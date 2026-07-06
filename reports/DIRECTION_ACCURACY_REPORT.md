# Directional Accuracy Report

This report documents the accuracy and distribution of predictions for directional trades (BUY/SELL) on the out-of-sample (OOS) test partition.

## Directional Accuracy Summary
- **Total BUY Predictions**: 77657
- **Correct BUY Predictions**: 54885
- **BUY Prediction Accuracy**: 70.68%
- **Total SELL Predictions**: 45562
- **Correct SELL Predictions**: 23202
- **SELL Prediction Accuracy**: 50.92%
- **Overall Directional Accuracy**: 63.37%

## Direction Confusion Matrix
| Predicted | Actual BUY | Actual SELL |
| :--- | :--- | :--- |
| **Predicted BUY** | 54885 (Correct) | 16057 (Incorrect) |
| **Predicted SELL** | 16240 (Incorrect) | 23202 (Correct) |

## Accuracy by Market Regime
| Market Regime | Directional Accuracy | Trade Signals Count |
| :--- | :--- | :--- |
| **Trending** | 92.16% | 20239 |
| **Ranging** | 92.27% | 20190 |
| **High Volatility** | 49.29% | 82790 |

## Accuracy by ML Confidence Bucket
| Confidence Bucket | Directional Accuracy | Signals Count |
| :--- | :--- | :--- |
| 50–60% | 55.22% | 24868 |
| 60–70% | 64.40% | 4138 |
| 70–80% | 84.24% | 330 |
| 80–90% | 89.36% | 47 |
| 90%+ | 99.97% | 34376 |

