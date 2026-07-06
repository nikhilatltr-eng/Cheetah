# Model Performance Report

## Champion Classification Performance (OOS Test)
- **Out-of-Sample Accuracy**: 65.02%
- **Precision (Trade vs Hold)**: 89.58%
- **Recall (Trade vs Hold)**: 99.37%
- **F1 Score**: 0.9422

## Feature Importance (Top 10 Contributors)
| Feature | SHAP/Importance Score |
| :--- | :--- |
| `trend_duration` | 0.6063 |
| `atr_14` | 0.4846 |
| `volume` | 0.4631 |
| `atr_21` | 0.3937 |
| `atr_55` | 0.0837 |
| `ema_55` | 0.0498 |
| `swing_high` | 0.0382 |
| `swing_low` | 0.0360 |
| `rsi` | 0.0349 |
| `realized_vol` | 0.0278 |
