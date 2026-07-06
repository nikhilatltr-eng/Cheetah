# MT5 Backtest Report

Historical backtesting results on Vantage MT5 broker data (0.28 spread, 0.05 slippage).

## 📊 Core Performance Summary
| Config | Trades | Win Rate | Profit Factor | Net PnL | Max DD | Daily Avg |
| --- | --- | --- | --- | --- | --- | --- |
| Thresh 70%, EMA21 exit | 11246 | 35.69% [34.8% - 36.6%] | 0.73 | $-28613.90 | 284.44% | 175.72 |
| Thresh 70%, ATR trailing stop | 4177 | 44.70% [43.2% - 46.2%] | 1.83 | $51464.76 | 5.74% | 65.27 |
| Thresh 60%, ATR trailing stop | 5808 | 43.85% [42.6% - 45.1%] | 1.69 | $62481.61 | 5.99% | 90.75 |

## 🛡️ Risk & Validation Checklist
- **No Same-Candle Exit**: Verified.
- **One Active Position Limit**: Verified.
- **No overlapping concurrent entries**: Verified.
