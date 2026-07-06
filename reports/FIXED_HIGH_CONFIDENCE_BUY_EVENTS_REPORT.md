# High Confidence BUY Signal Holding Policies Evaluation Report

This report evaluates performance metrics of high-confidence BUY predictions under five different holding/exit policies. 

## Simulation Controls
- **One Active Trade Limit**: Only one position can be open at any time.
- **No Active Re-entry**: Signal triggers while a position is open are locked out.
- **Execution Cost Inclusions**: BUY entries are executed at Ask (`Close + Spread + Slippage`), and BUY exits are executed at Bid (`Close/Stop - Slippage`).
- **Spread & Slippage Constants**: Spread = 15 points (0.15), Slippage = 5 points (0.05).

---

## 📊 Confidence Threshold >= 99%

| Exit Policy | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD | Avg Dur (min) | Avg Raw Signals/Trade |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EMA21 exit | 0 | N/A | N/A | $0.00 | 0.00% | N/A | N/A |
| EMA55 exit | 0 | N/A | N/A | $0.00 | 0.00% | N/A | N/A |
| ATR trailing stop | 0 | N/A | N/A | $0.00 | 0.00% | N/A | N/A |
| TP 1.5 ATR / SL 1 ATR | 0 | N/A | N/A | $0.00 | 0.00% | N/A | N/A |
| TP 2 ATR / SL 1 ATR | 0 | N/A | N/A | $0.00 | 0.00% | N/A | N/A |
---## 📊 Confidence Threshold >= 90%

| Exit Policy | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD | Avg Dur (min) | Avg Raw Signals/Trade |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EMA21 exit | 13 | 53.85% | 1.59 | $61.13 | 0.32% | 1.0 | 1.00 |
| EMA55 exit | 13 | 53.85% | 1.59 | $61.13 | 0.32% | 1.0 | 1.00 |
| ATR trailing stop | 10 | 50.00% | 2.04 | $146.34 | 0.83% | 12.3 | 1.30 |
| TP 1.5 ATR / SL 1 ATR | 12 | 41.67% | 2.06 | $218.60 | 0.76% | 3.8 | 1.08 |
| TP 2 ATR / SL 1 ATR | 12 | 33.33% | 1.68 | $181.23 | 0.76% | 9.5 | 1.08 |
---## 📊 Confidence Threshold >= 80%

| Exit Policy | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD | Avg Dur (min) | Avg Raw Signals/Trade |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EMA21 exit | 28 | 60.71% | 2.56 | $230.23 | 0.32% | 2.2 | 1.07 |
| EMA55 exit | 28 | 57.14% | 1.97 | $175.73 | 0.62% | 5.4 | 1.07 |
| ATR trailing stop | 23 | 47.83% | 2.48 | $361.38 | 0.83% | 11.0 | 1.30 |
| TP 1.5 ATR / SL 1 ATR | 26 | 42.31% | 1.56 | $223.15 | 1.82% | 7.2 | 1.15 |
| TP 2 ATR / SL 1 ATR | 26 | 30.77% | 1.24 | $119.72 | 1.82% | 10.8 | 1.15 |
---## 📊 Confidence Threshold >= 70%

| Exit Policy | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD | Avg Dur (min) | Avg Raw Signals/Trade |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EMA21 exit | 352 | 82.39% | 10.90 | $7107.88 | 0.49% | 2.1 | 1.05 |
| EMA55 exit | 351 | 82.05% | 9.30 | $6861.74 | 0.50% | 2.6 | 1.05 |
| ATR trailing stop | 336 | 62.50% | 7.00 | $12400.13 | 0.93% | 32.3 | 16.21 |
| TP 1.5 ATR / SL 1 ATR | 346 | 67.63% | 3.16 | $6641.90 | 2.41% | 21.2 | 15.74 |
| TP 2 ATR / SL 1 ATR | 343 | 57.73% | 2.76 | $7044.17 | 2.56% | 23.4 | 15.88 |
---## 📊 Confidence Threshold >= 60%

| Exit Policy | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD | Avg Dur (min) | Avg Raw Signals/Trade |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EMA21 exit | 3099 | 75.93% | 6.65 | $36989.17 | 1.46% | 2.7 | 1.07 |
| EMA55 exit | 3009 | 75.44% | 5.16 | $36406.25 | 1.46% | 5.3 | 2.01 |
| ATR trailing stop | 2418 | 58.85% | 4.58 | $60131.53 | 1.31% | 21.5 | 7.66 |
| TP 1.5 ATR / SL 1 ATR | 2855 | 64.48% | 2.67 | $39377.13 | 2.11% | 9.2 | 4.71 |
| TP 2 ATR / SL 1 ATR | 2702 | 54.77% | 2.38 | $39508.16 | 2.05% | 12.6 | 5.92 |



# High Confidence BUY Signal Holding Policies Stress Test Report

This report evaluates cost resilience, trading session filters, and monthly performance breakdowns of our best-performing BUY-only holding strategies on the out-of-sample (OOS) test partition.

## Strategy Target Configurations
1. **EMA21 exit**: Long position closed immediately when M5 price closes below EMA21.
2. **ATR trailing stop**: Trailing stop set at `Highest Bid observed - 1.5 * ATR_14`.

---

## 📊 Confidence Threshold >= 70%

### 🛡️ Cost Stress Test: EMA21 exit

| Cost Level | Spread | Slippage | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1x cost | 0.15 | 0.05 | 352 | 82.39% | 10.90 | $7107.88 | 0.49% |
| 2x cost | 0.30 | 0.10 | 352 | 77.56% | 7.98 | $6227.88 | 0.54% |
| 3x cost | 0.45 | 0.15 | 352 | 72.73% | 5.80 | $5347.88 | 0.64% |
| 0.40 Spread Equivalent | 0.40 | 0.05 | 352 | 77.56% | 7.98 | $6227.88 | 0.54% |

### 🛡️ Cost Stress Test: ATR trailing stop

| Cost Level | Spread | Slippage | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1x cost | 0.15 | 0.05 | 336 | 62.50% | 7.00 | $12400.13 | 0.93% |
| 2x cost | 0.30 | 0.10 | 336 | 59.82% | 5.83 | $11560.13 | 1.03% |
| 3x cost | 0.45 | 0.15 | 336 | 57.74% | 4.91 | $10720.13 | 1.13% |
| 0.40 Spread Equivalent | 0.40 | 0.05 | 336 | 59.82% | 5.83 | $11560.13 | 1.03% |

### ⏰ Trading Sessions Stress Test: EMA21 exit (1x Cost)

| Session | Entry Hours (UTC) | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| London only | 08:00 - 16:00 | 114 | 89.47% | 20.39 | $2538.21 | 0.33% |
| New York only | 13:00 - 21:00 | 112 | 85.71% | 15.32 | $2547.44 | 0.45% |
| London + NY | 08:00 - 21:00 | 185 | 85.95% | 14.26 | $3648.85 | 0.42% |
| All sessions | 24h | 352 | 82.39% | 10.90 | $7107.88 | 0.49% |

### ⏰ Trading Sessions Stress Test: ATR trailing stop (1x Cost)

| Session | Entry Hours (UTC) | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| London only | 08:00 - 16:00 | 110 | 70.00% | 10.25 | $5432.74 | 0.47% |
| New York only | 13:00 - 21:00 | 110 | 60.00% | 7.08 | $4055.32 | 0.81% |
| London + NY | 08:00 - 21:00 | 180 | 63.33% | 7.86 | $6843.31 | 0.50% |
| All sessions | 24h | 336 | 62.50% | 7.00 | $12400.13 | 0.93% |

### 📅 Monthly Performance Breakdown: EMA21 exit (1x Cost, All Sessions)

| Month | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04 | 119 | 80.67% | 12.31 | $2863.23 | 0.35% |
| 2026-05 | 102 | 82.35% | 11.32 | $1802.51 | 0.41% |
| 2026-06 | 131 | 83.97% | 9.42 | $2442.14 | 0.70% |

### 📅 Monthly Performance Breakdown: ATR trailing stop (1x Cost, All Sessions)

| Month | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04 | 113 | 62.83% | 7.82 | $5375.56 | 0.93% |
| 2026-05 | 97 | 60.82% | 6.13 | $3157.14 | 1.10% |
| 2026-06 | 126 | 63.49% | 6.81 | $3867.43 | 0.60% |

---## 📊 Confidence Threshold >= 60%

### 🛡️ Cost Stress Test: EMA21 exit

| Cost Level | Spread | Slippage | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1x cost | 0.15 | 0.05 | 3099 | 75.93% | 6.65 | $36989.17 | 1.46% |
| 2x cost | 0.30 | 0.10 | 3099 | 66.92% | 4.35 | $29241.67 | 1.66% |
| 3x cost | 0.45 | 0.15 | 3099 | 57.95% | 2.85 | $21494.17 | 1.90% |
| 0.40 Spread Equivalent | 0.40 | 0.05 | 3099 | 66.92% | 4.35 | $29241.67 | 1.66% |

### 🛡️ Cost Stress Test: ATR trailing stop

| Cost Level | Spread | Slippage | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1x cost | 0.15 | 0.05 | 2418 | 58.85% | 4.58 | $60131.53 | 1.31% |
| 2x cost | 0.30 | 0.10 | 2418 | 55.62% | 3.79 | $54086.53 | 1.56% |
| 3x cost | 0.45 | 0.15 | 2418 | 53.06% | 3.17 | $48041.53 | 1.82% |
| 0.40 Spread Equivalent | 0.40 | 0.05 | 2418 | 55.62% | 3.79 | $54086.53 | 1.56% |

### ⏰ Trading Sessions Stress Test: EMA21 exit (1x Cost)

| Session | Entry Hours (UTC) | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| London only | 08:00 - 16:00 | 1176 | 78.74% | 8.56 | $15251.79 | 0.55% |
| New York only | 13:00 - 21:00 | 888 | 77.25% | 8.72 | $12269.01 | 1.07% |
| London + NY | 08:00 - 21:00 | 1689 | 77.03% | 7.55 | $20378.44 | 0.81% |
| All sessions | 24h | 3099 | 75.93% | 6.65 | $36989.17 | 1.46% |

### ⏰ Trading Sessions Stress Test: ATR trailing stop (1x Cost)

| Session | Entry Hours (UTC) | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| London only | 08:00 - 16:00 | 907 | 59.87% | 4.61 | $24782.14 | 0.84% |
| New York only | 13:00 - 21:00 | 706 | 57.79% | 4.56 | $19055.76 | 1.28% |
| London + NY | 08:00 - 21:00 | 1323 | 58.50% | 4.58 | $33911.62 | 1.01% |
| All sessions | 24h | 2418 | 58.85% | 4.58 | $60131.53 | 1.31% |

### 📅 Monthly Performance Breakdown: EMA21 exit (1x Cost, All Sessions)

| Month | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04 | 970 | 76.91% | 6.96 | $13829.98 | 1.46% |
| 2026-05 | 1008 | 75.40% | 6.89 | $10824.16 | 0.45% |
| 2026-06 | 1121 | 75.56% | 6.18 | $12335.03 | 1.28% |

### 📅 Monthly Performance Breakdown: ATR trailing stop (1x Cost, All Sessions)

| Month | Trade Count | Win Rate | Profit Factor | Cost-Adj PnL ($) | Max DD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-04 | 754 | 58.75% | 4.43 | $19548.55 | 1.31% |
| 2026-05 | 789 | 59.82% | 4.96 | $19395.32 | 0.91% |
| 2026-06 | 875 | 58.06% | 4.42 | $21187.66 | 1.82% |


