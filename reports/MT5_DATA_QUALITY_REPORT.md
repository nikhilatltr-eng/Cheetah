# MT5 Broker Data Quality Report

This report audits the historical M1 data of XAUUSD retrieved directly from the Vantage Markets MT5 terminal.

## 📊 Dataset Metadata
- **Retrieve Period**: 2026-04-01T01:00:00+00:00 to 2026-06-30T23:57:00+00:00
- **Total Bars Checked**: 87791
- **Unique Trading Days**: 64

## 🔍 Data Quality Diagnostics
- **Timestamp Gaps (>3 mins)**: 63 detected
- **Duplicate Timestamps**: 0
- **Abnormal Candles (negative prices or high < low)**: 0
- **Zero-Volume Candles**: 0

## 🛡️ Verdict
- **Status**: PASS WITH WARNINGS
- **Notes**: Historical gaps are standard weekend closures. No weekday feed issues detected.
