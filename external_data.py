import logging
import time
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class ExternalDataManager:
    def __init__(self, cache_ttl_seconds=900):
        """
        Manages fetching external market and economic news data.
        Includes a simple in-memory TTL cache to prevent rate-limiting by yfinance.
        """
        self.ttl = cache_ttl_seconds
        self.price_cache = {}  # {ticker: (price, expiry_time)}
        self.history_cache = {}  # {key: (dataframe, expiry_time)}

    def get_latest_price(self, ticker_symbol) -> float:
        """Fetch current price of a ticker (e.g. DX-Y.NYB or ^TNX) with caching."""
        now = time.time()
        if ticker_symbol in self.price_cache:
            price, expiry = self.price_cache[ticker_symbol]
            if now < expiry:
                return price
                
        try:
            logger.info(f"Fetching latest price for {ticker_symbol} from yfinance...")
            ticker = yf.Ticker(ticker_symbol)
            price = None
            if hasattr(ticker, "fast_info") and "lastPrice" in ticker.fast_info:
                price = ticker.fast_info["lastPrice"]
            if price is None:
                df = ticker.history(period="1d")
                if not df.empty:
                    price = df["Close"].iloc[-1]
            
            if price is not None:
                price_val = float(price)
                self.price_cache[ticker_symbol] = (price_val, now + self.ttl)
                return price_val
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker_symbol}: {e}")
            
        # Fallback to stale value if available
        if ticker_symbol in self.price_cache:
            logger.warning(f"Using stale price cache for {ticker_symbol}.")
            return self.price_cache[ticker_symbol][0]
        return None

    def fetch_history(self, ticker_symbol, start_time, end_time, timeframe="1h") -> pd.DataFrame:
        """Fetch historical close prices for DXY or US 10Y Yield with caching."""
        now = time.time()
        
        # Ensure timestamps are localized or standard to build consistent cache keys
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
            
        cache_key = f"{ticker_symbol}_{start_time.isoformat()}_{end_time.isoformat()}_{timeframe}"
        if cache_key in self.history_cache:
            df, expiry = self.history_cache[cache_key]
            if now < expiry:
                return df
                
        # Map timeframe to yfinance intervals
        interval_map = {
            "M1": "1m",
            "M5": "5m",
            "M15": "15m",
            "H1": "1h",
            "H4": "1h",
            "D1": "1d"
        }
        yf_interval = interval_map.get(timeframe, "1d")
        
        # Safe limits for yfinance
        now_ts = pd.Timestamp.now(tz="UTC")
        delta_days = (now_ts - start_time).days
        if yf_interval == "1m" and delta_days > 7:
            yf_interval = "5m"
            delta_days = (now_ts - start_time).days
            if delta_days > 60:
                yf_interval = "1h"
                if delta_days > 730:
                    yf_interval = "1d"
                    
        try:
            logger.info(f"Fetching history for {ticker_symbol} ({yf_interval}) from yfinance...")
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start_time, end=end_time, interval=yf_interval)
            
            if df.empty:
                # Fallback to recent 1 month
                logger.warning(f"No history returned for range. Fetching recent 1mo...")
                df = ticker.history(period="1mo", interval=yf_interval)
                
            if not df.empty:
                df = df.reset_index()
                time_col = "Datetime" if "Datetime" in df.columns else ("Date" if "Date" in df.columns else df.columns[0])
                df = df.rename(columns={time_col: "timestamp", "Close": "close"})
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                res_df = df[["timestamp", "close"]].copy()
                self.history_cache[cache_key] = (res_df, now + self.ttl)
                return res_df
        except Exception as e:
            logger.error(f"Error fetching history for {ticker_symbol}: {e}")
            
        # Return empty df to prevent pipeline failure
        return pd.DataFrame(columns=["timestamp", "close"])

    def get_upcoming_events(self, within_minutes=1440):
        """
        Get upcoming economic events.
        TODO: Wire a real calendar API (e.g. Forex Factory, TradingEconomics, or Finnhub)
        using an API key here.
        For now, returns a list of mock events.
        """
        now = pd.Timestamp.now(tz="UTC")
        mock_events = [
            {
                "event": "US CPI MoM",
                "timestamp": now + pd.Timedelta(minutes=45),
                "impact": "HIGH"
            },
            {
                "event": "FOMC Meeting Minutes",
                "timestamp": now + pd.Timedelta(hours=4),
                "impact": "HIGH"
            },
            {
                "event": "US Initial Jobless Claims",
                "timestamp": now + pd.Timedelta(days=1),
                "impact": "MEDIUM"
            }
        ]
        target_time = now + pd.Timedelta(minutes=within_minutes)
        # Filter: event is in the future, up to target_time
        return [e for e in mock_events if now <= e["timestamp"] <= target_time]

    def minutes_to_next_high_impact_event(self, from_timestamp=None) -> float:
        """
        Returns the number of minutes until the next high-impact economic news event.
        If no event is upcoming within 24 hours, returns a default large value (e.g., 9999.0).
        """
        if from_timestamp is None:
            from_timestamp = pd.Timestamp.now(tz="UTC")
        elif isinstance(from_timestamp, str):
            from_timestamp = pd.to_datetime(from_timestamp)
            if from_timestamp.tzinfo is None:
                from_timestamp = from_timestamp.tz_localize("UTC")
                
        events = self.get_upcoming_events(within_minutes=1440)
        high_impact_events = [e for e in events if e["impact"] == "HIGH" and e["timestamp"] > from_timestamp]
        
        if not high_impact_events:
            return 9999.0
            
        next_event = min(high_impact_events, key=lambda e: e["timestamp"])
        time_diff = next_event["timestamp"] - from_timestamp
        return float(time_diff.total_seconds() / 60.0)
