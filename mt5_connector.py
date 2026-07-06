import logging
import time
import yaml
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import MT5 package safely for macOS/Linux environments
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logger.warning("MetaTrader5 package is not available. Running in Mock/Fallback mode.")

# Standard MT5 timeframe constants mapped manually to allow import on any platform
TIMEFRAME_MAP = {
    "M1": 1,        # mt5.TIMEFRAME_M1
    "M5": 5,        # mt5.TIMEFRAME_M5
    "M15": 15,      # mt5.TIMEFRAME_M15
    "H1": 16385,    # mt5.TIMEFRAME_H1
    "H4": 16388,    # mt5.TIMEFRAME_H4
    "D1": 16408,    # mt5.TIMEFRAME_D1
}

class MT5Connector:
    def __init__(self, config_path="config.yaml", mock=False):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.mt5_cfg = self.config.get("mt5", {})
        self.symbol = self.config.get("symbol", "XAUUSD")
        self.connected = False
        self.mock = mock
        
        if mt5 is None:
            self.mock = True
            logger.info("Forcing MOCK mode because MetaTrader5 package is not installed/supported.")

    def connect(self):
        """Initialize and login to MT5 using credentials with exponential backoff."""
        if self.mock:
            self.connected = True
            logger.info("[MOCK] Connected and authorized to mock broker server.")
            return True
            
        if mt5 is None:
            raise ImportError(
                "MetaTrader5 package is not available on this platform. "
                "Ensure you are running on Windows with MT5 terminal installed."
            )
            
        login = self.mt5_cfg.get("login")
        password = self.mt5_cfg.get("password")
        server = self.mt5_cfg.get("server")
        
        # Verify credentials format
        if login is not None:
            login = int(login)
            
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            logger.info(f"Attempting MT5 initialization (Attempt {attempt+1}/{max_retries})...")
            if not mt5.initialize():
                err = mt5.last_error()
                logger.error(f"mt5.initialize() failed: {err}")
                time.sleep(base_delay * (2 ** attempt))
                continue
                
            logger.info("Attempting login...")
            authorized = mt5.login(login=login, password=password, server=server)
            if authorized:
                self.connected = True
                logger.info("Successfully authorized with MT5 broker server.")
                try:
                    self.verify_symbol()
                except Exception as e:
                    logger.error(f"Symbol verification failed: {e}")
                    self.disconnect()
                    raise e
                return True
            else:
                err = mt5.last_error()
                logger.error(f"mt5.login() failed: {err}")
                mt5.shutdown()
                time.sleep(base_delay * (2 ** attempt))
                
        raise ConnectionError("Failed to connect to MT5 after multiple retries.")
        
    def verify_symbol(self):
        """Validate symbol exists via mt5.symbol_info(), failing loudly if not found."""
        if self.mock:
            logger.info(f"[MOCK] Symbol '{self.symbol}' verified successfully.")
            return
            
        if mt5 is None:
            raise ImportError("MetaTrader5 is not available.")
            
        sym_info = mt5.symbol_info(self.symbol)
        if sym_info is None:
            raise ValueError(f"Symbol '{self.symbol}' could not be resolved or found by the broker.")
            
        logger.info(f"Symbol '{self.symbol}' verified successfully.")
        
        # Make the symbol visible in Market Watch
        if not sym_info.visible:
            logger.info(f"Symbol '{self.symbol}' is not visible. Adding to Market Watch...")
            if not mt5.symbol_select(self.symbol, True):
                raise ValueError(f"Failed to select symbol '{self.symbol}' in Market Watch.")

    def fetch_ohlcv(self, symbol, timeframe, n_bars) -> pd.DataFrame:
        """Fetch historical bars (OHLCV) for a symbol/timeframe."""
        if self.mock:
            return self._generate_mock_ohlcv(symbol, timeframe, n_bars)
            
        if mt5 is None:
            raise ImportError("MetaTrader5 is not available.")
            
        tf_val = TIMEFRAME_MAP.get(timeframe)
        if tf_val is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        logger.debug(f"Fetching {n_bars} bars for {symbol} on timeframe {timeframe}...")
        rates = mt5.copy_rates_from_pos(symbol, tf_val, 0, n_bars)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"No OHLCV rates returned for symbol {symbol} on {timeframe}.")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        
        # Standardize columns
        cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        df = df[cols]
        return df

    def fetch_ticks(self, symbol, from_time, to_time) -> pd.DataFrame:
        """Fetch historical ticks for a symbol between two timestamps."""
        if self.mock:
            return self._generate_mock_ticks(symbol, from_time, to_time)
            
        if mt5 is None:
            raise ImportError("MetaTrader5 is not available.")
            
        if isinstance(from_time, str):
            from_time = pd.to_datetime(from_time)
        if isinstance(to_time, str):
            to_time = pd.to_datetime(to_time)
            
        logger.debug(f"Fetching ticks for {symbol} from {from_time} to {to_time}...")
        ticks = mt5.copy_ticks_range(symbol, from_time, to_time, mt5.COPY_TICKS_ALL)
        
        if ticks is None or len(ticks) == 0:
            logger.warning(f"No tick data returned for symbol {symbol} in the requested range.")
            return pd.DataFrame()
            
        df = pd.DataFrame(ticks)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "time_msc" in df.columns:
            df["timestamp_ms"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
            
        return df

    def poll_latest_tick(self, symbol) -> dict:
        """Poll the latest tick for a symbol. Returns dict with timestamp, bid, ask, last, volume."""
        if self.mock:
            now = pd.Timestamp.now(tz="UTC")
            # Generate a mock tick around gold spot price
            bid = 2350.0 + np.random.normal(0, 0.5)
            ask = bid + 0.3
            return {
                "timestamp": now,
                "bid": bid,
                "ask": ask,
                "last": bid,
                "volume": float(np.random.randint(1, 10)),
            }
            
        if mt5 is None:
            raise ImportError("MetaTrader5 is not available.")
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.warning(f"Failed to poll latest tick for symbol {symbol}.")
            return {}
            
        return {
            "timestamp": pd.to_datetime(tick.time_msc, unit="ms", utc=True),
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
        }

    def disconnect(self):
        """Shutdown MetaTrader 5 connection."""
        if self.mock:
            self.connected = False
            logger.info("[MOCK] Shut down mock connector.")
            return
            
        if mt5 is not None and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MetaTrader 5 connection shut down.")

    def _generate_mock_ohlcv(self, symbol, timeframe, n_bars) -> pd.DataFrame:
        """Generates synthetic OHLCV data using a random walk + sine wave for test validation."""
        now = pd.Timestamp.now(tz="UTC").floor("min")
        
        freq_map = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1D"
        }
        freq = freq_map.get(timeframe, "1min")
        timestamps = pd.date_range(end=now, periods=n_bars, freq=freq, tz="UTC")
        
        # Deterministic seed based on symbol/timeframe so it remains consistent
        seed_val = int(abs(hash(symbol + timeframe)) % (2**31))
        np.random.seed(seed_val)
        
        # Construct synthetic price sequence (Sine wave trend to guarantee crosses)
        x = np.linspace(0, 8 * np.pi, n_bars)
        sine_trend = 15 * np.sin(x)
        random_walk = np.cumsum(np.random.normal(0, 0.4, n_bars))
        
        close = 2350.0 + random_walk + sine_trend
        open_px = close - np.random.normal(0, 0.15, n_bars)
        high = np.maximum(open_px, close) + np.abs(np.random.normal(0, 0.25, n_bars))
        low = np.minimum(open_px, close) - np.abs(np.random.normal(0, 0.25, n_bars))
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": np.random.randint(10, 100, n_bars).astype(float),
            "spread": np.random.randint(1, 4, n_bars).astype(float),
            "real_volume": np.random.randint(100, 1000, n_bars).astype(float)
        })
        return df

    def _generate_mock_ticks(self, symbol, from_time, to_time) -> pd.DataFrame:
        """Generates synthetic tick data between two timestamps."""
        from_ts = pd.to_datetime(from_time, utc=True)
        to_ts = pd.to_datetime(to_time, utc=True)
        
        # Generate a tick every 5 seconds
        timestamps = pd.date_range(start=from_ts, end=to_ts, freq="5s", tz="UTC")
        n_ticks = len(timestamps)
        
        if n_ticks == 0:
            return pd.DataFrame()
            
        np.random.seed(42)
        prices = 2350.0 + np.cumsum(np.random.normal(0, 0.05, n_ticks))
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "time": timestamps.astype(np.int64) // 10**9,
            "bid": prices,
            "ask": prices + 0.2,
            "last": prices,
            "volume": np.random.randint(1, 5, n_ticks).astype(float)
        })
        return df
