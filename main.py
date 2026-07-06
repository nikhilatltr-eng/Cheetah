import os
import sys
import time
import logging
import yaml
import pandas as pd

from mt5_connector import MT5Connector
from external_data import ExternalDataManager
from storage import ParquetStorage
from features import compute_all_features
from baseline_strategy import generate_signal
from telegram_alerts import TelegramAlerts

# Logger definition (handlers are configured inside main())
logger = logging.getLogger("cheetah_main")

def main():
    # Setup structured logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("cheetah.log"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger.info("Initializing Cheetah XAUUSD MT5 Trading Bot (Phase 1)...")
    
    # Load configuration
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    symbol = config.get("symbol", "XAUUSD")
    timeframes = config.get("timeframes", ["M1", "M5", "M15", "H1", "H4", "D1"])
    polling_interval = config.get("polling_interval_seconds", 5)
    
    # Initialize components
    connector = MT5Connector(config_path=config_path)
    external_mgr = ExternalDataManager(cache_ttl_seconds=900)
    storage = ParquetStorage(base_dir="data_store")
    alerts = TelegramAlerts(config_path=config_path)
    
    # 1. Connect to MT5
    try:
        connector.connect()
    except Exception as e:
        logger.error(f"Initialization failure: Could not connect to MT5. Error: {e}")
        if not connector.mock:
            sys.exit(1)
            
    # 2. Warm up historical data (pull 600 bars for all configured timeframes)
    logger.info("Starting historical data warmup...")
    for tf in timeframes:
        try:
            logger.info(f"Warmup: Fetching 600 bars for {symbol} ({tf})...")
            df_warmup = connector.fetch_ohlcv(symbol, tf, 600)
            if not df_warmup.empty:
                storage.append_bars(symbol, tf, df_warmup)
                logger.info(f"Warmup: Stored {len(df_warmup)} bars for {symbol} ({tf}).")
            else:
                logger.warning(f"Warmup: No bars returned for {symbol} ({tf}).")
        except Exception as e:
            logger.error(f"Warmup: Failed to fetch/store data for {tf}: {e}")
            
    # Get initial active M1 candle to key the polling loop
    last_active_m1_opentime = None
    try:
        df_m1 = connector.fetch_ohlcv(symbol, "M1", 2)
        if not df_m1.empty:
            last_active_m1_opentime = df_m1["timestamp"].iloc[-1]
            logger.info(f"Keyed loop to active candle starting at: {last_active_m1_opentime} UTC")
    except Exception as e:
        logger.error(f"Failed to fetch initial M1 bar for loop key: {e}")
        
    if last_active_m1_opentime is None:
        # Fallback to current time if MT5 failed to return data
        last_active_m1_opentime = pd.Timestamp.now(tz="UTC").floor("min")
        logger.warning(f"Using default fallback candle key: {last_active_m1_opentime} UTC")
        
    logger.info("Startup complete. Entering main polling loop...")
    alerts.send_message("Cheetah MT5 Bot successfully started in Phase 1 (Data & Feature Pipeline)!")
    
    # Flag to stop the loop (can be wired to signal handler)
    running = True
    
    while running:
        try:
            time.sleep(polling_interval)
            
            # Check connection
            if not connector.connected:
                logger.warning("MT5 connection offline. Attempting to reconnect...")
                connector.connect()
                continue
                
            # Fetch last 2 M1 bars to check for new candle close
            df_check = connector.fetch_ohlcv(symbol, "M1", 2)
            if df_check.empty:
                logger.warning("Failed to fetch polling rates from MT5. Skipping cycle...")
                continue
                
            active_m1_opentime = df_check["timestamp"].iloc[-1]
            
            # If the current active candle's open time has increased,
            # the previous active candle (last_active_m1_opentime) is now closed!
            if active_m1_opentime > last_active_m1_opentime:
                closed_candle_time = last_active_m1_opentime
                logger.info(f"--- [NEW CANDLE CLOSED] Time: {closed_candle_time} | Active Time: {active_m1_opentime} ---")
                
                # 1. Update storage for all timeframes (pull last 5 bars to catch closes/updates)
                for tf in timeframes:
                    try:
                        df_update = connector.fetch_ohlcv(symbol, tf, 5)
                        if not df_update.empty:
                            storage.append_bars(symbol, tf, df_update)
                    except Exception as e:
                        logger.error(f"Error updating storage for timeframe {tf}: {e}")
                        
                # 2. Fetch history from storage for M1 features (last 500 bars)
                df_history = storage.read_bars(symbol, "M1")
                if df_history.empty or len(df_history) < 200:
                    logger.warning(f"Insufficient history in storage for feature engineering (Length: {len(df_history)}). Skipping features...")
                    last_active_m1_opentime = active_m1_opentime
                    continue
                    
                # Take last 500 rows to speed up calculation and keep it focused
                df_history = df_history.tail(500).reset_index(drop=True)
                
                # 3. Pull DXY history for rolling correlation
                start_time = df_history["timestamp"].iloc[0]
                end_time = df_history["timestamp"].iloc[-1]
                
                dxy_df = pd.DataFrame()
                try:
                    dxy_df = external_mgr.fetch_history("DX-Y.NYB", start_time, end_time, timeframe="M1")
                except Exception as e:
                    logger.error(f"Error fetching DXY history: {e}")
                    
                # 4. Compute all features
                logger.info("Computing technical and macro features...")
                feature_df = compute_all_features(df_history, dxy_df=dxy_df, external_mgr=external_mgr)
                
                # Check for NaNs in latest row
                latest_features = feature_df.iloc[-1]
                logger.info(
                    f"Features Computed | Close: {latest_features['close']:.2f} | "
                    f"EMA8: {latest_features['ema_8']:.2f} | EMA21: {latest_features['ema_21']:.2f} | "
                    f"ADX: {latest_features['adx']:.2f} | RSI: {latest_features['rsi']:.2f} | "
                    f"Volatility: {latest_features['realized_vol']:.5f} | NewsProximity: {latest_features['time_to_news']:.1f}m"
                )
                
                # 5. Run Strategy Signal Evaluation
                signal = generate_signal(feature_df)
                
                if signal["direction"] != "HOLD":
                    logger.info(f"Signal Generated: {signal['direction']} @ {signal['price']} | Triggering features: {signal['triggering_features']}")
                    # Dispatch alert
                    alerts.send_alert(signal)
                else:
                    logger.info("Decision: HOLD. Reason: No strategy rules triggered.")
                    
                # Update key for the next closed candle
                last_active_m1_opentime = active_m1_opentime
                
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
            running = False
        except Exception as e:
            logger.error(f"Exception in main execution loop: {e}", exc_info=True)
            time.sleep(polling_interval)  # Backoff sleep before retrying
            
    connector.disconnect()
    logger.info("Cheetah bot terminated cleanly.")

if __name__ == "__main__":
    main()
