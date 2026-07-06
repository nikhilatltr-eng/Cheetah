import os
import sys
import logging
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mt5_connector import MT5Connector
from features import compute_all_features
from storage import ParquetStorage
from regime_model import RegimeDetector
from model_registry import ModelRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_regime")

def main():
    logger.info("Starting Cheetah regime model training...")
    
    # 1. Initialize components
    connector = MT5Connector(mock=True)
    storage = ParquetStorage(base_dir="data_store")
    registry = ModelRegistry(registry_dir="models")
    
    symbol = "XAUUSD"
    timeframe = "M1"
    
    # Check if we have data in storage, if not, generate mock data to train
    df = storage.read_bars(symbol, timeframe)
    if df.empty or len(df) < 500:
        logger.info("Storage is empty or insufficient. Fetching warmup mock bars to seed storage...")
        df_warmup = connector.fetch_ohlcv(symbol, timeframe, 1000)
        storage.append_bars(symbol, timeframe, df_warmup)
        df = storage.read_bars(symbol, timeframe)
        
    logger.info(f"Loaded {len(df)} historical bars for training.")
    
    # 2. Compute features
    logger.info("Computing features for regime classification...")
    feature_df = compute_all_features(df)
    
    # 3. Fit Regime Detector
    detector = RegimeDetector()
    try:
        detector.fit(feature_df)
    except Exception as e:
        logger.error(f"Failed to fit regime detector: {e}")
        sys.exit(1)
        
    # Check that HMM states are distinct
    means = detector.hmm.means_
    logger.info("Verifying state uniqueness (means):")
    for state_idx, name in detector.state_map.items():
        state_mean = means[state_idx]
        logger.info(f"Regime State '{name}': Volatility={state_mean[0]:.6f}, ADX={state_mean[1]:.2f}, Duration={state_mean[2]:.1f}")
        
    # 4. Save to model registry
    metrics = {
        "ranging_state_means": list(means[list(detector.state_map.keys())[0]]),
        "trending_state_means": list(means[list(detector.state_map.keys())[1]]),
        "volatile_state_means": list(means[list(detector.state_map.keys())[2]]),
        "n_samples": len(df)
    }
    
    model_path, meta_path = registry.save_model(
        model_name="regime_hmm",
        model_obj=detector,
        data_df=df,
        metrics_dict=metrics,
        version=1
    )
    
    logger.info(f"Regime training complete. Registry files:\nModel: {model_path}\nMetadata: {meta_path}")

if __name__ == "__main__":
    main()
