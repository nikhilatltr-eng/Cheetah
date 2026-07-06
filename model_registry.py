import os
import json
import hashlib
import logging
import datetime
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, registry_dir: str = "models"):
        """
        Versioned model registry. Saves models as joblib files
        and outputs matching JSON files containing performance metrics,
        training parameters, data hashes, and git commit versioning.
        """
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Computes MD5 hash of the training DataFrame to guarantee data provenance."""
        if df is None or df.empty:
            return ""
        # Hash the string representation of DataFrame values
        try:
            hash_val = hashlib.md5(df.to_string().encode("utf-8")).hexdigest()
            return hash_val
        except Exception as e:
            logger.warning(f"Error computing data hash: {e}")
            return "hash_error"

    def save_model(self, model_name: str, model_obj, data_df: pd.DataFrame, metrics_dict: dict, version: int = 1) -> tuple:
        """
        Saves a model binary and creates a version-matched JSON metadata file.
        Returns paths: (model_path, metadata_path)
        """
        model_filename = f"{model_name}_v{version}.joblib"
        meta_filename = f"{model_name}_v{version}.json"
        
        model_path = os.path.join(self.registry_dir, model_filename)
        meta_path = os.path.join(self.registry_dir, meta_filename)
        
        # 1. Save binary
        logger.info(f"Saving model binary to {model_path}...")
        joblib.dump(model_obj, model_path)
        
        # 2. Extract metadata parameters
        start_date = ""
        end_date = ""
        if data_df is not None and "timestamp" in data_df.columns:
            ts_series = pd.to_datetime(data_df["timestamp"])
            start_date = ts_series.min().isoformat()
            end_date = ts_series.max().isoformat()
            
        git_commit = "unknown"
        try:
            import subprocess
            # Query standard git log for current head revision hash
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            # Silence error if running outside a git repo
            pass
            
        metadata = {
            "model_name": model_name,
            "version": version,
            "save_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_range": {
                "start": start_date,
                "end": end_date
            },
            "data_hash": self._compute_data_hash(data_df),
            "git_commit": git_commit,
            "metrics": metrics_dict
        }
        
        # 3. Save JSON metadata sidecar
        logger.info(f"Saving metadata sidecar to {meta_path}...")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Successfully registered model '{model_name}' (Version: {version})")
        return model_path, meta_path

    def load_model(self, model_name: str, version: int = 1) -> tuple:
        """
        Loads a registered model binary and its matching JSON metadata.
        Returns: (model_obj, metadata_dict)
        """
        model_filename = f"{model_name}_v{version}.joblib"
        meta_filename = f"{model_name}_v{version}.json"
        
        model_path = os.path.join(self.registry_dir, model_filename)
        meta_path = os.path.join(self.registry_dir, meta_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found in registry.")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file '{meta_path}' not found in registry.")
            
        logger.info(f"Loading registered model '{model_name}' (Version: {version})...")
        model_obj = joblib.load(model_path)
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
        logger.info("Model and metadata successfully loaded.")
        return model_obj, metadata
