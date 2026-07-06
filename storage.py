import os
import shutil
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

class ParquetStorage:
    def __init__(self, base_dir="data_store"):
        """
        Manages historical OHLCV data storage using Parquet partitioned by:
        symbol / timeframe / date.
        """
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
    def _get_partition_path(self, symbol: str, timeframe: str, date_str: str) -> str:
        """Returns the partition directory path following Hive conventions."""
        return os.path.join(
            self.base_dir, 
            f"symbol={symbol}", 
            f"timeframe={timeframe}", 
            f"date={date_str}"
        )

    def append_bars(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        Append bars to the Parquet dataset. 
        Idempotent: merges new data with existing data in each partition,
        removes duplicates based on timestamp, and rewrites the partition files.
        """
        if df.empty:
            logger.info(f"Empty dataframe passed to append_bars for {symbol} / {timeframe}.")
            return
            
        df = df.copy()
        # Normalize timestamps to timezone-aware UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # Create partition key 'date'
        df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
        
        # Standard columns to write
        cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
        
        # Group by date to process partition by partition
        grouped = df.groupby("date")
        for date_str, group in grouped:
            partition_path = self._get_partition_path(symbol, timeframe, date_str)
            
            existing_df = pd.DataFrame()
            if os.path.exists(partition_path):
                try:
                    # Load all Parquet files in this partition folder
                    existing_df = pq.read_table(partition_path).to_pandas()
                except Exception as e:
                    logger.warning(
                        f"Error reading partition {partition_path} (could be corrupted): {e}. "
                        f"Overwriting instead."
                    )
            
            if not existing_df.empty:
                # Align timezones
                existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)
                # Concatenate existing and new data
                combined_df = pd.concat([existing_df, group[cols + ["date"]]], ignore_index=True)
            else:
                combined_df = group[cols + ["date"]].copy()
                
            # Remove duplicate timestamps (keep the last one to allow updating current bar)
            combined_df = combined_df.drop_duplicates(subset=["timestamp"], keep="last")
            # Sort chronologically
            combined_df = combined_df.sort_values(by="timestamp").reset_index(drop=True)
            
            # Re-create partition directory cleanly
            if os.path.exists(partition_path):
                shutil.rmtree(partition_path)
            os.makedirs(partition_path, exist_ok=True)
            
            # Write parquet file (drop date column to avoid Hive partition mismatch)
            write_df = combined_df.drop(columns=["date"])
            file_path = os.path.join(partition_path, "data.parquet")
            table = pa.Table.from_pandas(write_df, preserve_index=False)
            pq.write_table(table, file_path)
            logger.debug(f"Wrote partition: {partition_path} with {len(write_df)} rows.")

    def read_bars(self, symbol: str, timeframe: str, start_time=None, end_time=None) -> pd.DataFrame:
        """
        Read bars from Parquet dataset, applying symbol and timeframe filters,
        and restricting by start and end timestamps.
        """
        if not os.path.exists(self.base_dir):
            logger.warning(f"Storage directory {self.base_dir} does not exist yet.")
            return pd.DataFrame()
            
        try:
            # Discover and load Hive dataset
            dataset = ds.dataset(self.base_dir, partitioning="hive")
            if len(dataset.files) == 0:
                logger.debug("No files found in Parquet storage dataset.")
                return pd.DataFrame()
                
            # Filter by Hive partition fields
            filter_expr = (ds.field("symbol") == symbol) & (ds.field("timeframe") == timeframe)
            
            # Load to pyarrow table and convert to pandas
            table = dataset.to_table(filter=filter_expr)
            df = table.to_pandas()
            
            if df.empty:
                return df
                
            # Normalize timestamp timezone for comparison
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values(by="timestamp").reset_index(drop=True)
            
            # Apply time filtering
            if start_time is not None:
                start_time = pd.to_datetime(start_time, utc=True)
                df = df[df["timestamp"] >= start_time]
            if end_time is not None:
                end_time = pd.to_datetime(end_time, utc=True)
                df = df[df["timestamp"] <= end_time]
                
            # Keep standard columns only
            standard_cols = ["timestamp", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            cols_to_keep = [c for c in standard_cols if c in df.columns]
            df = df[cols_to_keep].reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.error(f"Failed to read bars from Parquet storage: {e}")
            return pd.DataFrame()
