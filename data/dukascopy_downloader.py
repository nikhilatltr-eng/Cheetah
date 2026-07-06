import os
import lzma
import struct
import logging
import datetime
import time
import urllib.request
import pandas as pd
import numpy as np

logger = logging.getLogger("cheetah_dukascopy")

class DukascopyDownloader:
    def __init__(self, symbol: str = "XAUUSD", raw_dir: str = "data/raw/dukascopy"):
        """
        Downloads and parses daily 1-minute candle files (LZMA bi5 format) from Dukascopy.
        """
        self.symbol = symbol.upper()
        # Adjust symbol naming convention (Dukascopy Gold is XAUUSD)
        if self.symbol == "GOLD":
            self.symbol = "XAUUSD"
            
        self.raw_dir = os.path.join(raw_dir, "m1")
        os.makedirs(self.raw_dir, exist_ok=True)
        
        self.base_url = "https://datafeed.dukascopy.com/datafeed"

    def _get_url_and_path(self, date_val: datetime.date) -> tuple:
        # Dukascopy expects 0-indexed month (00 for January, 11 for December)
        month_0 = date_val.month - 1
        url = f"{self.base_url}/{self.symbol}/{date_val.year}/{month_0:02d}/{date_val.day:02d}/BID_candles_min_1.bi5"
        
        file_name = f"{date_val.year}_{date_val.month:02d}_{date_val.day:02d}.bi5"
        file_path = os.path.join(self.raw_dir, file_name)
        return url, file_path

    def download_day(self, date_val: datetime.date, max_retries: int = 3) -> str:
        """Downloads the bi5 file for a single day, supporting retries and resume."""
        if getattr(self, "server_offline", False):
            return ""
            
        url, file_path = self._get_url_and_path(date_val)
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.debug(f"Dukascopy: File already exists for {date_val.isoformat()}. Skipping.")
            return file_path
            
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        
        delay = 2.0
        for attempt in range(max_retries):
            try:
                logger.info(f"Dukascopy: Downloading {url} (Attempt {attempt+1}/{max_retries})...")
                with urllib.request.urlopen(req, timeout=10) as response:
                    content = response.read()
                    
                with open(file_path, "wb") as f:
                    f.write(content)
                logger.info(f"Dukascopy: Saved {file_path}")
                return file_path
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # Some weekend days or holidays naturally do not have data
                    logger.warning(f"Dukascopy: No data available for {date_val.isoformat()} (404 Not Found).")
                    return ""
                logger.error(f"Dukascopy: HTTP Error {e.code} on {date_val.isoformat()}: {e}")
                if e.code in [500, 502, 503, 504]:
                    logger.warning("Dukascopy: Server returned server error. Setting server_offline = True.")
                    self.server_offline = True
                    return ""
            except Exception as e:
                logger.error(f"Dukascopy: Network error on {date_val.isoformat()}: {e}")
                logger.warning("Dukascopy: Network/timeout error. Setting server_offline = True.")
                self.server_offline = True
                return ""
                
            time.sleep(delay)
            delay *= 2
            
        return ""

    def parse_bi5(self, file_path: str, date_val: datetime.date) -> pd.DataFrame:
        """
        Decompresses and unpacks a BID_candles_min_1.bi5 file.
        Utilizes self-calibration to automatically resolve OHLC column order and price scaling.
        """
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            return pd.DataFrame()
            
        try:
            with open(file_path, "rb") as f:
                compressed_data = f.read()
            decompressed = lzma.decompress(compressed_data)
        except Exception as e:
            logger.error(f"Dukascopy: Failed to decompress {file_path}: {e}")
            return pd.DataFrame()
            
        fmt = ">IIIIIf"  # Big-endian: 5 unsigned ints, 1 float (24 bytes)
        record_size = struct.calcsize(fmt)
        n_records = len(decompressed) // record_size
        
        if n_records == 0:
            return pd.DataFrame()
            
        records = []
        for i in range(n_records):
            chunk = decompressed[i * record_size : (i + 1) * record_size]
            records.append(struct.unpack(fmt, chunk))
            
        # Parse into a numpy array for vector operations
        arr = np.array(records)
        time_offsets = arr[:, 0]
        v_raw = arr[:, 5]
        
        # For XAUUSD, Dukascopy uses 1000.0 as price scaling factor
        scaler = 1000.0
        if arr[0, 1] > 10000000:
            scaler = 100000.0
        elif arr[0, 1] > 100000:
            scaler = 1000.0
        elif arr[0, 1] > 10000:
            scaler = 100.0
            
        # Self-calibrate column layout order
        # Option A: Open, Close, Low, High (indexes 1, 2, 3, 4)
        # Option B: Open, High, Low, Close (indexes 1, 4, 3, 2)
        # We check which layout satisfies: High >= max(Open, Close) and Low <= min(Open, Close)
        opt_A_high = arr[:, 4] / scaler
        opt_A_low = arr[:, 3] / scaler
        opt_A_open = arr[:, 1] / scaler
        opt_A_close = arr[:, 2] / scaler
        
        valid_A = np.all(opt_A_high >= np.maximum(opt_A_open, opt_A_close)) and np.all(opt_A_low <= np.minimum(opt_A_open, opt_A_close))
        
        if valid_A:
            o = opt_A_open
            h = opt_A_high
            l = opt_A_low
            c = opt_A_close
        else:
            # Fallback to Option B
            o = arr[:, 1] / scaler
            h = arr[:, 2] / scaler
            l = arr[:, 3] / scaler
            c = arr[:, 4] / scaler
            
        # Reconstruct absolute UTC datetime
        base_dt = datetime.datetime(date_val.year, date_val.month, date_val.day, tzinfo=datetime.timezone.utc)
        timestamps = [base_dt + datetime.timedelta(seconds=int(offset)) for offset in time_offsets]
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v_raw
        })
        
        return df

    def download_range(self, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """Downloads and aggregates a complete date range using threadpool concurrency."""
        from concurrent.futures import ThreadPoolExecutor
        
        curr = start_date
        dates = []
        while curr <= end_date:
            dates.append(curr)
            curr += datetime.timedelta(days=1)
            
        logger.info(f"Dukascopy: Concurrently downloading {len(dates)} days...")
        
        # 1. Download in parallel using thread pool
        with ThreadPoolExecutor(max_workers=16) as executor:
            file_paths = list(executor.map(lambda d: (d, self.download_day(d)), dates))
            
        # 2. Parse sequentially (decompression and mapping)
        all_dfs = []
        for d, path in file_paths:
            if path:
                df_day = self.parse_bi5(path, d)
                if not df_day.empty:
                    all_dfs.append(df_day)
                    
        if not all_dfs:
            logger.warning(f"Dukascopy: No data successfully downloaded in range {start_date} to {end_date}")
            return pd.DataFrame()
            
        return pd.concat(all_dfs, ignore_index=True)
