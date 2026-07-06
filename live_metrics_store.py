import os
import sqlite3
import threading
import logging
import pandas as pd
import time
import datetime

logger = logging.getLogger(__name__)

class LiveMetricsStore:
    def __init__(self, db_path: str = "shared_state.db"):
        """
        Manages trade metrics persistence in SQLite.
        Used to feed statistical drift tests and the frontend dashboard.
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS live_trades (
                    ticket INTEGER PRIMARY KEY,
                    entry_time TEXT,
                    exit_time TEXT,
                    type TEXT,
                    volume REAL,
                    open_price REAL,
                    close_price REAL,
                    confidence REAL,
                    regime TEXT,
                    pnl REAL,
                    r_multiple REAL,
                    hold_time REAL
                )
            """)
            conn.commit()
            conn.close()

    def add_trade(self, ticket: int, entry_time: str, trade_type: str, volume: float, 
                  open_price: float, confidence: float, regime: str):
        """Records initial trade entry context."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO live_trades (ticket, entry_time, type, volume, open_price, confidence, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticket) DO UPDATE SET
                    entry_time=excluded.entry_time, type=excluded.type, volume=excluded.volume,
                    open_price=excluded.open_price, confidence=excluded.confidence, regime=excluded.regime
            """, (ticket, entry_time, trade_type, volume, open_price, confidence, regime))
            conn.commit()
            conn.close()
            logger.info(f"LiveMetricsStore: Registered trade entry. Ticket: {ticket}")

    def close_trade(self, ticket: int, exit_time: str, close_price: float, pnl: float, r_multiple: float):
        """Updates trade parameters at close, calculating hold time."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Fetch entry_time to calculate hold_time
            cursor.execute("SELECT entry_time FROM live_trades WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            
            hold_time = 0.0
            if row:
                try:
                    entry_dt = datetime.datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                    exit_dt = datetime.datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                    hold_time = (exit_dt - entry_dt).total_seconds()
                except Exception as e:
                    logger.warning(f"LiveMetricsStore: Error parsing times for hold calculation: {e}")
                    
            cursor.execute("""
                UPDATE live_trades
                SET exit_time = ?, close_price = ?, pnl = ?, r_multiple = ?, hold_time = ?
                WHERE ticket = ?
            """, (exit_time, close_price, pnl, r_multiple, hold_time, ticket))
            conn.commit()
            conn.close()
            logger.info(f"LiveMetricsStore: Registered trade close. Ticket: {ticket} | PnL: ${pnl:.2f} | R: {r_multiple:.2f} | Hold: {hold_time:.1f}s")

    def get_all_trades(self) -> pd.DataFrame:
        """Retrieves a DataFrame of all trades."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM live_trades ORDER BY ticket ASC", conn)
            conn.close()
        return df

    def get_completed_trades(self) -> pd.DataFrame:
        """Retrieves only completed trades (where exit_time is not null)."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM live_trades WHERE exit_time IS NOT NULL ORDER BY exit_time ASC", conn)
            conn.close()
        return df
        
    def clear(self):
        """Deletes all persistent trade records."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM live_trades")
            conn.commit()
            conn.close()
            logger.info("LiveMetricsStore: Database cleared.")
