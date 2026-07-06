import os
import sqlite3
import threading
import logging
import json
import time

logger = logging.getLogger(__name__)

class SharedState:
    def __init__(self, db_path: str = "shared_state.db"):
        """
        Persistent SQLite-backed key-value store to safely share state
        between the slow loop and fast loop. Thread-safe using threading.Lock.
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.commit()
            conn.close()

    def get(self, key: str, default=None):
        """Retrieve a value by key. Auto-decodes JSON to preserve types."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM state WHERE key = ?", (key,))
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return default
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]

    def set(self, key: str, value):
        """Saves a value by key. Auto-serializes to JSON."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            val_str = json.dumps(value)
            cursor.execute("""
                INSERT INTO state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, val_str))
            conn.commit()
            conn.close()

    # Convenience Properties
    @property
    def current_regime(self) -> str:
        return self.get("current_regime", "ranging")

    @current_regime.setter
    def current_regime(self, val: str):
        self.set("current_regime", val)

    @property
    def current_bias(self) -> str:
        # Returns: 'no_trade', 'long', or 'short' (or numeric direction)
        return self.get("current_bias", "no_trade")

    @current_bias.setter
    def current_bias(self, val: str):
        self.set("current_bias", val)

    @property
    def reversal_armed(self) -> bool:
        return self.get("reversal_armed", False)

    @reversal_armed.setter
    def reversal_armed(self, val: bool):
        self.set("reversal_armed", val)

    @property
    def last_updated_slow(self) -> float:
        return self.get("last_updated_slow", 0.0)

    @last_updated_slow.setter
    def last_updated_slow(self, val: float):
        self.set("last_updated_slow", val)

    @property
    def last_updated_fast(self) -> float:
        return self.get("last_updated_fast", 0.0)

    @last_updated_fast.setter
    def last_updated_fast(self, val: float):
        self.set("last_updated_fast", val)

    def clear(self):
        """Clears all stored states."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM state")
            conn.commit()
            conn.close()
            logger.info("SharedState database cleared.")
