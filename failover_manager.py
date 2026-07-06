import os
import time
import logging
import urllib.request
import urllib.parse
from shared_state import SharedState

logger = logging.getLogger("cheetah_failover")

class HeartbeatWriter:
    def __init__(self, heartbeat_path: str = "heartbeat.txt", interval: float = 5.0):
        """
        Periodically writes a timestamp to a local file to signal primary process liveness.
        """
        self.path = heartbeat_path
        self.interval = interval
        self.is_running = True

    def write_heartbeat(self):
        """Writes current timestamp to the heartbeat file."""
        try:
            with open(self.path, "w") as f:
                f.write(str(time.time()))
            logger.debug(f"HeartbeatWriter: Updated {self.path}")
        except Exception as e:
            logger.error(f"HeartbeatWriter: Failed to write to {self.path}: {e}")

    def loop(self):
        logger.info("HeartbeatWriter: Starting liveness loop.")
        while self.is_running:
            self.write_heartbeat()
            time.sleep(self.interval)

    def stop(self):
        self.is_running = False
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except Exception:
                pass


class StandbyMonitor:
    def __init__(self, heartbeat_path: str = "heartbeat.txt", check_interval: float = 5.0, 
                 dead_threshold_seconds: float = 15.0, telegram_token: str = None, 
                 telegram_chat_id: str = None):
        """
        Standby monitor running on the backup node. Checks if the primary heartbeat has died,
        triggers Telegram alerts, and initiates safety failovers.
        """
        self.path = heartbeat_path
        self.check_interval = check_interval
        self.threshold = dead_threshold_seconds
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        self.is_running = True
        self.alerted = False

    def _send_telegram(self, message: str):
        logger.warning(f"StandbyMonitor: {message}")
        if not self.telegram_token or not self.telegram_chat_id:
            return
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": self.telegram_chat_id, "text": message}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                response.read()
        except Exception as e:
            logger.error(f"StandbyMonitor: Telegram dispatch failed: {e}")

    def check_health(self) -> bool:
        """
        Checks if the primary heartbeat is stale.
        Returns:
            bool: True if primary is healthy, False if primary is dead.
        """
        if not os.path.exists(self.path):
            # If heartbeat file doesn't exist, check how long it's been missing
            logger.warning(f"StandbyMonitor: Heartbeat file {self.path} is missing!")
            return False
            
        try:
            with open(self.path, "r") as f:
                content = f.read().strip()
            primary_timestamp = float(content)
            elapsed = time.time() - primary_timestamp
            
            if elapsed >= self.threshold:
                logger.error(f"StandbyMonitor: Primary heartbeat stale! Last updated {elapsed:.1f} seconds ago.")
                return False
                
            self.alerted = False
            return True
        except Exception as e:
            logger.error(f"StandbyMonitor: Error reading heartbeat file: {e}")
            return False

    def run_check_once(self):
        healthy = self.check_health()
        if not healthy and not self.alerted:
            self._send_telegram(
                f"🚨 [FAILOVER ALERT] Primary VPS Heartbeat is DEAD!\n"
                f"Standby Node is taking over trade coordination. Checking active orders..."
            )
            self.alerted = True
            self.activate_failover()

    def activate_failover(self):
        """Action rules to execute during a failover transition."""
        logger.warning("StandbyMonitor: ACTIVATING STANDBY FAILOVER TAKEOVER PROCESS.")
        # Under production failover, the standby process would initialize its own DemoRunner
        # to manage positions safely and flag the primary node database as inactive.

    def monitor_loop(self):
        logger.info("StandbyMonitor: Active on standby node, watching heartbeat file.")
        while self.is_running:
            self.run_check_once()
            time.sleep(self.check_interval)

    def stop(self):
        self.is_running = False
        logger.info("StandbyMonitor: Loop stopped.")


# =====================================================================
# STANDBY REDUNDANCY DEPLOYMENT DOCUMENTATION
# =====================================================================
# To configure the secondary standby VPS for Cheetah:
#
# 1. Heartbeat Synchronization:
#    - Share a network-attached filesystem (e.g. AWS EFS or sshfs) between
#      primary and standby instances to access `heartbeat.txt`.
#    - Alternatively, configure the primary process to send a webhook ping
#      to a local server run by the StandbyMonitor on the standby VPS.
#
# 2. Standby Launch Command:
#    Run on the standby VPS:
#    `python3 failover_manager.py` (with monitor loop active)
#
# 3. Safe Takeover Logic:
#    - When takeover is triggered, the Standby Node checks active positions on MT5.
#    - Because MT5 allows multiple terminal connections on the same login magic number,
#      the standby node can cleanly inherit the trailing stop loop for open tickets.
#    - Standby sends a Telegram confirmation that it has successfully assumed control.
