import time
import logging
import urllib.request
import urllib.parse
from shared_state import SharedState

logger = logging.getLogger("cheetah_watchdog")

class ConnectionWatchdog:
    def __init__(self, connector, shared_state: SharedState, execution_engine, 
                 check_interval: float = 5.0, telegram_token: str = None, telegram_chat_id: str = None):
        """
        Non-blocking connection watchdog that periodically evaluates MT5 terminal liveness,
        enforces entry halts upon connection loss, and retries reconnection using non-blocking backoff.
        """
        self.connector = connector
        self.state = shared_state
        self.execution = execution_engine
        self.check_interval = check_interval
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        
        self.is_running = True
        self.alerted_outage = False
        
        # Non-blocking backoff variables
        self.backoff_delay = 2.0
        self.next_reconnect_time = 0.0

    def _send_telegram(self, message: str):
        """Dispatches Telegram notifications to the operator."""
        logger.warning(f"ConnectionWatchdog Alert: {message}")
        if not self.telegram_token or not self.telegram_chat_id:
            return
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": self.telegram_chat_id, "text": message}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as response:
                response.read()
            logger.info("ConnectionWatchdog: Telegram notification dispatched.")
        except Exception as e:
            logger.error(f"ConnectionWatchdog: Telegram alert send failed: {e}")

    def trigger_outage_alerts(self):
        """Queries active positions and sends safety alerts."""
        if self.alerted_outage:
            return
            
        active_positions = []
        if self.execution.mock:
            active_positions = self.execution.mock_positions
        else:
            try:
                import MetaTrader5 as mt5
                pos_info = mt5.positions_get()
                if pos_info:
                    for p in pos_info:
                        active_positions.append({
                            "ticket": p.ticket,
                            "type": "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                            "volume": p.volume,
                            "open_price": p.price_open
                        })
            except Exception as e:
                logger.error(f"ConnectionWatchdog: Error collecting active live positions: {e}")
                
        if active_positions:
            msg = f"🚨 [CHEETAH ALERT] MT5 Disconnected while positions are ACTIVE!\n"
            for pos in active_positions:
                msg += f"- Ticket {pos['ticket']}: {pos['type']} {pos['volume']} lots open at {pos['open_price']:.2f}\n"
            msg += "Please monitor broker trades manually until connection is restored."
            self._send_telegram(msg)
        else:
            self._send_telegram("⚠️ [CHEETAH WARNING] MT5 Disconnected. Entry signals are paused.")
            
        self.alerted_outage = True

    def run_check(self) -> bool:
        """
        Performs a non-blocking check on connection status.
        Triggers halts and schedule retries if disconnected.
        """
        connected = self.connector.connected
        
        if not connected:
            # 1. Set halt state if not set
            if self.state.get("halt_reason") != "MT5 Connection Lost":
                self.state.set("is_halted", True)
                self.state.set("halt_reason", "MT5 Connection Lost")
                self.trigger_outage_alerts()
                
            # 2. Check if the next reconnect time is reached
            now = time.time()
            if now >= self.next_reconnect_time:
                logger.info("ConnectionWatchdog: Executing non-blocking reconnection check...")
                try:
                    self.connector.connect()
                except Exception as e:
                    logger.error(f"ConnectionWatchdog: Connect failed: {e}")
                    
                if self.connector.connected:
                    logger.info("ConnectionWatchdog: Connection restored successfully.")
                    self.state.set("is_halted", False)
                    self.state.set("halt_reason", "")
                    self.backoff_delay = 2.0
                    self.alerted_outage = False
                    self._send_telegram("🟢 [CHEETAH INFO] MT5 Connection restored. Trading resumed.")
                else:
                    # Update backoff parameters
                    self.next_reconnect_time = now + self.backoff_delay
                    logger.info(f"ConnectionWatchdog: Reconnection failed. Next retry in {self.backoff_delay}s.")
                    self.backoff_delay = min(self.backoff_delay * 2, 60.0)
            return False
            
        # Connection is healthy
        if self.state.get("halt_reason") == "MT5 Connection Lost":
            self.state.set("is_halted", False)
            self.state.set("halt_reason", "")
            self.backoff_delay = 2.0
            self.alerted_outage = False
            logger.info("ConnectionWatchdog: Connection restored. Trading resumed.")
            
        return True

    def monitor_loop(self):
        """Schedulers run_check on interval."""
        logger.info("ConnectionWatchdog: Starting non-blocking watchdog check loop.")
        while self.is_running:
            self.run_check()
            time.sleep(self.check_interval)

    def stop(self):
        self.is_running = False
        logger.info("ConnectionWatchdog: Monitor loop halted.")
