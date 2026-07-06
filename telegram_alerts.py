import time
import logging
import requests
import yaml

logger = logging.getLogger(__name__)

class TelegramAlerts:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.tel_cfg = self.config.get("telegram", {})
        self.bot_token = self.tel_cfg.get("bot_token")
        self.chat_id = self.tel_cfg.get("chat_id")
        
    def send_alert(self, signal: dict) -> bool:
        """
        Formats and sends a trading signal alert to Telegram.
        Retries up to 3 times with exponential backoff on failure.
        """
        if not self.bot_token or not self.chat_id or "YOUR_TELEGRAM" in str(self.bot_token):
            logger.warning("Telegram is not configured. Skipping signal alert transmission.")
            return False
            
        direction = signal.get("direction")
        price = signal.get("price")
        timestamp = signal.get("timestamp")
        trigger_feats = signal.get("triggering_features", {})
        
        # Format message as Markdown
        msg = (
            f"🚨 *Cheetah Trading Bot Signal* 🚨\n\n"
            f"*Asset:* XAUUSD\n"
            f"*Direction:* {direction}\n"
            f"*Price:* {price:.2f}\n"
            f"*Time (UTC):* {timestamp}\n\n"
            f"*Triggering Features:*\n"
        )
        for k, v in trigger_feats.items():
            if isinstance(v, float):
                msg += f"• `{k}`: {v:.4f}\n"
            else:
                msg += f"• `{k}`: {v}\n"
                
        return self.send_message(msg)

    def send_message(self, message: str) -> bool:
        """Sends a raw text message to Telegram with retry logic."""
        if not self.bot_token or not self.chat_id or "YOUR_TELEGRAM" in str(self.bot_token):
            logger.warning("Telegram credentials not set. Message not sent.")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        max_retries = 3
        delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info("Telegram message delivered successfully.")
                    return True
                else:
                    logger.error(
                        f"Telegram API returned error status {response.status_code}. "
                        f"Response: {response.text}"
                    )
            except Exception as e:
                logger.error(f"Exception during Telegram API request (attempt {attempt+1}/{max_retries}): {e}")
                
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                
        logger.error("Failed to send message to Telegram after multiple attempts.")
        return False
