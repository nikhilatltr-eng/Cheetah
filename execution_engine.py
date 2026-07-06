import logging
import time
import random
import pandas as pd
import numpy as np

# Try to import MT5 safely
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

logger = logging.getLogger("cheetah_execution")

class ExecutionEngine:
    def __init__(self, connector, symbol: str = "XAUUSD", magic_number: int = 888888, 
                 initial_balance: float = 10000.0, mock: bool = False, shared_state=None):
        """
        Manages trade executions via MT5.
        Includes a full mock-mode paper-trading simulator to track active trades, trailing stops,
        partial closes, and balance tracking on macOS.
        """
        self.connector = connector
        self.symbol = symbol
        self.magic_number = magic_number
        self.mock = mock or (mt5 is None or connector.mock)
        self.state = shared_state
        
        # Mock account parameters
        self.balance = initial_balance
        self.equity = initial_balance
        self.realized_pnl = 0.0
        self.mock_positions = []  # List of dicts representing open positions
        self.trade_history = []   # Logs of closed trades
        
        if self.mock:
            logger.info(f"ExecutionEngine: Starting in MOCK Paper-Trading mode. Balance: ${self.balance:.2f}")
        else:
            logger.info("ExecutionEngine: Initializing live MT5 Order execution wrapper.")
        self._update_shared_state()

    def _update_shared_state(self):
        if hasattr(self, "state") and self.state is not None:
            self.state.set("mock_positions", self.mock_positions)
            self.state.set("trade_history", self.trade_history)
            self.state.set("balance", self.balance)
            self.state.set("equity", self.equity)
            
            # Append equity history to draw a curve
            eq_hist = self.state.get("equity_history", [])
            # Store up to 500 records
            eq_hist.append({"time": time.time(), "equity": self.equity})
            if len(eq_hist) > 500:
                eq_hist.pop(0)
            self.state.set("equity_history", eq_hist)

    def execute_order(self, action: str, volume: float, price: float, sl: float, tp: float, reason_context: str = "") -> dict:
        """
        Executes a market order:
          - action: 'trigger_buy' or 'trigger_sell'
          - volume: trade lot size
          - price: entry price
          - sl: Stop Loss level
          - tp: Take Profit level
        """
        logger.info(f"ExecutionEngine: Executing {action.upper()} | Lot: {volume} | SL: {sl:.2f} | TP: {tp:.2f} | Reason: {reason_context}")
        
        if self.mock:
            # Generate a mock trade ticket
            ticket = random.randint(100000, 999999)
            trade_type = 0 if action == "trigger_buy" else 1  # 0: BUY, 1: SELL
            
            position = {
                "ticket": ticket,
                "symbol": self.symbol,
                "type": "BUY" if trade_type == 0 else "SELL",
                "volume": volume,
                "open_price": price,
                "sl": sl,
                "tp": tp,
                "highest_price": price,
                "lowest_price": price,
                "partially_closed": False,
                "magic": self.magic_number,
                "entry_time": time.time(),
                "reason": reason_context
            }
            self.mock_positions.append(position)
            logger.info(f"[MOCK PAPER TRADE] Order opened successfully. Ticket: {ticket}")
            self._update_shared_state()
            return {"status": "success", "ticket": ticket, "position": position}
            
        else:
            # Live MT5 order execution
            if not self.connector.connected:
                logger.error("ExecutionEngine: MT5 is disconnected. Execution aborted.")
                return {"status": "error", "reason": "disconnected"}
                
            trade_type = mt5.ORDER_TYPE_BUY if action == "trigger_buy" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": trade_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": f"Cheetah: {reason_context[:20]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"ExecutionEngine: Order Send Failed! Retcode={result.retcode} | Error={mt5.last_error()}")
                return {"status": "failed", "retcode": result.retcode, "reason": str(mt5.last_error())}
                
            logger.info(f"ExecutionEngine: Order Deal successfully placed. Ticket: {result.order}")
            return {"status": "success", "ticket": result.order}

    def close_position(self, ticket: int, price: float) -> float:
        """Closes a position completely, returning the realized PnL."""
        if self.mock:
            # Find mock position
            for pos in self.mock_positions:
                if pos["ticket"] == ticket:
                    # Calculate PnL: XAUUSD standard lot size is 100 ounces
                    volume = pos["volume"]
                    multiplier = 100.0
                    
                    if pos["type"] == "BUY":
                        pnl = (price - pos["open_price"]) * volume * multiplier
                    else:
                        pnl = (pos["open_price"] - price) * volume * multiplier
                        
                    self.realized_pnl += pnl
                    self.balance += pnl
                    self.mock_positions.remove(pos)
                    
                    pos["close_price"] = price
                    pos["close_time"] = time.time()
                    pos["pnl"] = pnl
                    self.trade_history.append(pos)
                    
                    logger.info(f"[MOCK PAPER TRADE] Closed Ticket {ticket} at {price:.2f}. Realized PnL: ${pnl:.2f}")
                    self._update_shared_state()
                    return pnl
            return 0.0
        else:
            # Live MT5 position close logic
            # Fetch active positions to match ticket
            pos_info = mt5.positions_get(ticket=ticket)
            if pos_info is None or len(pos_info) == 0:
                logger.error(f"ExecutionEngine: Ticket {ticket} position not found.")
                return 0.0
                
            pos = pos_info[0]
            trade_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": trade_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": "Cheetah Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"ExecutionEngine: Close Order Failed for ticket {ticket}!")
                return 0.0
                
            logger.info(f"ExecutionEngine: Live Ticket {ticket} successfully closed.")
            return float(result.profit)

    def partial_close(self, ticket: int, price: float, fraction: float = 0.5):
        """Closes a fraction of a position (standard for scalp quick targets)."""
        logger.info(f"ExecutionEngine: Requesting partial close ({fraction*100}%) for ticket {ticket}...")
        
        if self.mock:
            for pos in self.mock_positions:
                if pos["ticket"] == ticket and not pos["partially_closed"]:
                    orig_volume = pos["volume"]
                    close_vol = orig_volume * fraction
                    keep_vol = orig_volume - close_vol
                    
                    # Realize PnL for closed part
                    multiplier = 100.0
                    if pos["type"] == "BUY":
                        pnl = (price - pos["open_price"]) * close_vol * multiplier
                    else:
                        pnl = (pos["open_price"] - price) * close_vol * multiplier
                        
                    self.realized_pnl += pnl
                    self.balance += pnl
                    
                    # Update volume
                    pos["volume"] = keep_vol
                    pos["partially_closed"] = True
                    logger.info(f"[MOCK PAPER TRADE] Partially closed {close_vol:.2f} lots of ticket {ticket}. Realized PnL: ${pnl:.2f}. Remaining volume: {keep_vol:.2f}")
                    self._update_shared_state()
                    break
        else:
            # Live partial close on MT5 involves sending a DEAL order with volume = close_vol
            # referencing the open position ticket number.
            pos_info = mt5.positions_get(ticket=ticket)
            if pos_info and len(pos_info) > 0:
                pos = pos_info[0]
                close_vol = float(np.round(pos.volume * fraction, 2))
                if close_vol >= 0.01:
                    trade_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": close_vol,
                        "type": trade_type,
                        "position": ticket,
                        "price": price,
                        "deviation": 20,
                        "magic": self.magic_number,
                        "comment": "Cheetah Partial Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"ExecutionEngine: Partial close of {close_vol} lots completed for ticket {ticket}.")
                    else:
                        logger.error(f"ExecutionEngine: Live Partial Close Send Failed: {result.comment}")

    def update_positions_on_tick(self, bid: float, ask: float, atr: float = 2.0):
        """
        Updates active trailing stops and triggers stop-outs (SL) or target takes (TP) in Mock Mode.
        """
        if not self.mock:
            # In live mode, MT5 handles SL/TP stops automatically on broker servers.
            # Trailing stops must be adjusted by client commands:
            self._update_live_trailing_stops(bid, ask, atr)
            return
            
        # Mock mode processing
        active_positions = list(self.mock_positions)
        self.equity = self.balance
        
        multiplier = 100.0
        for pos in active_positions:
            price = bid if pos["type"] == "BUY" else ask
            
            # Update paper equity in real-time
            if pos["type"] == "BUY":
                pnl = (price - pos["open_price"]) * pos["volume"] * multiplier
                pos["highest_price"] = max(pos["highest_price"], price)
            else:
                pnl = (pos["open_price"] - price) * pos["volume"] * multiplier
                pos["lowest_price"] = min(pos["lowest_price"], price)
                
            self.equity += pnl
            
            # Check Stop Loss (SL) Trigger
            if pos["type"] == "BUY" and price <= pos["sl"]:
                logger.info(f"[MOCK] Stop Loss hit for BUY ticket {pos['ticket']} at {price:.2f}")
                self.close_position(pos["ticket"], pos["sl"])
                continue
            elif pos["type"] == "SELL" and price >= pos["sl"]:
                logger.info(f"[MOCK] Stop Loss hit for SELL ticket {pos['ticket']} at {price:.2f}")
                self.close_position(pos["ticket"], pos["sl"])
                continue
                
            # Check Take Profit (TP) Trigger
            if pos["type"] == "BUY" and price >= pos["tp"]:
                logger.info(f"[MOCK] Take Profit hit for BUY ticket {pos['ticket']} at {price:.2f}")
                self.close_position(pos["ticket"], pos["tp"])
                continue
            elif pos["type"] == "SELL" and price <= pos["tp"]:
                logger.info(f"[MOCK] Take Profit hit for SELL ticket {pos['ticket']} at {price:.2f}")
                self.close_position(pos["ticket"], pos["tp"])
                continue
                
            # Apply Trailing Stop rules (moves stop loss into profit as trend advances)
            # Trailing stop rule: if price moves > 1.5 * ATR in profit, lock in 0.5 * ATR profit
            if pos["type"] == "BUY":
                profit_distance = price - pos["open_price"]
                if profit_distance > 1.5 * atr:
                    new_sl = price - 1.0 * atr  # trailing stop buffer of 1 ATR
                    if new_sl > pos["sl"]:
                        pos["sl"] = new_sl
                        logger.info(f"[MOCK] Trailing Stop updated for BUY ticket {pos['ticket']} to {new_sl:.2f}")
            else:
                profit_distance = pos["open_price"] - price
                if profit_distance > 1.5 * atr:
                    new_sl = price + 1.0 * atr
                    if new_sl < pos["sl"]:
                        pos["sl"] = new_sl
                        logger.info(f"[MOCK] Trailing Stop updated for SELL ticket {pos['ticket']} to {new_sl:.2f}")
                        
            # Apply Partial Close rules: if price moves > 1.0 * ATR in profit and not partially closed yet
            if not pos["partially_closed"]:
                profit_distance = (price - pos["open_price"]) if pos["type"] == "BUY" else (pos["open_price"] - price)
                if profit_distance > 1.0 * atr:
                    logger.info(f"[MOCK] Profit Target 1 (1 ATR) reached for ticket {pos['ticket']}. Triggering partial close.")
                    self.partial_close(pos["ticket"], price, fraction=0.5)
            self._update_shared_state()

    def _update_live_trailing_stops(self, bid: float, ask: float, atr: float):
        """Adjusts open stop levels in real MT5 terminal positions."""
        if mt5 is None:
            return
        
        # Get active positions matching our magic number
        positions = mt5.positions_get(magic=self.magic_number)
        if not positions:
            return
            
        for pos in positions:
            price = bid if pos.type == mt5.POSITION_TYPE_BUY else ask
            atr_dist = atr if atr > 0 else 2.0
            
            if pos.type == mt5.POSITION_TYPE_BUY:
                profit_distance = price - pos.price_open
                if profit_distance > 1.5 * atr_dist:
                    new_sl = float(np.round(price - 1.0 * atr_dist, 2))
                    if new_sl > pos.sl:
                        # Request SL modification
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": pos.tp
                        }
                        result = mt5.order_send(request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"ExecutionEngine: Live Trailing Stop modified for Buy ticket {pos.ticket} to {new_sl}")
            else:
                # Sell Position
                profit_distance = pos.price_open - price
                if profit_distance > 1.5 * atr_dist:
                    new_sl = float(np.round(price + 1.0 * atr_dist, 2))
                    if new_sl < pos.sl or pos.sl == 0.0:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": pos.ticket,
                            "sl": new_sl,
                            "tp": pos.tp
                        }
                        result = mt5.order_send(request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"ExecutionEngine: Live Trailing Stop modified for Sell ticket {pos.ticket} to {new_sl}")
