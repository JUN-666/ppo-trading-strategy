
"""Pre-trade and intraday risk controls for HFT reinforcement learning."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque


@dataclass
class RiskState:
    position: int = 0
    realized_pnl_ntd: float = 0.0
    peak_assets_ntd: float = 0.0
    current_assets_ntd: float = 0.0


class RiskManager:
    """Lightweight risk gate used before an RL action is accepted.

    Actions follow the environment convention: 0 hold, 1 buy, 2 sell.
    """

    def __init__(self, max_position: int, max_daily_loss_ntd: float, kill_switch_drawdown_pct: float, max_order_rate_per_minute: int):
        self.max_position = max_position
        self.max_daily_loss_ntd = max_daily_loss_ntd
        self.kill_switch_drawdown_pct = kill_switch_drawdown_pct
        self.max_order_rate_per_minute = max_order_rate_per_minute
        self.order_timestamps: Deque[float] = deque()

    def filter_action(self, action: int, state: RiskState, timestamp_seconds: float | None = None) -> int:
        if action == 0:
            return 0
        if state.realized_pnl_ntd <= -abs(self.max_daily_loss_ntd):
            return 0
        if state.peak_assets_ntd > 0:
            drawdown = 1.0 - (state.current_assets_ntd / state.peak_assets_ntd)
            if drawdown >= self.kill_switch_drawdown_pct:
                return 0
        projected_position = state.position + (1 if action == 1 else -1)
        if abs(projected_position) > self.max_position:
            return 0
        if timestamp_seconds is not None and self._rate_limited(timestamp_seconds):
            return 0
        return action

    def record_order(self, timestamp_seconds: float) -> None:
        self.order_timestamps.append(timestamp_seconds)
        self._prune(timestamp_seconds)

    def _rate_limited(self, now: float) -> bool:
        self._prune(now)
        return len(self.order_timestamps) >= self.max_order_rate_per_minute

    def _prune(self, now: float) -> None:
        while self.order_timestamps and now - self.order_timestamps[0] > 60.0:
            self.order_timestamps.popleft()
