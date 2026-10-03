from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict, List

from .models import Market, MarketStateSnapshot, MarketTick


class RealTimeUniverseScanner:
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.history: Dict[str, Deque[float]] = {market.value: deque(maxlen=lookback) for market in Market}

    def scan(self, tick: MarketTick) -> MarketStateSnapshot:
        history = self.history[tick.market.value]
        history.append(tick.price)
        prices = list(history)

        ema_fast = self._ema(prices, 5)
        ema_slow = self._ema(prices, 12)
        rsi = self._rsi(prices)
        trend_bias = ema_fast - ema_slow

        quality_score = tick.quality_score
        data_complete = tick.price > 0 and tick.volume >= 0
        reason_codes: List[str] = []
        if not data_complete:
            reason_codes.append("data_incomplete")
        if quality_score < 0.75:
            reason_codes.append("low_quality")

        return MarketStateSnapshot(
            market=tick.market.value,
            as_of=tick.as_of,
            price=tick.price,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            rsi=rsi,
            trend_bias=trend_bias,
            quality_score=quality_score,
            data_complete=data_complete,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        if not values:
            return 0.0
        alpha = 2 / (period + 1)
        ema = values[0]
        for value in values[1:]:
            ema = (alpha * value) + ((1 - alpha) * ema)
        return ema

    @staticmethod
    def _rsi(values: List[float], period: int = 14) -> float:
        if len(values) < 2:
            return 50.0
        deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
        if len(deltas) < period:
            avg_gain = sum(max(d, 0) for d in deltas) / len(deltas) if deltas else 0.0
            avg_loss = sum(max(-d, 0) for d in deltas) / len(deltas) if deltas else 0.0
        else:
            recent = deltas[-period:]
            avg_gain = sum(max(d, 0) for d in recent) / len(recent)
            avg_loss = sum(max(-d, 0) for d in recent) / len(recent)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
