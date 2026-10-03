from __future__ import annotations

from datetime import datetime
from typing import Iterable, List

from .models import DataQualityState, MarketTick


class DataQualityEngine:
    def assess(self, tick: MarketTick, previous_tick: MarketTick | None = None) -> DataQualityState:
        reasons: List[str] = []
        quality_score = tick.quality_score

        if previous_tick is not None:
            delta = abs(tick.price - previous_tick.price)
            if tick.price <= 0:
                quality_score *= 0.4
                reasons.append("non_positive_price")
            if delta / max(previous_tick.price, 1.0) > 0.045:
                quality_score *= 0.7
                reasons.append("price_jump")
            if tick.bid > 0 and tick.ask > 0 and tick.bid >= tick.ask:
                quality_score *= 0.6
                reasons.append("crossed_book")

        if tick.quality_score < 0.75:
            quality_score *= 0.7
            reasons.append("provider_quality_low")

        freshness_ok = (datetime.utcnow() - tick.as_of).total_seconds() < 30
        complete = tick.price > 0 and tick.volume >= 0
        if not complete:
            reasons.append("incomplete_tick")
        if not freshness_ok:
            reasons.append("stale_market_data")

        return DataQualityState(
            market=tick.market.value,
            quality_score=max(0.0, min(1.0, quality_score)),
            complete=complete,
            freshness_ok=freshness_ok,
            reasons=reasons,
        )

    def assess_sequence(self, ticks: Iterable[MarketTick]) -> List[DataQualityState]:
        states: List[DataQualityState] = []
        previous: MarketTick | None = None
        for tick in ticks:
            states.append(self.assess(tick, previous))
            previous = tick
        return states
