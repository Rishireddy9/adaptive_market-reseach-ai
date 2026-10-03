from __future__ import annotations

from .models import DecisionType, MarketStateSnapshot, SignalCandidate


class StrategyEngine:
    def __init__(self, strategy_name: str = "EMA_RSI_V1"):
        self.strategy_name = strategy_name

    def evaluate(self, snapshot: MarketStateSnapshot) -> SignalCandidate:
        reasons: list[str] = []

        if snapshot.quality_score < 0.7:
            return SignalCandidate(
                strategy_name=self.strategy_name,
                market=snapshot.market,
                direction=DecisionType.NO_TRADE,
                confidence=0.0,
                expected_move=0.0,
                horizon_minutes=5,
                reason_codes=["low_data_confidence"],
            )

        if snapshot.rsi >= 70 and snapshot.trend_bias > 0:
            direction = DecisionType.BUY_CE
            expected_move = max(snapshot.price * 0.0025, 12.0)
            confidence = 0.72
            reasons.append("uptrend_with_overbought_context")
        elif snapshot.rsi <= 30 and snapshot.trend_bias < 0:
            direction = DecisionType.BUY_PE
            expected_move = max(snapshot.price * 0.0025, 12.0)
            confidence = 0.72
            reasons.append("downtrend_with_oversold_context")
        else:
            direction = DecisionType.WAIT
            expected_move = 0.0
            confidence = 0.45
            reasons.append("trend_not_ready")

        if snapshot.rsi < 35 or snapshot.rsi > 65:
            reasons.append("momentum_filter")

        return SignalCandidate(
            strategy_name=self.strategy_name,
            market=snapshot.market,
            direction=direction,
            confidence=confidence,
            expected_move=expected_move,
            horizon_minutes=5,
            reason_codes=reasons,
        )
