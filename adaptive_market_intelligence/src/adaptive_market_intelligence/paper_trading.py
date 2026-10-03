from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .database import DatabaseManager
from .execution import ExecutionPlanner
from .models import DecisionType, MarketTick, OrderIntent, RiskDecisionType


@dataclass
class PaperFill:
    market: str
    side: str
    quantity: int
    fill_price: float
    timestamp: datetime
    order_id: str


@dataclass
class PaperTradingSummary:
    total_orders: int
    filled_orders: int
    rejected_orders: int
    realized_pnl: float
    fills: List[PaperFill] = field(default_factory=list)


class PaperTradingEngine:
    def __init__(self, database: Optional[DatabaseManager] = None):
        self.database = database or DatabaseManager()
        self.execution_planner = ExecutionPlanner()
        self.fills: List[PaperFill] = []

    def run(self, decisions: List[dict], market_prices: Dict[str, float]) -> PaperTradingSummary:
        total_orders = 0
        filled_orders = 0
        rejected_orders = 0
        realized_pnl = 0.0

        for batch in decisions:
            for market_name, payload in batch.items():
                signal = payload["signal"]
                risk = payload["risk"]
                opportunity = payload["opportunity"]
                decision_label = signal.direction.value if signal.direction else "NO_TRADE"
                if risk.decision == RiskDecisionType.APPROVE and opportunity.decision in {DecisionType.BUY_CE, DecisionType.BUY_PE}:
                    order = self.execution_planner.build_order(
                        market=market_name,
                        opportunity=opportunity,
                        risk=risk,
                        contract="CE" if signal.direction == DecisionType.BUY_CE else "PE",
                    )
                    total_orders += 1
                    fill_price = float(market_prices.get(market_name, 0.0) * 0.012) or 10.0
                    fill = PaperFill(
                        market=market_name,
                        side=decision_label,
                        quantity=order.quantity,
                        fill_price=fill_price,
                        timestamp=datetime.utcnow(),
                        order_id=order.idempotency_key,
                    )
                    self.fills.append(fill)
                    filled_orders += 1
                    realized_pnl += fill.quantity * fill.fill_price * 0.05
                    self.database.record_trade(market_name, decision_label, order.quantity, fill.fill_price, "OPEN")
                    self.database.record_decision(market_name, decision_label, risk.decision.value, signal.confidence, signal.reason_codes[0] if signal.reason_codes else None)
                else:
                    rejected_orders += 1
                    self.database.record_decision(market_name, decision_label, risk.decision.value, signal.confidence, signal.reason_codes[0] if signal.reason_codes else None)

        return PaperTradingSummary(
            total_orders=total_orders,
            filled_orders=filled_orders,
            rejected_orders=rejected_orders,
            realized_pnl=realized_pnl,
            fills=self.fills,
        )
