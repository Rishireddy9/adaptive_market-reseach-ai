from __future__ import annotations

from uuid import uuid4

from .models import DecisionType, OpportunityDecision, OrderIntent, RiskDecision, RiskDecisionType


class ExecutionPlanner:
    def build_order(self, market: str, opportunity: OpportunityDecision, risk: RiskDecision, contract: str = "CE") -> OrderIntent:
        if opportunity.decision == DecisionType.NO_TRADE:
            raise ValueError("No executable order for a rejected or halted opportunity")
        if risk.decision in {RiskDecisionType.REJECT, RiskDecisionType.HALT}:
            raise ValueError("No executable order for a rejected or halted opportunity")

        qty = max(1, int(risk.approved_quantity * 0.8))
        price = max(10.0, abs(opportunity.expected_move) * 0.9)
        return OrderIntent(
            market=market,
            contract=f"{market}-{contract}",
            quantity=qty,
            order_type="LIMIT",
            price=price,
            decision_ttl_seconds=60,
            idempotency_key=f"{market}-{uuid4().hex[:10]}",
        )
