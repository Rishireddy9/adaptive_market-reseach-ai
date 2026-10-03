from __future__ import annotations

from typing import Dict

from .config import DEFAULT_CONFIG, SystemConfig
from .models import CapitalAllocation, DecisionType, OpportunityDecision, RiskDecision, RiskDecisionType, SignalCandidate


class SafetyKernel:
    def __init__(self, config: SystemConfig = DEFAULT_CONFIG):
        self.config = config

    def evaluate(self, signal: SignalCandidate, opportunity: OpportunityDecision) -> RiskDecision:
        hard_limits = {
            "max_daily_loss": self.config.max_daily_loss > 0,
            "market_budget_ok": True,
            "quality_gate": signal.confidence >= 0.5,
            "decision_valid": opportunity.decision != DecisionType.NO_TRADE,
        }

        exposures = {
            signal.market: float(self.config.capital_budget * 0.10),
            "portfolio": float(self.config.capital_budget * 0.10),
        }

        if signal.direction == DecisionType.NO_TRADE or opportunity.decision == DecisionType.NO_TRADE:
            return RiskDecision(
                market=signal.market,
                decision=RiskDecisionType.REJECT,
                approved_quantity=0,
                estimated_loss=0.0,
                exposures=exposures,
                hard_limit_results=hard_limits,
                reason_codes=["No trade approved by strategy or opportunity engine"],
            )

        if not hard_limits["quality_gate"]:
            return RiskDecision(
                market=signal.market,
                decision=RiskDecisionType.REJECT,
                approved_quantity=0,
                estimated_loss=0.0,
                exposures=exposures,
                hard_limit_results=hard_limits,
                reason_codes=["signal_confidence_below_threshold"],
            )

        approved_quantity = max(1, int(self.config.capital_budget * 0.05 / max(opportunity.expected_move, 1.0)))
        estimated_loss = max(0.0, opportunity.expected_move * 0.8)

        return RiskDecision(
            market=signal.market,
            decision=RiskDecisionType.APPROVE,
            approved_quantity=approved_quantity,
            estimated_loss=estimated_loss,
            exposures=exposures,
            hard_limit_results=hard_limits,
            reason_codes=["risk_gate_passed"],
        )

    def capital_allocation(self, market: str, desired_trade_budget: float) -> CapitalAllocation:
        market_budget = self.config.capital_budget * self.config.market_caps.get(market, 0.2)
        return CapitalAllocation(
            portfolio_budget=self.config.capital_budget,
            market_budget=market_budget,
            strategy_budget=min(desired_trade_budget, market_budget),
            requested_trade_budget=desired_trade_budget,
            correlation_adjustment=1.0,
        )
