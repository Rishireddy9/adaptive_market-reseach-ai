from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Market(str, Enum):
    NIFTY = "NIFTY"
    BANKNIFTY = "BANKNIFTY"
    SENSEX = "SENSEX"


class Environment(str, Enum):
    DEV = "dev"
    BACKTEST = "backtest"
    REPLAY = "replay"
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


class DecisionType(str, Enum):
    BUY_CE = "BUY_CE"
    BUY_PE = "BUY_PE"
    WAIT = "WAIT"
    NO_TRADE = "NO_TRADE"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    UNCERTAIN = "UNCERTAIN"


class RiskDecisionType(str, Enum):
    APPROVE = "APPROVE"
    RESIZE = "RESIZE"
    REJECT = "REJECT"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    HALT = "HALT"


@dataclass(frozen=True)
class MarketTick:
    market: Market
    as_of: datetime
    price: float
    volume: int
    oi: int = 0
    iv: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    quality_score: float = 1.0
    synthetic: bool = True


@dataclass
class DataQualityState:
    market: str
    quality_score: float
    complete: bool
    freshness_ok: bool
    reasons: List[str] = field(default_factory=list)


@dataclass
class MarketStateSnapshot:
    market: str
    as_of: datetime
    price: float
    ema_fast: float
    ema_slow: float
    rsi: float
    trend_bias: float
    quality_score: float
    data_complete: bool
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class SignalCandidate:
    strategy_name: str
    market: str
    direction: DecisionType
    confidence: float
    expected_move: float
    horizon_minutes: int
    reason_codes: List[str]


@dataclass
class OptionCandidate:
    contract: str
    option_type: str
    strike: float
    expiry: str
    iv: float
    theta: float
    liquidity: float
    spread: float
    estimated_cost: float
    expected_ev: float


@dataclass
class OpportunityDecision:
    market: str
    decision: DecisionType
    expected_move: float
    remaining_move: float
    stage: str
    ev_before_cost: float
    ev_after_cost: float
    reason_codes: List[str]


@dataclass
class CapitalAllocation:
    portfolio_budget: float
    market_budget: float
    strategy_budget: float
    requested_trade_budget: float
    correlation_adjustment: float


@dataclass
class RiskDecision:
    market: str
    decision: RiskDecisionType
    approved_quantity: int
    estimated_loss: float
    exposures: Dict[str, float]
    hard_limit_results: Dict[str, bool]
    risk_config_version: str = "v1"
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class OrderIntent:
    market: str
    contract: str
    quantity: int
    order_type: str
    price: float
    decision_ttl_seconds: int = 60
    idempotency_key: str = ""


@dataclass
class DecisionRecord:
    market: str
    signal: SignalCandidate
    opportunity: OpportunityDecision
    risk: RiskDecision
    order: Optional[OrderIntent]
    reason_codes: List[str]


@dataclass
class ExecutionResult:
    accepted: bool
    order: Optional[OrderIntent]
    status: str
    message: str
