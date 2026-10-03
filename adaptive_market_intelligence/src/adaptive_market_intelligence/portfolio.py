from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PortfolioState:
    capital: float
    active_positions: Dict[str, float]
    market_weights: Dict[str, float]
    daily_loss_limit: float
    max_notional_per_market: float


class PortfolioManager:
    def __init__(self, initial_capital: float = 250000.0):
        self.initial_capital = initial_capital
        self.state = PortfolioState(
            capital=initial_capital,
            active_positions={},
            market_weights={"NIFTY": 0.45, "BANKNIFTY": 0.35, "SENSEX": 0.20},
            daily_loss_limit=15000.0,
            max_notional_per_market=50000.0,
        )

    def allocate(self, market: str, requested: float) -> float:
        max_allowed = self.state.market_weights.get(market, 0.2) * self.state.capital
        return min(requested, min(max_allowed, self.state.max_notional_per_market))

    def update_position(self, market: str, notional_delta: float) -> None:
        self.state.active_positions[market] = self.state.active_positions.get(market, 0.0) + notional_delta
        self.state.capital -= notional_delta

    def health(self) -> Dict[str, float]:
        return {
            "capital": self.state.capital,
            "total_active_positions": sum(self.state.active_positions.values()),
            "daily_loss_limit": self.state.daily_loss_limit,
            "max_notional_per_market": self.state.max_notional_per_market,
        }
