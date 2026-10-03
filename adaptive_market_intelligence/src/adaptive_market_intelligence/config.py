from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class SystemConfig:
    capital_budget: float = 250000.0
    max_position_size: float = 50000.0
    max_daily_loss: float = 15000.0
    min_data_quality: float = 0.75
    min_option_liquidity: float = 0.6
    max_spread_ratio: float = 0.08
    live_mode: bool = False
    environment: str = "paper"
    risk_multiplier: float = 1.0
    allowed_markets: List[str] = field(default_factory=lambda: ["NIFTY", "BANKNIFTY", "SENSEX"])

    @property
    def market_caps(self) -> Dict[str, float]:
        return {
            "NIFTY": 0.45,
            "BANKNIFTY": 0.35,
            "SENSEX": 0.20,
        }


DEFAULT_CONFIG = SystemConfig()
