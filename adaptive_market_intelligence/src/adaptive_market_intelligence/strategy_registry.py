from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .models import DecisionType, SignalCandidate


class StrategyFamily(str, Enum):
    BASELINE = "baseline"
    CHALLENGER = "challenger"
    CHAMPION = "champion"


@dataclass
class StrategySpec:
    name: str
    family: StrategyFamily
    version: str
    description: str
    enabled: bool = True
    risk_mode: str = "safety_gated"


class StrategyRegistry:
    def __init__(self):
        self.registry: Dict[str, StrategySpec] = {
            "EMA_RSI_V0": StrategySpec(
                name="EMA_RSI_V0",
                family=StrategyFamily.BASELINE,
                version="v0",
                description="Baseline EMA14/20 + RSI14/20 multi-timeframe benchmark",
                enabled=True,
                risk_mode="safety_gated",
            ),
            "TREND_CLASSIFIER_V1": StrategySpec(
                name="TREND_CLASSIFIER_V1",
                family=StrategyFamily.CHALLENGER,
                version="v1",
                description="Historical-data research challenge model",
                enabled=True,
                risk_mode="paper_only",
            ),
            "CHAMPION_APPROVED": StrategySpec(
                name="CHAMPION_APPROVED",
                family=StrategyFamily.CHAMPION,
                version="approved",
                description="Approved live champion with explicit promotion gates",
                enabled=False,
                risk_mode="live_armed",
            ),
        }

    def list(self) -> List[StrategySpec]:
        return list(self.registry.values())

    def get(self, name: str) -> Optional[StrategySpec]:
        return self.registry.get(name)

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self.registry:
            self.registry[name].enabled = enabled

    def champion_status(self) -> Dict[str, bool]:
        return {name: spec.enabled for name, spec in self.registry.items()}
