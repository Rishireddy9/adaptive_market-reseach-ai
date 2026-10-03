"""Adaptive Market Intelligence framework."""

from .database import DatabaseManager
from .llm import HistoricalLLMTrainer, LLMAnalysisResult, LLMDecisionAdapter, LLMTrainingArtifact, build_llm_training_summary
from .main import main
from .paper_trading import PaperTradingEngine
from .phase_gates import PhaseGateManager
from .portfolio import PortfolioManager
from .strategy_registry import StrategyRegistry
from .system_lifecycle import LifecycleState, SystemLifecycle, SystemMode

__all__ = [
    "main",
    "DatabaseManager",
    "PortfolioManager",
    "PaperTradingEngine",
    "PhaseGateManager",
    "StrategyRegistry",
    "SystemLifecycle",
    "LifecycleState",
    "SystemMode",
    "LLMDecisionAdapter",
    "HistoricalLLMTrainer",
    "LLMAnalysisResult",
    "LLMTrainingArtifact",
    "build_llm_training_summary",
]
