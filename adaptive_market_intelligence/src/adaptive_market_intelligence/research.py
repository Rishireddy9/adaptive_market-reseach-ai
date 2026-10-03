from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .scanner import RealTimeUniverseScanner
from .simulator import SyntheticMarketGenerator


@dataclass
class ResearchArtifact:
    model_name: str
    trained_at: str
    accuracy: float
    feature_names: List[str]
    sample_count: int
    validation_score: float
    status: str = "trained"


class ResearchModelTrainer:
    """Historical-data research model that mirrors the doc's research-only design."""

    def __init__(self, model_name: str = "trend_classifier_v1"):
        self.model_name = model_name

    def train(self, series: Dict[str, List], test_size: float = 0.2) -> ResearchArtifact:
        rows: List[List[float]] = []
        labels: List[int] = []

        for market_name, ticks in series.items():
            prices = [tick.price for tick in ticks]
            for idx in range(10, len(prices) - 6):
                window = prices[idx - 10:idx]
                future = prices[idx:idx + 5]
                if len(window) < 10 or len(future) < 5:
                    continue

                ema_fast = self._ema(window, 5)
                ema_slow = self._ema(window, 9)
                momentum = future[-1] - prices[idx - 1]
                volatility = float(np.std(window[-8:]))
                rsi = self._rsi(window)
                spread = (ticks[idx].ask - ticks[idx].bid) if ticks[idx].ask and ticks[idx].bid else 0.0

                feature_vector = [
                    prices[idx - 1],
                    ema_fast,
                    ema_slow,
                    momentum,
                    volatility,
                    rsi,
                    spread,
                    ticks[idx].volume,
                ]
                label = 1 if (future[-1] - prices[idx - 1]) > 0 else 0
                rows.append(feature_vector)
                labels.append(label)

        if len(rows) < 10:
            raise ValueError("Insufficient research data for training.")

        X_train, X_test, y_train, y_test = train_test_split(rows, labels, test_size=test_size, random_state=42, stratify=labels)
        model = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=6)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        return ResearchArtifact(
            model_name=self.model_name,
            trained_at=datetime.utcnow().isoformat(timespec="seconds"),
            accuracy=float(accuracy),
            feature_names=["last_price", "ema_fast", "ema_slow", "momentum", "volatility", "rsi", "spread", "volume"],
            sample_count=len(rows),
            validation_score=float(accuracy),
        )

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        if not values:
            return 0.0
        alpha = 2.0 / (period + 1.0)
        ema = values[0]
        for value in values[1:]:
            ema = (alpha * value) + ((1.0 - alpha) * ema)
        return ema

    @staticmethod
    def _rsi(values: List[float], period: int = 14) -> float:
        if len(values) < 2:
            return 50.0
        deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
        if not deltas:
            return 50.0
        recent = deltas[-period:]
        gains = [max(d, 0) for d in recent]
        losses = [max(-d, 0) for d in recent]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


def train_research_model() -> ResearchArtifact:
    series = SyntheticMarketGenerator.generate_market_series(days=30, steps_per_day=40)
    trainer = ResearchModelTrainer("trend_classifier_v1")
    return trainer.train(series)
