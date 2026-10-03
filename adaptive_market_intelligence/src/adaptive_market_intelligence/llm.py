from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@dataclass
class LLMAnalysisResult:
    model_name: str
    recommendation: str
    confidence: float
    rationale: str
    prompt: str
    fallback_used: bool = False
    provider: str = "ollama"


@dataclass
class LLMTrainingArtifact:
    model_name: str
    trained_at: str
    accuracy: float
    sample_count: int
    status: str = "trained"
    metadata: Dict[str, Any] = field(default_factory=dict)


def market_features_from_ticks(ticks: List[Any], window_size: int = 10) -> Dict[str, float]:
    if not ticks:
        raise ValueError("At least one market tick is required for LLM analysis.")
    recent = ticks[-window_size:]
    current = recent[-1]
    prices = [float(tick.price) for tick in recent]
    first_price = prices[0]
    last_price = prices[-1]
    changes = [prices[index] - prices[index - 1] for index in range(1, len(prices))]
    gains = [change for change in changes if change > 0]
    losses = [-change for change in changes if change < 0]
    avg_gain = sum(gains) / max(1, len(changes))
    avg_loss = sum(losses) / max(1, len(changes))
    if avg_loss == 0:
        rsi = 100.0 if avg_gain > 0 else 50.0
    else:
        rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    spread = (
        (float(current.ask) - float(current.bid)) / max(1.0, float(current.price))
        if current.ask and current.bid and current.ask >= current.bid
        else 0.0
    )
    return {
        "last_price": last_price,
        "trend_bias": (last_price - first_price) / max(1.0, abs(first_price)),
        "volatility": (max(prices) - min(prices)) / max(1.0, abs(last_price)),
        "quality_score": float(current.quality_score),
        "liquidity": min(1.0, max(0.0, 1.0 - spread * 10.0)),
        "rsi": rsi,
        "spread": spread,
        "volume": float(current.volume),
    }


class LLMDecisionAdapter:
    """Research-only client for a local model served by Ollama."""

    RECOMMENDATIONS = {"BUY_CE", "BUY_PE", "WAIT", "NO_TRADE"}

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.api_base = (api_base or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def build_prompt(self, market: str, features: Dict[str, float], context: str = "") -> str:
        lines = [
            "Analyze an Indian index market snapshot for research only. This is not an order instruction.",
            "Use only the supplied point-in-time information. Do not invent prices, option-chain facts, news, or future outcomes.",
            f"Market: {market}",
            "State: " + context,
            "Inputs:",
        ]
        for key, value in sorted(features.items()):
            lines.append(f"- {key}: {value:.4f}")
        lines.extend([
            "Decision goals:",
            "1. Return BUY_CE, BUY_PE, WAIT, or NO_TRADE as a research hypothesis.",
            "2. Return confidence as a number from 0 to 1 and a concise rationale.",
            "3. Use WAIT or NO_TRADE when evidence is weak or incomplete.",
            "4. Never claim the recommendation has passed portfolio risk or execution checks.",
        ])
        return "\n".join(lines)

    @staticmethod
    def _parse_response(content: str) -> Dict[str, Any]:
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("Local LLM returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise ValueError("Local LLM response must be a JSON object.")

        recommendation = str(result.get("recommendation", "")).upper()
        try:
            confidence = float(result["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Local LLM response must include numeric confidence.") from exc
        rationale = result.get("rationale")
        if recommendation not in LLMDecisionAdapter.RECOMMENDATIONS:
            raise ValueError(f"Local LLM returned unsupported recommendation: {recommendation!r}.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Local LLM confidence must be between 0 and 1.")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("Local LLM response must include a non-empty rationale.")
        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "rationale": rationale.strip(),
        }

    def analyze(self, market: str, features: Dict[str, float], context: str = "") -> LLMAnalysisResult:
        prompt = self.build_prompt(market, features, context)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a cautious Indian index market research analyst. "
                        "Return only the requested JSON object. Your output is not an order."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": {
                "type": "object",
                "properties": {
                    "recommendation": {"type": "string", "enum": sorted(self.RECOMMENDATIONS)},
                    "confidence": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["recommendation", "confidence", "rationale"],
            },
            "options": {"temperature": 0.1},
        }

        req = urllib.request.Request(
            url=f"{self.api_base}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.api_base}. Start Ollama and pull model "
                f"{self.model_name!r}, or configure OLLAMA_BASE_URL/OLLAMA_MODEL."
            ) from exc

        try:
            response_data = json.loads(body)
            message = response_data["message"]
            content = message["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("Ollama returned an unexpected chat response.") from exc
        if not isinstance(content, str):
            raise ValueError("Ollama chat response content must be text.")
        result = self._parse_response(content)
        return LLMAnalysisResult(
            model_name=self.model_name,
            recommendation=result["recommendation"],
            confidence=result["confidence"],
            rationale=result["rationale"],
            prompt=prompt,
            fallback_used=False,
            provider="ollama",
        )


class HistoricalLLMTrainer:
    """Trains a text classifier; this is not a generative LLM or LLM fine-tuning."""

    def __init__(self, model_name: str = "market_narrative_classifier_v1"):
        self.model_name = model_name
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model: Optional[LogisticRegression] = None

    def _build_text(self, market: str, feature_map: Dict[str, float]) -> str:
        summary = [
            f"market {market}",
            f"trend_bias {feature_map.get('trend_bias', 0.0):.4f}",
            f"volatility {feature_map.get('volatility', 0.0):.4f}",
            f"quality_score {feature_map.get('quality_score', 0.5):.4f}",
            f"liquidity {feature_map.get('liquidity', 0.5):.4f}",
            f"rsi {feature_map.get('rsi', 50.0):.2f}",
            f"spread {feature_map.get('spread', 0.0):.4f}",
        ]
        return " ".join(summary)

    def _feature_map_from_ticks(self, market: str, ticks: List[Any], index: int) -> Dict[str, float]:
        if index < 0 or index >= len(ticks):
            raise IndexError("Tick index is outside the supplied market history.")
        return market_features_from_ticks(ticks[:index + 1])

    def train(self, series: Dict[str, List[Any]], test_size: float = 0.2) -> LLMTrainingArtifact:
        documents: List[str] = []
        labels: List[int] = []

        for market, ticks in series.items():
            for idx in range(10, len(ticks) - 5):
                current = ticks[idx]
                future = ticks[idx:idx + 5]
                if len(future) < 2:
                    continue
                feature_map = self._feature_map_from_ticks(market, ticks, idx)
                label = 1 if (future[-1].price - current.price) > 0 else 0
                documents.append(self._build_text(market, feature_map))
                labels.append(label)

        if len(documents) < 10:
            raise ValueError("Insufficient historical data for narrative-classifier training.")

        X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=test_size, random_state=42, stratify=labels)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        self.model = LogisticRegression(max_iter=2000, random_state=42)
        self.model.fit(X_train_vec, y_train)
        predictions = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, predictions)

        return LLMTrainingArtifact(
            model_name=self.model_name,
            trained_at=datetime.utcnow().isoformat(timespec="seconds"),
            accuracy=float(accuracy),
            sample_count=len(documents),
            status="trained",
            metadata={
                "market_count": len(series),
                "vectorizer": "TfidfVectorizer",
                "classifier": "LogisticRegression",
            },
        )

    def generate_analysis(self, market: str, history: List[Any]) -> LLMAnalysisResult:
        if self.vectorizer is None or self.model is None:
            raise ValueError("The trainer must be trained before generating an analysis.")

        feature_map = self._feature_map_from_ticks(market, history, len(history) - 1)
        prompt = self._build_text(market, feature_map)
        vector = self.vectorizer.transform([prompt])
        probabilities = self.model.predict_proba(vector)[0]
        predicted_index = int(probabilities.argmax())
        predicted_label = 1 if predicted_index == 1 else 0
        confidence = float(probabilities[predicted_index])
        recommendation = "BUY_CE" if predicted_label == 1 else "WAIT"
        rationale = (
            f"Narrative model scores {recommendation} with {confidence:.2%} confidence "
            f"based on trend bias, quality, liquidity, and volatility in the recent {market} context."
        )
        return LLMAnalysisResult(
            model_name=self.model_name,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
            prompt=prompt,
            fallback_used=False,
        )


def build_llm_training_summary(series: Dict[str, List[Any]], model_name: str = "market_narrative_classifier_v1") -> Tuple[LLMTrainingArtifact, HistoricalLLMTrainer]:
    trainer = HistoricalLLMTrainer(model_name=model_name)
    artifact = trainer.train(series)
    return artifact, trainer
