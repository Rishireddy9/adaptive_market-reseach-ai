import unittest
import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from adaptive_market_intelligence.llm import HistoricalLLMTrainer, LLMDecisionAdapter, market_features_from_ticks
from adaptive_market_intelligence.simulator import SyntheticMarketGenerator


class LLMResearchTests(unittest.TestCase):
    def test_llm_trainer_builds_reliable_artifact(self):
        series = SyntheticMarketGenerator.generate_market_series(days=6, steps_per_day=20)
        trainer = HistoricalLLMTrainer("market_narrative_classifier_v1")
        artifact = trainer.train(series)
        self.assertGreater(artifact.sample_count, 0)
        self.assertGreaterEqual(artifact.accuracy, 0.0)
        self.assertLessEqual(artifact.accuracy, 1.0)

    def test_llm_analysis_returns_actionable_recommendation(self):
        series = SyntheticMarketGenerator.generate_market_series(days=4, steps_per_day=12)
        trainer = HistoricalLLMTrainer("market_narrative_classifier_v1")
        trainer.train(series)
        result = trainer.generate_analysis("NIFTY", series["NIFTY"])
        self.assertIn(result.recommendation, {"BUY_CE", "BUY_PE", "WAIT", "NO_TRADE"})
        self.assertGreaterEqual(result.confidence, 0.0)

    def test_local_llm_adapter_calls_ollama_and_parses_structured_output(self):
        payload = {
            "message": {
                "content": json.dumps({
                    "recommendation": "WAIT",
                    "confidence": 0.73,
                    "rationale": "The supplied snapshot does not show a strong enough edge.",
                })
            }
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()

        with patch("adaptive_market_intelligence.llm.urllib.request.urlopen", return_value=response) as urlopen:
            result = LLMDecisionAdapter("qwen2.5:7b").analyze(
                "NIFTY",
                {"trend_bias": 0.01, "quality_score": 0.9, "volatility": 0.02, "liquidity": 0.8},
            )

        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.recommendation, "WAIT")
        self.assertFalse(result.fallback_used)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertNotIn("Authorization", request.headers)
        body = json.loads(request.data)
        self.assertEqual(body["model"], "qwen2.5:7b")
        self.assertFalse(body["stream"])

    def test_local_llm_adapter_rejects_invalid_model_output(self):
        payload = {"message": {"content": json.dumps({
            "recommendation": "SEND_ORDER",
            "confidence": 1.2,
            "rationale": "Not a supported research output.",
        })}}
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode()

        with patch("adaptive_market_intelligence.llm.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "unsupported recommendation"):
                LLMDecisionAdapter().analyze("NIFTY", {"trend_bias": 0.0})

    def test_local_llm_adapter_reports_unavailable_ollama(self):
        with patch(
            "adaptive_market_intelligence.llm.urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Start Ollama and pull model"):
                LLMDecisionAdapter("qwen2.5:7b").analyze("NIFTY", {"trend_bias": 0.0})

    def test_market_features_require_non_empty_history(self):
        with self.assertRaisesRegex(ValueError, "At least one market tick"):
            market_features_from_ticks([])


if __name__ == "__main__":
    unittest.main()
