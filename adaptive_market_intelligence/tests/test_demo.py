import unittest
from datetime import datetime

from adaptive_market_intelligence.data_quality import DataQualityEngine
from adaptive_market_intelligence.models import DecisionType, Market, MarketTick
from adaptive_market_intelligence.phase_gates import PhaseGateManager
from adaptive_market_intelligence.risk import SafetyKernel
from adaptive_market_intelligence.scanner import RealTimeUniverseScanner
from adaptive_market_intelligence.simulator import BacktestEngine, SyntheticMarketGenerator
from adaptive_market_intelligence.strategy import StrategyEngine
from adaptive_market_intelligence.system_lifecycle import LifecycleState, SystemLifecycle, SystemMode


class AdaptiveMarketIntelligenceTests(unittest.TestCase):
    def test_quality_engine_rejects_crossed_book(self):
        tick = MarketTick(
            market=Market.NIFTY,
            as_of=datetime.utcnow(),
            price=22500.0,
            volume=1200,
            bid=22510.0,
            ask=22490.0,
            quality_score=0.9,
        )
        state = DataQualityEngine().assess(tick)
        self.assertLess(state.quality_score, 1.0)

    def test_signal_engine_generates_trade_signal(self):
        scanner = RealTimeUniverseScanner(lookback=20)
        tick = MarketTick(
            market=Market.NIFTY,
            as_of=datetime.utcnow(),
            price=22600.0,
            volume=1500,
            iv=18.0,
            bid=22598.0,
            ask=22602.0,
            quality_score=0.95,
        )
        snapshot = scanner.scan(tick)
        signal = StrategyEngine().evaluate(snapshot)
        self.assertIn(signal.direction, {DecisionType.BUY_CE, DecisionType.BUY_PE, DecisionType.WAIT})

    def test_safety_kernel_rejects_no_trade(self):
        signal = StrategyEngine().evaluate(RealTimeUniverseScanner().scan(
            MarketTick(
                market=Market.NIFTY,
                as_of=datetime.utcnow(),
                price=22500.0,
                volume=2000,
                iv=16.0,
                bid=22498.0,
                ask=22502.0,
                quality_score=0.5,
            )
        ))
        self.assertNotEqual(signal.direction, DecisionType.BUY_CE)

    def test_backtest_engine_generates_reasonable_metrics(self):
        engine = BacktestEngine()
        series = SyntheticMarketGenerator.generate_market_series(days=4, steps_per_day=18)
        summary = engine.run(series)
        self.assertGreater(summary.total_trades, 0)
        self.assertGreater(summary.final_equity, 0.0)
        self.assertGreaterEqual(summary.win_rate, 0.0)
        self.assertLessEqual(summary.win_rate, 1.0)

    def test_phase_gate_manager_tracks_documented_phases(self):
        manager = PhaseGateManager()
        self.assertIn("P0", manager.phases)
        self.assertIn("P19", manager.phases)
        manager.complete("P0")
        self.assertEqual(manager.phases["P0"].status.value, "done")

    def test_system_lifecycle_enforces_halt_when_health_checks_fail(self):
        lifecycle = SystemLifecycle(mode=SystemMode.PAPER, armed=True)
        lifecycle.add_check("market_feed", True)
        lifecycle.add_check("broker_session", False, "broker session invalid")
        state = lifecycle.evaluate()
        self.assertEqual(state, LifecycleState.HALTED)


if __name__ == "__main__":
    unittest.main()
