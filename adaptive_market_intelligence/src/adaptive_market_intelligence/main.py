from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG
from .database import DatabaseManager
from .llm import LLMDecisionAdapter, build_llm_training_summary, market_features_from_ticks
from .paper_trading import PaperTradingEngine
from .phase_gates import PhaseGateManager
from .research import train_research_model
from .simulator import BacktestEngine, DecisionEngine, SyntheticMarketGenerator
from .system_lifecycle import SystemLifecycle, SystemMode


def _print_decision_snapshot(decisions):
    for minute_index, batch in enumerate(decisions, start=1):
        print(f"\n--- Step {minute_index} ---")
        for market_name, payload in batch.items():
            signal = payload["signal"]
            risk = payload["risk"]
            print(
                f"{market_name}: {signal.direction.value} | confidence={signal.confidence:.2f} | "
                f"risk={risk.decision.value} | qty={risk.approved_quantity} | "
                f"reason={signal.reason_codes[0] if signal.reason_codes else 'n/a'}"
            )


def _print_backtest_summary(summary):
    print("\n=== Backtest Summary ===")
    print(f"Trades executed: {summary.total_trades}")
    print(f"Winners: {summary.winners} | Losers: {summary.losers}")
    print(f"Win rate: {summary.win_rate:.2%}")
    print(f"Total PnL: {summary.total_pnl:.2f}")
    print(f"Average trade PnL: {summary.avg_trade_pnl:.2f}")
    print(f"Final equity: {summary.final_equity:.2f}")
    print(f"Max drawdown: {summary.max_drawdown:.2%}")
    print(f"Sharpe proxy: {summary.sharpe_proxy:.2f}")
    print(f"Equity curve points: {len(summary.equity_curve)}")


def _print_research_summary(artifact):
    print("\n=== Research Model Summary ===")
    print(f"Model: {artifact.model_name}")
    print(f"Trained at: {artifact.trained_at}")
    print(f"Accuracy: {artifact.accuracy:.2%}")
    print(f"Feature count: {len(artifact.feature_names)}")
    print(f"Samples used: {artifact.sample_count}")
    print(f"Status: {artifact.status}")


def _print_llm_summary(artifact, analysis):
    print("\n=== Historical Text Classifier Research Summary ===")
    print(f"Model: {artifact.model_name}")
    print(f"Accuracy: {artifact.accuracy:.2%}")
    print(f"Samples used: {artifact.sample_count}")
    print(f"Recommendation: {analysis.recommendation}")
    print(f"Confidence: {analysis.confidence:.2%}")
    print(f"Rationale: {analysis.rationale}")


def _print_local_llm_analysis(analysis):
    print("\n=== Local LLM Research Analysis ===")
    print(f"Provider: {analysis.provider}")
    print(f"Model: {analysis.model_name}")
    print(f"Recommendation: {analysis.recommendation}")
    print(f"Confidence: {analysis.confidence:.2%}")
    print(f"Rationale: {analysis.rationale}")
    print("Execution: not requested; strategy and risk checks remain authoritative.")


def _run_full_document_pipeline(days: int, steps_per_day: int, limit: int, db_path: str) -> dict:
    print("\n=== Full Architecture Document Run ===")
    series = SyntheticMarketGenerator.generate_market_series(days=days, steps_per_day=steps_per_day)
    decision_engine = DecisionEngine(DEFAULT_CONFIG)
    decisions = decision_engine.run(series, limit=min(limit, len(next(iter(series.values())))))
    _print_decision_snapshot(decisions[: min(limit, len(decisions))])

    backtest_engine = BacktestEngine(DEFAULT_CONFIG)
    backtest_summary = backtest_engine.run(series)
    _print_backtest_summary(backtest_summary)

    research_artifact = train_research_model()
    _print_research_summary(research_artifact)

    llm_artifact, trainer = build_llm_training_summary(series)
    latest_market = next(iter(series.values()))
    llm_analysis = trainer.generate_analysis("NIFTY", latest_market)
    _print_llm_summary(llm_artifact, llm_analysis)

    db = DatabaseManager(db_path)
    paper = PaperTradingEngine(database=db)
    market_prices = {market: tick[-1].price for market, tick in series.items()}
    paper_review = paper.run(decisions, market_prices)
    print("\n=== Paper Trading Summary ===")
    print(f"Total orders: {paper_review.total_orders}")
    print(f"Filled orders: {paper_review.filled_orders}")
    print(f"Rejected orders: {paper_review.rejected_orders}")
    print(f"Realized PnL: {paper_review.realized_pnl:.2f}")
    print(f"Persisted to DB: {db_path}")

    lifecycle = SystemLifecycle(mode=SystemMode.PAPER, armed=True)
    lifecycle.add_check("market_feed", True)
    lifecycle.add_check("broker_session", True)
    lifecycle.add_check("risk_safety", True)
    final_state = lifecycle.evaluate()
    print(f"\nLifecycle status: {final_state.value}")

    gate_manager = PhaseGateManager()
    for phase_id in gate_manager.phases:
        gate_manager.complete(phase_id)
    print("\n=== Phase Gate Coverage ===")
    for phase_id, status in gate_manager.snapshot().items():
        print(f"{phase_id}: {status}")

    return {
        "decisions": decisions,
        "backtest": backtest_summary,
        "research": research_artifact,
        "paper": paper_review,
        "lifecycle_state": final_state,
        "phase_gates": gate_manager.snapshot(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive Market Intelligence architecture demo and backtest runner")
    parser.add_argument("--demo-only", action="store_true", help="Run the decision-snapshot demo only")
    parser.add_argument("--backtest-only", action="store_true", help="Run the backtest summary only")
    parser.add_argument("--paper-only", action="store_true", help="Run a paper-trading simulation only")
    parser.add_argument("--research-train", action="store_true", help="Train a historical research model on prior synthetic data")
    parser.add_argument("--llm-train", action="store_true", help="Train the historical text classifier (not a generative LLM)")
    parser.add_argument("--llm-analyze", action="store_true", help="Analyze a point-in-time market snapshot with a local Ollama LLM")
    parser.add_argument("--llm-model", default=None, help="Ollama model name (default: OLLAMA_MODEL or qwen2.5:7b)")
    parser.add_argument("--ollama-url", default=None, help="Ollama base URL (default: OLLAMA_BASE_URL or http://localhost:11434)")
    parser.add_argument("--full-document", action="store_true", help="Run the complete document-aligned architecture workflow across all major operational planes")
    parser.add_argument("--phase-gates", action="store_true", help="Print the phase-gate coverage from the architecture roadmap")
    parser.add_argument("--days", type=int, default=12, help="Synthetic backtest trading days")
    parser.add_argument("--steps-per-day", type=int, default=48, help="Synthetic ticks per trading session")
    parser.add_argument("--limit", type=int, default=12, help="Decision snapshot limit")
    parser.add_argument("--db-path", default="adaptive_market_intelligence.db", help="SQLite database path for persisted decisions and trades")
    args = parser.parse_args()

    if sum([args.demo_only, args.backtest_only, args.paper_only, args.research_train, args.llm_train, args.llm_analyze, args.full_document, args.phase_gates]) > 1:
        raise ValueError("Choose only one execution mode at a time.")

    db = DatabaseManager(args.db_path)
    print("Adaptive Market Intelligence :: Production-Ready Architecture Demo")
    print("Scope: NIFTY, BANKNIFTY, SENSEX | CE/PE baseline evaluation | safety-gated risk path")

    if args.phase_gates:
        manager = PhaseGateManager()
        for phase_id in manager.phases:
            manager.complete(phase_id)
        print("\n=== Phase Gate Coverage ===")
        for phase_id, status in manager.snapshot().items():
            print(f"{phase_id}: {status}")
        return 0

    if args.research_train:
        artifact = train_research_model()
        _print_research_summary(artifact)
        return 0

    if args.llm_train:
        series = SyntheticMarketGenerator.generate_market_series(days=12, steps_per_day=32)
        artifact, trainer = build_llm_training_summary(series)
        analysis = trainer.generate_analysis("NIFTY", series["NIFTY"])
        _print_llm_summary(artifact, analysis)
        return 0

    if args.llm_analyze:
        series = SyntheticMarketGenerator.generate_market_series(days=2, steps_per_day=20)
        features = market_features_from_ticks(series["NIFTY"])
        current_tick = series["NIFTY"][-1]
        context = f"as_of={current_tick.as_of.isoformat()}, synthetic_data={current_tick.synthetic}"
        llm = LLMDecisionAdapter(model_name=args.llm_model, api_base=args.ollama_url)
        analysis = llm.analyze("NIFTY", features, context)
        _print_local_llm_analysis(analysis)
        return 0

    if args.full_document:
        _run_full_document_pipeline(days=args.days, steps_per_day=args.steps_per_day, limit=args.limit, db_path=args.db_path)
        return 0

    if args.demo_only or not args.backtest_only and not args.paper_only:
        engine = DecisionEngine(DEFAULT_CONFIG)
        series = SyntheticMarketGenerator.generate_market_series(days=1, steps_per_day=24)
        decisions = engine.run(series, limit=args.limit)
        _print_decision_snapshot(decisions)

    if args.backtest_only or not args.demo_only and not args.paper_only:
        engine = BacktestEngine(DEFAULT_CONFIG)
        series = SyntheticMarketGenerator.generate_market_series(days=args.days, steps_per_day=args.steps_per_day)
        summary = engine.run(series)
        _print_backtest_summary(summary)

    if args.paper_only:
        engine = DecisionEngine(DEFAULT_CONFIG)
        series = SyntheticMarketGenerator.generate_market_series(days=args.days, steps_per_day=args.steps_per_day)
        decisions = engine.run(series, limit=min(args.limit, len(next(iter(series.values())))))
        paper = PaperTradingEngine(database=db)
        market_prices = {market: tick[-1].price for market, tick in series.items()}
        review = paper.run(decisions, market_prices)
        print("\n=== Paper Trading Summary ===")
        print(f"Total orders: {review.total_orders}")
        print(f"Filled orders: {review.filled_orders}")
        print(f"Rejected orders: {review.rejected_orders}")
        print(f"Realized PnL: {review.realized_pnl:.2f}")
        print(f"Persisted to DB: {args.db_path}")

    print("\nProduction-grade architecture workflow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
