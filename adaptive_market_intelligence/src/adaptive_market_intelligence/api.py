from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .config import DEFAULT_CONFIG
from .database import DatabaseManager
from .paper_trading import PaperTradingEngine
from .portfolio import PortfolioManager
from .settings import API_HOST, API_PORT
from .simulator import BacktestEngine, DecisionEngine, SyntheticMarketGenerator

app = FastAPI(title="Adaptive Market Intelligence API", version="0.2.0")

db = DatabaseManager()
portfolio_manager = PortfolioManager()


class BacktestRequest(BaseModel):
    days: int = 6
    steps_per_day: int = 24


class PaperTradingRequest(BaseModel):
    days: int = 2
    steps_per_day: int = 12


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "adaptive-market-intelligence",
        "database": db.backend(),
        "environment": "dev",
    }


@app.get("/decisions")
def get_recent_decisions(limit: int = 20) -> list[dict]:
    return db.get_recent_decisions(limit=limit)


@app.get("/portfolio")
def get_portfolio() -> dict:
    return portfolio_manager.health()


@app.post("/backtest")
def run_backtest(payload: BacktestRequest) -> dict:
    summary = BacktestEngine(DEFAULT_CONFIG).run(
        SyntheticMarketGenerator.generate_market_series(days=payload.days, steps_per_day=payload.steps_per_day)
    )
    return {
        "total_trades": summary.total_trades,
        "winners": summary.winners,
        "losers": summary.losers,
        "win_rate": summary.win_rate,
        "total_pnl": summary.total_pnl,
        "avg_trade_pnl": summary.avg_trade_pnl,
        "final_equity": summary.final_equity,
        "max_drawdown": summary.max_drawdown,
        "sharpe_proxy": summary.sharpe_proxy,
    }


@app.post("/paper-trading")
def run_paper_trading(payload: PaperTradingRequest) -> dict:
    series = SyntheticMarketGenerator.generate_market_series(days=payload.days, steps_per_day=payload.steps_per_day)
    decisions = DecisionEngine(DEFAULT_CONFIG).run(series, limit=len(next(iter(series.values()))))
    market_prices = {market: tick[-1].price for market, tick in series.items()}
    review = PaperTradingEngine(database=db).run(decisions, market_prices)
    return {
        "total_orders": review.total_orders,
        "filled_orders": review.filled_orders,
        "rejected_orders": review.rejected_orders,
        "realized_pnl": review.realized_pnl,
        "fills": [
            {
                "market": fill.market,
                "side": fill.side,
                "quantity": fill.quantity,
                "fill_price": fill.fill_price,
                "timestamp": fill.timestamp.isoformat(),
            }
            for fill in review.fills
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("adaptive_market_intelligence.api:app", host=API_HOST, port=API_PORT, reload=False)
