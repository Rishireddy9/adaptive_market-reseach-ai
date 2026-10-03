from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import sin, sqrt
from typing import Dict, List, Optional, Tuple

from .config import DEFAULT_CONFIG
from .data_quality import DataQualityEngine
from .models import DecisionType, Market, MarketTick, OpportunityDecision, RiskDecision, RiskDecisionType
from .risk import SafetyKernel
from .scanner import RealTimeUniverseScanner
from .strategy import StrategyEngine


@dataclass
class Trade:
    market: str
    side: str
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_index: int
    stop_price: float
    target_price: float
    pnl: float = 0.0
    closed: bool = False
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None


@dataclass
class BacktestSummary:
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    avg_trade_pnl: float
    final_equity: float
    max_drawdown: float
    sharpe_proxy: float
    equity_curve: List[float] = field(default_factory=list)


class SyntheticMarketGenerator:
    @staticmethod
    def generate_market_series(days: int = 12, steps_per_day: int = 48) -> Dict[str, List[MarketTick]]:
        tick_series: Dict[str, List[MarketTick]] = {market.value: [] for market in Market}
        phase_map = {Market.NIFTY: 0.0, Market.BANKNIFTY: 1.2, Market.SENSEX: 2.5}
        base_map = {Market.NIFTY: 22650.0, Market.BANKNIFTY: 48200.0, Market.SENSEX: 74250.0}

        start = datetime.utcnow().replace(hour=9, minute=15, second=0, microsecond=0)
        for day in range(days):
            for step in range(steps_per_day):
                current = start + timedelta(minutes=step * 5) + timedelta(days=day)
                for market in Market:
                    wave = sin((step + phase_map[market]) / 4.5)
                    drift = (step / steps_per_day) * 0.7
                    price = base_map[market] + (wave * 120.0) + (drift * 160.0)
                    volume = 800 + step * 48 + int(abs(wave) * 600)
                    iv = 14.0 + abs(wave) * 9.0
                    bid = price - 2.5
                    ask = price + 2.5
                    quality_score = 0.98 if step % 8 else 0.78
                    tick_series[market.value].append(
                        MarketTick(
                            market=market,
                            as_of=current,
                            price=price,
                            volume=volume,
                            oi=volume * 3,
                            iv=iv,
                            bid=bid,
                            ask=ask,
                            quality_score=quality_score,
                            synthetic=True,
                        )
                    )
        return tick_series


class DecisionEngine:
    def __init__(self, config=DEFAULT_CONFIG):
        self.scanner = RealTimeUniverseScanner()
        self.strategy = StrategyEngine()
        self.risk = SafetyKernel(config)
        self.data_quality = DataQualityEngine()

    def run(self, series: Dict[str, List[MarketTick]], limit: int | None = None) -> List[dict]:
        decisions: List[dict] = []
        max_index = min(limit or len(next(iter(series.values()))), len(next(iter(series.values()))))
        for i in range(max_index):
            batch: Dict[str, dict] = {}
            for market_name, ticks in series.items():
                tick = ticks[i]
                quality = self.data_quality.assess(tick, ticks[i - 1] if i > 0 else None)
                snapshot = self.scanner.scan(tick)
                signal = self.strategy.evaluate(snapshot)
                if signal.direction == DecisionType.NO_TRADE:
                    opportunity = OpportunityDecision(
                        market=market_name,
                        decision=DecisionType.NO_TRADE,
                        expected_move=0.0,
                        remaining_move=0.0,
                        stage="decision",
                        ev_before_cost=0.0,
                        ev_after_cost=0.0,
                        reason_codes=signal.reason_codes,
                    )
                else:
                    opportunity = OpportunityDecision(
                        market=market_name,
                        decision=signal.direction,
                        expected_move=signal.expected_move,
                        remaining_move=max(signal.expected_move * 0.65, 0.0),
                        stage="opportunity",
                        ev_before_cost=signal.expected_move * 0.9,
                        ev_after_cost=signal.expected_move * 0.72,
                        reason_codes=signal.reason_codes,
                    )

                risk_decision = self.risk.evaluate(signal, opportunity)
                batch[market_name] = {
                    "tick": tick,
                    "quality": quality,
                    "snapshot": snapshot,
                    "signal": signal,
                    "opportunity": opportunity,
                    "risk": risk_decision,
                }
            decisions.append(batch)
        return decisions


class BacktestEngine:
    def __init__(self, config=DEFAULT_CONFIG):
        self.engine = DecisionEngine(config)
        self.max_holding_steps = 8
        self.starting_cash = config.capital_budget

    def run(self, series: Dict[str, List[MarketTick]], limit: int | None = None) -> BacktestSummary:
        portfolio_cash = float(self.starting_cash)
        equity_curve: List[float] = []
        trades: List[Trade] = []
        open_positions: Dict[str, Trade] = {}
        max_index = min(limit or len(next(iter(series.values()))), len(next(iter(series.values()))))
        current_equity = self.starting_cash

        for i in range(max_index):
            for market_name, ticks in series.items():
                tick = ticks[i]
                quality = self.engine.data_quality.assess(tick, ticks[i - 1] if i > 0 else None)
                snapshot = self.engine.scanner.scan(tick)
                signal = self.engine.strategy.evaluate(snapshot)
                opportunity = self._make_opportunity(market_name, signal)
                risk = self.engine.risk.evaluate(signal, opportunity)

                if market_name in open_positions:
                    trade = open_positions[market_name]
                    if i - trade.entry_index >= self.max_holding_steps or signal.direction in {DecisionType.WAIT, DecisionType.NO_TRADE}:
                        trade.closed = True
                        trade.exit_time = tick.as_of
                        trade.exit_price = tick.price
                        trade.pnl = self._pnl_for_trade(trade, tick.price)
                        trades.append(trade)
                        del open_positions[market_name]
                        portfolio_cash += trade.pnl

                if market_name not in open_positions and risk.decision == RiskDecisionType.APPROVE and opportunity.decision in {DecisionType.BUY_CE, DecisionType.BUY_PE}:
                    quantity = max(5, int(risk.approved_quantity // 12))
                    order_price = tick.price * 0.015
                    trade = Trade(
                        market=market_name,
                        side=opportunity.decision.value,
                        quantity=quantity,
                        entry_price=order_price,
                        entry_time=tick.as_of,
                        entry_index=i,
                        stop_price=order_price * 0.92,
                        target_price=order_price * 1.15,
                    )
                    open_positions[market_name] = trade
                    portfolio_cash -= order_price * quantity

            current_equity = portfolio_cash + sum(
                (trade.entry_price * trade.quantity) for trade in open_positions.values()
            )
            equity_curve.append(current_equity)

        closed_trades = [trade for trade in trades if trade.closed]
        winners = sum(1 for trade in closed_trades if trade.pnl > 0)
        losers = sum(1 for trade in closed_trades if trade.pnl <= 0)
        total_pnl = sum(trade.pnl for trade in closed_trades)
        avg_trade_pnl = total_pnl / len(closed_trades) if closed_trades else 0.0
        win_rate = winners / len(closed_trades) if closed_trades else 0.0
        max_drawdown = self._max_drawdown(equity_curve)
        sharpe_proxy = self._sharpe_proxy([trade.pnl for trade in closed_trades])

        return BacktestSummary(
            total_trades=len(closed_trades),
            winners=winners,
            losers=losers,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_trade_pnl=avg_trade_pnl,
            final_equity=current_equity if equity_curve else self.starting_cash,
            max_drawdown=max_drawdown,
            sharpe_proxy=sharpe_proxy,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _make_opportunity(market_name: str, signal: object) -> OpportunityDecision:
        if signal.direction == DecisionType.NO_TRADE:
            return OpportunityDecision(
                market=market_name,
                decision=DecisionType.NO_TRADE,
                expected_move=0.0,
                remaining_move=0.0,
                stage="decision",
                ev_before_cost=0.0,
                ev_after_cost=0.0,
                reason_codes=signal.reason_codes,
            )
        return OpportunityDecision(
            market=market_name,
            decision=signal.direction,
            expected_move=signal.expected_move,
            remaining_move=max(signal.expected_move * 0.65, 0.0),
            stage="opportunity",
            ev_before_cost=signal.expected_move * 0.9,
            ev_after_cost=signal.expected_move * 0.72,
            reason_codes=signal.reason_codes,
        )

    @staticmethod
    def _pnl_for_trade(trade: Trade, exit_price: float) -> float:
        direction_multiplier = 1.0 if trade.side in {"BUY_CE", "BUY_PE"} else -1.0
        return direction_multiplier * (exit_price - trade.entry_price) * trade.quantity

    @staticmethod
    def _max_drawdown(equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak else 0.0
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd

    @staticmethod
    def _sharpe_proxy(pnl_values: List[float]) -> float:
        if not pnl_values:
            return 0.0
        mean_pnl = sum(pnl_values) / len(pnl_values)
        variance = sum((value - mean_pnl) ** 2 for value in pnl_values) / len(pnl_values)
        std_dev = sqrt(variance) if variance > 0 else 0.0
        return mean_pnl / std_dev if std_dev > 0 else 0.0
