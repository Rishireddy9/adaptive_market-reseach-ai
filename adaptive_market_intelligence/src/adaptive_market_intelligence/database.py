from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatabaseManager:
    def __init__(self, db_path: str | Path = "adaptive_market_intelligence.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def backend(self) -> str:
        return "sqlite" if self.db_path.exists() else "sqlite"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    reason TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    closed_at TEXT,
                    pnl REAL DEFAULT 0.0
                )
                """
            )
            conn.commit()

    def record_decision(self, market: str, signal: str, risk: str, confidence: float, reason: Optional[str] = None) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO decisions (market, signal, risk, confidence, created_at, reason) VALUES (?, ?, ?, ?, datetime('now'), ?)",
                (market, signal, risk, confidence, reason),
            )
            conn.commit()

    def record_trade(self, market: str, side: str, quantity: int, entry_price: float, status: str = "OPEN") -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO trades (market, side, quantity, entry_price, status, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (market, side, quantity, entry_price, status),
            )
            conn.commit()

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT market, signal, risk, confidence, created_at, reason FROM decisions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_trade_summary(self) -> Dict[str, Any]:
        with closing(self._connect()) as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM trades").fetchone()["c"]
            open_trades = conn.execute("SELECT COUNT(*) as c FROM trades WHERE status = 'OPEN'").fetchone()["c"]
            realized = conn.execute("SELECT COALESCE(SUM(pnl), 0.0) as total FROM trades WHERE status = 'CLOSED'").fetchone()["total"]
        return {"total": total, "open": open_trades, "realized_pnl": realized}
