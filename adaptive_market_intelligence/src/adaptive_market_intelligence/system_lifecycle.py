from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SystemMode(str, Enum):
    DEV = "dev"
    BACKTEST = "backtest"
    REPLAY = "replay"
    PAPER = "paper"
    SHADOW = "shadow"
    LIVE = "live"


class LifecycleState(str, Enum):
    INIT = "INIT"
    READY = "READY"
    ARMED = "ARMED"
    ACTIVE = "ACTIVE"
    EXIT_ONLY = "EXIT_ONLY"
    HALTED = "HALTED"
    EMERGENCY = "EMERGENCY"


@dataclass
class HealthCheck:
    name: str
    passed_: bool
    details: str = ""


@dataclass
class SystemLifecycle:
    mode: SystemMode = SystemMode.DEV
    state: LifecycleState = LifecycleState.INIT
    checks: List[HealthCheck] = field(default_factory=list)
    armed: bool = False
    last_reason: Optional[str] = None

    def add_check(self, name: str, passed_: bool, details: str = "") -> None:
        self.checks.append(HealthCheck(name=name, passed_=passed_, details=details))

    def evaluate(self) -> LifecycleState:
        all_good = all(check.passed_ for check in self.checks)
        if not all_good:
            self.state = LifecycleState.HALTED
            self.last_reason = "health_checks_failed"
            return self.state
        if self.mode == SystemMode.LIVE and not self.armed:
            self.state = LifecycleState.READY
            self.last_reason = "live_not_armed"
            return self.state
        self.state = LifecycleState.ARMED if self.armed else LifecycleState.READY
        if self.state == LifecycleState.ARMED:
            self.last_reason = "ready_for_execution"
        return self.state

    def arm(self) -> None:
        self.armed = True
        self.state = LifecycleState.ARMED

    def exit_only(self) -> None:
        self.armed = False
        self.state = LifecycleState.EXIT_ONLY
        self.last_reason = "exit_only_mode"

    def halt(self, reason: str) -> None:
        self.armed = False
        self.state = LifecycleState.HALTED
        self.last_reason = reason
