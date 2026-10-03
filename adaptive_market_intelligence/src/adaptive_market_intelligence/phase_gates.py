from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class PhaseStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    BLOCKED = "blocked"
    DONE = "done"


@dataclass
class PhaseGate:
    phase: str
    description: str
    status: PhaseStatus = PhaseStatus.PENDING
    exit_criteria: List[str] = None

    def __post_init__(self):
        if self.exit_criteria is None:
            self.exit_criteria = []

    def mark_ready(self) -> None:
        self.status = PhaseStatus.READY

    def mark_done(self) -> None:
        self.status = PhaseStatus.DONE

    def block(self, reason: str) -> None:
        self.status = PhaseStatus.BLOCKED
        self.exit_criteria.append(reason)


class PhaseGateManager:
    def __init__(self):
        self.phases: Dict[str, PhaseGate] = {
            "P0": PhaseGate("P0", "Architecture freeze and canonical contracts", ["Architecture approved enough to code"]),
            "P1": PhaseGate("P1", "Repository and engineering foundation", ["Fresh clone runs lint and tests"]),
            "P2": PhaseGate("P2", "Core domain and reference time system", ["Instrument/session correctness tests pass"]),
            "P3": PhaseGate("P3", "Historical/live data adapters and raw recorder", ["Can ingest and replay a day reproducibly"]),
            "P4": PhaseGate("P4", "Data quality, watermark and canonical market state", ["Bad data cannot silently enter decisions"]),
            "P5": PhaseGate("P5", "Parallel universe scanner and multi-timeframe aggregation", ["All markets update concurrently within performance budget"]),
            "P6": PhaseGate("P6", "Baseline features and strategy V0", ["Baseline reproducible; no broker code involved"]),
            "P7": PhaseGate("P7", "Event-driven backtest engine", ["Leakage/cost/regression tests pass"]),
            "P8": PhaseGate("P8", "Market intelligence", ["Only evidence-backed features remain candidates"]),
            "P9": PhaseGate("P9", "Options intelligence", ["No strong underlying signal can bypass poor option economics"]),
            "P10": PhaseGate("P10", "Opportunity / uncertainty / global ranking", ["Deterministic decisions and reason codes for frozen inputs"]),
            "P11": PhaseGate("P11", "Portfolio / capital / risk / safety / exit", ["Risk can veto or resize"]),
            "P12": PhaseGate("P12", "Broker execution / orders / positions / reconciliation", ["Timeout/duplicate tests pass"]),
            "P13": PhaseGate("P13", "Replay / counterfactual / TCA / attribution", ["Replay decisions match expected deterministic state"]),
            "P14": PhaseGate("P14", "Live paper trading", ["Multiple full sessions stable without manual intervention"]),
            "P15": PhaseGate("P15", "Reliability / fault / recovery / performance", ["Critical failures fail closed and recover deterministically"]),
            "P16": PhaseGate("P16", "AWS V1 deployment", ["Fresh deployment from code; no manual patching"]),
            "P17": PhaseGate("P17", "Shadow + champion/challenger", ["Stable live evidence and no critical discrepancies"]),
            "P18": PhaseGate("P18", "Tiny controlled live", ["Human-approved tiny-live release"]),
            "P19": PhaseGate("P19", "Controlled production / HA when earned", ["Production SLOs and capacity plan"]),
        }

    def release(self, phase_id: str) -> PhaseGate:
        gate = self.phases[phase_id]
        gate.mark_ready()
        return gate

    def complete(self, phase_id: str) -> PhaseGate:
        gate = self.phases[phase_id]
        gate.mark_done()
        return gate

    def snapshot(self) -> Dict[str, str]:
        return {phase_id: gate.status.value for phase_id, gate in self.phases.items()}
