"""A hard daily spend cap.

The lesson's rule: a public demo needs a ceiling, not a graph. This ledger is
in-memory and per process, which is honest for a single-instance demo and is
listed as a limitation - two replicas do not share the budget.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date


class BudgetExceeded(RuntimeError):
    """Raised before an LLM call that would cross the daily cap."""


@dataclass
class BudgetLedger:
    limit_usd: float
    spent_usd: float = 0.0
    calls: int = 0
    day: date = field(default_factory=date.today)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _roll(self) -> None:
        today = date.today()
        if today != self.day:
            self.day = today
            self.spent_usd = 0.0
            self.calls = 0

    def check(self, estimated_cost: float = 0.0) -> None:
        with self._lock:
            self._roll()
            if self.spent_usd + estimated_cost > self.limit_usd:
                raise BudgetExceeded(
                    f"daily budget of {self.limit_usd:.2f} USD exhausted"
                )

    def charge(self, cost_usd: float) -> None:
        with self._lock:
            self._roll()
            self.spent_usd = round(self.spent_usd + cost_usd, 6)
            self.calls += 1

    def snapshot(self) -> dict:
        with self._lock:
            self._roll()
            return {
                "limit_usd": self.limit_usd,
                "spent_usd": round(self.spent_usd, 6),
                "remaining_usd": round(max(0.0, self.limit_usd - self.spent_usd), 6),
                "calls": self.calls,
                "day": self.day.isoformat(),
            }
