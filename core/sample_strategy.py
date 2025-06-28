"""Simple sample strategy implementation used for tests and demos."""
from __future__ import annotations

import pickle
from typing import Any, List, Tuple, Optional


class SampleGoStrategy:
    """Extremely naive Go strategy.

    The strategy merely selects the first empty point on the board when asked
    to :meth:`predict`.  It supports ``save``/``load`` methods so it can be
    persisted by :class:`~core.strategy_manager.StrategyManager`.
    """

    def __init__(self, name: str = "sample", stats: Optional[dict] = None) -> None:
        self.name = name
        self.stats = stats or {}

    # ------------------------------------------------------------------
    def predict(self, board: List[List[int]]) -> Tuple[int, int] | None:
        """Return the coordinates of the first empty position in ``board``."""
        for y, row in enumerate(board):
            for x, val in enumerate(row):
                if val == 0:
                    return (x, y)
        return None

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist strategy state to ``path`` using pickle."""
        with open(path, "wb") as fh:
            pickle.dump({"name": self.name, "stats": self.stats}, fh)

    @classmethod
    def load(cls, path: str) -> "SampleGoStrategy":
        """Load a strategy instance previously saved with :meth:`save`."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, dict):
            return cls(name=data.get("name", "sample"), stats=data.get("stats"))
        # Fallback for direct object pickle
        return data


__all__ = ["SampleGoStrategy"]
