"""Main engine tying together strategy management and learning."""
from __future__ import annotations

from .strategy_manager import StrategyManager
from .auto_learner import AutoLearner


class Engine:
    """Facade for high level operations.

    The engine provides helper methods to manage strategies and perform very
    basic move decision.  It remains intentionally lightweight so that higher
    level interfaces (REST API, GTP etc.) can reuse it.
    """

    def __init__(self, model_dir: str) -> None:
        self.strategy_manager = StrategyManager(model_dir)
        self.auto_learner = AutoLearner(self.strategy_manager)
        self.current_strategy_name: str | None = None
        self.current_strategy: dict | None = None

    # ------------------------------------------------------------------
    # Strategy management helpers
    # ------------------------------------------------------------------
    def list_strategies(self) -> list[str]:
        """Return all available strategy names sorted alphabetically."""
        return self.strategy_manager.list_strategies()

    def load_strategy(self, name: str) -> dict:
        """Load ``name`` and set it as the active strategy."""
        data = self.strategy_manager.load_strategy(name)
        self.current_strategy_name = name
        self.current_strategy = data
        return data

    def train(self, data_dir: str) -> str:
        """Train a new strategy from SGF files in ``data_dir``.

        The newly created strategy will also become the active strategy.
        """
        name = self.auto_learner.train_and_save(data_dir)
        self.load_strategy(name)
        return name

    # ------------------------------------------------------------------
    # Decision making
    # ------------------------------------------------------------------
    def decide_move(self, board: list[list[int]], color: str) -> tuple[int, int] | None:
        """Return a very naive move decision for ``color`` on ``board``.

        The current implementation simply selects the first empty point found
        when scanning the board from the top left.  This is sufficient for
        tests and placeholders until proper models are implemented.
        """

        size = len(board)
        for y in range(size):
            for x in range(size):
                if board[y][x] == 0:
                    return x, y
        return None
