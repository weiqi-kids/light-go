"""Manage loading and saving of strategy models."""
from __future__ import annotations

import os
import pickle
from typing import Dict, List


class StrategyManager:
    """Simple manager for strategy files stored as pickles."""

    def __init__(self, strategies_path: str) -> None:
        self.strategies_path = strategies_path
        os.makedirs(self.strategies_path, exist_ok=True)

    def list_strategies(self) -> List[str]:
        """Return a sorted list of available strategy names."""
        names = []
        for fname in os.listdir(self.strategies_path):
            if fname.endswith(".pkl"):
                names.append(os.path.splitext(fname)[0])
        return sorted(set(names))

    def load_strategy(self, name: str) -> Dict:
        """Load a strategy by name."""
        path = os.path.join(self.strategies_path, f"{name}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_strategy(self, name: str, data: Dict) -> None:
        """Save strategy data to both .pkl and .pt files."""
        pkl_path = os.path.join(self.strategies_path, f"{name}.pkl")
        pt_path = os.path.join(self.strategies_path, f"{name}.pt")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)
        with open(pt_path, "wb") as f:
            pickle.dump(data, f)
