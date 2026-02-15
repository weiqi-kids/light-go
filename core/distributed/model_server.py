"""Model versioning and distribution for distributed self-play.

This module manages model versions and provides file-based model sharing
between the coordinator and workers.
"""

from __future__ import annotations

import os
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a model version.

    Attributes
    ----------
    version : int
        Version number
    path : str
        Path to model file
    timestamp : float
        Unix timestamp when version was created
    metadata : Dict[str, Any]
        Additional metadata (win_rate, generation, etc.)
    """

    version: int
    path: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """Manages model versioning and distribution.

    Uses file-based model sharing:
    - Model saved to shared path
    - Workers notified via queue
    - Workers load model from path

    Parameters
    ----------
    model_dir : str
        Directory for model storage
    max_versions : int
        Maximum versions to keep (default 5)
    """

    def __init__(
        self,
        model_dir: str = "data/models/shared",
        max_versions: int = 5
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.max_versions = max_versions
        self.current_version = 0
        self.version_history: List[ModelVersion] = []

        # Load existing versions
        self._load_version_history()

    def _load_version_history(self) -> None:
        """Load existing version history from disk."""
        model_files = sorted(
            self.model_dir.glob("model_v*.pt"),
            key=lambda p: int(p.stem.split('_v')[1])
        )

        for model_file in model_files:
            version = int(model_file.stem.split('_v')[1])
            self.version_history.append(ModelVersion(
                version=version,
                path=str(model_file),
                timestamp=model_file.stat().st_mtime,
                metadata={}
            ))

        if self.version_history:
            self.current_version = self.version_history[-1].version

    def save_model(
        self,
        model: torch.nn.Module,
        genome: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save new model version.

        Parameters
        ----------
        model : torch.nn.Module
            Model to save
        genome : Optional[Any]
            ArchitectureGenome (optional)
        metadata : Optional[Dict[str, Any]]
            Additional metadata

        Returns
        -------
        str
            Path to saved model
        """
        self.current_version += 1
        filename = f"model_v{self.current_version}.pt"
        filepath = self.model_dir / filename

        # Save model state
        save_dict = {
            'model_state_dict': model.state_dict(),
            'version': self.current_version,
            'timestamp': time.time()
        }

        if genome is not None:
            try:
                save_dict['genome'] = genome.to_dict()
            except Exception:
                pass

        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, filepath)

        # Create version record
        version_info = ModelVersion(
            version=self.current_version,
            path=str(filepath),
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.version_history.append(version_info)

        # Update current model symlink/copy
        current_path = self.model_dir / "current_model.pt"
        shutil.copy2(filepath, current_path)

        # Cleanup old versions
        self._cleanup_old_versions()

        logger.info(f"Saved model v{self.current_version} to {filepath}")
        return str(filepath)

    def get_current_model_path(self) -> str:
        """Get path to current (latest) model.

        Returns
        -------
        str
            Path to current model file
        """
        current_path = self.model_dir / "current_model.pt"

        if current_path.exists():
            return str(current_path)

        # Fallback to latest versioned file
        if self.version_history:
            return self.version_history[-1].path

        raise FileNotFoundError("No model available")

    def get_model_version(self) -> int:
        """Get current model version number.

        Returns
        -------
        int
            Current version number
        """
        return self.current_version

    def load_model(
        self,
        model: torch.nn.Module,
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """Load model state into provided model.

        Parameters
        ----------
        model : torch.nn.Module
            Model to load state into
        version : Optional[int]
            Specific version to load (default: current)

        Returns
        -------
        Dict[str, Any]
            Loaded checkpoint data
        """
        if version is None:
            filepath = self.get_current_model_path()
        else:
            filepath = self.model_dir / f"model_v{version}.pt"
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Model version {version} not found")

        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Loaded model from {filepath}")
        return checkpoint

    def get_version_info(self, version: Optional[int] = None) -> Optional[ModelVersion]:
        """Get version information.

        Parameters
        ----------
        version : Optional[int]
            Version to query (default: current)

        Returns
        -------
        Optional[ModelVersion]
            Version info or None if not found
        """
        if version is None:
            version = self.current_version

        for v in self.version_history:
            if v.version == version:
                return v

        return None

    def _cleanup_old_versions(self) -> None:
        """Remove old model versions beyond max_versions."""
        while len(self.version_history) > self.max_versions:
            old_version = self.version_history.pop(0)

            try:
                Path(old_version.path).unlink()
                logger.debug(f"Removed old model v{old_version.version}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_version.path}: {e}")

    def get_all_versions(self) -> List[ModelVersion]:
        """Get all available versions.

        Returns
        -------
        List[ModelVersion]
            List of version info
        """
        return self.version_history.copy()

    def model_exists(self, version: Optional[int] = None) -> bool:
        """Check if model version exists.

        Parameters
        ----------
        version : Optional[int]
            Version to check (default: any)

        Returns
        -------
        bool
            True if model exists
        """
        if version is None:
            return len(self.version_history) > 0

        return any(v.version == version for v in self.version_history)


class ModelNotification:
    """Notification message for model updates.

    Used to notify workers of new model versions via queue.
    """

    def __init__(
        self,
        version: int,
        path: str,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.version = version
        self.path = path
        self.timestamp = timestamp
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'type': 'model_update',
            'version': self.version,
            'path': self.path,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelNotification':
        """Create from dictionary."""
        return cls(
            version=data['version'],
            path=data['path'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )
