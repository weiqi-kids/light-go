"""Training sample store: the single owner of the LightGo NPZ schema.

Every LightGo training file is an ``.npz`` with three arrays:

``inputs``          float32 ``(N, C, H, W)``  -- encoded board features
``policy_targets``  float32 ``(N, H*W + 1)``  -- move distribution (last = pass)
``value_targets``   float32 ``(N, 1)``        -- game outcome in [-1, 1]

Writers call :func:`write_samples`; readers call :func:`open_sample_dataset`
(directory of files, one of three loading modes) or wrap arrays already in
memory with :class:`ArraySampleDataset`.  All datasets yield the dict the
MuZero supervised trainer consumes::

    {'features': Tensor(C, H, W), 'policy': Tensor(H*W + 1), 'value': Tensor()}

Samples containing NaN/Inf in any array are dropped at load time and counted
in the log, so bad data never reaches the trainer.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

INPUTS_KEY = "inputs"
POLICY_KEY = "policy_targets"
VALUE_KEY = "value_targets"

LOADING_MODES = ("batch", "streaming", "memory")

PathLike = Union[str, Path]
SampleArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Schema: validation, write, read
# ---------------------------------------------------------------------------

def _normalize(inputs: np.ndarray, policy_targets: np.ndarray,
               value_targets: np.ndarray) -> SampleArrays:
    """Validate shapes and return float32 arrays with value shape (N,)."""
    inputs = np.asarray(inputs, dtype=np.float32)
    policies = np.asarray(policy_targets, dtype=np.float32)
    values = np.asarray(value_targets, dtype=np.float32)

    if inputs.ndim != 4:
        raise ValueError(f"{INPUTS_KEY} must be (N, C, H, W), got {inputs.shape}")
    n, _, h, w = inputs.shape
    if policies.shape != (n, h * w + 1):
        raise ValueError(
            f"{POLICY_KEY} must be ({n}, {h * w + 1}), got {policies.shape}")
    if values.shape not in ((n,), (n, 1)):
        raise ValueError(f"{VALUE_KEY} must be ({n}, 1), got {values.shape}")
    return inputs, policies, values.reshape(n)


def _finite_mask(arrays: SampleArrays) -> np.ndarray:
    """Per-sample mask: True where every value of the sample is finite."""
    inputs, policies, values = arrays
    n = len(inputs)
    return (np.isfinite(inputs.reshape(n, -1)).all(axis=1)
            & np.isfinite(policies).all(axis=1)
            & np.isfinite(values))


def write_samples(path: PathLike, inputs: np.ndarray,
                  policy_targets: np.ndarray, value_targets: np.ndarray) -> int:
    """Write samples to ``path`` in the LightGo NPZ schema.

    Returns the number of samples written.  Raises ``ValueError`` on a shape
    mismatch so a malformed file is never produced.
    """
    inputs, policies, values = _normalize(inputs, policy_targets, value_targets)
    np.savez_compressed(
        path,
        **{INPUTS_KEY: inputs, POLICY_KEY: policies, VALUE_KEY: values[:, None]},
    )
    return len(inputs)


def read_samples(path: PathLike) -> Tuple[SampleArrays, int]:
    """Read one NPZ file, dropping non-finite samples.

    Returns ``((inputs, policies, values), rejected_count)`` where ``values``
    has shape ``(N,)``.  Raises on a missing key or shape mismatch.
    """
    with np.load(path) as data:
        arrays = _normalize(data[INPUTS_KEY], data[POLICY_KEY], data[VALUE_KEY])
    mask = _finite_mask(arrays)
    rejected = int(len(mask) - mask.sum())
    if rejected:
        arrays = tuple(a[mask] for a in arrays)  # type: ignore[assignment]
    return arrays, rejected


def _read_dir(files: List[Path], max_samples: Optional[int] = None
              ) -> Tuple[List[SampleArrays], int, int]:
    """Read files in order until ``max_samples`` is reached.

    Returns ``(chunks, files_read, rejected)``; unreadable files are skipped.
    """
    chunks: List[SampleArrays] = []
    rejected = failed = total = 0
    for f in files:
        try:
            arrays, bad = read_samples(f)
        except Exception as e:  # corrupt / foreign file: skip, keep training
            failed += 1
            logger.debug(f"載入 {f} 失敗: {e}")
            continue
        rejected += bad
        if len(arrays[0]):
            chunks.append(arrays)
            total += len(arrays[0])
        if max_samples is not None and total >= max_samples:
            break
    _log_skips(rejected, failed)
    return chunks, len(chunks), rejected


def _log_skips(rejected: int, failed: int) -> None:
    if rejected:
        logger.warning(f"略過 {rejected} 個含 NaN/Inf 的樣本")
    if failed:
        logger.warning(f"略過 {failed} 個無法讀取的 NPZ 檔案")


def _concat(chunks: List[SampleArrays], limit: Optional[int] = None) -> SampleArrays:
    if not chunks:
        return (np.zeros((0, 0, 0, 0), np.float32), np.zeros((0, 0), np.float32),
                np.zeros((0,), np.float32))
    return tuple(np.concatenate([c[k] for c in chunks])[:limit]  # type: ignore[return-value]
                 for k in range(3))


def _item(features: np.ndarray, policy: np.ndarray, value: float
          ) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.from_numpy(np.ascontiguousarray(features)),
        "policy": torch.from_numpy(np.ascontiguousarray(policy)),
        "value": torch.tensor(float(value), dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ArraySampleDataset(Dataset):
    """Dataset over sample arrays already in memory.

    ``refresh()`` is a no-op returning 0, so callers can treat every sample
    dataset uniformly.
    """

    def __init__(self, inputs: np.ndarray, policy_targets: np.ndarray,
                 value_targets: np.ndarray, *, drop_non_finite: bool = True):
        arrays = _normalize(inputs, policy_targets, value_targets)
        if drop_non_finite and len(arrays[0]):
            mask = _finite_mask(arrays)
            _log_skips(int(len(mask) - mask.sum()), 0)
            arrays = tuple(a[mask] for a in arrays)  # type: ignore[assignment]
        self.features, self.policies, self.values = arrays

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return _item(self.features[idx], self.policies[idx], self.values[idx])

    def refresh(self) -> int:
        """Nothing to reload for in-memory data."""
        return 0


class SampleDataset(Dataset):
    """Dataset over a directory of LightGo NPZ files.

    Create through :func:`open_sample_dataset`.  ``refresh()`` is called
    between epochs; it returns the number of samples newly made available
    (0 = unchanged, so the caller can keep its DataLoader).
    """

    def __init__(self, data_dir: PathLike, mode: str, *, max_files: Optional[int],
                 num_files: int, max_samples: int, cache_size: int):
        if mode not in LOADING_MODES:
            raise ValueError(f"mode 必須是 {LOADING_MODES} 之一，收到 {mode!r}")
        self.data_dir = Path(data_dir)
        self.mode = mode
        self._max_files = max_files
        self._num_files = num_files
        self._max_samples = max_samples
        self._cache_size = cache_size

        # memory / batch: samples held as arrays
        self._arrays: SampleArrays = _concat([])
        # streaming: (file_idx, sample_idx) index over valid samples + LRU cache
        self._files: List[Path] = []
        self._index: List[Tuple[int, int]] = []
        self._cache: Dict[int, SampleArrays] = {}
        self._cache_order: List[int] = []

        files = self._scan()
        if mode == "batch" and not files:
            raise ValueError(f"在 {self.data_dir} 找不到任何 NPZ 檔案")
        logger.info(f"找到 {len(files)} 個 NPZ 檔案（{mode} 模式）")
        self._load()

    # -- public ------------------------------------------------------------

    def __len__(self) -> int:
        if self.mode == "streaming":
            return len(self._index)
        return len(self._arrays[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "streaming":
            file_idx, sample_idx = self._index[idx]
            inputs, policies, values = self._cached(file_idx)
            return _item(inputs[sample_idx], policies[sample_idx], values[sample_idx])
        inputs, policies, values = self._arrays
        return _item(inputs[idx], policies[idx], values[idx])

    def refresh(self) -> int:
        """Prepare data for the next epoch.

        - ``batch``: draw a new random subset of files; returns samples loaded.
        - ``streaming``: rescan the directory for new files; returns samples added.
        - ``memory``: no-op; returns 0.
        """
        if self.mode == "memory":
            return 0
        if self.mode == "batch":
            return self._load()
        old = len(self._index)
        self._cache.clear()
        self._cache_order.clear()
        self._load()
        added = len(self._index) - old
        if added > 0:
            logger.info(f"新增 {added} 個樣本（總計 {len(self._index)}）")
        return max(added, 0)

    # -- internals ---------------------------------------------------------

    def _scan(self) -> List[Path]:
        files = sorted(self.data_dir.glob("*.npz"))
        if self._max_files:
            files = files[:self._max_files]
        self._files = files
        return files

    def _load(self) -> int:
        if self.mode == "memory":
            chunks, _, _ = _read_dir(self._files)
            self._arrays = _concat(chunks)
        elif self.mode == "batch":
            picked = random.sample(self._files, min(self._num_files, len(self._files)))
            chunks, n_read, _ = _read_dir(picked, self._max_samples)
            if not chunks:
                raise ValueError("無法載入任何資料")
            self._arrays = _concat(chunks, self._max_samples)
            logger.info(f"已載入 {len(self)} 個樣本（從 {n_read} 個檔案）")
            return len(self)
        else:
            self._build_index()
        logger.info(f"總共 {len(self)} 個訓練樣本（{self.mode} 模式）")
        return len(self)

    def _build_index(self) -> None:
        """Index valid samples; reads each file once to screen out NaN/Inf."""
        self._scan()
        self._index = []
        rejected = failed = 0
        for file_idx, f in enumerate(self._files):
            try:
                arrays = _normalize(*self._raw(f))
            except Exception as e:
                failed += 1
                logger.debug(f"掃描 {f} 失敗: {e}")
                continue
            valid = np.flatnonzero(_finite_mask(arrays))
            rejected += len(arrays[0]) - len(valid)
            self._index.extend((file_idx, int(i)) for i in valid)
            if (file_idx + 1) % 5000 == 0:
                logger.info(f"  已掃描 {file_idx + 1}/{len(self._files)} 檔案, "
                            f"{len(self._index)} 個樣本")
        _log_skips(rejected, failed)

    @staticmethod
    def _raw(path: Path) -> SampleArrays:
        with np.load(path) as data:
            return data[INPUTS_KEY], data[POLICY_KEY], data[VALUE_KEY]

    def _cached(self, file_idx: int) -> SampleArrays:
        """File arrays with LRU caching (streaming mode)."""
        if file_idx in self._cache:
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._cache[file_idx]
        arrays = _normalize(*self._raw(self._files[file_idx]))
        self._cache[file_idx] = arrays
        self._cache_order.append(file_idx)
        while len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]
        return arrays


def open_sample_dataset(
    data_dir: PathLike,
    mode: str = "batch",
    *,
    max_files: Optional[int] = None,
    num_files: int = 2000,
    max_samples: int = 200_000,
    cache_size: int = 100,
) -> SampleDataset:
    """Open a dataset over every ``*.npz`` in ``data_dir``.

    Parameters
    ----------
    mode : str
        ``"batch"`` (default): each load/refresh randomly picks ``num_files``
        files and keeps at most ``max_samples`` samples in memory.
        ``"streaming"``: index all files, read on demand with an LRU cache of
        ``cache_size`` files; ``refresh()`` picks up new files.
        ``"memory"``: load every file (first ``max_files`` if given) up front.
    """
    return SampleDataset(data_dir, mode, max_files=max_files, num_files=num_files,
                         max_samples=max_samples, cache_size=cache_size)
