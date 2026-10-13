"""Tests for core.sample_store (LightGo NPZ training sample store).

Covers schema write/read roundtrip, each directory loading mode
(memory / streaming / batch), NaN/Inf rejection and the in-memory
ArraySampleDataset used by SupervisedDataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from core.muzero.trainer import SupervisedDataset
from core.sample_store import (
    INPUTS_KEY,
    POLICY_KEY,
    VALUE_KEY,
    ArraySampleDataset,
    open_sample_dataset,
    read_samples,
    write_samples,
)

SampleArrays = Tuple[np.ndarray, np.ndarray, np.ndarray]

SIZE = 5  # small board keeps files tiny; schema is size-agnostic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_samples() -> Callable[..., SampleArrays]:
    """Factory: deterministic sample arrays; value[i] == offset + i."""
    def _create(n: int, offset: int = 0, size: int = SIZE) -> SampleArrays:
        inputs = np.full((n, 2, size, size), 0.5, dtype=np.float32)
        policy = np.full((n, size * size + 1), 1.0 / (size * size + 1), dtype=np.float32)
        value = (offset + np.arange(n, dtype=np.float32)).reshape(n, 1)
        return inputs, policy, value
    return _create


@pytest.fixture
def temp_npz_dir(tmp_path: Path, make_samples: Callable[..., SampleArrays]) -> Path:
    """Directory with 3 files x 4 samples (values 0..11)."""
    for i in range(3):
        write_samples(tmp_path / f"part{i}.npz", *make_samples(4, offset=4 * i))
    return tmp_path


def _values(dataset) -> list:
    return sorted(float(dataset[i]["value"]) for i in range(len(dataset)))


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------

class TestSampleStoreSchema:
    """write_samples / read_samples own the NPZ schema."""

    def test_roundtrip(self, tmp_path: Path, make_samples: Callable[..., SampleArrays]):
        """Written samples read back identical, with the documented keys."""
        inputs, policy, value = make_samples(3)
        path = tmp_path / "s.npz"

        assert write_samples(path, inputs, policy, value) == 3

        with np.load(path) as data:
            assert set(data.files) == {INPUTS_KEY, POLICY_KEY, VALUE_KEY}
            assert data[VALUE_KEY].shape == (3, 1)
        (r_inputs, r_policy, r_value), rejected = read_samples(path)
        assert rejected == 0
        np.testing.assert_array_equal(r_inputs, inputs)
        np.testing.assert_array_equal(r_policy, policy)
        np.testing.assert_array_equal(r_value, value.reshape(3))

    def test_write_accepts_flat_values(self, tmp_path: Path,
                                       make_samples: Callable[..., SampleArrays]):
        """Value targets of shape (N,) are stored as (N, 1)."""
        inputs, policy, value = make_samples(2)
        write_samples(tmp_path / "s.npz", inputs, policy, value.reshape(2))

        with np.load(tmp_path / "s.npz") as data:
            assert data[VALUE_KEY].shape == (2, 1)

    @pytest.mark.parametrize(
        "bad",
        ["policy_width", "value_count", "input_rank"],
        ids=["policy_width", "value_count", "input_rank"],
    )
    def test_write_rejects_bad_shape(self, tmp_path: Path,
                                     make_samples: Callable[..., SampleArrays], bad: str):
        """Shape mismatches raise instead of producing a malformed file."""
        inputs, policy, value = make_samples(2)
        if bad == "policy_width":
            policy = policy[:, :-1]
        elif bad == "value_count":
            value = value[:1]
        else:
            inputs = inputs[0]

        with pytest.raises(ValueError):
            write_samples(tmp_path / "s.npz", inputs, policy, value)
        assert not (tmp_path / "s.npz").exists()


class TestSampleStoreLoadingModes:
    """open_sample_dataset in each loading mode."""

    @pytest.mark.parametrize("mode", ["memory", "streaming", "batch"],
                             ids=["memory", "streaming", "batch"])
    def test_loads_all_samples(self, temp_npz_dir: Path, mode: str):
        """Every mode exposes all 12 samples in the trainer's dict format."""
        dataset = open_sample_dataset(temp_npz_dir, mode)

        assert len(dataset) == 12
        assert _values(dataset) == [float(v) for v in range(12)]
        item = dataset[0]
        assert set(item) == {"features", "policy", "value"}
        assert item["features"].shape == (2, SIZE, SIZE)
        assert item["policy"].shape == (SIZE * SIZE + 1,)
        assert item["value"].shape == ()
        assert item["features"].dtype == torch.float32

    @pytest.mark.parametrize("mode", ["memory", "streaming", "batch"],
                             ids=["memory", "streaming", "batch"])
    def test_dataloader_batches(self, temp_npz_dir: Path, mode: str):
        """Default collation produces the batch shapes the trainer expects."""
        batch = next(iter(DataLoader(open_sample_dataset(temp_npz_dir, mode), batch_size=5)))

        assert batch["features"].shape == (5, 2, SIZE, SIZE)
        assert batch["policy"].shape == (5, SIZE * SIZE + 1)
        assert batch["value"].shape == (5,)

    def test_memory_max_files(self, temp_npz_dir: Path):
        """max_files limits memory mode to the first files in sorted order."""
        dataset = open_sample_dataset(temp_npz_dir, "memory", max_files=1)

        assert _values(dataset) == [0.0, 1.0, 2.0, 3.0]

    def test_memory_refresh_noop(self, temp_npz_dir: Path):
        """Memory mode never reloads."""
        assert open_sample_dataset(temp_npz_dir, "memory").refresh() == 0

    def test_streaming_refresh_picks_up_new_files(
        self, temp_npz_dir: Path, make_samples: Callable[..., SampleArrays]
    ):
        """Streaming refresh rescans and reports newly added samples."""
        dataset = open_sample_dataset(temp_npz_dir, "streaming", cache_size=1)
        assert dataset.refresh() == 0

        write_samples(temp_npz_dir / "part9.npz", *make_samples(2, offset=100))

        assert dataset.refresh() == 2
        assert len(dataset) == 14
        assert 101.0 in _values(dataset)

    def test_batch_limits(self, temp_npz_dir: Path):
        """Batch mode honours num_files and max_samples; refresh redraws."""
        dataset = open_sample_dataset(temp_npz_dir, "batch", num_files=2, max_samples=6)

        assert len(dataset) == 6
        assert dataset.refresh() == 6

    def test_batch_empty_dir_raises(self, tmp_path: Path):
        """Batch mode refuses an empty directory."""
        with pytest.raises(ValueError):
            open_sample_dataset(tmp_path, "batch")

    def test_unknown_mode_raises(self, temp_npz_dir: Path):
        """Unknown loading mode is rejected."""
        with pytest.raises(ValueError):
            open_sample_dataset(temp_npz_dir, "lazy")

    def test_skips_unreadable_file(self, temp_npz_dir: Path):
        """A corrupt or foreign NPZ is skipped, not fatal."""
        (temp_npz_dir / "zz_corrupt.npz").write_bytes(b"not a zip")
        np.savez(temp_npz_dir / "zz_foreign.npz", other=np.zeros(3))

        for mode in ("memory", "streaming", "batch"):
            assert len(open_sample_dataset(temp_npz_dir, mode)) == 12


class TestSampleStoreNaNRejection:
    """Samples with NaN/Inf are dropped at load time."""

    @pytest.fixture
    def nan_npz_dir(self, tmp_path: Path, make_samples: Callable[..., SampleArrays]) -> Path:
        """One file of 5 samples: #1 NaN input, #2 Inf policy, #3 NaN value."""
        inputs, policy, value = make_samples(5)
        inputs[1, 0, 0, 0] = np.nan
        policy[2, 3] = np.inf
        value[3, 0] = np.nan
        write_samples(tmp_path / "bad.npz", inputs, policy, value)
        return tmp_path

    def test_read_samples_rejects(self, nan_npz_dir: Path):
        """read_samples drops the bad samples and counts them."""
        (inputs, policy, value), rejected = read_samples(nan_npz_dir / "bad.npz")

        assert rejected == 3
        assert value.tolist() == [0.0, 4.0]

    @pytest.mark.parametrize("mode", ["memory", "streaming", "batch"],
                             ids=["memory", "streaming", "batch"])
    def test_dataset_rejects(self, nan_npz_dir: Path, mode: str,
                             caplog: pytest.LogCaptureFixture):
        """Every loading mode keeps only finite samples and logs the count."""
        with caplog.at_level("WARNING", logger="core.sample_store"):
            dataset = open_sample_dataset(nan_npz_dir, mode)

        assert _values(dataset) == [0.0, 4.0]
        assert "3" in caplog.text and "NaN/Inf" in caplog.text
        for i in range(len(dataset)):
            item = dataset[i]
            assert all(torch.isfinite(t).all() for t in item.values())


class TestArraySampleDataset:
    """In-memory dataset and its SupervisedDataset alias."""

    def test_supervised_dataset_format(self, make_samples: Callable[..., SampleArrays]):
        """SupervisedDataset keeps its (features, policies, values) signature."""
        inputs, policy, value = make_samples(4)
        dataset = SupervisedDataset(inputs, policy, value.reshape(4))

        assert isinstance(dataset, ArraySampleDataset)
        assert len(dataset) == 4
        assert float(dataset[2]["value"]) == 2.0
        assert dataset[0]["value"].shape == ()

    def test_drops_non_finite(self, make_samples: Callable[..., SampleArrays]):
        """Non-finite samples are dropped from in-memory arrays too."""
        inputs, policy, value = make_samples(3)
        value[0, 0] = np.inf

        assert len(ArraySampleDataset(inputs, policy, value)) == 2
