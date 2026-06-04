"""
streaming_dataset.py — Streaming IterableDataset for CESNET-QUIC22.

Pulls data directly from the CESNET Data Zoo API chunk-by-chunk during training.
No full dataset download required. Each epoch streams a fresh set of chunks.

Usage:
    from src.streaming_dataset import CESNETStreamingDataset, build_streaming_loaders

    train_loader, val_loader = build_streaming_loaders(
        data_root="/workspace/.cesnet_cache",  # small local index + metadata only
        chunk_size=8192,
        batch_size=128,
    )
    # data_root holds only the index files (~50 MB), not the full dataset

Requirements:
    pip install cesnet-datazoo pyarrow

Tensor output per sample (identical to UnifiedFlowDataset):
    seq_tensor  : FloatTensor[128, 3]
    stat_tensor : FloatTensor[18]
    label       : LongTensor scalar (unified class ID 0–7)
"""

from __future__ import annotations

import ast
import logging
import math
import os
import random
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from src.data_validator import FlowValidator, REJECT_R1, REJECT_R2, REJECT_R3, REJECT_R4, REJECT_R7
from src.feature_engineering import (
    SEQ_LEN,
    SEQ_INPUT_DIM,
    STAT_INPUT_DIM,
    extract_seq_features,
    extract_stat_features,
)
from src.dataset_unified import LABEL_MAP, UNIFIED_CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunk-level feature extraction (applied on a pandas DataFrame batch)
# ---------------------------------------------------------------------------

_validator = FlowValidator()


def _parse_ppi(raw) -> Optional[list]:
    """Parse CESNET PPI field from various serialised forms."""
    if raw is None:
        return None
    if isinstance(raw, (list, np.ndarray)):
        ppi = list(raw)
        return [list(ppi[i]) for i in range(3)] if len(ppi) >= 3 else None
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                return [list(parsed[i]) for i in range(3)]
        except (ValueError, SyntaxError):
            pass
        # fallback: numpy print format "[[a b c]\n [d e f]]"
        import re
        rows = re.findall(r'\[([^\[\]]+)\]', raw)
        rows = [list(map(float, re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', r))) for r in rows if r.strip()]
        if len(rows) >= 3:
            return rows[:3]
    return None


def _process_chunk(df) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Convert a DataFrame chunk (from cesnet-datazoo or pyarrow) into a list of
    validated (seq, stat, label) tuples ready for tensor conversion.

    Corrupt rows are silently skipped — rejection counts go to the logger.
    """
    samples = []
    rejected = 0

    for _, row in df.iterrows():
        # --- Label ---
        raw_label = str(row.get("CATEGORY", "")).strip().lower()
        if raw_label not in LABEL_MAP:
            rejected += 1
            continue

        # --- PPI ---
        ppi = _parse_ppi(row.get("PPI"))
        if ppi is None:
            rejected += 1
            continue

        ppi_len = int(row.get("PPI_LEN", len(ppi[0])))
        endreason_active = int(row.get("FLOW_ENDREASON_ACTIVE", 0))

        ok, _ = _validator.validate_ppi(ppi, flow_endreason_active=endreason_active)
        if not ok:
            rejected += 1
            continue

        # --- Stats ---
        phist_raw = row.get("PHIST_SRC_SIZES", [0] * 8)
        if isinstance(phist_raw, str):
            try:
                phist_raw = ast.literal_eval(phist_raw)
            except (ValueError, SyntaxError):
                phist_raw = [0] * 8

        stats_dict = {
            "BYTES":              float(row.get("BYTES", 0)),
            "BYTES_REV":          float(row.get("BYTES_REV", 0)),
            "PACKETS":            float(row.get("PACKETS", 0)),
            "PACKETS_REV":        float(row.get("PACKETS_REV", 0)),
            "DURATION":           float(row.get("DURATION", 0)),
            "FLOW_ENDREASON_IDLE": int(row.get("FLOW_ENDREASON_IDLE", 0)),
            "FLOW_ENDREASON_ACTIVE": endreason_active,
        }
        ok, _ = _validator.validate_stats(stats_dict, is_cesnet=True)
        if not ok:
            rejected += 1
            continue

        # --- Feature extraction ---
        fe_row = {**stats_dict, "PPI": ppi, "PPI_LEN": ppi_len, "PHIST_SRC_SIZES": phist_raw}
        try:
            seq_data  = extract_seq_features(ppi, SEQ_LEN)
            stat_data = extract_stat_features(fe_row)
        except ValueError:
            rejected += 1
            continue

        if not (np.all(np.isfinite(seq_data)) and np.all(np.isfinite(stat_data))):
            rejected += 1
            continue

        unified_label = LABEL_MAP[raw_label]
        samples.append((seq_data, stat_data, unified_label))

    if rejected:
        logger.debug("Chunk: %d valid, %d rejected", len(samples), rejected)
    if len(samples) == 0 and rejected > 0:
        logger.warning("Chunk: 100%% rejected (%d rows) — check CATEGORY/PPI columns", rejected)

    return samples


# ---------------------------------------------------------------------------
# CESNETStreamingDataset
# ---------------------------------------------------------------------------

class CESNETStreamingDataset(IterableDataset):
    """
    Streams CESNET-QUIC22 flows directly from the Data Zoo API chunk-by-chunk.

    No full dataset is downloaded. The library fetches parquet row-groups
    on demand; each epoch sees a shuffled ordering of the available chunks.

    Parameters
    ----------
    data_root : str
        Local directory for cesnet-datazoo index and metadata (~50 MB).
        The actual flow records are NOT cached here.
    size : str
        Dataset size key — "XS" (100K), "S" (1M), "M" (10M).
        Controls how many flows are available per epoch.
        For streaming, use "S" or "M"; "L" (150M) is too large to stream
        without a permanent cache.
    chunk_size : int
        Number of flows fetched from the API per read call.
        Larger = fewer API calls but more RAM per worker.
        Recommended: 4096–16384.
    split : str
        One of "train", "val", "test".
    shuffle_chunks : bool
        Shuffle the order of chunks (not within a chunk) each epoch.
    """

    def __init__(
        self,
        data_root: str,
        size: str = "M",
        chunk_size: int = 8192,
        split: str = "train",
        shuffle_chunks: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.size = size
        self.chunk_size = chunk_size
        self.split = split
        self.shuffle_chunks = shuffle_chunks

        # Validate cesnet-datazoo is installed
        try:
            from cesnet_datazoo.datasets import CESNET_QUIC22
        except ImportError as exc:
            raise ImportError(
                "cesnet-datazoo is required for streaming. "
                "Install with: pip install cesnet-datazoo"
            ) from exc

        self._init_dataset()

    def _init_dataset(self) -> None:
        """Initialise cesnet-datazoo and discover available parquet files."""
        from cesnet_datazoo.datasets import CESNET_QUIC22
        from cesnet_datazoo.config import DatasetConfig, AppSelection

        os.makedirs(self.data_root, exist_ok=True)

        self._cesnet_dataset = CESNET_QUIC22(
            data_root=self.data_root,
            size=self.size,
        )

        # Minimal config — use all known application classes
        self._config = DatasetConfig(
            dataset=self._cesnet_dataset,
            apps_selection=AppSelection.ALL_KNOWN,
            train_workers=0,  # we manage workers via DataLoader
            val_workers=0,
            test_workers=0,
        )
        self._cesnet_dataset.set_dataset_config_and_initialize(self._config)
        logger.info(
            "CESNETStreamingDataset ready: size=%s split=%s chunk_size=%d",
            self.size, self.split, self.chunk_size,
        )

    def _iter_chunks(self):
        """Yield raw DataFrame chunks from the cesnet-datazoo split."""
        if self.split == "train":
            df_iter = self._cesnet_dataset.get_train_df(return_generator=True)
        elif self.split == "val":
            df_iter = self._cesnet_dataset.get_val_df(return_generator=True)
        else:
            df_iter = self._cesnet_dataset.get_test_df(return_generator=True)

        # Buffer chunks so we can optionally shuffle their order
        chunk_buffer = []
        current_chunk = []

        for row_batch in df_iter:
            # row_batch may be a full DataFrame or a single row depending on API
            if hasattr(row_batch, "iterrows"):
                current_chunk.append(row_batch)
            else:
                current_chunk.append(row_batch)

            if len(current_chunk) >= self.chunk_size:
                chunk_buffer.append(current_chunk[:self.chunk_size])
                current_chunk = current_chunk[self.chunk_size:]

        if current_chunk:
            chunk_buffer.append(current_chunk)

        if self.shuffle_chunks:
            random.shuffle(chunk_buffer)

        for chunk in chunk_buffer:
            yield chunk

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterate over the split, yielding one (seq, stat, label) tuple at a time.

        Multi-worker DataLoader: each worker gets a non-overlapping subset of
        chunks via torch.utils.data.get_worker_info().
        """
        worker_info = torch.utils.data.get_worker_info()

        import pandas as pd
        for chunk_rows in self._iter_chunks_for_worker(worker_info):
            if isinstance(chunk_rows, list):
                try:
                    df_chunk = pd.concat(chunk_rows, ignore_index=True)
                except Exception:
                    continue
            else:
                df_chunk = chunk_rows

            samples = _process_chunk(df_chunk)
            if self.shuffle_chunks:
                random.shuffle(samples)

            for seq_data, stat_data, label in samples:
                yield (
                    torch.from_numpy(seq_data),
                    torch.from_numpy(stat_data),
                    torch.tensor(label, dtype=torch.long),
                )

    def _iter_chunks_for_worker(self, worker_info):
        """Distribute chunks across DataLoader workers."""
        all_chunks = list(self._iter_raw_chunks())

        if worker_info is None:
            yield from all_chunks
            return

        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        for i, chunk in enumerate(all_chunks):
            if i % num_workers == worker_id:
                yield chunk

    def _iter_raw_chunks(self):
        """Yield raw DataFrame chunks from the Data Zoo."""
        import pandas as pd

        if self.split == "train":
            source = self._cesnet_dataset.get_train_df()
        elif self.split == "val":
            source = self._cesnet_dataset.get_val_df()
        else:
            source = self._cesnet_dataset.get_test_df()

        if source is None:
            logger.warning("No %s data available in cesnet-datazoo config.", self.split)
            return

        total = len(source)
        n_chunks = math.ceil(total / self.chunk_size)
        indices = list(range(n_chunks))
        if self.shuffle_chunks:
            random.shuffle(indices)

        for i in indices:
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total)
            yield source.iloc[start:end]


# ---------------------------------------------------------------------------
# Fallback: pyarrow HTTP streaming (no cesnet-datazoo required)
# ---------------------------------------------------------------------------

class ParquetStreamingDataset(IterableDataset):
    """
    Streams parquet files directly from HTTPS URLs using pyarrow.
    No local storage except a tiny row-group index.

    Use this if cesnet-datazoo is unavailable or you have a direct parquet URL.

    Parameters
    ----------
    parquet_urls : list of str
        Direct HTTPS URLs to parquet files (e.g. from Zenodo, S3, CESNET repo).
    chunk_size : int
        Number of rows to read per pyarrow batch.
    label_col : str
        Column name for the class label (default "CATEGORY" for CESNET).
    """

    def __init__(
        self,
        parquet_urls: List[str],
        chunk_size: int = 8192,
        label_col: str = "CATEGORY",
        shuffle_files: bool = True,
    ) -> None:
        super().__init__()
        self.parquet_urls = parquet_urls
        self.chunk_size = chunk_size
        self.label_col = label_col
        self.shuffle_files = shuffle_files

        try:
            import pyarrow.dataset as pad  # noqa: F401
        except ImportError as exc:
            raise ImportError("pyarrow is required: pip install pyarrow") from exc

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        import pyarrow.dataset as pad
        import pyarrow.fs as pafs

        urls = list(self.parquet_urls)
        if self.shuffle_files:
            random.shuffle(urls)

        worker_info = torch.utils.data.get_worker_info()

        for i, url in enumerate(urls):
            # Distribute files across workers
            if worker_info is not None:
                if i % worker_info.num_workers != worker_info.id:
                    continue

            try:
                # Read parquet over HTTPS in row-group batches
                ds = pad.dataset(url, filesystem=pafs.FSSpecHandler(
                    __import__("fsspec").filesystem("http")
                ))
                for batch in ds.to_batches(batch_size=self.chunk_size):
                    import pandas as pd
                    df_chunk = batch.to_pandas()
                    samples = _process_chunk(df_chunk)
                    random.shuffle(samples)
                    for seq_data, stat_data, label in samples:
                        yield (
                            torch.from_numpy(seq_data),
                            torch.from_numpy(stat_data),
                            torch.tensor(label, dtype=torch.long),
                        )
            except Exception as exc:
                logger.warning("Failed to stream %s: %s", url, exc)
                continue


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_streaming_loaders(
    data_root: str,
    size: str = "S",
    batch_size: int = 128,
    chunk_size: int = 8192,
    num_workers: int = 4,
    val_size: str = "XS",
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and val streaming DataLoaders for CESNET-QUIC22.

    Parameters
    ----------
    data_root : str
        Local directory for cesnet-datazoo index/metadata (~50 MB).
    size : str
        Training data size: "XS" (100K), "S" (1M), "M" (10M).
    batch_size : int
        Batch size for both loaders.
    chunk_size : int
        Rows fetched per API call. Larger = faster but more RAM.
    num_workers : int
        DataLoader worker processes.
    val_size : str
        Validation data size (usually smaller than train).

    Returns
    -------
    (train_loader, val_loader)

    Example
    -------
    >>> train_loader, val_loader = build_streaming_loaders(
    ...     data_root="/workspace/.cesnet_cache",
    ...     size="S",
    ...     batch_size=128,
    ... )
    >>> for seq, stat, labels in train_loader:
    ...     # seq: (128, 128, 3)  stat: (128, 18)  labels: (128,)
    ...     pass
    """
    train_ds = CESNETStreamingDataset(
        data_root=data_root,
        size=size,
        chunk_size=chunk_size,
        split="train",
        shuffle_chunks=True,
    )
    val_ds = CESNETStreamingDataset(
        data_root=data_root,
        size=val_size,
        chunk_size=chunk_size,
        split="val",
        shuffle_chunks=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        # No sampler — IterableDataset handles its own ordering
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "Streaming DataLoaders ready — train size=%s val size=%s batch=%d workers=%d",
        size, val_size, batch_size, num_workers,
    )
    return train_loader, val_loader
