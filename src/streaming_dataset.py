"""Streaming IterableDataset for CESNET-QUIC22 via cesnet-datazoo."""

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

from src.data_validator import FlowValidator
from src.feature_engineering import SEQ_LEN, extract_seq_features, extract_stat_features
from src.dataset_unified import LABEL_MAP

logger = logging.getLogger(__name__)

_validator = FlowValidator()

# cesnet-datazoo v0.2.0 exposes APP as int, not CATEGORY text
_CESNET_CAT_MAP: dict = {
    "streaming media": 0, "music": 1, "games": 2, "social": 3,
    "file sharing": 4, "search": 5, "blogs & news": 5, "e-commerce": 5,
    "information systems": 5, "instant messaging": 6, "mail": 6, "videoconferencing": 6,
}


def _build_app_int_map(cesnet_dataset) -> dict:
    import pandas as pd
    try:
        smap = pd.read_csv(cesnet_dataset.servicemap_path)
        tag_to_cat = {str(r["Tag"]).lower(): str(r["Service Category"]).lower() for _, r in smap.iterrows()}
        result = {int(k): _CESNET_CAT_MAP[tag_to_cat.get(str(v).lower(), "")]
                  for k, v in cesnet_dataset._tables_app_enum.items()
                  if tag_to_cat.get(str(v).lower(), "") in _CESNET_CAT_MAP}
        logger.info("APP_INT_MAP: %d apps → %d classes", len(result), len(set(result.values())))
        return result
    except Exception as exc:
        logger.warning("Could not build APP_INT_MAP: %s", exc)
        return {}


def _parse_ppi(raw) -> Optional[list]:
    if raw is None:
        return None
    if isinstance(raw, (list, np.ndarray)):
        arr = np.asarray(raw)
        if arr.ndim == 2:
            if arr.shape[0] == 3:
                return [list(arr[0]), list(arr[1]), list(arr[2])]
            if arr.shape[1] == 3:  # (N,3) transposed
                return [list(arr[:, 0]), list(arr[:, 1]), list(arr[:, 2])]
        ppi = list(raw)
        return [list(ppi[i]) for i in range(3)] if len(ppi) >= 3 else None
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                return [list(parsed[i]) for i in range(3)]
        except (ValueError, SyntaxError):
            pass
        import re
        rows = re.findall(r'\[([^\[\]]+)\]', raw)
        rows = [list(map(float, re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', r))) for r in rows if r.strip()]
        if len(rows) >= 3:
            return rows[:3]
    return None


# Classes held out from training to prove zero-day generalization:
#   1 = audio/music streaming,  2 = gaming
_ZERO_DAY_CLASSES: frozenset = frozenset({1, 2})


def _process_chunk(df, app_int_map=None, split: str = "train") -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    samples, rejected = [], 0

    for _, row in df.iterrows():
        if app_int_map is not None:
            app_int = int(row.get("APP", -1))
            if app_int not in app_int_map:
                rejected += 1
                continue
            unified_label = app_int_map[app_int]
        else:
            raw_label = str(row.get("CATEGORY", "")).strip().lower()
            if raw_label not in LABEL_MAP:
                rejected += 1
                continue
            unified_label = LABEL_MAP[raw_label]

        # Patch 4: strictly exclude zero-day classes from training split
        if split == "train" and unified_label in _ZERO_DAY_CLASSES:
            rejected += 1
            continue

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

        phist_raw = row.get("PHIST_SRC_SIZES", [0] * 8)
        if isinstance(phist_raw, str):
            try:
                phist_raw = ast.literal_eval(phist_raw)
            except (ValueError, SyntaxError):
                phist_raw = [0] * 8

        stats_dict = {
            "BYTES":               float(row.get("BYTES", 0)),
            "BYTES_REV":           float(row.get("BYTES_REV", 0)),
            "PACKETS":             float(row.get("PACKETS", 0)),
            "PACKETS_REV":         float(row.get("PACKETS_REV", 0)),
            "DURATION":            float(row.get("DURATION", 0)),
            "FLOW_ENDREASON_IDLE": int(row.get("FLOW_ENDREASON_IDLE", 0)),
            "FLOW_ENDREASON_ACTIVE": endreason_active,
        }
        ok, _ = _validator.validate_stats(stats_dict, is_cesnet=True)
        if not ok:
            rejected += 1
            continue

        fe_row = {**stats_dict, "PPI": ppi, "PPI_LEN": ppi_len, "PHIST_SRC_SIZES": phist_raw}
        try:
            seq_data = extract_seq_features(ppi, SEQ_LEN)
            stat_data = extract_stat_features(fe_row)
        except ValueError:
            rejected += 1
            continue

        if not (np.all(np.isfinite(seq_data)) and np.all(np.isfinite(stat_data))):
            rejected += 1
            continue

        # Extract 5-tuple ports (fallback to 0 if column absent)
        src_port = int(row.get("SRC_PORT", row.get("SOURCE_PORT", 0)))
        dst_port = int(row.get("DST_PORT", row.get("DESTINATION_PORT", 0)))
        ports_data = np.array([src_port, dst_port], dtype=np.int64)

        samples.append((seq_data, stat_data, ports_data, unified_label))

    if rejected:
        logger.debug("Chunk: %d valid, %d rejected", len(samples), rejected)

    return samples


class CESNETStreamingDataset(IterableDataset):

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
        self._init_dataset()

    def _init_dataset(self) -> None:
        from cesnet_datazoo.datasets import CESNET_QUIC22
        from cesnet_datazoo.config import DatasetConfig, AppSelection

        os.makedirs(self.data_root, exist_ok=True)
        self._cesnet_dataset = CESNET_QUIC22(data_root=self.data_root, size=self.size)
        self._config = DatasetConfig(
            dataset=self._cesnet_dataset,
            apps_selection=AppSelection.ALL_KNOWN,
            train_workers=0,
            val_workers=0,
            test_workers=0,
        )
        self._cesnet_dataset.set_dataset_config_and_initialize(self._config)
        self._app_int_map = _build_app_int_map(self._cesnet_dataset)

    def _iter_raw_chunks(self):
        import pandas as pd
        if self.split == "train":
            source = self._cesnet_dataset.get_train_df()
        elif self.split == "val":
            source = self._cesnet_dataset.get_val_df()
        else:
            source = self._cesnet_dataset.get_test_df()

        if source is None or len(source) == 0:
            return

        indices = list(range(math.ceil(len(source) / self.chunk_size)))
        if self.shuffle_chunks:
            random.shuffle(indices)
        for i in indices:
            start = i * self.chunk_size
            yield source.iloc[start: start + self.chunk_size]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        # num_workers=0 only: stream chunks lazily without materialising all of them.
        for chunk in self._iter_raw_chunks():
            samples = _process_chunk(chunk, self._app_int_map, split=self.split)
            if self.shuffle_chunks:
                random.shuffle(samples)
            for seq_data, stat_data, ports_data, label in samples:
                yield (
                    torch.from_numpy(seq_data),
                    torch.from_numpy(stat_data),
                    torch.from_numpy(ports_data),
                    torch.tensor(label, dtype=torch.long),
                )


def build_streaming_loaders(
    data_root: str,
    size: str = "S",
    batch_size: int = 128,
    chunk_size: int = 8192,
    num_workers: int = 0,
    val_size: str = "XS",
) -> Tuple[DataLoader, DataLoader]:
    train_ds = CESNETStreamingDataset(data_root=data_root, size=size, chunk_size=chunk_size, split="train", shuffle_chunks=True)
    val_ds = CESNETStreamingDataset(data_root=data_root, size=val_size, chunk_size=chunk_size, split="val", shuffle_chunks=False)

    # persistent_workers MUST be False for IterableDataset: with it True the
    # persistent workers do not re-run __iter__ on epoch 2+, yielding an empty
    # loader and silently ending training after the first epoch.
    # drop_last=True: prevents a trailing size-1 batch from crashing BatchNorm1d
    # in the projection head (BN requires >1 sample in train mode).
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, persistent_workers=False)

    logger.info("Streaming loaders ready — train=%s val=%s batch=%d workers=%d", size, val_size, batch_size, num_workers)
    return train_loader, val_loader
