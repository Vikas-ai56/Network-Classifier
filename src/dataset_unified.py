"""
dataset_unified.py — Unified PyTorch Dataset for multi-source network traffic classification.

Supports:
  1. CESNET-QUIC22  — Parquet or CSV; uses PPI field for Branch A, flow stats for Branch B
  2. ISCXVPN2016    — JSON files ({"lengths": [...], "intervals": [...]})
  3. 5G Kaggle      — CSV with flow-level columns; detected at load time

Output tensors per sample:
  seq_tensor  : FloatTensor[SEQ_LEN=30, SEQ_INPUT_DIM=3]
  stat_tensor : FloatTensor[STAT_INPUT_DIM=18]
  label       : int (unified class ID, 0–7)

Unified label taxonomy: 8 classes mapped from both dataset source labels.
Spec reference: Feature Engineering Specification v1.0.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data_validator import (
    DatasetReport,
    FlowValidator,
    REJECT_R1,
    REJECT_R2,
    REJECT_R3,
    REJECT_R4,
    REJECT_R5,
    REJECT_R6,
    REJECT_R7,
    REJECT_R8,
)
from src.feature_engineering import (
    SEQ_LEN,
    SEQ_INPUT_DIM,
    STAT_INPUT_DIM,
    compute_class_weights,
    extract_seq_features,
    extract_seq_from_iscxvpn,
    extract_stat_features,
    extract_stat_from_iscxvpn,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified label taxonomy (spec §3.1)
# ---------------------------------------------------------------------------

# Maps lower-cased source label strings to unified class IDs.
# CESNET entries use CATEGORY values; ISCXVPN2016 entries use directory names.
LABEL_MAP: Dict[str, int] = {
    # class 0 — video_streaming
    "video": 0,
    "youtube": 0,
    "netflix": 0,
    "vimeo": 0,
    # ISCXVPN2016: "streaming" dir contains captured video/audio streaming
    # sessions (YouTube, Vimeo). Source: Draper-Gil et al. (2016) Table 1.
    "streaming": 0,
    # class 1 — audio_streaming
    "audio": 1,
    "spotify": 1,
    "voip": 1,
    # class 2 — gaming
    "gaming": 2,
    "steam": 2,
    "game": 2,
    # class 3 — social_media
    "social": 3,
    "facebook": 3,
    "instagram": 3,
    "twitter": 3,
    # class 4 — file_transfer
    "file-transfer": 4,
    "cloud": 4,
    "sftp": 4,
    "ftps": 4,
    "scp": 4,
    # ISCXVPN2016: "ftp" dir contains FTP bulk data transfers.
    # "p2p" dir contains BitTorrent peer-to-peer file sharing.
    # Both are file transfer protocols. Source: Draper-Gil et al. (2016) Table 1.
    "ftp": 4,
    "p2p": 4,
    # class 5 — browsing
    "web": 5,
    "browsing": 5,
    "http": 5,
    "https": 5,
    # class 6 — communication
    "communication": 6,
    "skype": 6,
    "hangouts": 6,
    "zoom": 6,
    # ISCXVPN2016: "chat" dir contains instant messaging sessions (Skype chat,
    # ICQ, AIM, Facebook chat). "email" dir contains SMTP/IMAP mail sessions.
    # Both are person-to-person communication protocols.
    # Source: Draper-Gil et al. (2016) Table 1.
    "chat": 6,
    "email": 6,
    # class 7 — vpn_tunnel
    "vpn-voip": 7,
    "vpn-streaming": 7,
    "vpn-file-transfer": 7,
    "vpn-browsing": 7,
    "vpn-email": 7,
    "vpn-chat": 7,
}

UNIFIED_CLASS_NAMES: Dict[int, str] = {
    0: "video_streaming",
    1: "audio_streaming",
    2: "gaming",
    3: "social_media",
    4: "file_transfer",
    5: "browsing",
    6: "communication",
    7: "vpn_tunnel",
}

NUM_CLASSES: int = 8
MIN_SAMPLES_PER_CLASS: int = 200

# Set of lower-cased source labels recognized by the taxonomy (for validator)
_KNOWN_SOURCE_LABELS: frozenset = frozenset(LABEL_MAP.keys())

# ---------------------------------------------------------------------------
# 5G Kaggle column detection helpers
# ---------------------------------------------------------------------------

# Canonical column name sets for each required logical field
_5G_COL_BYTES_FWD = {"bytes_fwd", "fwd_bytes", "total_fwd_bytes", "src_bytes"}
_5G_COL_BYTES_REV = {"bytes_rev", "bwd_bytes", "total_bwd_bytes", "dst_bytes"}
_5G_COL_PKTS_FWD = {"packets_fwd", "fwd_packets", "total_fwd_packets", "src_pkts"}
_5G_COL_PKTS_REV = {"packets_rev", "bwd_packets", "total_bwd_packets", "dst_pkts"}
_5G_COL_DURATION = {"duration", "flow_duration", "duration_ms"}
_5G_COL_LABEL = {"label", "class", "category", "traffic_class", "app"}
_5G_COL_PKT_SIZES = {"pkt_sizes", "packet_sizes", "pkt_size_list", "flow_pkt_sizes"}
_5G_COL_PKT_IATS = {"pkt_iats", "inter_arrival_times", "iat_list", "flow_iats", "intervals"}


def _find_col(df_columns: List[str], candidates: frozenset) -> Optional[str]:
    """Return the first column name (lowercased) present in candidates, or None."""
    lower_map = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


# ---------------------------------------------------------------------------
# Internal sample tuple
# ---------------------------------------------------------------------------

# Each loaded sample is (seq_data: ndarray, stat_data: ndarray, label: int)
# The 5-tuple identity (ports/IPs) is intentionally NOT carried as a feature.
_Sample = Tuple[np.ndarray, np.ndarray, int]


# ---------------------------------------------------------------------------
# UnifiedFlowDataset
# ---------------------------------------------------------------------------


class UnifiedFlowDataset(Dataset):
    """
    Unified Dataset combining CESNET-QUIC22, ISCXVPN2016, and 5G Kaggle flows.

    Parameters
    ----------
    data_dir : str or Path
        Root directory. The loader inspects the contents to determine source type:
          - ``*.parquet`` or ``*.csv`` files at the root or one level deep → CESNET or 5G Kaggle
          - Subdirectories containing ``*.json`` → ISCXVPN2016
        Multiple source types may coexist under the same root.
    seq_len : int
        Sequence length for Branch A. Default 128.
    min_samples_per_class : int
        Classes below this count are excluded post-scan (rule R6). Default 200.
    source_hint : str or None
        One of {"cesnet", "iscxvpn", "5g", None}.
        When None the loader auto-detects per sub-directory/file.
    transform : callable or None
        Optional transform applied to (seq_tensor, stat_tensor) tuples.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int = SEQ_LEN,
        min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
        source_hint: Optional[str] = None,
        transform=None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.min_samples_per_class = min_samples_per_class
        self.source_hint = source_hint
        self.transform = transform

        self._validator = FlowValidator()
        self.report = DatasetReport()

        # Populated during _load()
        self._samples: List[_Sample] = []
        self.labels: np.ndarray = np.array([], dtype=np.int64)

        self._load()

        # Emit validation report
        self.report.log_summary()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(UNIFIED_CLASS_NAMES)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_data, stat_data, label = self._samples[idx]
        seq_tensor   = torch.from_numpy(seq_data)            # (seq_len, 3)
        stat_tensor  = torch.from_numpy(stat_data)           # (16,)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            seq_tensor, stat_tensor = self.transform(seq_tensor, stat_tensor)

        return seq_tensor, stat_tensor, label_tensor

    # ------------------------------------------------------------------
    # Internal loading orchestrator
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """
        Discover and load all samples under self.data_dir.

        Detection order:
          1. Parquet files → CESNET-QUIC22
          2. JSON files in sub-directories → ISCXVPN2016
          3. CSV files → 5G Kaggle (column-detected) or CESNET fallback
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        parquet_files = list(self.data_dir.rglob("*.parquet"))
        json_files = list(self.data_dir.rglob("*.json"))
        csv_files = list(self.data_dir.rglob("*.csv"))

        hint = (self.source_hint or "").lower()

        if hint == "iscxvpn" or (not hint and json_files and not parquet_files):
            self._load_iscxvpn(json_files)
        elif hint == "cesnet" or (not hint and parquet_files):
            self._load_cesnet(parquet_files)
        elif hint == "5g" or (not hint and csv_files and not parquet_files and not json_files):
            self._load_5g_kaggle(csv_files)
        else:
            # Mixed root: load each type independently
            if parquet_files:
                self._load_cesnet(parquet_files)
            if json_files:
                self._load_iscxvpn(json_files)
            if csv_files and not parquet_files:
                self._load_5g_kaggle(csv_files)

        # R6: drop classes with fewer than min_samples_per_class
        self._apply_min_samples_filter()

        # Build label array for sampler construction
        self.labels = np.array([s[2] for s in self._samples], dtype=np.int64)

    # ------------------------------------------------------------------
    # ISCXVPN2016 loader
    # ------------------------------------------------------------------

    def _load_iscxvpn(self, json_files: List[Path]) -> None:
        """Load ISCXVPN2016 JSON flow files."""
        for file_path in json_files:
            source_path = str(file_path)

            # Label from immediate parent directory name
            raw_label = file_path.parent.name.lower()

            # R5 — label check
            ok, reason = self._validator.validate_label(raw_label, _KNOWN_SOURCE_LABELS)
            if not ok:
                self.report.record_rejection(reason)
                continue

            try:
                with file_path.open("r") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read %s: %s", source_path, exc)
                self.report.record_rejection(REJECT_R7)
                continue

            lengths = data.get("lengths", [])
            intervals = data.get("intervals", [])

            # Validate sequence
            ok, reason = self._validator.validate_iscxvpn_sequence(
                lengths, intervals, source_path=source_path
            )
            if not ok:
                self.report.record_rejection(reason)
                continue

            try:
                seq_data = extract_seq_from_iscxvpn(lengths, intervals, self.seq_len)
                stat_data = extract_stat_from_iscxvpn(lengths, intervals)
            except ValueError as exc:
                logger.error("Feature extraction failed for %s: %s", source_path, exc)
                self.report.record_rejection(REJECT_R7)
                continue

            # R7 — NaN/Inf check
            ok, reason = self._validator.validate_feature_array(
                seq_data, source_path, "seq_data"
            )
            if not ok:
                self.report.record_rejection(reason)
                continue
            ok, reason = self._validator.validate_feature_array(
                stat_data, source_path, "stat_data"
            )
            if not ok:
                self.report.record_rejection(reason)
                continue

            unified_label = LABEL_MAP[raw_label]
            unified_class_name = UNIFIED_CLASS_NAMES[unified_label]
            self._samples.append((seq_data, stat_data, unified_label))
            self.report.record_load(unified_class_name)

    # ------------------------------------------------------------------
    # CESNET-QUIC22 loader
    # ------------------------------------------------------------------

    def _load_cesnet(self, parquet_files: List[Path]) -> None:
        """Load CESNET-QUIC22 Parquet (or CSV) flow records."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to load CESNET Parquet files") from exc

        for file_path in parquet_files:
            source_path = str(file_path)
            try:
                if file_path.suffix == ".parquet":
                    df = pd.read_parquet(file_path)
                else:
                    df = pd.read_csv(file_path)
            except Exception as exc:
                logger.warning("Cannot read %s: %s", source_path, exc)
                continue

            # Normalize column names to uppercase for consistency
            df.columns = [str(c).upper() if isinstance(c, str) else str(c) for c in df.columns]

            self._process_cesnet_dataframe(df, source_path)

    def _process_cesnet_dataframe(self, df, source_path: str) -> None:
        """Iterate rows of a CESNET DataFrame and extract features."""
        import ast

        for row_idx, row in df.iterrows():
            row_id = f"{source_path}:row{row_idx}"

            # R5 — label from CATEGORY column
            raw_label = str(row.get("CATEGORY", "")).strip().lower()
            ok, reason = self._validator.validate_label(raw_label, _KNOWN_SOURCE_LABELS)
            if not ok:
                self.report.record_rejection(reason)
                continue

            # Parse PPI — may be stored as a string representation of a list
            ppi_raw = row.get("PPI")
            ppi = self._parse_ppi_field(ppi_raw, row_id)
            if ppi is None:
                self.report.record_rejection(REJECT_R4)
                continue

            ppi_len = int(row.get("PPI_LEN", len(ppi[0])))
            flow_endreason_active = int(row.get("FLOW_ENDREASON_ACTIVE", 0))
            flow_endreason_idle = int(row.get("FLOW_ENDREASON_IDLE", 0))

            # Validate PPI structure
            ok, reason = self._validator.validate_ppi(
                ppi,
                source_path=row_id,
                flow_endreason_active=flow_endreason_active,
            )
            if not ok:
                self.report.record_rejection(reason)
                continue

            # Build stats_dict for validate_stats
            stats_dict = {k: row.get(k, 0) for k in
                          ("BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV",
                           "DURATION", "PHIST_SRC_SIZES", "FLOW_ENDREASON_IDLE",
                           "FLOW_ENDREASON_ACTIVE")}
            ok, reason = self._validator.validate_stats(
                stats_dict, source_path=row_id, is_cesnet=True
            )
            if not ok:
                self.report.record_rejection(reason)
                continue

            # Parse PHIST_SRC_SIZES
            phist_raw = row.get("PHIST_SRC_SIZES", [0] * 8)
            phist_parsed = self._parse_array_field(phist_raw, 8, row_id)

            # Build the row dict for feature extraction
            fe_row = {
                "BYTES": float(row.get("BYTES", 0)),
                "BYTES_REV": float(row.get("BYTES_REV", 0)),
                "PACKETS": float(row.get("PACKETS", 0)),
                "PACKETS_REV": float(row.get("PACKETS_REV", 0)),
                "DURATION": float(row.get("DURATION", 0)),
                "PPI": ppi,
                "PPI_LEN": ppi_len,
                "PHIST_SRC_SIZES": phist_parsed,
                "FLOW_ENDREASON_IDLE": flow_endreason_idle,
                "FLOW_ENDREASON_ACTIVE": flow_endreason_active,
            }

            try:
                seq_data = extract_seq_features(ppi, self.seq_len)
                stat_data = extract_stat_features(fe_row)
            except ValueError as exc:
                logger.error("Feature extraction failed for %s: %s", row_id, exc)
                self.report.record_rejection(REJECT_R7)
                continue

            # R7 — NaN/Inf check
            ok, reason = self._validator.validate_feature_array(seq_data, row_id, "seq_data")
            if not ok:
                self.report.record_rejection(reason)
                continue
            ok, reason = self._validator.validate_feature_array(stat_data, row_id, "stat_data")
            if not ok:
                self.report.record_rejection(reason)
                continue

            unified_label = LABEL_MAP[raw_label]
            unified_class_name = UNIFIED_CLASS_NAMES[unified_label]
            self._samples.append((seq_data, stat_data, unified_label))
            self.report.record_load(unified_class_name)

    # ------------------------------------------------------------------
    # 5G Kaggle loader
    # ------------------------------------------------------------------

    def _load_5g_kaggle(self, csv_files: List[Path]) -> None:
        """
        Load 5G Kaggle CSV files.

        Column names are detected at load time using a prioritized lookup
        against known candidate column name sets. Treats the flow similarly
        to CESNET when per-packet arrays are available, or falls back to
        summary statistics when they are absent.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required to load 5G Kaggle CSV files") from exc

        for file_path in csv_files:
            source_path = str(file_path)
            try:
                df = pd.read_csv(file_path, low_memory=False)
            except Exception as exc:
                logger.warning("Cannot read %s: %s", source_path, exc)
                continue

            columns = list(df.columns)

            # Detect required columns
            col_label = _find_col(columns, frozenset(_5G_COL_LABEL))
            col_bytes_fwd = _find_col(columns, frozenset(_5G_COL_BYTES_FWD))
            col_bytes_rev = _find_col(columns, frozenset(_5G_COL_BYTES_REV))
            col_pkts_fwd = _find_col(columns, frozenset(_5G_COL_PKTS_FWD))
            col_pkts_rev = _find_col(columns, frozenset(_5G_COL_PKTS_REV))
            col_duration = _find_col(columns, frozenset(_5G_COL_DURATION))
            col_pkt_sizes = _find_col(columns, frozenset(_5G_COL_PKT_SIZES))
            col_pkt_iats = _find_col(columns, frozenset(_5G_COL_PKT_IATS))

            if col_label is None:
                logger.warning("No label column detected in %s; skipping", source_path)
                continue

            has_sequence_cols = col_pkt_sizes is not None and col_pkt_iats is not None

            for row_idx, row in df.iterrows():
                row_id = f"{source_path}:row{row_idx}"

                raw_label = str(row[col_label]).strip().lower()
                ok, reason = self._validator.validate_label(raw_label, _KNOWN_SOURCE_LABELS)
                if not ok:
                    self.report.record_rejection(reason)
                    continue

                if has_sequence_cols:
                    # Use packet-level arrays — treat like ISCXVPN2016 JSON
                    lengths = self._parse_array_field(row[col_pkt_sizes], None, row_id)
                    intervals = self._parse_array_field(row[col_pkt_iats], None, row_id)

                    ok, reason = self._validator.validate_iscxvpn_sequence(
                        lengths, intervals, source_path=row_id
                    )
                    if not ok:
                        self.report.record_rejection(reason)
                        continue

                    try:
                        seq_data = extract_seq_from_iscxvpn(lengths, intervals, self.seq_len)
                        stat_data = extract_stat_from_iscxvpn(lengths, intervals)
                    except ValueError as exc:
                        logger.error("Feature extraction failed %s: %s", row_id, exc)
                        self.report.record_rejection(REJECT_R7)
                        continue
                else:
                    # Summary-only path: synthesize seq from summary stats
                    seq_data, stat_data = self._extract_5g_summary_only(
                        row, col_bytes_fwd, col_bytes_rev,
                        col_pkts_fwd, col_pkts_rev, col_duration, row_id
                    )
                    if seq_data is None:
                        continue  # rejection already recorded inside the method

                # R7
                ok, reason = self._validator.validate_feature_array(seq_data, row_id, "seq_data")
                if not ok:
                    self.report.record_rejection(reason)
                    continue
                ok, reason = self._validator.validate_feature_array(stat_data, row_id, "stat_data")
                if not ok:
                    self.report.record_rejection(reason)
                    continue

                unified_label = LABEL_MAP[raw_label]
                unified_class_name = UNIFIED_CLASS_NAMES[unified_label]
                self._samples.append((seq_data, stat_data, unified_label))
                self.report.record_load(unified_class_name)

    def _extract_5g_summary_only(
        self,
        row,
        col_bytes_fwd: Optional[str],
        col_bytes_rev: Optional[str],
        col_pkts_fwd: Optional[str],
        col_pkts_rev: Optional[str],
        col_duration: Optional[str],
        row_id: str,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build Branch A and B tensors from 5G Kaggle summary columns only
        (no per-packet arrays available).

        Branch A is synthesized as a sequence of identical packets derived from
        summary stats — mean_size repeated pkts_total times, up to seq_len, with
        uniform IPT spacing. Direction slot is 0.0 (unknown).

        Returns (None, None) and records rejection if required fields are missing
        or validation fails.
        """
        def _safe_float(col: Optional[str], default: float = 0.0) -> float:
            if col is None:
                return default
            try:
                v = float(row[col])
                return v if np.isfinite(v) else default
            except (TypeError, ValueError):
                return default

        bytes_fwd = _safe_float(col_bytes_fwd)
        bytes_rev = _safe_float(col_bytes_rev)
        pkts_fwd = _safe_float(col_pkts_fwd)
        pkts_rev = _safe_float(col_pkts_rev)
        duration_ms = _safe_float(col_duration)

        bytes_total = bytes_fwd + bytes_rev
        pkts_total = pkts_fwd + pkts_rev

        # R2
        if duration_ms <= 0.0:
            self.report.record_rejection(REJECT_R2)
            return None, None

        # R3
        if bytes_total == 0.0:
            self.report.record_rejection(REJECT_R3)
            return None, None

        n_pkts = max(1, int(pkts_total))

        # Synthesize per-packet arrays from summary statistics
        mean_size = bytes_total / n_pkts if n_pkts > 0 else 0.0
        mean_ipt = duration_ms / n_pkts if n_pkts > 0 else 0.0

        lengths = [mean_size] * n_pkts
        intervals = [mean_ipt] * n_pkts

        # R1
        if n_pkts < FlowValidator.MIN_PACKETS:
            self.report.record_rejection(REJECT_R1)
            return None, None

        try:
            seq_data = extract_seq_from_iscxvpn(lengths, intervals, self.seq_len)
        except ValueError as exc:
            logger.error("seq extraction failed for 5G summary %s: %s", row_id, exc)
            self.report.record_rejection(REJECT_R7)
            return None, None

        # Build stat_data directly for summary path
        try:
            stat_data = self._build_stat_from_summary(
                bytes_fwd, bytes_rev, pkts_fwd, pkts_rev,
                duration_ms, mean_size, 0.0, mean_ipt, 0.0,
                len(lengths)
            )
        except ValueError as exc:
            logger.error("stat extraction failed for 5G summary %s: %s", row_id, exc)
            self.report.record_rejection(REJECT_R7)
            return None, None

        return seq_data, stat_data

    @staticmethod
    def _build_stat_from_summary(
        bytes_fwd: float,
        bytes_rev: float,
        pkts_fwd: float,
        pkts_rev: float,
        duration_ms: float,
        mean_size: float,
        std_size: float,
        mean_ipt: float,
        std_ipt: float,
        n_packets: int,
    ) -> np.ndarray:
        """
        Build a stat_data array from flow-level summary fields (no PHIST available).

        PHIST bins are derived by placing all packets into the mean_size bin.
        The result is shape (18,) float32.
        """
        import math as _math

        _lp_bytes = math.log1p
        MAX_BYTES = 10_000_000.0
        MAX_PACKETS = 10_000.0
        MAX_DURATION_MS = 300_000.0
        MAX_PKT = 1500.0
        MAX_IPT = 5000.0
        _log1p_mb = math.log1p(MAX_BYTES)
        _log1p_mp = math.log1p(MAX_PACKETS)
        _log1p_md = math.log1p(MAX_DURATION_MS)
        _log1p_mpkt = math.log1p(MAX_PKT)
        _log1p_mipt = math.log1p(MAX_IPT)

        bytes_total = bytes_fwd + bytes_rev
        pkts_total = pkts_fwd + pkts_rev

        bytes_total_norm = math.log1p(min(bytes_total, MAX_BYTES)) / _log1p_mb
        bytes_ratio = bytes_fwd / (bytes_total + 1.0)
        packets_total_norm = math.log1p(min(pkts_total, MAX_PACKETS)) / _log1p_mp
        packets_ratio = pkts_fwd / (pkts_total + 1.0)
        duration_norm = math.log1p(min(duration_ms, MAX_DURATION_MS)) / _log1p_md
        mean_pkt_size_norm = math.log1p(min(mean_size, MAX_PKT)) / _log1p_mpkt
        std_pkt_size_norm = math.log1p(min(std_size, MAX_PKT)) / _log1p_mpkt
        mean_ipt_norm = math.log1p(min(mean_ipt, MAX_IPT)) / _log1p_mipt
        std_ipt_norm = math.log1p(min(std_ipt, MAX_IPT)) / _log1p_mipt

        # Derive PHIST from mean_size only — place all n_packets in correct bin
        from src.feature_engineering import compute_phist_from_lengths
        phist = compute_phist_from_lengths([mean_size] * n_packets)

        ppi_len_norm = min(n_packets, SEQ_LEN) / float(SEQ_LEN)

        stat_data = np.array(
            [
                bytes_total_norm, bytes_ratio,
                packets_total_norm, packets_ratio,
                duration_norm,
                mean_pkt_size_norm, std_pkt_size_norm,
                mean_ipt_norm, std_ipt_norm,
                phist[0], phist[1], phist[2], phist[3],
                phist[4], phist[5], phist[6], phist[7],
                ppi_len_norm,
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(stat_data)):
            raise ValueError("Non-finite value in synthesized stat_data")

        return stat_data

    # ------------------------------------------------------------------
    # R6 — minimum samples per class filter
    # ------------------------------------------------------------------

    def _apply_min_samples_filter(self) -> None:
        """
        Remove all samples belonging to classes with fewer than
        min_samples_per_class samples. Records R6 rejections.
        """
        class_counts: Dict[int, int] = defaultdict(int)
        for _, _, label in self._samples:
            class_counts[label] += 1

        dropped_classes = {
            cls_id
            for cls_id, count in class_counts.items()
            if count < self.min_samples_per_class
        }

        if not dropped_classes:
            return

        kept: List[_Sample] = []
        n_dropped = 0
        for sample in self._samples:
            if sample[2] in dropped_classes:
                n_dropped += 1
            else:
                kept.append(sample)

        for cls_id in dropped_classes:
            cls_name = UNIFIED_CLASS_NAMES.get(cls_id, str(cls_id))
            logger.warning(
                "R6: class '%s' (id=%d) dropped — only %d samples (< %d required)",
                cls_name, cls_id, class_counts[cls_id], self.min_samples_per_class,
            )

        # Record bulk R6 rejections
        for _ in range(n_dropped):
            self.report.record_rejection(REJECT_R6)

        # Remove R6-dropped class counts from per_class_counts in the report
        for cls_id in dropped_classes:
            cls_name = UNIFIED_CLASS_NAMES.get(cls_id, str(cls_id))
            if cls_name in self.report.per_class_counts:
                del self.report.per_class_counts[cls_name]

        self._samples = kept

    # ------------------------------------------------------------------
    # PPI / array field parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ppi_field(ppi_raw, row_id: str):
        """
        Parse a PPI field that may be stored as a Python list, numpy array,
        or a string representation (e.g., from CSV serialization).

        Returns a 3-element list of lists, or None on failure.
        """
        import ast

        if ppi_raw is None:
            return None

        if isinstance(ppi_raw, (list, np.ndarray)):
            ppi = list(ppi_raw)
            if len(ppi) >= 3:
                return [list(ppi[i]) for i in range(3)]
            return None

        if isinstance(ppi_raw, str):
            try:
                parsed = ast.literal_eval(ppi_raw)
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                    return [list(parsed[i]) for i in range(3)]
            except (ValueError, SyntaxError):
                pass
            # fallback: numpy print format "[[a b c]\n [d e f]]"
            import re
            rows = re.findall(r'\[([^\[\]]+)\]', ppi_raw)
            rows = [list(map(float, re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', r))) for r in rows if r.strip()]
            if len(rows) >= 3:
                return rows[:3]
            logger.warning("Cannot parse PPI string at %s", row_id)
            return None

        # Pandas Series or other sequence
        try:
            as_list = list(ppi_raw)
            if len(as_list) >= 3:
                return [list(as_list[i]) for i in range(3)]
        except TypeError:
            pass

        return None

    @staticmethod
    def _parse_array_field(raw, expected_len: Optional[int], row_id: str) -> list:
        """
        Parse an array field that may be a list, numpy array, or string.
        Returns a Python list (possibly empty).
        """
        import ast

        if raw is None:
            return []
        if isinstance(raw, (list, np.ndarray)):
            result = list(raw)
            if expected_len is not None:
                result = result[:expected_len]
            return [float(x) for x in result]
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)):
                    result = list(parsed)
                    if expected_len is not None:
                        result = result[:expected_len]
                    return [float(x) for x in result]
            except (ValueError, SyntaxError) as exc:
                logger.warning("Cannot parse array field at %s: %s", row_id, exc)
        return []


# ---------------------------------------------------------------------------
# build_dataloaders factory
# ---------------------------------------------------------------------------


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seq_len: int = SEQ_LEN,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
    source_hint: Optional[str] = None,
    use_weighted_sampler: bool = True,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    Build train, validation, and test DataLoaders from a UnifiedFlowDataset.

    The dataset is split deterministically by class-stratified index assignment
    so that every class maintains its proportion across splits.

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the dataset.
    batch_size : int
        Batch size for the train DataLoader.
    val_split : float
        Fraction of data assigned to validation (default 0.15).
    test_split : float
        Fraction of data assigned to test (default 0.10).
    seq_len : int
        Sequence length passed to UnifiedFlowDataset.
    min_samples_per_class : int
        Minimum samples per class (R6 threshold).
    source_hint : str or None
        Optional source type override ("cesnet", "iscxvpn", "5g").
    use_weighted_sampler : bool
        If True, use WeightedRandomSampler on the train split to balance classes.
    num_workers : int
        Number of DataLoader worker processes.
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, np.ndarray]
        (train_loader, val_loader, test_loader, class_weights)
        class_weights is shape (NUM_CLASSES,) float32 — can be passed to
        nn.CrossEntropyLoss(weight=...) for the fine-tune / eval stage.
    """
    from torch.utils.data import Subset

    dataset = UnifiedFlowDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        min_samples_per_class=min_samples_per_class,
        source_hint=source_hint,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No valid samples loaded from {data_dir}. "
            "Check data path, label taxonomy, and validation logs."
        )

    rng = np.random.default_rng(seed)
    labels = dataset.labels

    # Stratified split: collect indices per class then assign proportionally
    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        class_to_indices[int(lbl)].append(idx)

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for cls_id, indices in class_to_indices.items():
        shuffled = rng.permutation(indices)
        n = len(shuffled)
        n_test = max(1, int(n * test_split))
        n_val = max(1, int(n * val_split))
        n_train = n - n_test - n_val

        if n_train < 1:
            # Not enough samples for a clean 3-way split; assign all to train
            train_indices.extend(shuffled.tolist())
            continue

        train_indices.extend(shuffled[:n_train].tolist())
        val_indices.extend(shuffled[n_train: n_train + n_val].tolist())
        test_indices.extend(shuffled[n_train + n_val:].tolist())

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    # Compute class weights from training labels only
    train_labels = labels[train_indices]
    class_weights = compute_class_weights(train_labels, NUM_CLASSES)

    # Build train DataLoader (weighted sampler or shuffle)
    if use_weighted_sampler and len(train_indices) > 0:
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        "DataLoaders built — train=%d  val=%d  test=%d",
        len(train_set), len(val_set), len(test_set),
    )

    return train_loader, val_loader, test_loader, class_weights
