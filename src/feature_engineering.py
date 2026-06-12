"""
feature_engineering.py — Pure, stateless feature extraction and normalization functions.

All functions are side-effect-free. No global mutable state. All normalizers are
computed inline per-call using constants defined at module level.

Spec reference: DualBranchEncoder Feature Engineering Specification v1.0
  Branch A: seq_input_dim = 3  (size_norm, ipt_norm, dir)
  Branch B: stat_input_dim = 18
"""

from __future__ import annotations

import math
import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level normalization constants (never mutated)
# ---------------------------------------------------------------------------

SEQ_LEN: int = 30
SEQ_INPUT_DIM: int = 3
STAT_INPUT_DIM: int = 16   # bytes_total and duration_ms removed (shortcut-learning risk)

MAX_PACKET_SIZE: float = 1500.0
MAX_IPT_MS: float = 5000.0
MAX_BYTES: float = 10_000_000.0
MAX_PACKETS: float = 10_000.0
MAX_DURATION_MS: float = 300_000.0

_LOG1P_MAX_PKT_SIZE: float = math.log1p(MAX_PACKET_SIZE)   # log(1501) ≈ 7.3132
_LOG1P_MAX_IPT: float = math.log1p(MAX_IPT_MS)             # log(5001) ≈ 8.5173
_LOG1P_MAX_BYTES: float = math.log1p(MAX_BYTES)
_LOG1P_MAX_PACKETS: float = math.log1p(MAX_PACKETS)
_LOG1P_MAX_DURATION: float = math.log1p(MAX_DURATION_MS)

# PHIST bin edges — 8 bins matching CESNET-QUIC22 PHIST_SRC_SIZES layout
PHIST_BIN_EDGES: List[float] = [0, 16, 32, 64, 128, 256, 512, 1024, float("inf")]

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pad_truncate_array(arr: np.ndarray, target_len: int) -> np.ndarray:
    """
    Truncate to target_len from the head, or zero-pad at the tail.
    Always returns a 1-D float32 array of exactly target_len elements.
    """
    n = len(arr)
    if n >= target_len:
        return arr[:target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    out[:n] = arr
    return out


def _size_norm(sizes: np.ndarray) -> np.ndarray:
    """Slot 0: log-scale normalization for packet sizes."""
    clipped = np.clip(sizes, 0.0, MAX_PACKET_SIZE)
    return np.log1p(clipped).astype(np.float32) / _LOG1P_MAX_PKT_SIZE


def _ipt_norm(ipts: np.ndarray) -> np.ndarray:
    """Slot 1: log-scale normalization for inter-packet times (ms)."""
    clipped = np.clip(ipts, 0.0, MAX_IPT_MS)
    return np.log1p(clipped).astype(np.float32) / _LOG1P_MAX_IPT


def _scalar_size_norm(v: float) -> float:
    return math.log1p(min(v, MAX_PACKET_SIZE)) / _LOG1P_MAX_PKT_SIZE


def _scalar_ipt_norm(v: float) -> float:
    return math.log1p(min(v, MAX_IPT_MS)) / _LOG1P_MAX_IPT


def _scalar_bytes_norm(v: float) -> float:
    return math.log1p(min(v, MAX_BYTES)) / _LOG1P_MAX_BYTES


def _scalar_packets_norm(v: float) -> float:
    return math.log1p(min(v, MAX_PACKETS)) / _LOG1P_MAX_PACKETS


def _scalar_duration_norm(v: float) -> float:
    return math.log1p(min(v, MAX_DURATION_MS)) / _LOG1P_MAX_DURATION


# ---------------------------------------------------------------------------
# PHIST helpers
# ---------------------------------------------------------------------------


def compute_phist_from_lengths(lengths: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Reconstruct an 8-bin source-size histogram from a raw lengths array.
    Used for ISCXVPN2016 which has no precomputed PHIST fields.

    Bin edges: [0,16), [16,32), [32,64), [64,128), [128,256),
               [256,512), [512,1024), [1024,inf)

    Returns float32 array of shape (8,) where each element is
    counts[i] / (sum(counts) + 1).
    """
    counts = np.zeros(8, dtype=np.float32)
    for size in lengths:
        for i in range(8):
            if PHIST_BIN_EDGES[i] <= size < PHIST_BIN_EDGES[i + 1]:
                counts[i] += 1.0
                break
    total = counts.sum() + 1.0
    return counts / total


def _normalize_phist_src(raw_counts: np.ndarray) -> np.ndarray:
    """
    Normalize CESNET PHIST_SRC_SIZES raw counts (8 values).
    Returns float32 array of shape (8,).
    """
    arr = np.asarray(raw_counts, dtype=np.float32)
    if len(arr) < 8:
        padded = np.zeros(8, dtype=np.float32)
        padded[: len(arr)] = arr
        arr = padded
    total = arr[:8].sum() + 1.0
    return arr[:8] / total


# ---------------------------------------------------------------------------
# Branch A — Sequential features
# ---------------------------------------------------------------------------


def extract_seq_features(ppi: list, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Extract and normalize Branch A sequential features from a CESNET-QUIC22 PPI.

    PPI layout (list of 3 sub-arrays):
      ppi[0] — inter_packet_times (float, ms)
      ppi[1] — packet_directions  ({+1, -1})
      ppi[2] — packet_sizes       (unsigned int, bytes)

    The first IPT entry is forced to 0.0 (no preceding packet).

    Parameters
    ----------
    ppi : list
        Three-element list where each element is a sequence of numeric values.
    seq_len : int
        Target sequence length. Default SEQ_LEN=128.

    Returns
    -------
    np.ndarray
        Shape (seq_len, 3), dtype float32.
        Columns: [size_norm, ipt_norm, dir].

    Raises
    ------
    ValueError
        If ppi does not have at least 3 sub-arrays or any sub-array is empty.
    """
    if len(ppi) < 3:
        raise ValueError(f"PPI must have 3 sub-arrays; got {len(ppi)}")
    if any(len(ppi[i]) == 0 for i in range(3)):
        raise ValueError("One or more PPI sub-arrays have length 0")

    raw_sizes = np.asarray(ppi[2], dtype=np.float32)
    raw_ipts = np.asarray(ppi[0], dtype=np.float32)
    raw_dirs = np.asarray(ppi[1], dtype=np.float32)

    # Enforce ipt[0] = 0.0 (no preceding packet for first entry)
    if len(raw_ipts) > 0:
        raw_ipts[0] = 0.0

    # Emit soft warnings before clipping (logged, not raised)
    if np.any(raw_sizes > MAX_PACKET_SIZE):
        logger.debug("OVERSIZE_PKT: max_size=%.1f", float(raw_sizes.max()))
    if np.any(raw_ipts > MAX_IPT_MS):
        logger.debug("LONG_IPT: max_ipt=%.1fms", float(raw_ipts.max()))

    sizes_padded = _pad_truncate_array(raw_sizes, seq_len)
    ipts_padded = _pad_truncate_array(raw_ipts, seq_len)
    dirs_padded = _pad_truncate_array(raw_dirs, seq_len)

    sizes_norm = _size_norm(sizes_padded)
    ipts_norm = _ipt_norm(ipts_padded)
    dirs_out = dirs_padded.astype(np.float32)

    seq_data = np.stack([sizes_norm, ipts_norm, dirs_out], axis=-1)  # (seq_len, 3)

    if not np.all(np.isfinite(seq_data)):
        raise ValueError("Non-finite value in seq_data after normalization")

    return seq_data


def extract_seq_from_iscxvpn(
    lengths: List[float],
    intervals: List[float],
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """
    Convert ISCXVPN2016 JSON fields to Branch A sequential tensor.

    ISCXVPN2016 has no direction field; direction slot is filled with 0.0
    (a sentinel value outside {+1, -1}, signaling "direction unknown").

    Parameters
    ----------
    lengths : list of float
        Raw packet sizes in bytes.
    intervals : list of float
        Inter-packet times in milliseconds.
    seq_len : int
        Target sequence length. Default SEQ_LEN=128.

    Returns
    -------
    np.ndarray
        Shape (seq_len, 3), dtype float32.
        Columns: [size_norm, ipt_norm, 0.0].

    Raises
    ------
    ValueError
        If resulting array contains non-finite values.
    """
    raw_sizes = np.asarray(lengths, dtype=np.float32)
    raw_ipts = np.asarray(intervals, dtype=np.float32)

    # Force first IPT to 0.0
    if len(raw_ipts) > 0:
        raw_ipts[0] = 0.0

    if np.any(raw_sizes > MAX_PACKET_SIZE):
        logger.warning("OVERSIZE_PKT (ISCXVPN): max_size=%.1f", float(raw_sizes.max()))
    if np.any(raw_ipts > MAX_IPT_MS):
        logger.warning("LONG_IPT (ISCXVPN): max_ipt=%.1fms", float(raw_ipts.max()))

    sizes_padded = _pad_truncate_array(raw_sizes, seq_len)
    ipts_padded = _pad_truncate_array(raw_ipts, seq_len)
    dirs_filled = np.zeros(seq_len, dtype=np.float32)  # direction unknown sentinel

    sizes_norm = _size_norm(sizes_padded)
    ipts_norm = _ipt_norm(ipts_padded)

    seq_data = np.stack([sizes_norm, ipts_norm, dirs_filled], axis=-1)  # (seq_len, 3)

    if not np.all(np.isfinite(seq_data)):
        raise ValueError("Non-finite value in seq_data (ISCXVPN) after normalization")

    return seq_data


# ---------------------------------------------------------------------------
# Branch B — Statistical features
# ---------------------------------------------------------------------------


def extract_stat_features(row: dict) -> np.ndarray:
    """
    Extract and normalize Branch B statistical features from a CESNET-QUIC22 row.

    Expected keys in ``row``:
      BYTES, BYTES_REV, PACKETS, PACKETS_REV, DURATION (ms),
      PPI (3-element list of arrays), PPI_LEN (int),
      PHIST_SRC_SIZES (list/array of 8 raw counts)

    Optional CESNET flow-end reason keys (used only for warnings, not rejection):
      FLOW_ENDREASON_IDLE, FLOW_ENDREASON_ACTIVE

    Returns
    -------
    np.ndarray
        Shape (18,), dtype float32.

    Raises
    ------
    ValueError
        If any required key is missing, or output contains non-finite values.
    """
    required = ("BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV", "DURATION",
                "PPI", "PPI_LEN", "PHIST_SRC_SIZES")
    for key in required:
        if key not in row:
            raise ValueError(f"Missing required key '{key}' in row dict")

    bytes_fwd = float(row["BYTES"])
    bytes_rev = float(row["BYTES_REV"])
    pkts_fwd = float(row["PACKETS"])
    pkts_rev = float(row["PACKETS_REV"])
    duration_ms = float(row["DURATION"])
    ppi = row["PPI"]
    ppi_len = int(row["PPI_LEN"])
    phist_raw = row["PHIST_SRC_SIZES"]

    bytes_total = bytes_fwd + bytes_rev
    pkts_total = pkts_fwd + pkts_rev

    # index 0
    bytes_total_norm = _scalar_bytes_norm(bytes_total)
    # index 1
    bytes_ratio = bytes_fwd / (bytes_total + 1.0)
    # index 2
    packets_total_norm = _scalar_packets_norm(pkts_total)
    # index 3
    packets_ratio = pkts_fwd / (pkts_total + 1.0)
    # index 4
    duration_norm = _scalar_duration_norm(duration_ms)

    # PPI sub-arrays: ppi[2]=sizes, ppi[0]=ipts
    valid_len = min(ppi_len, len(ppi[2]), len(ppi[0]))
    raw_sizes = np.asarray(ppi[2][:valid_len], dtype=np.float32)
    raw_ipts = np.asarray(ppi[0][:valid_len], dtype=np.float32)

    if len(raw_sizes) == 0:
        mean_size = 0.0
        std_size = 0.0
    else:
        mean_size = float(np.mean(raw_sizes))
        std_size = float(np.std(raw_sizes))
        if std_size == 0.0:
            logger.warning("ZERO_STD: feature=pkt_size")

    if len(raw_ipts) == 0:
        mean_ipt = 0.0
        jitter_estimate = 0.0
    else:
        mean_ipt = float(np.mean(raw_ipts))
        jitter_estimate = float(np.std(raw_ipts))   # std of IPT ≈ jitter (QoS rubric)
        if jitter_estimate == 0.0:
            logger.warning("ZERO_STD: feature=ipt")

    # index 3
    mean_pkt_size_norm = _scalar_size_norm(mean_size)
    # index 4
    std_pkt_size_norm = _scalar_size_norm(std_size)
    # index 5
    mean_ipt_norm = _scalar_ipt_norm(mean_ipt)
    # index 6
    jitter_norm = _scalar_ipt_norm(jitter_estimate)

    # indices 7–14: PHIST_SRC_SIZES normalized
    phist_src_norm = _normalize_phist_src(np.asarray(phist_raw, dtype=np.float32))
    if np.asarray(phist_raw).sum() == 0:
        logger.warning("EMPTY_PHIST: PHIST_SRC_SIZES sums to zero")

    # index 15
    ppi_len_norm = min(ppi_len, SEQ_LEN) / float(SEQ_LEN)

    # bytes_total_norm and duration_norm intentionally excluded:
    # they are volume/time shortcuts that correlate with class size, not class identity.
    stat_data = np.array(
        [
            bytes_ratio,            # 0  scale-invariant directional ratio
            packets_total_norm,     # 1
            packets_ratio,          # 2
            mean_pkt_size_norm,     # 3
            std_pkt_size_norm,      # 4
            mean_ipt_norm,          # 5
            jitter_norm,            # 6  QoS jitter estimate
            phist_src_norm[0],      # 7
            phist_src_norm[1],      # 8
            phist_src_norm[2],      # 9
            phist_src_norm[3],      # 10
            phist_src_norm[4],      # 11
            phist_src_norm[5],      # 12
            phist_src_norm[6],      # 13
            phist_src_norm[7],      # 14
            ppi_len_norm,           # 15
        ],
        dtype=np.float32,
    )

    if not np.all(np.isfinite(stat_data)):
        raise ValueError("Non-finite value in stat_data after normalization")

    return stat_data


def extract_stat_from_iscxvpn(
    lengths: List[float],
    intervals: List[float],
) -> np.ndarray:
    """
    Convert ISCXVPN2016 JSON fields to Branch B statistical tensor.

    Since ISCXVPN2016 has no directional split:
      - bytes_ratio = 0.5  (direction unknown)
      - packets_ratio = 0.5
      - duration_ms = sum(intervals)
      - PHIST derived from lengths via compute_phist_from_lengths()

    Parameters
    ----------
    lengths : list of float
        Raw packet sizes in bytes.
    intervals : list of float
        Inter-packet times in milliseconds.

    Returns
    -------
    np.ndarray
        Shape (18,), dtype float32.

    Raises
    ------
    ValueError
        If output contains non-finite values.
    """
    raw_sizes = np.asarray(lengths, dtype=np.float32)
    raw_ipts = np.asarray(intervals, dtype=np.float32)

    bytes_total = float(raw_sizes.sum())
    pkts_total = float(len(raw_sizes))
    duration_ms = float(raw_ipts.sum())

    # index 0
    bytes_total_norm = _scalar_bytes_norm(bytes_total)
    # index 1 — direction unknown
    bytes_ratio = 0.5
    # index 2
    packets_total_norm = _scalar_packets_norm(pkts_total)
    # index 3 — direction unknown
    packets_ratio = 0.5
    # index 4
    duration_norm = _scalar_duration_norm(duration_ms)

    if len(raw_sizes) == 0:
        mean_size = 0.0
        std_size = 0.0
    else:
        mean_size = float(np.mean(raw_sizes))
        std_size = float(np.std(raw_sizes))
        if std_size == 0.0:
            logger.warning("ZERO_STD: feature=pkt_size (ISCXVPN)")

    if len(raw_ipts) == 0:
        mean_ipt = 0.0
        jitter_estimate = 0.0
    else:
        mean_ipt = float(np.mean(raw_ipts))
        jitter_estimate = float(np.std(raw_ipts))
        if jitter_estimate == 0.0:
            logger.warning("ZERO_STD: feature=ipt (ISCXVPN)")

    mean_pkt_size_norm = _scalar_size_norm(mean_size)
    std_pkt_size_norm = _scalar_size_norm(std_size)
    mean_ipt_norm = _scalar_ipt_norm(mean_ipt)
    jitter_norm = _scalar_ipt_norm(jitter_estimate)

    phist_src_norm = compute_phist_from_lengths(lengths)
    ppi_len_norm = min(len(lengths), SEQ_LEN) / float(SEQ_LEN)

    stat_data = np.array(
        [
            bytes_ratio,
            packets_total_norm,
            packets_ratio,
            mean_pkt_size_norm,
            std_pkt_size_norm,
            mean_ipt_norm,
            jitter_norm,
            phist_src_norm[0],
            phist_src_norm[1],
            phist_src_norm[2],
            phist_src_norm[3],
            phist_src_norm[4],
            phist_src_norm[5],
            phist_src_norm[6],
            phist_src_norm[7],
            ppi_len_norm,
        ],
        dtype=np.float32,
    )

    if not np.all(np.isfinite(stat_data)):
        raise ValueError("Non-finite value in stat_data (ISCXVPN) after normalization")

    return stat_data


# ---------------------------------------------------------------------------
# Class-weight utility (spec §5.1)
# ---------------------------------------------------------------------------


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute inverse-frequency class weights normalized so they sum to num_classes.

    Parameters
    ----------
    labels : np.ndarray
        1-D integer array of class indices, shape (N,).
    num_classes : int
        Total number of classes.

    Returns
    -------
    np.ndarray
        float32 array of shape (num_classes,). Classes with zero samples
        receive weight 0.0.
    """
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    mask = class_counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[mask] = 1.0 / class_counts[mask]
    total = weights.sum()
    if total > 0:
        weights = weights * num_classes / total
    return weights.astype(np.float32)
