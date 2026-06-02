"""
data_validator.py — Validation layer for network flow samples.

FlowValidator enforces hard rejection rules (R1–R8) and emits structured
soft warnings (W1–W6) from the Feature Engineering Specification v1.0.

DatasetReport is a dataclass that accumulates per-rejection-reason counts
so the dataset loader can emit a concise validation summary at init time.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Taxonomy of rejection / warning reasons
# ---------------------------------------------------------------------------

REJECT_R1 = "R1_TOO_FEW_PACKETS"
REJECT_R2 = "R2_ZERO_DURATION"
REJECT_R3 = "R3_ZERO_BYTES"
REJECT_R4 = "R4_EMPTY_PPI_ARRAY"
REJECT_R5 = "R5_UNKNOWN_LABEL"
REJECT_R6 = "R6_CLASS_BELOW_MIN_SAMPLES"
REJECT_R7 = "R7_NAN_OR_INF_FEATURE"
REJECT_R8 = "R8_ACTIVE_TIMEOUT_SHORT_FLOW"

WARN_W1 = "W1_SHORT_FLOW"
WARN_W2 = "W2_OVERSIZE_PACKET"
WARN_W3 = "W3_LONG_IPT"
WARN_W4 = "W4_ZERO_STD"
WARN_W5 = "W5_IDLE_TIMEOUT"
WARN_W6 = "W6_EMPTY_PHIST"

# ---------------------------------------------------------------------------
# DatasetReport
# ---------------------------------------------------------------------------


@dataclass
class DatasetReport:
    """
    Accumulates counts of per-rejection-reason discards and per-class loads.

    Attributes
    ----------
    n_loaded : int
        Number of samples successfully loaded into the dataset.
    n_rejected : int
        Total number of samples discarded.
    rejection_counts : dict
        Maps rejection reason string to count.
    per_class_counts : dict
        Maps unified class name to loaded sample count.
    warning_counts : dict
        Maps warning code to count (samples are retained despite warnings).
    """

    n_loaded: int = 0
    n_rejected: int = 0
    rejection_counts: Dict[str, int] = field(default_factory=dict)
    per_class_counts: Dict[str, int] = field(default_factory=dict)
    warning_counts: Dict[str, int] = field(default_factory=dict)

    def record_rejection(self, reason: str) -> None:
        self.n_rejected += 1
        self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1

    def record_load(self, class_name: str) -> None:
        self.n_loaded += 1
        self.per_class_counts[class_name] = self.per_class_counts.get(class_name, 0) + 1

    def record_warning(self, code: str) -> None:
        self.warning_counts[code] = self.warning_counts.get(code, 0) + 1

    def log_summary(self) -> None:
        logger.info(
            "DatasetReport — loaded=%d  rejected=%d",
            self.n_loaded,
            self.n_rejected,
        )
        if self.rejection_counts:
            for reason, count in sorted(self.rejection_counts.items()):
                logger.info("  REJECT %-35s %d", reason, count)
        if self.per_class_counts:
            logger.info("  Per-class sample counts:")
            for cls, cnt in sorted(self.per_class_counts.items()):
                logger.info("    %-30s %d", cls, cnt)
        if self.warning_counts:
            logger.info("  Soft warnings:")
            for code, cnt in sorted(self.warning_counts.items()):
                logger.info("    %-35s %d", code, cnt)

    def to_dict(self) -> dict:
        return {
            "n_loaded": self.n_loaded,
            "n_rejected": self.n_rejected,
            "rejection_counts": dict(self.rejection_counts),
            "per_class_counts": dict(self.per_class_counts),
            "warning_counts": dict(self.warning_counts),
        }


# ---------------------------------------------------------------------------
# FlowValidator
# ---------------------------------------------------------------------------


class FlowValidator:
    """
    Validates individual flow samples before feature extraction.

    Each validate_* method returns a (passed: bool, reason: str) tuple.
    When passed=True, reason is an empty string.
    When passed=False, reason is one of the REJECT_R* constants.

    Soft warnings are emitted via the logging system at WARNING level.
    They do not affect the return value (the sample is retained).
    """

    # Hard thresholds
    MIN_PACKETS: int = 4
    MIN_ACTIVE_TIMEOUT_PACKETS: int = 10

    # Packet size validity range (bytes)
    PKT_SIZE_MIN: float = 1.0
    PKT_SIZE_MAX: float = 65535.0

    # Valid direction values
    VALID_DIRECTIONS: Set[float] = {-1.0, 0.0, 1.0}

    # -----------------------------------------------------------------------
    # PPI validation (CESNET-QUIC22)
    # -----------------------------------------------------------------------

    def validate_ppi(
        self,
        ppi: Any,
        source_path: str = "",
        flow_endreason_active: int = 0,
    ) -> Tuple[bool, str]:
        """
        Validate the PPI structure from a CESNET-QUIC22 record.

        PPI layout:
          ppi[0] — inter_packet_times (float, ms, non-negative)
          ppi[1] — packet_directions  ({+1, -1})
          ppi[2] — packet_sizes       (1–65535 bytes)

        Parameters
        ----------
        ppi : list
            Three-element list of equal-length numeric sequences.
        source_path : str
            File path or identifier for logging context.
        flow_endreason_active : int
            FLOW_ENDREASON_ACTIVE flag from the CESNET record (0 or 1).

        Returns
        -------
        (bool, str)
            (True, "") if valid.
            (False, REJECT_Rx) if a hard rejection rule fires.
        """
        # R4 — PPI must be a 3-element structure, no sub-array empty
        if ppi is None or not hasattr(ppi, "__len__") or len(ppi) < 3:
            return False, REJECT_R4
        for i in range(3):
            sub = ppi[i]
            if sub is None or len(sub) == 0:
                return False, REJECT_R4

        ipts = ppi[0]
        dirs = ppi[1]
        sizes = ppi[2]
        n = len(sizes)

        # R1 — minimum packet count
        if n < self.MIN_PACKETS:
            return False, REJECT_R1

        # R8 — active-timeout short flows
        if flow_endreason_active == 1 and n < self.MIN_ACTIVE_TIMEOUT_PACKETS:
            return False, REJECT_R8

        # Soft warnings (do not reject)
        if n < 10:
            logger.warning("SHORT_FLOW: %s, len=%d", source_path, n)

        sizes_arr = np.asarray(sizes, dtype=np.float32)
        ipts_arr = np.asarray(ipts, dtype=np.float32)
        dirs_arr = np.asarray(dirs, dtype=np.float32)

        if np.any(sizes_arr > 1500.0):
            logger.warning(
                "OVERSIZE_PKT: %s, max_size=%.1f", source_path, float(sizes_arr.max())
            )
        if np.any(ipts_arr > 5000.0):
            logger.warning(
                "LONG_IPT: %s, max_ipt=%.1fms", source_path, float(ipts_arr.max())
            )

        # Validate individual values (non-hard; log only)
        if np.any(ipts_arr < 0.0):
            logger.warning(
                "NEGATIVE_IPT: %s, min_ipt=%.3f — clamping to 0", source_path, float(ipts_arr.min())
            )

        unique_dirs = set(float(d) for d in dirs_arr)
        invalid_dirs = unique_dirs - self.VALID_DIRECTIONS
        if invalid_dirs:
            logger.warning(
                "INVALID_DIRECTION: %s, values=%s", source_path, invalid_dirs
            )

        return True, ""

    # -----------------------------------------------------------------------
    # Statistical feature validation
    # -----------------------------------------------------------------------

    def validate_stats(
        self,
        stats_dict: Dict[str, Any],
        source_path: str = "",
        is_cesnet: bool = True,
    ) -> Tuple[bool, str]:
        """
        Validate all statistical fields needed for Branch B feature extraction.

        For CESNET (is_cesnet=True) the following keys are required:
          BYTES, BYTES_REV, PACKETS, PACKETS_REV, DURATION,
          PPI_LEN, PHIST_SRC_SIZES

        For ISCXVPN2016 (is_cesnet=False) only lengths and intervals matter;
        this method validates the derived quantities instead.

        Parameters
        ----------
        stats_dict : dict
            Record dictionary (may be a pandas Series or plain dict).
        source_path : str
            File or row identifier for logging.
        is_cesnet : bool
            True for CESNET-QUIC22 records; False for ISCXVPN2016 records.

        Returns
        -------
        (bool, str)
            (True, "") if valid.
            (False, REJECT_Rx) if a hard rule fires.
        """
        def _get(key: str, default: Any = None) -> Any:
            try:
                return stats_dict[key]
            except (KeyError, TypeError):
                return default

        if is_cesnet:
            # R2 — duration must be positive
            duration = _get("DURATION")
            if duration is None or not math.isfinite(float(duration)) or float(duration) <= 0.0:
                return False, REJECT_R2

            # R3 — total bytes must be non-zero
            bytes_fwd = _get("BYTES", 0.0)
            bytes_rev = _get("BYTES_REV", 0.0)
            try:
                if float(bytes_fwd) + float(bytes_rev) == 0.0:
                    return False, REJECT_R3
            except (TypeError, ValueError):
                return False, REJECT_R3

            # Check PHIST_SRC_SIZES
            phist = _get("PHIST_SRC_SIZES")
            if phist is not None:
                try:
                    phist_arr = np.asarray(phist, dtype=np.float32)
                    if phist_arr.sum() == 0:
                        logger.warning("EMPTY_PHIST: %s", source_path)
                except (TypeError, ValueError):
                    pass

            # Check flow end reason warnings
            if _get("FLOW_ENDREASON_IDLE", 0) == 1:
                logger.warning("IDLE_TIMEOUT: %s", source_path)

        else:
            # ISCXVPN2016 path
            lengths = _get("lengths", [])
            intervals = _get("intervals", [])

            try:
                lengths_list = list(lengths)
                intervals_list = list(intervals)
            except TypeError:
                return False, REJECT_R2

            # R1 — minimum packets
            if len(lengths_list) < self.MIN_PACKETS:
                return False, REJECT_R1

            # R2 — total duration proxy must be positive
            if len(intervals_list) == 0 or sum(float(x) for x in intervals_list) <= 0.0:
                return False, REJECT_R2

            if len(lengths_list) < 10:
                logger.warning("SHORT_FLOW: %s, len=%d", source_path, len(lengths_list))

        return True, ""

    # -----------------------------------------------------------------------
    # Label validation
    # -----------------------------------------------------------------------

    def validate_label(
        self,
        label: str,
        known_labels: Set[str],
    ) -> Tuple[bool, str]:
        """
        Validate that a raw label string maps to a known unified class.

        Parameters
        ----------
        label : str
            Raw label string (directory name or CATEGORY value).
        known_labels : set of str
            Set of all valid raw label values in the unified taxonomy,
            pre-lower-cased by the caller.

        Returns
        -------
        (bool, str)
            (True, "") if valid.
            (False, REJECT_R5) if not in known_labels.
        """
        if not isinstance(label, str) or label.strip() == "":
            return False, REJECT_R5
        if label.lower() not in known_labels:
            return False, REJECT_R5
        return True, ""

    # -----------------------------------------------------------------------
    # Feature array NaN/Inf guard
    # -----------------------------------------------------------------------

    def validate_feature_array(
        self,
        arr: np.ndarray,
        source_path: str = "",
        array_name: str = "feature",
    ) -> Tuple[bool, str]:
        """
        Check that a numpy array contains only finite values.

        Parameters
        ----------
        arr : np.ndarray
            Feature array to check.
        source_path : str
            Identifier for logging.
        array_name : str
            Human-readable array name for the log message.

        Returns
        -------
        (bool, str)
        """
        if not np.all(np.isfinite(arr)):
            logger.error(
                "NAN_OR_INF in %s at %s — discarding sample",
                array_name,
                source_path,
            )
            return False, REJECT_R7
        return True, ""

    # -----------------------------------------------------------------------
    # ISCXVPN2016 sequence-level checks
    # -----------------------------------------------------------------------

    def validate_iscxvpn_sequence(
        self,
        lengths: List[float],
        intervals: List[float],
        source_path: str = "",
    ) -> Tuple[bool, str]:
        """
        Combined hard + soft check for an ISCXVPN2016 JSON sample.

        Parameters
        ----------
        lengths : list of float
        intervals : list of float
        source_path : str

        Returns
        -------
        (bool, str)
        """
        n = len(lengths)

        # R1
        if n < self.MIN_PACKETS:
            return False, REJECT_R1

        # R2
        duration_proxy = sum(float(x) for x in intervals) if intervals else 0.0
        if duration_proxy <= 0.0:
            return False, REJECT_R2

        # Soft warnings
        if n < 10:
            logger.warning("SHORT_FLOW: %s, len=%d", source_path, n)

        sizes_arr = np.asarray(lengths, dtype=np.float32)
        ipts_arr = np.asarray(intervals, dtype=np.float32)

        if np.any(sizes_arr > 1500.0):
            logger.warning(
                "OVERSIZE_PKT: %s, max_size=%.1f", source_path, float(sizes_arr.max())
            )
        if np.any(ipts_arr > 5000.0):
            logger.warning(
                "LONG_IPT: %s, max_ipt=%.1fms", source_path, float(ipts_arr.max())
            )
        if len(sizes_arr) > 1 and float(np.std(sizes_arr)) == 0.0:
            logger.warning("ZERO_STD: %s, feature=lengths", source_path)
        if len(ipts_arr) > 1 and float(np.std(ipts_arr)) == 0.0:
            logger.warning("ZERO_STD: %s, feature=intervals", source_path)

        return True, ""
