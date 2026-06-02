import math
import numpy as np
import pandas as pd
from typing import List

try:
    import nfstream
    HAS_NFSTREAM = True
except ImportError:
    HAS_NFSTREAM = False

# Normalization constants — must match feature_engineering.py exactly
_MAX_PACKET_SIZE = 1500.0
_MAX_IPT_MS = 5000.0
_LOG1P_MAX_PKT = math.log1p(_MAX_PACKET_SIZE)
_LOG1P_MAX_IPT = math.log1p(_MAX_IPT_MS)


def _size_norm(v: float) -> float:
    return math.log1p(min(max(v, 0.0), _MAX_PACKET_SIZE)) / _LOG1P_MAX_PKT


def _ipt_norm(v: float) -> float:
    return math.log1p(min(max(v, 0.0), _MAX_IPT_MS)) / _LOG1P_MAX_IPT


class NFStreamExtractor:
    """
    Extracts behavioral features from PCAPs or live traffic for inference.

    Implements the 5-Tuple Purge (no src/dst IP, ports, protocol) and
    Handshake Masking guardrails from the THETA architecture spec.

    Output tensor shapes match the training pipeline exactly:
      seq_data:  (batch, 128, 3)  — [size_norm, ipt_norm, direction]
      stat_data: (batch, 18)      — (not fully implemented; returns 4-elem proxy
                                     for live inference; batch training uses
                                     feature_engineering.extract_stat_features)
    """
    def __init__(self, splt_analysis: int = 128):
        self.splt_analysis = splt_analysis

    def extract_features(self, pcap_path: str) -> pd.DataFrame:
        """Process a PCAP file and return a DataFrame of per-flow features."""
        if not HAS_NFSTREAM:
            raise ImportError("nfstream is not installed. Run: pip install nfstream")

        streamer = nfstream.NFStreamer(
            source=pcap_path,
            n_dissections=0,       # 5-tuple purge: no DPI
            splt_analysis=self.splt_analysis,
            accounting_mode=0,
        )

        flow_data = []
        for flow in streamer:
            # Filter degenerate connections (< 10 bidirectional packets)
            if flow.bidirectional_packets < 10:
                continue

            # --- Branch A: Sequence (size, IAT, direction) ---
            seq_ps = list(getattr(flow, "splt_ps", []) or [])
            seq_iat = list(getattr(flow, "splt_iat", []) or [])
            # direction: +1 = client→server (positive ps), -1 = server→client (negative ps)
            seq_dir = [1.0 if ps >= 0 else -1.0 for ps in seq_ps]
            # Use absolute size for normalization
            seq_ps_abs = [abs(ps) for ps in seq_ps]

            seq_ps_abs = self._pad_truncate(seq_ps_abs, self.splt_analysis)
            seq_iat = self._pad_truncate(seq_iat, self.splt_analysis)
            seq_dir = self._pad_truncate(seq_dir, self.splt_analysis)

            # Handshake masking: zero-out first 5 positions to prevent SNI leakage
            for i in range(min(5, self.splt_analysis)):
                seq_ps_abs[i] = 0.0
                seq_iat[i] = 0.0
                seq_dir[i] = 0.0

            # --- Branch B: Statistical proxy (4 scalars for live inference) ---
            duration_ms = getattr(flow, "bidirectional_duration_ms", 1.0) or 1.0
            total_bytes = getattr(flow, "bidirectional_bytes", 0)
            bidirectional_packets = max(flow.bidirectional_packets, 1)
            packet_rate = bidirectional_packets / (duration_ms / 1000.0)
            mean_ipt = duration_ms / bidirectional_packets

            flow_data.append({
                "id": flow.id,
                "seq_ps": seq_ps_abs,
                "seq_iat": seq_iat,
                "seq_dir": seq_dir,
                "total_bytes": total_bytes,
                "packet_rate": packet_rate,
                "mean_ipt_ms": mean_ipt,
                "duration_ms": duration_ms,
            })

        return pd.DataFrame(flow_data)

    def _pad_truncate(self, data: List[float], target_len: int) -> List[float]:
        if len(data) >= target_len:
            return list(data[:target_len])
        return list(data) + [0.0] * (target_len - len(data))

    def prepare_for_model(self, df: pd.DataFrame):
        """
        Convert DataFrame to tensors matching DualBranchEncoder input contract.

        Returns:
          seq_tensor:  FloatTensor (batch, 128, 3)  — [size_norm, ipt_norm, dir]
          stat_tensor: FloatTensor (batch, 4)        — proxy stats for live inference
                       (full 18-feature vector requires compute_phist_from_lengths;
                        use feature_engineering.extract_stat_from_iscxvpn for batch)
        """
        import torch

        batch = len(df)
        seq_data = np.zeros((batch, self.splt_analysis, 3), dtype=np.float32)

        for i, row in df.iterrows():
            ps = np.array(row["seq_ps"], dtype=np.float32)
            iat = np.array(row["seq_iat"], dtype=np.float32)
            d = np.array(row["seq_dir"], dtype=np.float32)
            # Apply same log1p normalization as feature_engineering.py
            ps_norm = np.array([_size_norm(float(v)) for v in ps], dtype=np.float32)
            iat_norm = np.array([_ipt_norm(float(v)) for v in iat], dtype=np.float32)
            seq_data[i] = np.stack([ps_norm, iat_norm, d], axis=-1)

        # 4-element proxy stat vector for live inference
        stat_cols = ["total_bytes", "packet_rate", "mean_ipt_ms", "duration_ms"]
        stat_data = df[stat_cols].values.astype(np.float32)

        return (
            torch.tensor(seq_data, dtype=torch.float32),
            torch.tensor(stat_data, dtype=torch.float32),
        )


if __name__ == "__main__":
    extractor = NFStreamExtractor(splt_analysis=128)
    print(f"NFStreamExtractor initialized. nfstream available: {HAS_NFSTREAM}")
    print("seq output shape per flow: (128, 3) — [size_norm, ipt_norm, direction]")
    print("Use prepare_for_model() to get tensors for DualBranchEncoder inference.")
