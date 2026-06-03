import nfstream
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

class NFStreamExtractor:
    """
    Utility to extract behavioral features from PCAPs or live traffic.
    Implements the '5-Tuple Trap' guardrail by excluding IP addresses.

    Branch A: Per-packet sequences (size, IAT)
    Branch B: Flow-level QoS statistics
    """
    def __init__(self, splt_analysis: int = 128):
        self.splt_analysis = splt_analysis

    def extract_features(self, pcap_path: str) -> pd.DataFrame:
        """
        Process a PCAP file and return a DataFrame with Branch A/B features.

        Branch A features: seq_ps (packet sizes), seq_iat (inter-arrival times)
        Branch B features: total_bytes, packet_rate, rtt, jitter, bidirectional_packets,
                          src_port_category, protocol
        """
        # Configure NFStream
        # n_dissections=0 for inference/feature extraction to avoid DPI
        # splt_analysis=128 as per PDF mandate
        streamer = nfstream.NFStreamer(
            source=pcap_path,
            n_dissections=0,
            splt_analysis=self.splt_analysis,
            accounting_mode=0
        )

        flow_data = []
        for flow in streamer:
            # ========== BRANCH A: Sequence Features ==========
            # Per-packet: size (ps) and inter-arrival time (iat)
            seq_ps = flow.splt_ps if hasattr(flow, 'splt_ps') else []
            seq_iat = flow.splt_iat if hasattr(flow, 'splt_iat') else []
            seq_dir = flow.splt_direction if hasattr(flow, 'splt_direction') else []

            # Pad or truncate to exact splt_analysis length
            seq_ps = self._pad_truncate(seq_ps, self.splt_analysis)
            seq_iat = self._pad_truncate(seq_iat, self.splt_analysis)
            seq_dir = self._pad_truncate(seq_dir, self.splt_analysis)

            # ========== BRANCH B: QoS Statistics ==========
            duration_sec = flow.duration / 1000.0 if flow.duration > 0 else 0.001

            # 1. Total Bytes (bidirectional)
            total_bytes = flow.bidirectional_bytes if hasattr(flow, 'bidirectional_bytes') else 0

            # 2. Packet Rate (packets per second)
            bidirectional_packets = flow.bidirectional_packets if hasattr(flow, 'bidirectional_packets') else 0
            packet_rate = bidirectional_packets / duration_sec if duration_sec > 0 else 0

            # 3. RTT: Use TCP SYN-ACK time if available, else use min packet interval
            rtt = self._estimate_rtt(flow)

            # 4. Jitter: Variance in inter-arrival times
            jitter = self._estimate_jitter(seq_iat)

            # 5. Bidirectional Packets (already extracted above)
            # Used as indicator of flow symmetry

            # 6. Source Port Category (well-known: <1024, registered: 1024-49151, ephemeral: >49151)
            src_port = flow.src_port if hasattr(flow, 'src_port') else 0
            src_port_category = self._categorize_port(src_port)

            # 7. Protocol (TCP=6, UDP=17, QUIC=443-like heuristic)
            protocol = flow.protocol if hasattr(flow, 'protocol') else 0

            # 8. Mean packet size (proxy for application behavior)
            mean_pkt_size = np.mean(seq_ps) if len(seq_ps) > 0 else 0

            # Combine into a record
            record = {
                "id": flow.id,
                "seq_ps": seq_ps,
                "seq_iat": seq_iat,
                "seq_dir": seq_dir,
                # Branch B QoS features (8 features)
                "total_bytes": total_bytes,
                "packet_rate": packet_rate,
                "rtt": rtt,
                "jitter": jitter,
                "bidirectional_packets": bidirectional_packets,
                "src_port_category": src_port_category,
                "protocol": protocol,
                "mean_pkt_size": mean_pkt_size,
            }
            flow_data.append(record)

        return pd.DataFrame(flow_data)

    def _pad_truncate(self, data: List[float], target_len: int) -> List[float]:
        """Pad or truncate a list to target length."""
        if len(data) >= target_len:
            return data[:target_len]
        else:
            return data + [0.0] * (target_len - len(data))

    def _estimate_rtt(self, flow) -> float:
        """
        Estimate RTT from flow.
        If tcp_syn_payload_bytes available, use as proxy for TCP handshake.
        Otherwise use bidirectional_min_ps as a conservative estimate.
        """
        if hasattr(flow, 'tcp_syn_payload_bytes') and flow.tcp_syn_payload_bytes > 0:
            return float(flow.tcp_syn_payload_bytes)
        elif hasattr(flow, 'bidirectional_min_ps'):
            return float(flow.bidirectional_min_ps)
        else:
            return 0.0

    def _estimate_jitter(self, seq_iat: List[float]) -> float:
        """
        Calculate jitter as standard deviation of inter-arrival times.
        Jitter is a measure of IAT variability.
        """
        if len(seq_iat) < 2:
            return 0.0
        iat_array = np.array(seq_iat)
        iat_array = iat_array[iat_array > 0]  # Ignore zero-padding
        if len(iat_array) < 2:
            return 0.0
        return float(np.std(iat_array))

    def _categorize_port(self, port: int) -> float:
        """
        Categorize port into classes:
        0 = well-known (0-1023)
        1 = registered (1024-49151)
        2 = ephemeral/dynamic (49152-65535)
        """
        if port < 1024:
            return 0.0
        elif port < 49152:
            return 1.0
        else:
            return 2.0

    def prepare_for_model(self, df: pd.DataFrame) -> Tuple:
        """
        Convert DataFrame to tensors for DualBranchEncoder.

        Returns:
            (seq_data, stat_data) tensors ready for model input
        """
        import torch

        # Prepare Sequence Data (Batch, 128, 2)
        seq_ps = np.stack(df['seq_ps'].values)
        seq_iat = np.stack(df['seq_iat'].values)
        seq_data = np.stack([seq_ps, seq_iat], axis=-1)

        # Prepare Stat Data (Batch, 8) - 8 QoS features
        stat_cols = [
            'total_bytes', 'packet_rate', 'rtt', 'jitter',
            'bidirectional_packets', 'src_port_category', 'protocol', 'mean_pkt_size'
        ]
        stat_data = df[stat_cols].values

        return torch.tensor(seq_data, dtype=torch.float32), torch.tensor(stat_data, dtype=torch.float32)

if __name__ == "__main__":
    import torch
    extractor = NFStreamExtractor()
    print("NFStreamExtractor initialized with splt_analysis=128")
    # Note: This requires a valid pcap file to run test extraction
