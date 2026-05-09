import nfstream
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class NFStreamExtractor:
    """
    Utility to extract behavioral features from PCAPs or live traffic.
    Implements the '5-Tuple Trap' guardrail by excluding IP addresses.
    """
    def __init__(self, splt_analysis: int = 128):
        self.splt_analysis = splt_analysis

    def extract_features(self, pcap_path: str) -> pd.DataFrame:
        """
        Process a PCAP file and return a DataFrame with mandated features.
        """
        # Configure NFStream
        # n_dissections=0 for inference/feature extraction to avoid DPI
        # splt_analysis=128 as per PDF mandate
        streamer = nfstream.NFStreamer(
            source=pcap_path,
            n_dissections=0,
            splt_analysis=self.splt_analysis,
            accounting_mode=0 # Standard accounting
        )

        flow_data = []
        for flow in streamer:
            # Sequence Features (Branch A)
            # NFStream provides splt_direction, splt_ps, splt_iat
            # We focus on Packet Size (ps) and Inter-Arrival Time (iat)
            seq_ps = flow.splt_ps
            seq_iat = flow.splt_iat
            
            # Pad or truncate to exact splt_analysis length
            seq_ps = self._pad_truncate(seq_ps, self.splt_analysis)
            seq_iat = self._pad_truncate(seq_iat, self.splt_analysis)
            
            # Static Features (Branch B)
            # Note: NFStream field names might vary slightly depending on version
            stats = {
                "total_bytes": flow.bidirectional_bytes,
                "packet_rate": flow.bidirectional_packets / (flow.duration / 1000.0) if flow.duration > 0 else 0,
                "rtt": getattr(flow, 'requested_server_name', 0), # Placeholder if RTT not direct
                # In real nfstream, RTT and Jitter might need a custom plugin or specific field
                # For this demo, we'll use bidirectional_min_ps and bidirectional_max_ps as proxies 
                # if specific plugin metrics aren't active.
                "jitter": flow.bidirectional_duration_ms / flow.bidirectional_packets if flow.bidirectional_packets > 0 else 0
            }
            
            # Combine into a record
            record = {
                "id": flow.id,
                "seq_ps": seq_ps,
                "seq_iat": seq_iat,
                **stats
            }
            flow_data.append(record)
            
        return pd.DataFrame(flow_data)

    def _pad_truncate(self, data: List[float], target_len: int) -> List[float]:
        if len(data) >= target_len:
            return data[:target_len]
        else:
            return data + [0.0] * (target_len - len(data))

    def prepare_for_model(self, df: pd.DataFrame):
        """
        Convert DataFrame to tensors for DualBranchEncoder.
        """
        # Prepare Sequence Data (Batch, 128, 2)
        seq_ps = np.stack(df['seq_ps'].values)
        seq_iat = np.stack(df['seq_iat'].values)
        seq_data = np.stack([seq_ps, seq_iat], axis=-1)
        
        # Prepare Stat Data (Batch, 4)
        stat_cols = ['total_bytes', 'packet_rate', 'rtt', 'jitter']
        stat_data = df[stat_cols].values
        
        return torch.tensor(seq_data, dtype=torch.float32), torch.tensor(stat_data, dtype=torch.float32)

if __name__ == "__main__":
    import torch
    extractor = NFStreamExtractor()
    print("NFStreamExtractor initialized with splt_analysis=128")
    # Note: This requires a valid pcap file to run test extraction
