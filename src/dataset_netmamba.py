import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob

class NetMambaDataset(Dataset):
    """
    Dataset class to load NetMamba pre-processed JSON files.
    JSON structure: {"lengths": [...], "intervals": [...]}
    """
    def __init__(self, root_dir, seq_len=128, split='all', val_ratio=0.15, test_ratio=0.15,
                 transform=None, seed=42):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform

        all_paths = sorted(glob(os.path.join(root_dir, "**/*.json"), recursive=True))
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in all_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        all_labels = [self.class_to_idx[os.path.basename(os.path.dirname(p))] for p in all_paths]

        if split == 'all':
            self.file_paths = all_paths
            self.labels = all_labels
        else:
            rng = np.random.default_rng(seed)
            indices = np.arange(len(all_paths))
            rng.shuffle(indices)
            n = len(indices)
            n_test = int(n * test_ratio)
            n_val = int(n * val_ratio)
            test_idx = indices[:n_test]
            val_idx = indices[n_test:n_test + n_val]
            train_idx = indices[n_test + n_val:]

            if split == 'train':
                chosen = train_idx
            elif split == 'val':
                chosen = val_idx
            elif split == 'test':
                chosen = test_idx
            else:
                raise ValueError(f"split must be 'train', 'val', 'test', or 'all', got {split}")

            self.file_paths = [all_paths[i] for i in chosen]
            self.labels = [all_labels[i] for i in chosen]

        print(f"[NetMambaDataset] split={split}: {len(self.file_paths)} samples, "
              f"{len(self.classes)} classes.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract features
        lengths = data.get('lengths', [])
        intervals = data.get('intervals', [])
        
        # Pad or truncate to seq_len
        lengths = self._pad_truncate(lengths, self.seq_len)
        intervals = self._pad_truncate(intervals, self.seq_len)
        
        # Branch A Data: (seq_len, 2)
        seq_data = np.stack([lengths, intervals], axis=-1).astype(np.float32)
        
        # Branch B Data: (8) - QoS statistics
        # real_feature_dims: [total_bytes, packet_rate, rtt, jitter, bidirectional_packets,
        #                     src_port_category, protocol, mean_pkt_size]
        # Since JSON doesn't have these, derive from sequences as behavioral proxies
        lengths_arr = np.array(lengths, dtype=np.float32)
        intervals_arr = np.array(intervals, dtype=np.float32)

        stat_data = np.array([
            np.sum(lengths_arr) if len(lengths_arr) > 0 else 0,          # total_bytes proxy
            len(lengths) / (np.sum(intervals_arr) / 1000.0) if np.sum(intervals_arr) > 0 else 0,  # packet_rate
            np.mean(intervals_arr) if len(intervals_arr) > 0 else 0,     # rtt proxy (mean IAT)
            np.std(intervals_arr) if len(intervals_arr) > 0 else 0,      # jitter (std IAT)
            len(lengths),                                                 # bidirectional_packets
            1.0,                                                          # src_port_category (default to registered)
            6.0,                                                          # protocol (default to TCP=6)
            np.mean(lengths_arr) if len(lengths_arr) > 0 else 0,         # mean_pkt_size
        ]).astype(np.float32)
        
        return torch.tensor(seq_data), torch.tensor(stat_data), torch.tensor(label)

    def _pad_truncate(self, data, target_len):
        if len(data) >= target_len:
            return data[:target_len]
        else:
            return data + [0.0] * (target_len - len(data))

if __name__ == "__main__":
    # Test loading
    dataset_path = "datasets/netmamba/ISCXVPN2016/images_sampled_new"
    if os.path.exists(dataset_path):
        ds = NetMambaDataset(dataset_path)
        seq, stat, lbl = ds[0]
        print(f"Sample Sequence Shape: {seq.shape}")
        print(f"Sample Stat Shape: {stat.shape}")
        print(f"Label: {lbl} ({ds.classes[lbl]})")
    else:
        print(f"Path not found: {dataset_path}")
