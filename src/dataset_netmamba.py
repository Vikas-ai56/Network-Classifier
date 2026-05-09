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
    def __init__(self, root_dir, seq_len=128, transform=None):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        
        # Find all JSON files and extract labels from directory names
        self.file_paths = glob(os.path.join(root_dir, "**/*.json"), recursive=True)
        self.classes = sorted(list(set([os.path.basename(os.path.dirname(p)) for p in self.file_paths])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Loaded {len(self.file_paths)} samples across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label_name = os.path.basename(os.path.dirname(file_path))
        label = self.class_to_idx[label_name]
        
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
        
        # Branch B Data: (4) - Using placeholders as the dataset lacks static stats
        # In a real scenario, these would be [TotalBytes, PacketRate, RTT, Jitter]
        # We use [MeanLength, VarLength, MeanInterval, VarInterval] as behavioral proxies
        stat_data = np.array([
            np.mean(lengths) if lengths else 0,
            np.std(lengths) if lengths else 0,
            np.mean(intervals) if intervals else 0,
            np.std(intervals) if intervals else 0
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
