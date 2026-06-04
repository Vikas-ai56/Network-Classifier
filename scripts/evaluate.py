"""
Evaluate trained DualBranchEncoder on CESNET-QUIC22.

Two modes:
  --csv   : run locally on a CSV file (no HDF5 needed)
  --stream: run on CESNET HDF5 via cesnet-datazoo (requires downloaded dataset)

Pipeline:
  1. Load checkpoint + prototypes (pre-built on GPU instance)
  2. Classify flows by nearest prototype (cosine similarity)
  3. Report per-class and overall accuracy
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models_dual_branch import DualBranchEncoder
from src.streaming_dataset import _parse_ppi, _process_chunk, _build_app_int_map
from src.feature_engineering import SEQ_LEN, extract_seq_features, extract_stat_features
from src.dataset_unified import UNIFIED_CLASS_NAMES, NUM_CLASSES, LABEL_MAP
from src.data_validator import FlowValidator


def load_model(checkpoint_path, device):
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=18, d_model=128, embed_dim=256)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint — epoch {ckpt.get('epoch', 0) + 1}, loss {ckpt.get('loss', '?'):.4f}")
    return model


def classify(model, prototypes, seq_batch, stat_batch, device):
    classes = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[c] for c in classes]).to(device)
    with torch.no_grad():
        embs = model(seq_batch.to(device), stat_batch.to(device))
        sims = torch.matmul(embs, proto_matrix.T)
        return [classes[i] for i in sims.argmax(dim=1).tolist()]


def evaluate_csv(model, prototypes, csv_path, device):
    """Evaluate on a local CSV file (cesnet_stream_500.csv format)."""
    from src.streaming_dataset import _build_app_int_map

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Build APP_INT_MAP from servicemap if available alongside the CSV
    # Fall back to LABEL_MAP (CATEGORY column) if no servicemap
    app_int_map = None

    samples = _process_chunk(df, app_int_map)
    if not samples:
        print("No valid samples found in CSV — check CATEGORY/APP columns")
        return

    seq = torch.from_numpy(np.stack([s[0] for s in samples]))
    stat = torch.from_numpy(np.stack([s[1] for s in samples]))
    labels = [s[2] for s in samples]

    correct = {c: 0 for c in range(NUM_CLASSES)}
    total   = {c: 0 for c in range(NUM_CLASSES)}

    batch_size = 64
    for i in range(0, len(samples), batch_size):
        preds = classify(model, prototypes, seq[i:i+batch_size], stat[i:i+batch_size], device)
        for pred, true in zip(preds, labels[i:i+batch_size]):
            total[true] += 1
            if pred == true:
                correct[true] += 1

    _print_results(correct, total)


def evaluate_stream(model, prototypes, data_root, size, device):
    """Evaluate on CESNET test split via cesnet-datazoo."""
    from src.streaming_dataset import CESNETStreamingDataset
    from cesnet_datazoo.datasets import CESNET_QUIC22
    from cesnet_datazoo.config import DatasetConfig, AppSelection

    ds = CESNETStreamingDataset(data_root=data_root, size=size, split="test", shuffle_chunks=False)
    loader = DataLoader(ds, batch_size=256, num_workers=2, persistent_workers=True)

    classes = sorted(prototypes.keys())
    proto_matrix = torch.stack([prototypes[c] for c in classes]).to(device)

    correct = {c: 0 for c in range(NUM_CLASSES)}
    total   = {c: 0 for c in range(NUM_CLASSES)}

    model.eval()
    with torch.no_grad():
        for seq, stat, labels in loader:
            embs = model(seq.to(device), stat.to(device))
            sims = torch.matmul(embs, proto_matrix.T)
            preds = [classes[i] for i in sims.argmax(dim=1).tolist()]
            for pred, true in zip(preds, labels.tolist()):
                total[true] += 1
                if pred == true:
                    correct[true] += 1

    _print_results(correct, total)


def _print_results(correct, total):
    print("\n--- Per-Class Accuracy ---")
    overall_c, overall_t = 0, 0
    for c in sorted(total):
        if total[c] == 0:
            continue
        acc = correct[c] / total[c]
        name = UNIFIED_CLASS_NAMES.get(c, str(c))
        print(f"  {name:22s}  {acc:.3f}  ({correct[c]}/{total[c]})")
        overall_c += correct[c]
        overall_t += total[c]
    if overall_t:
        print(f"\n  Overall accuracy: {overall_c / overall_t:.4f}  ({overall_c}/{overall_t})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="model/checkpoint_latest.pth")
    parser.add_argument("--prototypes",  default="model/prototypes.pth")
    parser.add_argument("--csv",         default=None, help="Local CSV file to evaluate on")
    parser.add_argument("--data_root",   default="/workspace/cesnet_cache")
    parser.add_argument("--size",        default="XS", choices=["XS", "S", "M"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    if not os.path.exists(args.prototypes):
        print(f"Prototypes not found at {args.prototypes}")
        print("Run on the GPU instance first to build prototypes, then push to git.")
        return

    prototypes = torch.load(args.prototypes, map_location=device)
    print(f"Loaded prototypes for classes: {sorted(prototypes.keys())}")

    if args.csv:
        evaluate_csv(model, prototypes, args.csv, device)
    else:
        evaluate_stream(model, prototypes, args.data_root, args.size, device)


if __name__ == "__main__":
    main()
