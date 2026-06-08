"""
KPI: Zero-day generalization >= 85%.

Simulates encountering a completely unseen traffic class at inference time.

Protocol:
  1. Train prototypes for N-1 known classes from the full dataset.
  2. Hold out one class entirely (the "zero-day" class).
  3. Give the model K labeled examples of the zero-day class (K-shot).
  4. Build its prototype from those K examples.
  5. Classify all remaining zero-day samples using nearest-prototype.
  6. Repeat for every class as the held-out class and report mean accuracy.

A model that truly generalizes learns embeddings where traffic semantics
cluster naturally, allowing new classes to slot in with just a few examples.

Usage:
  python scripts/zero_day_test.py --checkpoint model/best_model.pth
  python scripts/zero_day_test.py --checkpoint model/best_model.pth --k_shot 5
"""

import argparse, os, sys, json, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models_dual_branch import DualBranchEncoder
from src.feature_engineering import extract_seq_from_iscxvpn, extract_stat_from_iscxvpn
from src.dataset_unified import LABEL_MAP, UNIFIED_CLASS_NAMES

DATA_DIR = "dataset/netmamba/ISCXVPN2016/images_sampled_new"


# ---------------------------------------------------------------------------
# Dataset — uses same feature extraction as training (feature_engineering.py)
# ---------------------------------------------------------------------------

class ISCXDataset(Dataset):
    def __init__(self, root_dir):
        all_files = sorted(glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True))
        self.samples = []   # (path, unified_label)
        skipped = 0
        for path in all_files:
            raw_label = os.path.basename(os.path.dirname(path)).lower()
            if raw_label not in LABEL_MAP:
                skipped += 1
                continue
            self.samples.append((path, LABEL_MAP[raw_label]))
        present_ids = sorted(set(lbl for _, lbl in self.samples))
        self.classes = [UNIFIED_CLASS_NAMES[i] for i in present_ids]
        self.labels  = [lbl for _, lbl in self.samples]
        if skipped:
            print(f"  {skipped} samples skipped (unknown label)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path) as f:
            d = json.load(f)
        lengths   = d.get("lengths",   [])
        intervals = d.get("intervals", [])
        seq  = extract_seq_from_iscxvpn(lengths, intervals)   # (30, 3)
        stat = extract_stat_from_iscxvpn(lengths, intervals)  # (18,)
        return torch.tensor(seq), torch.tensor(stat), torch.tensor(label)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_all(model, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labels = [], []
    model.eval()
    with torch.no_grad():
        for seq, stat, lbl in loader:
            e = model(seq.to(device), stat.to(device))
            embs.append(e.cpu())
            labels.append(lbl)
    return torch.cat(embs), torch.cat(labels)


# ---------------------------------------------------------------------------
# Zero-day evaluation
# ---------------------------------------------------------------------------

def zero_day_eval(embs, labels, classes, class_ids, k_shot):
    """
    For each class as the held-out zero-day class:
      - Build prototypes for all OTHER classes from their full embeddings.
      - Randomly pick k_shot examples of the held-out class -> build its prototype.
      - Classify all remaining held-out samples by nearest prototype.
    """
    results = {}

    for held_out_id, held_out_name in zip(class_ids, classes):
        held_mask  = labels == held_out_id
        known_mask = ~held_mask

        known_embs   = embs[known_mask]
        known_labels = labels[known_mask]
        held_embs    = embs[held_mask]

        if held_embs.shape[0] <= k_shot:
            print(f"  Skipping {held_out_name}: not enough samples ({held_embs.shape[0]})")
            continue

        # Build prototypes for known classes
        prototypes = {}
        for c in class_ids:
            if c == held_out_id:
                continue
            mask = known_labels == c
            if mask.any():
                prototypes[c] = F.normalize(known_embs[mask].mean(0), dim=0)

        # K-shot prototype for zero-day class
        perm    = torch.randperm(held_embs.shape[0])
        support = held_embs[perm[:k_shot]]
        query   = held_embs[perm[k_shot:]]
        prototypes[held_out_id] = F.normalize(support.mean(0), dim=0)

        # Nearest-prototype classification on query set
        sorted_ids   = sorted(prototypes.keys())
        proto_matrix = torch.stack([prototypes[c] for c in sorted_ids])  # (n, d)

        sims         = torch.matmul(query, proto_matrix.T)               # (Q, n)
        preds        = sims.argmax(dim=1)
        pred_classes = [sorted_ids[p.item()] for p in preds]
        acc = sum(p == held_out_id for p in pred_classes) / len(pred_classes)
        results[held_out_name] = acc

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model/best_model.pth")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    parser.add_argument("--k_shot",     type=int, default=5, help="Labeled examples of new class")
    parser.add_argument("--n_trials",   type=int, default=10, help="Repeat each hold-out N times")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"K-shot     : {args.k_shot}  (labeled examples of unseen class)")
    print(f"Trials     : {args.n_trials} per held-out class\n")

    # Load model
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=18, d_model=256, embed_dim=256)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()

    # Load dataset & extract embeddings
    dataset   = ISCXDataset(args.data_dir)
    class_ids = sorted(set(dataset.labels))
    print(f"Dataset    : {len(dataset)} samples | {len(dataset.classes)} classes: {dataset.classes}")
    print("Extracting embeddings...")
    embs, labels = extract_all(model, dataset, device)
    print(f"Embeddings : {embs.shape}\n")

    # Run zero-day eval over multiple trials for stability
    all_trials = {cls: [] for cls in dataset.classes}
    for trial in range(args.n_trials):
        trial_results = zero_day_eval(embs, labels, dataset.classes, class_ids, args.k_shot)
        for cls, acc in trial_results.items():
            all_trials[cls].append(acc)

    # Report
    print("--- Per-Class Zero-Day Accuracy ---")
    print(f"  {'Class':<18}  {'Mean Acc':>9}  {'Std':>7}  {'KPI >=85%'}")
    print(f"  {'-'*50}")
    all_accs = []
    for cls in dataset.classes:
        if all_trials[cls]:
            mean_acc = np.mean(all_trials[cls])
            std_acc  = np.std(all_trials[cls])
            kpi      = "✓" if mean_acc >= 0.85 else "✗"
            print(f"  {cls:<18}  {mean_acc*100:>8.2f}%  {std_acc*100:>6.2f}%  {kpi}")
            all_accs.append(mean_acc)

    overall = np.mean(all_accs) if all_accs else 0.0
    passed  = overall >= 0.85
    print(f"\n{'='*52}")
    print(f"  KPI: Zero-day generalization >= 85%")
    print(f"  Mean across all held-out classes: {overall*100:.2f}%")
    print(f"  Result: {'✓  KPI MET' if passed else '✗  KPI not met'}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
