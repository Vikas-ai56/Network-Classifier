"""
KPI: Zero-Day Generalization >= 85%
Holds out N classes from training, builds prototypes from k-shot examples only,
then evaluates classification accuracy on the held-out (unseen) classes.
Usage: python zero_day_test.py [--data_dir ...] [--model_path ...]
"""

import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.getLogger("src.feature_engineering").setLevel(logging.ERROR)
logging.getLogger("src.data_validator").setLevel(logging.ERROR)

from src.models_dual_branch import DualBranchEncoder
from src.dataset_unified import UnifiedFlowDataset, UNIFIED_CLASS_NAMES


def extract_embeddings(model, loader, device):
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for seq, stat, y in loader:
            emb = model(seq.to(device), stat.to(device))
            all_embs.append(emb.cpu())
            all_labels.append(y)
    return torch.cat(all_embs), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="dataset/netmamba/ISCXVPN2016/images_sampled_new")
    parser.add_argument("--model_path",   default="model/best_model.pth")
    parser.add_argument("--batch_size",   type=int, default=256)
    parser.add_argument("--k_shot",       type=int, default=5,  help="support shots for zero-day classes")
    parser.add_argument("--n_zero_day",   type=int, default=2,  help="number of held-out zero-day classes")
    parser.add_argument("--n_trials",     type=int, default=50, help="random trials to average over")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=18, d_model=256, embed_dim=256)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 0) + 1}")

    # Load ISCXVPN2016
    dataset = UnifiedFlowDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    unique_classes = list(np.unique(dataset.labels))
    n_classes = len(unique_classes)
    print(f"Dataset: {len(dataset)} samples | {n_classes} classes")
    print(f"Zero-day test: hold out {args.n_zero_day} classes, {args.k_shot}-shot, {args.n_trials} trials")

    if n_classes <= args.n_zero_day:
        print(f"ERROR: need more than {args.n_zero_day} classes for zero-day test")
        return

    # Extract all embeddings once
    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings(model, loader, device)

    rng = np.random.default_rng(0)
    trial_accs = []

    for trial in range(args.n_trials):
        # Pick zero-day classes
        zero_day = rng.choice(unique_classes, args.n_zero_day, replace=False).tolist()
        known    = [c for c in unique_classes if c not in zero_day]

        # Build prototypes — known classes: mean of ALL their embeddings
        #                     zero-day classes: mean of k_shot embeddings only
        proto_vecs, proto_class_ids = [], []

        for c in known:
            mask = labels == c
            proto_vecs.append(embeddings[mask].mean(0))
            proto_class_ids.append(c)

        for c in zero_day:
            idx = (labels == c).nonzero(as_tuple=True)[0]
            support_idx = idx[:args.k_shot]
            proto_vecs.append(embeddings[support_idx].mean(0))
            proto_class_ids.append(c)

        prototypes = F.normalize(torch.stack(proto_vecs), p=2, dim=1)
        proto_ids  = torch.tensor(proto_class_ids)

        # Evaluate on zero-day query samples (everything after k_shot support)
        correct, total = 0, 0
        for c in zero_day:
            idx = (labels == c).nonzero(as_tuple=True)[0]
            query_idx = idx[args.k_shot:]
            if len(query_idx) == 0:
                continue
            query_emb  = embeddings[query_idx]
            dists      = torch.cdist(query_emb, prototypes)
            pred_proto = (-dists).argmax(dim=1)
            preds      = proto_ids[pred_proto]
            correct   += (preds == c).sum().item()
            total     += len(query_idx)

        if total > 0:
            trial_accs.append(correct / total)

    mean_acc = float(np.mean(trial_accs))
    std_acc  = float(np.std(trial_accs))

    print(f"\n=== Zero-Day Generalization ===")
    print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}  over {len(trial_accs)} trials")
    print(f"KPI      : {'✓ ≥85% MET' if mean_acc >= 0.85 else '✗ below 85% KPI'}")

    # Per-trial breakdown
    print(f"\nMin: {min(trial_accs):.4f}  Max: {max(trial_accs):.4f}  Median: {float(np.median(trial_accs)):.4f}")


if __name__ == "__main__":
    main()
