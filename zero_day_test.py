"""
KPI: Zero-Day Generalization >= 85%

Simulates zero-day detection: holds out N classes, then classifies them using
only k_shot labeled examples (no retraining). This tests whether the pre-trained
encoder generalizes to traffic types unseen during training.

Evaluation uses k-NN over k_shot reference samples (more reliable than a single
centroid for high-variance traffic clusters).

Usage:
  python zero_day_test.py [--model_path ...] [--k_shot 50] [--n_zero_day 2]
"""

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier

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
            all_embs.append(emb.cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_embs), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="dataset/netmamba/ISCXVPN2016/images_sampled_new")
    parser.add_argument("--model_path", default="model/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--k_shot",     type=int, default=50,
                        help="labeled examples per zero-day class (reference gallery)")
    parser.add_argument("--n_zero_day", type=int, default=2,
                        help="number of held-out zero-day classes per trial")
    parser.add_argument("--n_trials",   type=int, default=50)
    parser.add_argument("--knn_k",      type=int, default=5,
                        help="k for the k-NN classifier over the shot gallery")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=18, d_model=256, embed_dim=256)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    key = "encoder_state_dict" if "encoder_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 0) + 1}  (key={key})")

    # Load ISCXVPN2016
    dataset = UnifiedFlowDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    unique_classes = list(np.unique(dataset.labels))
    n_classes = len(unique_classes)
    print(f"Dataset: {len(dataset)} samples | {n_classes} classes")
    print(f"Zero-day protocol: {args.n_zero_day} held-out classes | "
          f"{args.k_shot}-shot k-NN gallery | {args.n_trials} trials")

    if n_classes <= args.n_zero_day:
        print(f"ERROR: need more than {args.n_zero_day} classes")
        return

    print("Extracting embeddings...")
    X, y = extract_embeddings(model, loader, device)

    rng = np.random.default_rng(0)
    trial_accs = []

    for trial in range(args.n_trials):
        # Hold out zero-day classes
        zero_day = rng.choice(unique_classes, args.n_zero_day, replace=False).tolist()
        known    = [c for c in unique_classes if c not in zero_day]

        # Build k-NN gallery:
        #   known classes → all samples (they were "seen" during adaptation)
        #   zero-day classes → k_shot samples only (simulates few-shot availability)
        gallery_X, gallery_y = [], []

        for c in known:
            mask = y == c
            gallery_X.append(X[mask])
            gallery_y.extend([c] * mask.sum())

        for c in zero_day:
            idx = np.where(y == c)[0]
            shot_idx = idx[:args.k_shot]
            gallery_X.append(X[shot_idx])
            gallery_y.extend([c] * len(shot_idx))

        gallery_X = np.concatenate(gallery_X)
        gallery_y = np.array(gallery_y)

        knn = KNeighborsClassifier(n_neighbors=min(args.knn_k, len(gallery_y)),
                                   metric="cosine", n_jobs=1)
        knn.fit(gallery_X, gallery_y)

        # Evaluate on zero-day QUERY samples (all samples beyond k_shot)
        correct, total = 0, 0
        for c in zero_day:
            idx = np.where(y == c)[0]
            query_idx = idx[args.k_shot:]
            if len(query_idx) == 0:
                continue
            preds    = knn.predict(X[query_idx])
            correct += (preds == c).sum()
            total   += len(query_idx)

        if total > 0:
            trial_accs.append(correct / total)

    mean_acc = float(np.mean(trial_accs))
    std_acc  = float(np.std(trial_accs))
    kpi_ok   = mean_acc >= 0.85

    print(f"\n=== Zero-Day Generalization ===")
    print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}  over {len(trial_accs)} trials")
    print(f"KPI      : {'✓ ≥85% MET' if kpi_ok else '✗ below 85% KPI'}")
    print(f"Min: {min(trial_accs):.4f}  Max: {max(trial_accs):.4f}  "
          f"Median: {float(np.median(trial_accs)):.4f}")


if __name__ == "__main__":
    main()
