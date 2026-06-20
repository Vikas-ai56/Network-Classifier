"""
KPI: Zero-Day Generalization >= 85%

Zero-day protocol: since the model was trained on CESNET-QUIC22 and ALL ISCXVPN2016
classes are unseen, zero-day generalization means classifying new traffic types using
only k_shot labeled examples per class (no retraining).

Balanced k-shot evaluation:
  - k_shot samples per class → reference gallery (equal for all classes)
  - Remaining samples → query set
  - k-NN classifies queries using the gallery
  - Report: overall accuracy + per-class accuracy

Usage:
  python zero_day_test.py [--model_path ...] [--k_shot 50]
"""

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

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
                        help="labeled examples per class for zero-day gallery")
    parser.add_argument("--n_trials",   type=int, default=50,
                        help="random trials to average (different k_shot subsets)")
    parser.add_argument("--knn_k",      type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=256, embed_dim=256)
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
    class_names = [UNIFIED_CLASS_NAMES[c] for c in unique_classes]
    print(f"Dataset: {len(dataset)} samples | {n_classes} classes")
    print(f"Zero-day protocol: {args.k_shot}-shot balanced gallery | "
          f"{args.n_trials} trials (different random subsets)\n")

    print("Extracting embeddings...")
    X, y = extract_embeddings(model, loader, device)

    rng = np.random.default_rng(0)
    trial_accs = []
    all_preds_all, all_true_all = [], []

    for trial in range(args.n_trials):
        # Balanced gallery: k_shot random samples per class (ALL classes treated as zero-day)
        gallery_X, gallery_y = [], []
        query_X,   query_y   = [], []

        for c in unique_classes:
            idx = np.where(y == c)[0]
            perm = rng.permutation(idx)
            k = min(args.k_shot, len(perm) - 1)  # keep at least 1 query
            gallery_X.append(X[perm[:k]])
            gallery_y.extend([c] * k)
            query_X.append(X[perm[k:]])
            query_y.extend([c] * (len(perm) - k))

        gallery_X = np.concatenate(gallery_X)
        gallery_y = np.array(gallery_y)
        query_X   = np.concatenate(query_X)
        query_y   = np.array(query_y)

        knn = KNeighborsClassifier(
            n_neighbors=min(args.knn_k, len(gallery_y)),
            metric="cosine", n_jobs=1,
        )
        knn.fit(gallery_X, gallery_y)
        preds = knn.predict(query_X)

        trial_accs.append(accuracy_score(query_y, preds))
        all_preds_all.extend(preds.tolist())
        all_true_all.extend(query_y.tolist())

    mean_acc = float(np.mean(trial_accs))
    std_acc  = float(np.std(trial_accs))
    kpi_ok   = mean_acc >= 0.85

    print(f"=== Zero-Day Generalization ({args.k_shot}-shot balanced k-NN) ===")
    print(f"Accuracy : {mean_acc:.4f} ± {std_acc:.4f}  over {args.n_trials} trials")
    print(f"KPI      : {'✓ ≥85% MET' if kpi_ok else '✗ below 85% KPI'}")
    print(f"Min: {min(trial_accs):.4f}  Max: {max(trial_accs):.4f}  "
          f"Median: {float(np.median(trial_accs)):.4f}")

    # Per-class breakdown from all trials combined
    print(f"\n=== Per-class accuracy (averaged across {args.n_trials} trials) ===")
    print(classification_report(
        all_true_all, all_preds_all,
        labels=unique_classes, target_names=class_names,
    ))


if __name__ == "__main__":
    main()
