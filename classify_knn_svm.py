"""
KPI: Classification Accuracy >= 90%
Extracts embeddings from best_model.pth, then evaluates k-NN and SVM on ISCXVPN2016.
Usage: python classify_knn_svm.py [--data_dir ...] [--model_path ...]
"""

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

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
    parser.add_argument("--k",          type=int, default=5, help="k for k-NN")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model (handles both pretrain and finetune checkpoint formats)
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=18, d_model=256, embed_dim=256)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    key = "encoder_state_dict" if "encoder_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 0) + 1}  (key={key})")

    # Load ISCXVPN2016
    dataset = UnifiedFlowDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    n_classes = len(np.unique(dataset.labels))
    print(f"Dataset: {len(dataset)} samples | {n_classes} classes")

    # Extract embeddings
    print("Extracting embeddings...")
    X, y = extract_embeddings(model, loader, device)

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    present_classes = sorted(np.unique(y_test))
    target_names = [UNIFIED_CLASS_NAMES.get(i, str(i)) for i in present_classes]

    # k-NN
    print(f"\n=== k-NN (k={args.k}, metric=cosine) ===")
    knn = KNeighborsClassifier(n_neighbors=args.k, metric="cosine", n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print(f"Accuracy : {acc_knn:.4f}  ({'✓ KPI MET' if acc_knn >= 0.90 else '✗ below KPI'})")
    print(classification_report(y_test, y_pred_knn, target_names=target_names))

    # SVM
    print("\n=== SVM (kernel=rbf, C=10) ===")
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"Accuracy : {acc_svm:.4f}  ({'✓ KPI MET' if acc_svm >= 0.90 else '✗ below KPI'})")
    print(classification_report(y_test, y_pred_svm, target_names=target_names))

    # Logistic Regression
    print("\n=== Logistic Regression (max_iter=2000) ===")
    lr_clf = LogisticRegression(C=10, max_iter=2000, solver="lbfgs",
                                multi_class="multinomial", random_state=42)
    lr_clf.fit(X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy : {acc_lr:.4f}  ({'✓ KPI MET' if acc_lr >= 0.90 else '✗ below KPI'})")
    print(classification_report(y_test, y_pred_lr, target_names=target_names))

    best_acc = max(acc_knn, acc_svm, acc_lr)
    print("\n=== KPI Summary ===")
    print(f"k-NN : {acc_knn:.4f}  {'✓ KPI MET' if acc_knn >= 0.90 else '✗ below KPI'}")
    print(f"SVM  : {acc_svm:.4f}  {'✓ KPI MET' if acc_svm >= 0.90 else '✗ below KPI'}")
    print(f"LR   : {acc_lr:.4f}  {'✓ KPI MET' if acc_lr >= 0.90 else '✗ below KPI'}")
    print(f"Best : {best_acc:.4f}  {'✓ ≥90% KPI MET' if best_acc >= 0.90 else '✗ below 90% KPI'}")


if __name__ == "__main__":
    main()
