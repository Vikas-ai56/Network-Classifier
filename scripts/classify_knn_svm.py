"""
KPI: Classification accuracy >= 90% using k-NN and SVM downstream classifiers.

Pipeline:
  1. Load checkpoint (trained DualBranchEncoder).
  2. Extract embeddings for all samples (frozen encoder, no grad).
  3. 80/20 stratified split.
  4. Train k-NN and SVM classifiers on training embeddings.
  5. Evaluate on test embeddings — report per-class and overall accuracy.

Usage:
  python scripts/classify_knn_svm.py
  python scripts/classify_knn_svm.py --checkpoint model/best_model.pth
"""

import argparse, os, sys, json, glob, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models_dual_branch import DualBranchEncoder

DATA_DIR = "dataset/netmamba/ISCXVPN2016/images_sampled_new"


# ---------------------------------------------------------------------------
# Dataset (inline, no removed-module dependency)
# ---------------------------------------------------------------------------

class _ISCXDataset(Dataset):
    SEQ_LEN = 30

    def __init__(self, root_dir):
        self.files   = sorted(glob.glob(os.path.join(root_dir, "**/*.json"), recursive=True))
        self.classes = sorted(set(os.path.basename(os.path.dirname(p)) for p in self.files))
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        print(f"  {len(self.files)} samples | {len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.files)

    def _pad(self, lst, n):
        a = lst[:n]; a += [0.0] * (n - len(a)); return a

    def __getitem__(self, idx):
        path  = self.files[idx]
        label = self.cls2idx[os.path.basename(os.path.dirname(path))]
        with open(path) as f:
            d = json.load(f)
        L = self._pad(d.get("lengths",   []), self.SEQ_LEN)
        I = self._pad(d.get("intervals", []), self.SEQ_LEN)
        sizes = np.array(L, dtype=np.float32)
        ipts  = np.array(I, dtype=np.float32)
        dirs  = np.sign(np.diff(sizes, prepend=sizes[0])).astype(np.float32)
        seq   = np.stack([
            np.log1p(np.clip(sizes, 0, 1500)) / np.log1p(1500),
            np.log1p(np.clip(ipts,  0, 5000)) / np.log1p(5000),
            dirs,
        ], axis=-1).astype(np.float32)
        stat = np.array([
            np.mean(sizes), np.std(sizes), np.mean(ipts), np.std(ipts),
            np.max(sizes),  np.min(ipts),
            float(len([x for x in L if x > 0])), float(np.sum(sizes)),
            float(np.percentile(sizes, 25)), float(np.percentile(sizes, 75)),
            float(np.percentile(ipts,  25)), float(np.percentile(ipts,  75)),
            float(np.sum(ipts > 0)), float(np.mean(dirs)),
            float(np.std(dirs)),     float(np.max(ipts)),
            float(np.min(sizes[sizes > 0]) if any(s > 0 for s in sizes) else 0.0),
            float(np.median(sizes)),
        ], dtype=np.float32)
        return torch.tensor(seq), torch.tensor(stat), torch.tensor(label)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(path, device):
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=18, d_model=256, embed_dim=256)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"  Loaded {path}  (epoch {ckpt.get('epoch', '?')})")
    return model.to(device).eval()


def extract_embeddings(model, data_dir, device, batch_size=256):
    dataset = _ISCXDataset(data_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, labels = [], []
    with torch.no_grad():
        for seq, stat, lbl in loader:
            embs.append(model(seq.to(device), stat.to(device)).cpu().numpy())
            labels.append(lbl.numpy())
    return np.concatenate(embs), np.concatenate(labels), dataset.classes


def run_classifiers(X_tr, X_te, y_tr, y_te, class_names):
    classifiers = {
        "k-NN (k=1, cosine)": KNeighborsClassifier(n_neighbors=1, metric="cosine", n_jobs=-1),
        "k-NN (k=5, cosine)": KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1),
        "SVM  (RBF,  C=10) ": SVC(kernel="rbf",    C=10.0, gamma="scale"),
        "SVM  (Linear, C=1)": SVC(kernel="linear", C=1.0),
    }
    results = {}
    for name, clf in classifiers.items():
        print(f"\n{'='*58}\n  {name}\n{'='*58}")
        t0    = time.time(); clf.fit(X_tr, y_tr);       t_fit  = time.time() - t0
        t0    = time.time(); preds = clf.predict(X_te); t_pred = time.time() - t0
        acc   = accuracy_score(y_te, preds)
        results[name] = acc
        print(f"  Fit time   : {t_fit:.2f}s")
        print(f"  Infer time : {t_pred*1000:.1f}ms  ({t_pred/len(X_te)*1000:.3f}ms/sample)")
        print(f"  Overall    : {acc*100:.2f}%\n")
        print(classification_report(y_te, preds, target_names=class_names, digits=3,
                                    zero_division=0))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model/best_model.pth")
    parser.add_argument("--data_dir",   default=DATA_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    print("[ 1/3 ] Loading model...")
    model = load_model(args.checkpoint, device)

    print("\n[ 2/3 ] Extracting embeddings...")
    t0 = time.time()
    X, y, class_names = extract_embeddings(model, args.data_dir, device)
    print(f"  Done in {time.time()-t0:.1f}s | shape={X.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")

    print("\n[ 3/3 ] Training & evaluating classifiers...")
    results = run_classifiers(X_tr, X_te, y_tr, y_te, class_names)

    print(f"\n{'='*58}")
    print(f"  SUMMARY  (KPI: >= 90% accuracy)")
    print(f"{'='*58}")
    for name, acc in results.items():
        status = "✓ KPI MET" if acc >= 0.90 else f"✗ gap: {(0.90-acc)*100:.1f}%"
        print(f"  {name}  {acc*100:6.2f}%   {status}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
