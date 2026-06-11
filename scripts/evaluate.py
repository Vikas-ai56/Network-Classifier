"""
Evaluate trained DualBranchEncoder on CESNET-QUIC22.

Two modes:
  --csv   : run locally on a CSV file (no HDF5 needed; needs pre-built FAISS store)
  stream  : run on CESNET HDF5 via cesnet-datazoo (default; builds FAISS store if absent)

Pipeline:
  1. Load checkpoint
  2. Build FAISS index from training embeddings (or load cached model/faiss_store.npz)
  3. Classify test/val flows with k=5 ANN majority vote (cosine sim via IndexFlatIP)
  4. Report per-class and overall accuracy

Zero-day classes (1=music, 2=gaming) are withheld from the FAISS index since they were
excluded from training. Their test flows are classified to the nearest seen class —
a non-zero score here proves zero-day generalization.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models_dual_branch import DualBranchEncoder
from src.streaming_dataset import _process_chunk, _build_app_int_map, _ZERO_DAY_CLASSES
from src.feature_engineering import SEQ_LEN
from src.dataset_unified import UNIFIED_CLASS_NAMES, NUM_CLASSES

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Maximum embeddings stored per class in the FAISS index (memory guard)
_MAX_PER_CLASS = 10_000


def load_model(checkpoint_path, device):
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=128, embed_dim=256)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint — epoch {ckpt.get('epoch', 0) + 1}, loss {ckpt.get('loss', 0):.4f}")
    return model


# ---------------------------------------------------------------------------
# FAISS index construction
# ---------------------------------------------------------------------------

def build_faiss_store(model, data_root, size, device, store_path):
    """
    Stream the training split and collect up to _MAX_PER_CLASS embeddings per
    non-zero-day class. Save as model/faiss_store.npz (embeddings + labels).
    Zero-day classes (music, gaming) are excluded — they were never seen in training.
    """
    if not HAS_FAISS:
        raise ImportError("faiss-cpu is required: pip install faiss-cpu")

    from torch.utils.data import DataLoader
    from src.streaming_dataset import CESNETStreamingDataset

    print(f"\nBuilding FAISS store from train split (max {_MAX_PER_CLASS}/class)...")
    ds = CESNETStreamingDataset(data_root=data_root, size=size, split="train", shuffle_chunks=False)
    loader = DataLoader(ds, batch_size=256, num_workers=2, persistent_workers=True)

    bucket_embs = {c: [] for c in range(NUM_CLASSES) if c not in _ZERO_DAY_CLASSES}

    model.eval()
    with torch.no_grad():
        for seq, stat, ports, labels in loader:
            embs = model(seq.to(device), stat.to(device), ports.to(device)).cpu().numpy()
            for emb, lbl in zip(embs, labels.tolist()):
                if lbl in _ZERO_DAY_CLASSES or lbl not in bucket_embs:
                    continue
                if len(bucket_embs[lbl]) < _MAX_PER_CLASS:
                    bucket_embs[lbl].append(emb)
            if all(len(bucket_embs[c]) >= _MAX_PER_CLASS for c in bucket_embs):
                break

    all_embs = np.concatenate(
        [np.stack(v) for v in bucket_embs.values() if v], axis=0
    ).astype(np.float32)
    all_lbls = np.concatenate(
        [np.full(len(v), c, dtype=np.int64) for c, v in bucket_embs.items() if v], axis=0
    )

    np.savez(store_path, embeddings=all_embs, labels=all_lbls)
    counts = {UNIFIED_CLASS_NAMES.get(c, str(c)): len(v) for c, v in bucket_embs.items() if v}
    print(f"FAISS store saved → {store_path}  ({len(all_embs)} total embeddings)")
    for name, n in sorted(counts.items()):
        print(f"  {name:22s}  {n}")
    return all_embs, all_lbls


def load_faiss_store(store_path):
    data = np.load(store_path)
    return data["embeddings"].astype(np.float32), data["labels"].astype(np.int64)


def build_index(all_embs: np.ndarray) -> "faiss.IndexFlatIP":
    """Build a FAISS IndexFlatIP over L2-normalized embeddings (IP = cosine sim)."""
    if not HAS_FAISS:
        raise ImportError("faiss-cpu is required: pip install faiss-cpu")
    index = faiss.IndexFlatIP(all_embs.shape[1])  # dim=256
    index.add(all_embs)
    return index


# ---------------------------------------------------------------------------
# k-NN classification
# ---------------------------------------------------------------------------

def classify_knn(index, train_labels: np.ndarray, embs_np: np.ndarray, k: int = 5):
    """
    Classify a batch of L2-normalized embeddings via k-NN majority vote.

    Uses FAISS IndexFlatIP: inner product on unit vectors equals cosine similarity.
    Returns list of predicted class IDs.
    """
    _, nn_indices = index.search(embs_np.astype(np.float32), k)  # (batch, k)
    predictions = []
    for row in nn_indices:
        neighbor_labels = train_labels[row]
        pred = int(np.bincount(neighbor_labels.astype(np.intp)).argmax())
        predictions.append(pred)
    return predictions


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def evaluate_csv(model, index, train_labels, csv_path, device):
    """Evaluate on a local CSV file using FAISS k-NN."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    samples = _process_chunk(df, app_int_map=None, excluded_labels=None)
    if not samples:
        print("No valid samples found in CSV — check CATEGORY/APP columns")
        return

    seq = torch.from_numpy(np.stack([s[0] for s in samples]))
    stat = torch.from_numpy(np.stack([s[1] for s in samples]))
    ports = torch.from_numpy(np.stack([s[2] for s in samples]))
    true_labels = [s[3] for s in samples]

    correct = {c: 0 for c in range(NUM_CLASSES)}
    total   = {c: 0 for c in range(NUM_CLASSES)}

    batch_size = 256
    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            embs = model(
                seq[i:i+batch_size].to(device),
                stat[i:i+batch_size].to(device),
                ports[i:i+batch_size].to(device),
            ).cpu().numpy()
            preds = classify_knn(index, train_labels, embs)
            for pred, true in zip(preds, true_labels[i:i+batch_size]):
                total[true] += 1
                if pred == true:
                    correct[true] += 1

    _print_results(correct, total)


def evaluate_stream(model, index, train_labels, data_root, size, device):
    """Evaluate on CESNET test split via cesnet-datazoo using FAISS k-NN."""
    from torch.utils.data import DataLoader
    from src.streaming_dataset import CESNETStreamingDataset

    ds = CESNETStreamingDataset(data_root=data_root, size=size, split="test", shuffle_chunks=False)
    loader = DataLoader(ds, batch_size=256, num_workers=2, persistent_workers=True)

    correct = {c: 0 for c in range(NUM_CLASSES)}
    total   = {c: 0 for c in range(NUM_CLASSES)}

    model.eval()
    with torch.no_grad():
        for seq, stat, ports, labels in loader:
            embs = model(seq.to(device), stat.to(device), ports.to(device)).cpu().numpy()
            preds = classify_knn(index, train_labels, embs)
            for pred, true in zip(preds, labels.tolist()):
                total[true] += 1
                if pred == true:
                    correct[true] += 1

    _print_results(correct, total)


def _print_results(correct, total):
    print("\n--- Per-Class Accuracy ---")
    overall_c, overall_t = 0, 0
    zero_day_c, zero_day_t = 0, 0
    for c in sorted(total):
        if total[c] == 0:
            continue
        acc = correct[c] / total[c]
        name = UNIFIED_CLASS_NAMES.get(c, str(c))
        tag = " [zero-day]" if c in _ZERO_DAY_CLASSES else ""
        print(f"  {name:22s}  {acc:.3f}  ({correct[c]}/{total[c]}){tag}")
        overall_c += correct[c]
        overall_t += total[c]
        if c in _ZERO_DAY_CLASSES:
            zero_day_c += correct[c]
            zero_day_t += total[c]
    if overall_t:
        print(f"\n  Overall accuracy:   {overall_c / overall_t:.4f}  ({overall_c}/{overall_t})")
    if zero_day_t:
        print(f"  Zero-day accuracy:  {zero_day_c / zero_day_t:.4f}  ({zero_day_c}/{zero_day_t})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if not HAS_FAISS:
        print("ERROR: faiss-cpu is not installed. Run: pip install faiss-cpu")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="model/checkpoint_latest.pth")
    parser.add_argument("--faiss_store", default="model/faiss_store.npz",
                        help="Cached FAISS embeddings+labels (built automatically if absent)")
    parser.add_argument("--csv",         default=None, help="Local CSV file to evaluate on")
    parser.add_argument("--data_root",   default="/workspace/cesnet_cache")
    parser.add_argument("--size",        default="XS", choices=["XS", "S", "M"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    # Load or build FAISS store
    if os.path.exists(args.faiss_store):
        print(f"Loading FAISS store from {args.faiss_store}")
        all_embs, all_lbls = load_faiss_store(args.faiss_store)
        print(f"  {len(all_embs)} training embeddings, classes: {sorted(set(all_lbls.tolist()))}")
    elif args.csv:
        print("ERROR: No FAISS store found and --csv mode cannot build one.")
        print(f"  Run without --csv first to build {args.faiss_store} from the training split.")
        sys.exit(1)
    else:
        all_embs, all_lbls = build_faiss_store(model, args.data_root, args.size, device, args.faiss_store)

    index = build_index(all_embs)

    if args.csv:
        evaluate_csv(model, index, all_lbls, args.csv, device)
    else:
        evaluate_stream(model, index, all_lbls, args.data_root, args.size, device)


if __name__ == "__main__":
    main()
