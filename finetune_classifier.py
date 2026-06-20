"""
Fine-tune encoder + linear head on ISCXVPN2016.

Loss = CrossEntropy + 0.5 * MarginBasedSupConLoss (contrastive)
Encoder runs at low LR (1e-5) to adapt without forgetting geometry.
Head runs at high LR (1e-3).

Fixes both:
  - Classification accuracy KPI (>= 90%)  via cross-entropy supervision
  - Zero-day generalization KPI (>= 85%)  via contrastive cluster tightening

Usage:
  python finetune_classifier.py [--data_dir ...] [--model_path ...]
"""

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

logging.getLogger("src.feature_engineering").setLevel(logging.ERROR)
logging.getLogger("src.data_validator").setLevel(logging.ERROR)

from src.models_dual_branch import DualBranchEncoder
from src.dataset_unified import UnifiedFlowDataset, UNIFIED_CLASS_NAMES
from src.train_supcon import MarginBasedSupConLoss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default="dataset/netmamba/ISCXVPN2016/images_sampled_new")
    parser.add_argument("--model_path",    default="model/best_model.pth")
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--lr_encoder",    type=float, default=1e-5,  help="encoder LR (keep small)")
    parser.add_argument("--lr_head",       type=float, default=1e-3,  help="classifier head LR")
    parser.add_argument("--supcon_weight", type=float, default=0.5,   help="weight for contrastive loss")
    parser.add_argument("--save_path",     default="model/finetuned_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load encoder (fine-tuned, not frozen)
    encoder = DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=256, embed_dim=256)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.to(device)
    print(f"Encoder loaded from epoch {ckpt.get('epoch', 0) + 1}  (will fine-tune at lr={args.lr_encoder})")

    # Load dataset
    dataset = UnifiedFlowDataset(args.data_dir)
    labels  = dataset.labels
    present_classes = sorted(np.unique(labels))
    n_classes = len(present_classes)
    class_names = [UNIFIED_CLASS_NAMES[c] for c in present_classes]
    print(f"Dataset: {len(dataset)} samples | {n_classes} classes: {class_names}")

    # Remap labels to 0..n_classes-1
    label_remap = {orig: new for new, orig in enumerate(present_classes)}
    remapped = np.array([label_remap[l] for l in labels])

    # Stratified 80/20 split
    rng = np.random.default_rng(42)
    train_idx, val_idx = [], []
    for c in range(n_classes):
        idx = np.where(remapped == c)[0]
        idx = rng.permutation(idx)
        n_val = max(1, int(len(idx) * 0.2))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    class RemappedSubset(torch.utils.data.Dataset):
        def __init__(self, base, indices, remapped_labels):
            self.base = base
            self.indices = indices
            self.remapped = remapped_labels
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            idx = self.indices[i]
            seq, stat, _ = self.base[idx]
            return seq, stat, torch.tensor(self.remapped[idx], dtype=torch.long)

    train_set = RemappedSubset(dataset, train_idx, remapped)
    val_set   = RemappedSubset(dataset, val_idx,   remapped)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}\n")

    # Classifier head
    head = nn.Sequential(
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, n_classes),
    ).to(device)

    # Two param groups: encoder (slow) + head (fast)
    optimizer = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": args.lr_encoder},
        {"params": head.parameters(),    "lr": args.lr_head},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ce_loss      = nn.CrossEntropyLoss()
    supcon_loss  = MarginBasedSupConLoss(lambda_pos=0.7, lambda_neg=0.3)
    scaler       = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        for seq, stat, y in train_loader:
            seq, stat, y = seq.to(device), stat.to(device), y.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    emb    = encoder(seq, stat)
                    logits = head(emb)
                    loss   = ce_loss(logits, y) + args.supcon_weight * supcon_loss(emb, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                emb    = encoder(seq, stat)
                logits = head(emb)
                loss   = ce_loss(logits, y) + args.supcon_weight * supcon_loss(emb, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
                optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

        scheduler.step()
        train_acc = correct / total

        # Validation
        encoder.eval()
        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for seq, stat, y in val_loader:
                emb  = encoder(seq.to(device), stat.to(device))
                pred = head(emb).argmax(1).cpu()
                val_correct += (pred == y).sum().item()
                val_total   += len(y)
        val_acc = val_correct / val_total

        flag = " ✓ NEW BEST" if val_acc > best_val_acc else ""
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f}{flag}"
              f"{'  ← KPI MET ✓' if val_acc >= 0.90 else ''}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "encoder_state_dict": encoder.state_dict(),
                "head_state_dict":    head.state_dict(),
                "label_remap":        label_remap,
                "present_classes":    present_classes,
                "val_acc":            val_acc,
                "epoch":              epoch,
            }, args.save_path)

    # Final evaluation
    print(f"\n=== Final Classification Report (best val acc = {best_val_acc:.4f}) ===")
    saved = torch.load(args.save_path, map_location=device, weights_only=False)
    encoder.load_state_dict(saved["encoder_state_dict"])
    head.load_state_dict(saved["head_state_dict"])
    encoder.eval()
    head.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for seq, stat, y in val_loader:
            emb  = encoder(seq.to(device), stat.to(device))
            pred = head(emb).argmax(1).cpu()
            all_preds.append(pred)
            all_true.append(y)
    all_preds = torch.cat(all_preds).numpy()
    all_true  = torch.cat(all_true).numpy()

    print(classification_report(all_true, all_preds, target_names=class_names))
    final_acc = accuracy_score(all_true, all_preds)
    kpi = "✓ ≥90% KPI MET" if final_acc >= 0.90 else "✗ below 90% KPI"
    print(f"Classification accuracy : {final_acc:.4f}  {kpi}")
    print(f"Saved best model        : {args.save_path}")
    print(f"\nNext: run zero_day_test.py --model_path {args.save_path} to check zero-day KPI")


if __name__ == "__main__":
    main()
