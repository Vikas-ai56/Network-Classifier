"""
Training entry point — Samsung EnnovateX.
Target KPIs:
  - Intra-class cosine sim > 0.7
  - Inter-class cosine sim < 0.3
  - Classification accuracy >= 90%
  - Zero-day generalization >= 85%
  - Inference latency < 100ms/flow
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logging.getLogger(__name__).warning("faiss not installed — falling back to numpy k-NN eval")

# Silence per-flow data-quality warnings (EMPTY_PHIST, LONG_IPT, ZERO_STD, ...)
# These fire thousands of times on CESNET and are expected/handled, not errors.
logging.getLogger("src.feature_engineering").setLevel(logging.ERROR)
logging.getLogger("src.data_validator").setLevel(logging.ERROR)

from src.models_dual_branch import DualBranchEncoder
from src.dataset_unified import UnifiedFlowDataset, build_dataloaders, NUM_CLASSES
from src.feature_engineering import STAT_INPUT_DIM
from src.train_supcon import (
    MarginBasedSupConLoss,
    EpisodicSampler,
    compute_prototypes,
    prototypical_loss,
    execute_validation_layer,
    get_cosine_schedule_with_warmup,
    augment_flow,
)
from src.streaming_dataset import build_streaming_loaders

os.environ["WANDB_MODE"] = "offline"
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_model(
    data_dir: str,
    epochs: int = 35,
    batch_size: int = 256,
    n_way: int = 5,
    k_shot: int = 5,
    k_query: int = 15,
    warmup_epochs: int = 0,
    lr: float = 1e-3,
    streaming: bool = False,
    streaming_data_root: str = "/workspace/.cesnet_cache",
    streaming_size: str = "XS",
) -> None:
    # ------------------------------------------------------------------ data
    if streaming:
        print(f"Streaming CESNET-QUIC22 (size={streaming_size}, batch={batch_size})")
        train_loader, val_loader = build_streaming_loaders(
            data_root=streaming_data_root,
            size=streaming_size,
            batch_size=batch_size,
            chunk_size=8192,
            num_workers=0,
        )
        eval_loader = None
        eval_n_way  = n_way
        try:
            dataset = UnifiedFlowDataset(data_dir) if os.path.exists(data_dir) else None
        except Exception:
            dataset = None
        if dataset and len(dataset) > 0:
            n_available = len(np.unique(dataset.labels))
            eval_n_way  = min(n_way, n_available)
            if eval_n_way >= 2:
                eval_sampler = EpisodicSampler(
                    labels=dataset.labels, n_way=eval_n_way,
                    k_shot=k_shot, k_query=k_query, iterations=100,
                )
                eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)
            else:
                print("  Not enough classes for ProtoNet eval — skipping episodic eval")
    else:
        eval_n_way = n_way
        print(f"Loading dataset from {data_dir}...")
        train_loader, val_loader, _, class_weights = build_dataloaders(
            data_dir=data_dir, batch_size=batch_size,
            seq_len=30, use_weighted_sampler=True,
        )
        dataset = UnifiedFlowDataset(data_dir)
        eval_sampler = EpisodicSampler(
            labels=dataset.labels, n_way=eval_n_way,
            k_shot=k_shot, k_query=k_query, iterations=100,
        )
        eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)

    # ----------------------------------------------------------------- model
    model = DualBranchEncoder(
        seq_input_dim=3,
        stat_input_dim=STAT_INPUT_DIM,
        d_model=256,
        embed_dim=256,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = MarginBasedSupConLoss(lambda_pos=0.7, lambda_neg=0.3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine LR schedule with linear warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Mixed precision on GPU
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device        : {gpu_name}")
    print(f"Parameters    : {n_params:,}")
    print(f"Epochs        : {epochs}  (warmup={warmup_epochs})")
    print(f"Batch size    : {batch_size}")
    print(f"LR            : {lr}  → cosine decay")

    if HAS_WANDB:
        wandb.init(project="samsung-ennovatex", config={
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "warmup_epochs": warmup_epochs, "d_model": 256, "embed_dim": 256,
            "loss": "MarginBasedSupConLoss",
        })

    best_acc = 0.0
    checkpoint_path = os.path.join("model", "checkpoint_latest.pth")

    # Resume from checkpoint if available
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_acc = ckpt.get("best_acc", 0.0)
            print(f"Resumed from epoch {start_epoch}  (best_acc={best_acc:.4f})")
        except Exception as e:
            print(f"Could not resume checkpoint ({e}) — starting fresh")

    for epoch in range(start_epoch, epochs):
        # ---------------------------------------------------- training
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch_idx, (seq, stat, ports, labels) in enumerate(train_loader):
            seq, stat, ports, labels = (seq.to(device), stat.to(device),
                                        ports.to(device), labels.to(device))

            # Flow augmentation (training only)
            seq, stat = augment_flow(seq, stat, noise_std=0.02, packet_drop_prob=0.05)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    embeddings = model(seq, stat, ports)
                    loss = criterion(embeddings, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                embeddings = model(seq, stat, ports)
                loss = criterion(embeddings, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

            if batch_idx % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}]  "
                      f"Batch [{batch_idx}]  "
                      f"Loss: {loss.item():.4f}  "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # --------------------------------- FAISS k-NN eval (replaces ProtoNet)
        model.eval()
        epoch_acc = 0.0
        if val_loader is not None:
            gallery_embs, gallery_labels = [], []
            query_embs,   query_labels   = [], []
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    seq_b, stat_b, ports_b, y_b = batch
                    emb = model(seq_b.to(device), stat_b.to(device), ports_b.to(device))
                    emb_np = emb.cpu().numpy()
                    y_np   = y_b.numpy()
                    if i % 2 == 0:          # even batches → gallery
                        gallery_embs.append(emb_np)
                        gallery_labels.append(y_np)
                    else:                   # odd batches → query
                        query_embs.append(emb_np)
                        query_labels.append(y_np)
                    if i >= 39:             # cap at 40 batches (10240 samples)
                        break

            if gallery_embs and query_embs:
                G = np.concatenate(gallery_embs).astype(np.float32)
                Q = np.concatenate(query_embs).astype(np.float32)
                Gy = np.concatenate(gallery_labels)
                Qy = np.concatenate(query_labels)

                if HAS_FAISS:
                    index = faiss.IndexFlatIP(G.shape[1])   # inner product = cosine for L2-normed
                    index.add(G)
                    _, I = index.search(Q, 5)               # k=5 nearest neighbours
                else:
                    # Numpy fallback: full dot-product matrix, top-5 per row
                    sims = Q @ G.T
                    I    = np.argsort(-sims, axis=1)[:, :5]

                # Majority vote of k=5 neighbours
                preds = np.array([np.bincount(Gy[I[j]], minlength=int(Gy.max())+1).argmax()
                                   for j in range(len(Q))])
                epoch_acc = float((preds == Qy).mean())

        # --------------------------------------- geometric KPI validation
        avg_intra, avg_inter = execute_validation_layer(model, val_loader, device)

        kpi_sim  = "✓" if avg_intra > 0.7 and avg_inter < 0.3 else "✗"
        kpi_acc  = "✓" if epoch_acc >= 0.90 else "✗"
        print(
            f"\n--- Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
            f"ProtoNet: {epoch_acc:.4f} {kpi_acc} | "
            f"Intra: {avg_intra:.4f} | Inter: {avg_inter:.4f} {kpi_sim} ---\n"
        )

        if HAS_WANDB:
            wandb.log({
                "train/loss": avg_loss,
                "eval/proto_acc": epoch_acc,
                "eval/intra_sim": avg_intra,
                "eval/inter_sim": avg_inter,
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            })

        # ------------------------------------------- save checkpoints
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "best_acc": best_acc,
        }
        torch.save(ckpt_data, checkpoint_path)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            ckpt_data["best_acc"] = best_acc
            torch.save(ckpt_data, os.path.join("model", "best_model.pth"))
            print(f"  --> New best  ProtoNet acc={best_acc:.4f}  (KPI {'MET ✓' if best_acc >= 0.90 else 'not yet'})")

    print("\n=== Training complete ===")
    print(f"Best ProtoNet accuracy : {best_acc:.4f}  ({'✓ ≥90% KPI MET' if best_acc >= 0.90 else '✗ below 90% KPI'})")
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",        default="dataset/netmamba/ISCXVPN2016/images_sampled_new")
    parser.add_argument("--epochs",          type=int,   default=35)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--warmup_epochs",   type=int,   default=0)
    parser.add_argument("--streaming",       action="store_true")
    parser.add_argument("--streaming_root",  default="/workspace/.cesnet_cache")
    parser.add_argument("--streaming_size",  default="XS", choices=["XS", "S", "M"])
    args = parser.parse_args()

    os.makedirs("model", exist_ok=True)
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        streaming=args.streaming,
        streaming_data_root=args.streaming_root,
        streaming_size=args.streaming_size,
    )
