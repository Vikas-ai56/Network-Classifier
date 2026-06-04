# ---------------------------------------------------------------------------------------------------
# epochs=25 batch_size=128  streaming_size=XS (~2.5GB)  — first iteration on RTX 4090
# ---------------------------------------------------------------------------------------------------

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models_dual_branch import DualBranchEncoder
from src.dataset_unified import UnifiedFlowDataset, build_dataloaders, NUM_CLASSES
from src.train_supcon import (
    MarginBasedSupConLoss,
    EpisodicSampler,
    compute_prototypes,
    prototypical_loss,
    execute_validation_layer,
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
    epochs: int = 25,
    batch_size: int = 128,
    n_way: int = 5,
    k_shot: int = 5,
    k_query: int = 15,
    streaming: bool = False,
    streaming_data_root: str = "/workspace/.cesnet_cache",
    streaming_size: str = "XS",
) -> None:
    """
    Train DualBranchEncoder with Margin-Based SupCon + ProtoNet eval.

    Parameters
    ----------
    data_dir : str
        Path to local dataset (ISCXVPN2016).
        Ignored when streaming=True.
    streaming : bool
        If True, streams CESNET-QUIC22 directly from Data Zoo batch-by-batch.
        No full dataset download required.
    streaming_data_root : str
        Local directory for cesnet-datazoo metadata only (~50 MB). Used when streaming=True.
    streaming_size : str
        "XS" (10M raw flows), "S" (25M), "M" (50M). Used when streaming=True.
    """
    if streaming:
        print(f"Streaming CESNET-QUIC22 from Data Zoo (size={streaming_size}, batch={batch_size})")
        train_loader, val_loader = build_streaming_loaders(
            data_root=streaming_data_root,
            size=streaming_size,
            batch_size=batch_size,
            chunk_size=8192,
            num_workers=4,
        )
        # For episodic eval we still need a label-indexed dataset; use a small local one
        # If no local data is available, skip ProtoNet eval and rely on geometric validation only
        eval_loader = None
        try:
            dataset = UnifiedFlowDataset(data_dir, seq_len=128) if os.path.exists(data_dir) else None
        except Exception:
            dataset = None

        if dataset is not None and len(dataset) > 0:
            eval_sampler = EpisodicSampler(
                labels=dataset.labels, n_way=n_way, k_shot=k_shot,
                k_query=k_query, iterations=100,
            )
            eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)
    else:
        print(f"Initializing dataset from {data_dir}...")
        train_loader, val_loader, _, class_weights = build_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            seq_len=128,
            use_weighted_sampler=True,
        )
        dataset = UnifiedFlowDataset(data_dir, seq_len=128)
        eval_sampler = EpisodicSampler(
            labels=dataset.labels,
            n_way=n_way,
            k_shot=k_shot,
            k_query=k_query,
            iterations=100,
        )
        eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)

    model = DualBranchEncoder(
        seq_input_dim=3,
        stat_input_dim=18,
        d_model=128,
        embed_dim=256,
    )
    criterion = MarginBasedSupConLoss(lambda_pos=0.7, lambda_neg=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training on {gpu_name}")

    if HAS_WANDB:
        wandb.init(
            project="netwok-classifier",
            config={
                "epochs": epochs, "batch_size": batch_size, "lr": 1e-3,
                "embed_dim": 256, "seq_len": 30, "seq_input_dim": 3,
                "stat_input_dim": 18, "lambda_pos": 0.7, "lambda_neg": 0.3,
                "loss": "MarginBasedSupCon",
            },
        )

    best_acc = 0.0
    start_epoch = 0

    checkpoint_path = os.path.join("model", "checkpoint_latest.pth")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        # --- Margin-Based SupCon Training ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx, (seq, stat, labels) in enumerate(train_loader):
            seq, stat, labels = seq.to(device), stat.to(device), labels.to(device)
            optimizer.zero_grad()
            embeddings = model(seq, stat)
            loss = criterion(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{batch_idx}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(n_batches, 1)

        # --- ProtoNet Episodic Eval ---
        model.eval()
        eval_accs = []
        if eval_loader is not None:
            with torch.no_grad():
                for seq, stat, _ in eval_loader:
                    embs = model(seq.to(device), stat.to(device))
                    support = embs[: n_way * k_shot]
                    query = embs[n_way * k_shot :]
                    prototypes = compute_prototypes(support, n_way, k_shot)
                    _, acc = prototypical_loss(prototypes, query, n_way, k_query)
                    eval_accs.append(acc.item())
        epoch_acc = np.mean(eval_accs) if eval_accs else 0.0

        # --- Geometric Validation Layer (KPI metrics) ---
        avg_intra, avg_inter = execute_validation_layer(model, val_loader, device)

        print(
            f"--- Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
            f"ProtoNet Acc: {epoch_acc:.4f} | "
            f"Intra-Sim: {avg_intra:.4f} (>0.7) | "
            f"Inter-Sim: {avg_inter:.4f} (<0.3) ---"
        )

        if HAS_WANDB:
            wandb.log({
                "train/loss": avg_loss,
                "eval/proto_acc": epoch_acc,
                "eval/intra_sim": avg_intra,
                "eval/inter_sim": avg_inter,
                "epoch": epoch + 1,
            })

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, os.path.join("model", "best_model.pth"))
            print(f"    --> Saved best model (ProtoNet acc={best_acc:.4f})")

        # Save checkpoint every epoch regardless of ProtoNet acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }, os.path.join("model", "checkpoint_latest.pth"))

    print("\n--- Training complete ---")
    if HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="datasets/netmamba/ISCXVPN2016/images_sampled_new")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--streaming", action="store_true",
                        help="Stream CESNET-QUIC22 from Data Zoo (no full download)")
    parser.add_argument("--streaming_root", default="/workspace/.cesnet_cache",
                        help="Local dir for cesnet-datazoo metadata (~50 MB)")
    parser.add_argument("--streaming_size", default="XS",
                        choices=["XS", "S", "M"],
                        help="XS=10M raw flows (~2.5GB)  S=25M (~6GB)  M=50M (~13GB)")
    args = parser.parse_args()

    os.makedirs("model", exist_ok=True)

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        streaming=args.streaming,
        streaming_data_root=args.streaming_root,
        streaming_size=args.streaming_size,
    )
