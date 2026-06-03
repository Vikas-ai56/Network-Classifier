import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from src.models_dual_branch import DualBranchEncoder
from src.dataset_netmamba import NetMambaDataset
from dataset.dataset_5g_kaggle import FiveGKaggleDataset
from src.train_supcon import SupConLoss, EpisodicSampler, compute_prototypes, prototypical_loss
from tqdm import tqdm
import os
import argparse
from pathlib import Path

def get_dataset(dataset_type, data_root, split='train', **kwargs):
    """Load dataset based on type."""
    if dataset_type == 'cesnet-quic22':
        from dataset.dataset_cesnet_quic22 import CESNETQuic22Dataset
        size = kwargs.get('size', 'S')
        print(f"Loading CESNET-QUIC22 ({size}) from {data_root}...")
        return CESNETQuic22Dataset(data_root=data_root, split=split, size=size)
    elif dataset_type == 'netmamba-json':
        return NetMambaDataset(data_root, split=split)
    elif dataset_type == '5g-kaggle':
        return FiveGKaggleDataset(data_root, split=split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def train_model(dataset_type, data_root, output_model, epochs=10, batch_size=32,
                n_way=5, k_shot=5, k_query=15, **kwargs):
    """
    Train DualBranchEncoder with SupCon + ProtoNet.

    Args:
        dataset_type: 'cesnet-quic22' or 'netmamba-json'
        data_root: Path to dataset
        output_model: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for SupCon training
        n_way: Number of classes per episode (ProtoNet)
        k_shot: Support samples per class (ProtoNet)
        k_query: Query samples per class (ProtoNet)
    """
    # 1. Load Training Dataset
    train_dataset = get_dataset(dataset_type, data_root, split='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2. Load Validation Dataset (if available)
    try:
        val_dataset = get_dataset(dataset_type, data_root, split='val', **kwargs)
        n_classes = len(val_dataset.classes)
        effective_n_way = min(n_way, n_classes)
        eval_sampler = EpisodicSampler(
            val_dataset.labels,
            n_way=effective_n_way, k_shot=k_shot, k_query=k_query, iterations=100
        )
        eval_loader = DataLoader(val_dataset, batch_sampler=eval_sampler, num_workers=0)
        print(f"Validation set: {len(val_dataset)} samples, {n_classes} classes, "
              f"{effective_n_way}-way episodes")
    except Exception as e:
        print(f"Warning: Could not load validation set: {e}")
        eval_loader = None
        effective_n_way = n_way

    # 3. Initialize Model & Loss
    print("Initializing DualBranchEncoder with 8 QoS features in Branch B...")
    model = DualBranchEncoder(seq_input_dim=2, stat_input_dim=8, embed_dim=128)
    criterion_supcon = SupConLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on {device}...")

    best_acc = 0.0
    training_log = []

    for epoch in range(epochs):
        # ========== SupCon Training Phase ==========
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        for batch_data in pbar:
            seq, stat, labels = batch_data
            seq, stat, labels = seq.to(device), stat.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = model(seq, stat)
            loss = criterion_supcon(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} | SupCon Loss: {avg_train_loss:.4f} "
              f"| LR: {scheduler.get_last_lr()[0]:.2e}")

        # ========== ProtoNet Evaluation Phase ==========
        if eval_loader is not None:
            model.eval()
            eval_accs = []

            with torch.no_grad():
                for batch_data in tqdm(eval_loader, desc="  ProtoNet eval", leave=False):
                    seq, stat, _ = batch_data
                    seq, stat = seq.to(device), stat.to(device)
                    embeddings = model(seq, stat)

                    n_samples = effective_n_way * (k_shot + k_query)
                    if len(embeddings) < n_samples:
                        continue

                    support_feat = embeddings[:effective_n_way * k_shot]
                    query_feat = embeddings[effective_n_way * k_shot:
                                           effective_n_way * (k_shot + k_query)]

                    prototypes = compute_prototypes(support_feat, effective_n_way, k_shot)
                    _, acc = prototypical_loss(prototypes, query_feat, effective_n_way, k_query)
                    eval_accs.append(acc.item())

            if eval_accs:
                epoch_acc = np.mean(eval_accs)
                print(f"  ProtoNet Val Accuracy: {epoch_acc:.4f}")

                training_log.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'val_acc': epoch_acc
                })

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    os.makedirs(os.path.dirname(output_model), exist_ok=True)
                    torch.save(model.state_dict(), output_model)
                    print(f"  Best model saved (acc: {best_acc:.4f}) -> {output_model}")
        else:
            os.makedirs(os.path.dirname(output_model), exist_ok=True)
            torch.save(model.state_dict(), output_model)
            print(f"  Checkpoint saved -> {output_model}")

    print(f"\n✓ Training complete! Best model saved to {output_model}")
    return model, training_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NetMamba for traffic classification')
    parser.add_argument('--dataset', type=str, choices=['cesnet-quic22', 'netmamba-json', '5g-kaggle'],
                       default='netmamba-json',
                       help='Dataset type to train on')
    parser.add_argument('--data-root', type=str,
                       default='dataset/netmamba/ISCXVPN2016/images_sampled_new',
                       help='Path to dataset root directory')
    parser.add_argument('--output-model', type=str, default='model/best_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for SupCon training')
    parser.add_argument('--dataset-size', type=str, choices=['S', 'M', 'L'], default='S',
                       help='Dataset size for CESNET-QUIC22 (S=small, M=medium, L=large)')

    args = parser.parse_args()

    # Create model directory
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    print(f"EnnovateX Training Pipeline")
    print(f"=" * 50)
    print(f"Dataset: {args.dataset}")
    print(f"Data Root: {args.data_root}")
    print(f"Output Model: {args.output_model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"=" * 50)

    if not os.path.exists(args.data_root):
        print(f"❌ Error: Dataset not found at {args.data_root}")
        print(f"Please ensure the dataset is downloaded to this path first.")
        exit(1)

    try:
        model, log = train_model(
            dataset_type=args.dataset,
            data_root=args.data_root,
            output_model=args.output_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            size=args.dataset_size
        )
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
