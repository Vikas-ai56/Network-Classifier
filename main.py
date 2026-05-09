import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models_dual_branch import DualBranchEncoder
from src.dataset_netmamba import NetMambaDataset
from src.train_supcon import SupConLoss, EpisodicSampler, compute_prototypes, prototypical_loss
import os

def train_model(data_dir, epochs=10, batch_size=32, n_way=5, k_shot=5, k_query=15):
    # 1. Load Dataset
    print(f"Initializing dataset from {data_dir}...")
    dataset = NetMambaDataset(data_dir)
    
    # DataLoader for SupCon Pre-training
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Sampler for ProtoNet Evaluation (Episodic)
    eval_sampler = EpisodicSampler(dataset.labels if hasattr(dataset, 'labels') else [ds[2].item() for ds in dataset], 
                                  n_way=n_way, k_shot=k_shot, k_query=k_query, iterations=100)
    eval_loader = DataLoader(dataset, batch_sampler=eval_sampler)

    # 2. Initialize Model & Loss
    model = DualBranchEncoder(seq_input_dim=2, stat_input_dim=4, embed_dim=128)
    criterion_supcon = SupConLoss(temperature=0.07)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (seq, stat, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass: Generate Embeddings
            embeddings = model(seq, stat)
            
            # Compute SupCon Loss
            loss = criterion_supcon(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Evaluation Phase (ProtoNet)
        model.eval()
        eval_accs = []
        with torch.no_grad():
            # This is a simplified episodic evaluation
            for episode_data in eval_loader:
                seq, stat, _ = episode_data
                embeddings = model(seq, stat)
                
                # Split into support and query sets
                support_feat = embeddings[:n_way * k_shot]
                query_feat = embeddings[n_way * k_shot:]
                
                prototypes = compute_prototypes(support_feat, n_way, k_shot)
                _, acc = prototypical_loss(prototypes, query_feat, n_way, k_query)
                eval_accs.append(acc.item())
        epoch_acc = np.mean(eval_accs)
        print(f"--- Epoch {epoch+1} Complete. Avg SupCon Loss: {total_loss/len(train_loader):.4f}, ProtoNet Eval Acc: {epoch_acc:.4f} ---")
        
        # Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join("model", "best_model.pth"))
            print(f"    --> Saved new best model with accuracy {best_acc:.4f}")

if __name__ == "__main__":
    import numpy as np
    dataset_path = "dataset/netmamba/ISCXVPN2016/images_sampled_new"
    # Create directory for saving models if it doesn't exist
    os.makedirs("model", exist_ok=True)
    if os.path.exists(dataset_path):
        train_model(dataset_path, epochs=5)
    else:
        print(f"Dataset not found at {dataset_path}. Please unzip the dataset first.")
