import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.
    Encourages embeddings of the same class to be closer than different classes.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (batch, embed_dim) - assumed normalized
        # labels: (batch)
        
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Compute dot product (cosine similarity since features are normalized)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive pairs
        # Prevent division by zero if a class has only one sample in the batch
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
        
        # Only average over samples that have at least one positive pair
        loss = -mean_log_prob_pos[mask_sum > 0].mean()
        return loss

class EpisodicSampler(torch.utils.data.Sampler):
    """
    Samples episodes (N-way, K-shot) for Prototypical Networks.
    """
    def __init__(self, labels, n_way, k_shot, k_query, iterations):
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.iterations = iterations
        
        # Group indices by class
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}

    def __iter__(self):
        for _ in range(self.iterations):
            # Select N random classes
            selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
            
            support_indices = []
            query_indices = []
            
            for c in selected_classes:
                indices = self.class_indices[c]
                # Select K samples for support set and K for query set
                selected_indices = np.random.choice(indices, self.k_shot + self.k_query, replace=False)
                support_indices.extend(selected_indices[:self.k_shot])
                query_indices.extend(selected_indices[self.k_shot:])
                
            yield support_indices + query_indices

    def __len__(self):
        return self.iterations

def compute_prototypes(support_features, n_way, k_shot):
    """
    Computes class prototypes as the mean of support embeddings.
    """
    # support_features: (n_way * k_shot, embed_dim)
    embed_dim = support_features.shape[-1]
    prototypes = support_features.view(n_way, k_shot, embed_dim).mean(1)
    return prototypes

def prototypical_loss(prototypes, query_features, n_way, k_query):
    """
    Classifies query features based on Euclidean distance to prototypes.
    """
    # prototypes: (n_way, embed_dim)
    # query_features: (n_way * k_query, embed_dim)
    
    # Compute squared Euclidean distance
    # (n_way * k_query, n_way)
    dists = torch.cdist(query_features, prototypes)
    
    # Create target labels for query set
    # labels: [0,0...0, 1,1...1, ..., n_way-1...n_way-1]
    target_labels = torch.arange(n_way).repeat_interleave(k_query).to(query_features.device)
    
    # Calculate CrossEntropy loss on the distances (negative distance as logit)
    loss = F.cross_entropy(-dists, target_labels)
    
    # Calculate accuracy
    _, predictions = torch.max(-dists, 1)
    acc = (predictions == target_labels).float().mean()
    
    return loss, acc

if __name__ == "__main__":
    print("Training modules initialized: SupConLoss, EpisodicSampler, ProtoNet Logic")
    # Sanity check for SupConLoss
    loss_fn = SupConLoss()
    feat = torch.randn(16, 128)
    feat = F.normalize(feat, p=2, dim=1)
    lbl = torch.randint(0, 4, (16,))
    loss = loss_fn(feat, lbl)
    print(f"SupCon Loss: {loss.item():.4f}")
