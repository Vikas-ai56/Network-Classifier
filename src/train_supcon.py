import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

logger = logging.getLogger(__name__)

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

class MarginBasedSupConLoss(nn.Module):
    """
    Margin-Based Supervised Contrastive Loss.

    Directly enforces the KPI geometric constraints:
      - Positive pairs: penalised only when cosine_sim < lambda_pos (default 0.7)
      - Negative pairs: penalised only when cosine_sim > lambda_neg (default 0.3)

    Once a pair satisfies its constraint the gradient contribution drops to zero,
    focusing all training capacity on hard boundary cases.

    Reference: Section 6 of Samsung EnnovateX System Architecture doc.
    """
    def __init__(self, lambda_pos: float = 0.7, lambda_neg: float = 0.3):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (batch, embed_dim) — must be L2-normalised before calling
        # labels:   (batch,)
        batch_size = features.shape[0]
        device = features.device

        # Cosine similarity matrix via dot product (works because features are L2-normalised)
        sim_matrix = torch.matmul(features, features.T)  # (batch, batch)

        labels_col = labels.contiguous().view(-1, 1)
        label_matrix = torch.eq(labels_col, labels_col.T)  # True where same class
        identity = torch.eye(batch_size, dtype=torch.bool, device=device)

        pos_mask = label_matrix & ~identity   # same-class, exclude self
        neg_mask = ~label_matrix              # different-class

        # Positive loss: penalise pairs below lambda_pos
        pos_loss = (pos_mask.float() * torch.clamp(self.lambda_pos - sim_matrix, min=0.0)).sum()
        pos_count = pos_mask.sum().clamp(min=1)

        # Negative loss: penalise pairs above lambda_neg
        neg_loss = (neg_mask.float() * torch.clamp(sim_matrix - self.lambda_neg, min=0.0)).sum()
        neg_count = neg_mask.sum().clamp(min=1)

        return pos_loss / pos_count + neg_loss / neg_count


@torch.no_grad()
def execute_validation_layer(
    model: nn.Module,
    val_loader,
    device: torch.device,
) -> tuple:
    """
    Geometric Validation Layer — computes pairwise cosine similarity statistics.

    Runs the full val set through the frozen encoder and returns:
      avg_intra: mean cosine similarity between same-class pairs  (target > 0.7)
      avg_inter: mean cosine similarity between different-class pairs (target < 0.3)

    Reference: Section 7 of Samsung EnnovateX System Architecture doc.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for i, (batch_seq, batch_stat, batch_y) in enumerate(val_loader):
        if i >= 500:  # cap to avoid OOM on sim_matrix computation
            break
        embeddings = model(batch_seq.to(device), batch_stat.to(device))
        all_embeddings.append(embeddings.cpu())
        all_labels.append(batch_y)

    if not all_embeddings:
        logger.warning("execute_validation_layer: val_loader yielded no samples — skipping")
        return 0.0, 0.0

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)

    sim_matrix = torch.matmul(embeddings, embeddings.T)
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity_mask = torch.eye(label_matrix.size(0), dtype=torch.bool)

    positive_mask = label_matrix & ~identity_mask
    negative_mask = ~label_matrix

    avg_intra = sim_matrix[positive_mask].mean().item() if positive_mask.any() else 0.0
    avg_inter = sim_matrix[negative_mask].mean().item() if negative_mask.any() else 0.0

    return avg_intra, avg_inter


if __name__ == "__main__":
    print("Training modules: SupConLoss, MarginBasedSupConLoss, EpisodicSampler, ProtoNet, GeometricValidation")

    # Sanity check SupConLoss
    loss_fn = SupConLoss()
    feat = F.normalize(torch.randn(16, 256), p=2, dim=1)
    lbl = torch.randint(0, 4, (16,))
    print(f"SupCon Loss: {loss_fn(feat, lbl).item():.4f}")

    # Sanity check MarginBasedSupConLoss
    margin_fn = MarginBasedSupConLoss(lambda_pos=0.7, lambda_neg=0.3)
    print(f"MarginBased Loss: {margin_fn(feat, lbl).item():.4f}")
