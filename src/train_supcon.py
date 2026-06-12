"""
Training utilities for Samsung EnnovateX — contrastive learning + prototypical networks.

Improvements over v1:
  - HardNegativeMarginLoss: mines top-k hardest negatives per batch instead of
    penalising all negative pairs equally. Focuses training capacity on the
    boundary cases that matter most for KPI compliance.
  - Combined loss: HardNegativeMarginLoss + SupConLoss weighted sum for
    complementary gradient signals.
  - get_cosine_schedule_with_warmup: standard warmup + cosine decay for GPU training.
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SupCon Loss (retained for compatibility)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(features.device), 0,
        )
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / torch.where(
            mask_sum > 0, mask_sum, torch.ones_like(mask_sum)
        )
        return -mean_log_prob_pos[mask_sum > 0].mean()


# ---------------------------------------------------------------------------
# Hard Negative Margin Loss — primary training loss
# ---------------------------------------------------------------------------

class HardNegativeMarginLoss(nn.Module):
    """
    Margin-Based SupCon Loss with hard negative mining.

    Directly enforces KPI geometric constraints:
      - Positive pairs: penalised when cosine_sim < lambda_pos (default 0.7)
      - Negative pairs: penalised when cosine_sim > lambda_neg (default 0.3)
                        BUT only for the top-k% hardest (most similar) negatives.

    Hard negative mining concentrates gradient on boundary violators, pushing
    convergence past the easy plateau that vanilla MarginBasedSupConLoss hits.

    Parameters
    ----------
    lambda_pos : float  Target minimum intra-class cosine similarity (KPI = 0.7)
    lambda_neg : float  Target maximum inter-class cosine similarity (KPI = 0.3)
    hard_neg_ratio : float  Fraction of negative pairs to penalise (top by similarity).
                            0.5 = focus on the hardest 50% of negatives.
    supcon_weight : float   Weight for the auxiliary SupCon loss term (0 to disable).
    temperature : float     Temperature for auxiliary SupCon term.
    """
    def __init__(
        self,
        lambda_pos: float = 0.7,
        lambda_neg: float = 0.3,
        hard_neg_ratio: float = 0.5,
        supcon_weight: float = 0.3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_pos      = lambda_pos
        self.lambda_neg      = lambda_neg
        self.hard_neg_ratio  = hard_neg_ratio
        self.supcon_weight   = supcon_weight
        self._supcon         = SupConLoss(temperature=temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        device = features.device

        sim_matrix   = torch.matmul(features, features.T)            # (B, B)
        labels_col   = labels.contiguous().view(-1, 1)
        label_matrix = torch.eq(labels_col, labels_col.T)
        identity     = torch.eye(batch_size, dtype=torch.bool, device=device)

        pos_mask = label_matrix & ~identity
        neg_mask = ~label_matrix

        # --- Positive loss ---
        pos_loss  = (pos_mask.float() * torch.clamp(self.lambda_pos - sim_matrix, min=0.0)).sum()
        pos_count = pos_mask.sum().clamp(min=1)

        # --- Hard negative mining ---
        # Select top-k most similar (hardest) negative pairs per anchor
        neg_sims = sim_matrix.detach() * neg_mask.float() - 1e9 * (~neg_mask).float()
        n_neg    = neg_mask.sum(dim=1)                               # (B,)
        k        = (n_neg.float() * self.hard_neg_ratio).long().clamp(min=1)

        hard_neg_mask = torch.zeros_like(neg_mask)
        for i in range(batch_size):
            ki = k[i].item()
            if ki > 0 and neg_mask[i].any():
                topk_idx = neg_sims[i].topk(ki).indices
                hard_neg_mask[i, topk_idx] = True

        neg_loss  = (hard_neg_mask.float() * torch.clamp(sim_matrix - self.lambda_neg, min=0.0)).sum()
        neg_count = hard_neg_mask.sum().clamp(min=1)

        margin_loss = pos_loss / pos_count + neg_loss / neg_count

        # --- Auxiliary SupCon loss ---
        if self.supcon_weight > 0:
            sc_loss   = self._supcon(features, labels)
            total     = margin_loss + self.supcon_weight * sc_loss
        else:
            total = margin_loss

        return total


class MarginBasedSupConLoss(nn.Module):
    """
    Vanilla margin loss — penalises ALL positive pairs below lambda_pos
    and ALL negative pairs above lambda_neg. No hard mining, no auxiliary loss.
    Converges faster and to lower loss values than HardNegativeMarginLoss.
    """
    def __init__(self, lambda_pos: float = 0.7, lambda_neg: float = 0.3):
        super().__init__()
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        device = features.device

        sim_matrix   = torch.matmul(features, features.T)
        labels_col   = labels.contiguous().view(-1, 1)
        label_matrix = torch.eq(labels_col, labels_col.T)
        identity     = torch.eye(batch_size, dtype=torch.bool, device=device)

        pos_mask = label_matrix & ~identity
        neg_mask = ~label_matrix

        pos_loss  = (pos_mask.float() * torch.clamp(self.lambda_pos - sim_matrix, min=0.0)).sum()
        pos_count = pos_mask.sum().clamp(min=1)
        neg_loss  = (neg_mask.float() * torch.clamp(sim_matrix - self.lambda_neg, min=0.0)).sum()
        neg_count = neg_mask.sum().clamp(min=1)

        return pos_loss / pos_count + neg_loss / neg_count


# ---------------------------------------------------------------------------
# LR Scheduler — cosine decay with linear warmup
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
):
    """
    Linear warmup for `warmup_epochs`, then cosine decay to min_lr_ratio * base_lr.
    Returns a torch LambdaLR scheduler.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Flow Augmentation — applied during training only
# ---------------------------------------------------------------------------

def augment_flow(
    seq: torch.Tensor,
    stat: torch.Tensor,
    noise_std: float = 0.02,
    packet_drop_prob: float = 0.05,
) -> tuple:
    """
    Lightweight flow-level augmentation for training robustness.

    seq  : (batch, seq_len, 3)
    stat : (batch, 18)

    Augmentations:
      - Gaussian noise on both branches (simulates measurement jitter)
      - Random packet dropout (zeros out packets, simulates capture loss)
    """
    if noise_std > 0:
        seq  = seq  + torch.randn_like(seq)  * noise_std
        stat = stat + torch.randn_like(stat) * noise_std

    if packet_drop_prob > 0:
        drop_mask = (torch.rand(seq.shape[0], seq.shape[1], 1, device=seq.device)
                     > packet_drop_prob).float()
        seq = seq * drop_mask

    return seq, stat


# ---------------------------------------------------------------------------
# Episodic Sampler (unchanged)
# ---------------------------------------------------------------------------

class EpisodicSampler(torch.utils.data.Sampler):
    def __init__(self, labels, n_way, k_shot, k_query, iterations):
        self.labels  = np.array(labels)
        self.classes = np.unique(self.labels)
        self.n_way   = n_way
        self.k_shot  = k_shot
        self.k_query = k_query
        self.iterations = iterations
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}

    def __iter__(self):
        for _ in range(self.iterations):
            selected = np.random.choice(self.classes, self.n_way, replace=False)
            sup, qry = [], []
            for c in selected:
                idx = np.random.choice(self.class_indices[c],
                                       self.k_shot + self.k_query, replace=False)
                sup.extend(idx[:self.k_shot])
                qry.extend(idx[self.k_shot:])
            yield sup + qry

    def __len__(self):
        return self.iterations


# ---------------------------------------------------------------------------
# ProtoNet utilities (unchanged)
# ---------------------------------------------------------------------------

def compute_prototypes(support_features: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
    embed_dim  = support_features.shape[-1]
    prototypes = support_features.view(n_way, k_shot, embed_dim).mean(1)
    return F.normalize(prototypes, p=2, dim=1)


def prototypical_loss(
    prototypes: torch.Tensor,
    query_features: torch.Tensor,
    n_way: int,
    k_query: int,
) -> tuple:
    dists        = torch.cdist(query_features, prototypes)
    target       = torch.arange(n_way).repeat_interleave(k_query).to(query_features.device)
    loss         = F.cross_entropy(-dists, target)
    predictions  = (-dists).argmax(dim=1)
    acc          = (predictions == target).float().mean()
    return loss, acc


# ---------------------------------------------------------------------------
# Geometric Validation Layer (unchanged interface, minor logging improvement)
# ---------------------------------------------------------------------------

@torch.no_grad()
def execute_validation_layer(model: nn.Module, val_loader, device: torch.device) -> tuple:
    """
    Computes pairwise cosine similarity statistics over the validation set.

    Returns:
      avg_intra : mean cosine similarity for same-class pairs    (KPI target > 0.7)
      avg_inter : mean cosine similarity for different-class pairs (KPI target < 0.3)
    """
    model.eval()
    all_embeddings, all_labels = [], []

    for i, batch in enumerate(val_loader):
        if i >= 20:   # 20 batches × 256 = 5120 samples → ~100MB sim matrix vs 9.7GB at 500
            break
        batch_seq, batch_stat, batch_ports, batch_y = batch
        embs = model(batch_seq.to(device), batch_stat.to(device), batch_ports.to(device))
        all_embeddings.append(embs.cpu())
        all_labels.append(batch_y)

    if not all_embeddings:
        logger.warning("execute_validation_layer: empty val_loader — skipping")
        return 0.0, 0.0

    embeddings   = torch.cat(all_embeddings)
    labels       = torch.cat(all_labels)
    sim_matrix   = torch.matmul(embeddings, embeddings.T)
    label_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
    identity     = torch.eye(label_matrix.size(0), dtype=torch.bool)
    pos_mask     = label_matrix & ~identity
    neg_mask     = ~label_matrix

    avg_intra = sim_matrix[pos_mask].mean().item() if pos_mask.any() else 0.0
    avg_inter = sim_matrix[neg_mask].mean().item() if neg_mask.any() else 0.0
    return avg_intra, avg_inter
