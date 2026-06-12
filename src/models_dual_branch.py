"""
DualBranchEncoder v2 — improved architecture for Samsung EnnovateX.

Key upgrades over v1:
  1. SequenceBranch: learnable attention pooling replaces mean pooling
  2. SequenceBranch: multi-head self-attention after Mamba/LSTM for global context
  3. Fusion: cross-attention lets stat features attend to sequence context
  4. ProjectionHead: residual connection + deeper MLP for better embedding geometry
  5. All branch outputs have LayerNorm before fusion for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ---------------------------------------------------------------------------
# Attention Pooling — replaces mean pooling, learns which packets matter most
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """
    Learnable weighted pooling over a sequence.
    Instead of averaging all timesteps equally, learns to focus on
    the most discriminative packets in the flow.
    Input:  (batch, seq_len, d_model)
    Output: (batch, d_model)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.score(x), dim=1)   # (batch, seq_len, 1)
        return (x * weights).sum(dim=1)             # (batch, d_model)


# ---------------------------------------------------------------------------
# Branch A — Temporal Sequence
# ---------------------------------------------------------------------------

class SequenceBranch(nn.Module):
    """
    Branch A: Temporal Pulse — processes per-packet features.

    Pipeline:
      input → linear projection → Mamba layers (or BiLSTM)
             → final hidden state extraction → LayerNorm

    Mamba path:  returns x[:, -1, :] — the final position's hidden state.
    BiLSTM path: returns cat(hn[-2], hn[-1]) — forward + backward final states.

    Input:  (batch, seq_len=30, input_dim=3)  [size_norm, ipt_norm, direction]
    Output: (batch, d_model=256)
    """
    def __init__(self, input_dim: int = 3, d_model: int = 256, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        if HAS_MAMBA:
            self.encoder = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ])
            self.is_mamba = True
        else:
            # hidden_size = d_model // 2 so that cat(fwd, bwd) = d_model
            self.encoder = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if n_layers > 1 else 0.0,
            )
            self.is_mamba = False

        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)    # (batch, seq_len, d_model)

        if self.is_mamba:
            for layer in self.encoder:
                x = layer(x)
            out = x[:, -1, :]           # final hidden state: (batch, d_model)
        else:
            # hn: (num_layers*2, batch, d_model//2)
            _, (hn, _) = self.encoder(x)
            # hn[-2] = last-layer forward, hn[-1] = last-layer backward
            out = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # (batch, d_model)

        return self.output_norm(out)


# ---------------------------------------------------------------------------
# Branch B — Statistical / Environmental
# ---------------------------------------------------------------------------

class StatBranch(nn.Module):
    """
    Branch B: Contextual Environment — processes macro flow statistics + 5-tuple ports.

    Port embedding: nn.Embedding(65536, 16) per port × 2 ports = 32 dims.
    Concatenated with the 16 scale-invariant stat features → 48-dim MLP input.

    Input:  stat (batch, input_dim=16),  ports (batch, 2) int64
    Output: (batch, d_model=256)
    """
    _PORT_EMBED_DIM: int = 16

    def __init__(self, input_dim: int = 16, d_model: int = 256):
        super().__init__()
        self.port_embedding = nn.Embedding(65536, self._PORT_EMBED_DIM)
        mlp_in = input_dim + 2 * self._PORT_EMBED_DIM   # 16 + 32 = 48
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor, ports: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ports is None:
            ports = torch.zeros(x.shape[0], 2, dtype=torch.long, device=x.device)
        # ports: (batch, 2) → embed → (batch, 2, 16) → (batch, 32)
        port_emb = self.port_embedding(ports.clamp(0, 65535))
        port_emb = port_emb.view(port_emb.shape[0], -1)
        x = torch.cat([x, port_emb], dim=1)   # (batch, 48)
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Cross-Attention Fusion — stat features attend to sequence context
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between sequence and stat branches.

    Allows the stat branch to query which parts of the sequence are most
    relevant to its flow-level summary, and vice versa.

    Inputs:  seq_feat  (batch, d_model)
             stat_feat (batch, d_model)
    Output:  fused     (batch, d_model * 2)
    """
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.seq_to_stat = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.stat_to_seq = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm_seq  = nn.LayerNorm(d_model)
        self.norm_stat = nn.LayerNorm(d_model)

    def forward(self, seq_feat: torch.Tensor, stat_feat: torch.Tensor) -> torch.Tensor:
        s  = seq_feat.unsqueeze(1)    # (batch, 1, d_model)
        st = stat_feat.unsqueeze(1)   # (batch, 1, d_model)

        # stat attends to seq
        out_stat, _ = self.seq_to_stat(st, s, s)
        stat_out = self.norm_stat(stat_feat + out_stat.squeeze(1))

        # seq attends to stat
        out_seq, _ = self.stat_to_seq(s, st, st)
        seq_out = self.norm_seq(seq_feat + out_seq.squeeze(1))

        return torch.cat([seq_out, stat_out], dim=1)   # (batch, d_model * 2)


# ---------------------------------------------------------------------------
# Full Encoder
# ---------------------------------------------------------------------------

class DualBranchEncoder(nn.Module):
    """
    DualBranchEncoder v2 — full encoder with cross-attention fusion.

    Tensor contract:
      seq_data:  (batch, seq_len, seq_input_dim)   e.g. (B, 30, 3)
      stat_data: (batch, stat_input_dim)            e.g. (B, 18)
      output:    (batch, embed_dim)                 L2-normalized

    Architecture:
      Branch A → Mamba/BiLSTM → Self-Attn → AttentionPool → LayerNorm
      Branch B → 3-layer MLP  → LayerNorm
      Fusion   → Cross-Attention → concat → ProjectionHead (residual) → L2-norm
    """
    def __init__(
        self,
        seq_input_dim: int = 3,
        stat_input_dim: int = 18,
        d_model: int = 256,
        embed_dim: int = 256,
        n_seq_layers: int = 2,
        n_attn_heads: int = 4,
    ):
        super().__init__()
        self.seq_branch  = SequenceBranch(input_dim=seq_input_dim,  d_model=d_model,
                                          n_layers=n_seq_layers, n_heads=n_attn_heads)
        self.stat_branch = StatBranch(input_dim=stat_input_dim, d_model=d_model)
        self.fusion      = CrossAttentionFusion(d_model=d_model, n_heads=n_attn_heads)

        fused_dim = d_model * 2  # after cross-attention concat

        # Projection head with residual — LayerNorm instead of BatchNorm1d to
        # prevent intra-batch label leakage during contrastive training.
        self.proj1    = nn.Linear(fused_dim, fused_dim)
        self.proj_bn  = nn.LayerNorm(fused_dim)
        self.proj2    = nn.Linear(fused_dim, embed_dim)
        self.proj_res = nn.Linear(fused_dim, embed_dim)   # residual shortcut

    def forward(
        self,
        seq_data: torch.Tensor,
        stat_data: torch.Tensor,
        ports_data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_feat  = self.seq_branch(seq_data)                    # (batch, d_model)
        stat_feat = self.stat_branch(stat_data, ports_data)      # (batch, d_model)
        fused     = self.fusion(seq_feat, stat_feat)             # (batch, d_model*2)

        # Projection head with residual
        h = F.gelu(self.proj_bn(self.proj1(fused)))              # (batch, fused_dim)
        embedding = self.proj2(h) + self.proj_res(fused)         # residual skip

        return F.normalize(embedding, p=2, dim=1)                # (batch, embed_dim)


if __name__ == "__main__":
    from src.feature_engineering import STAT_INPUT_DIM
    model = DualBranchEncoder(stat_input_dim=STAT_INPUT_DIM)
    mamba_str = "Mamba" if HAS_MAMBA else "BiLSTM"
    print(f"DualBranchEncoder v3 | Sequence encoder: {mamba_str} (final hidden state)")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    dummy_seq   = torch.randn(8, 30, 3)
    dummy_stat  = torch.randn(8, STAT_INPUT_DIM)
    dummy_ports = torch.randint(0, 65536, (8, 2))
    out = model(dummy_seq, dummy_stat, dummy_ports)
    print(f"Output shape: {out.shape}")             # (8, 256)
    print(f"L2 norms (all should be 1.0): {out.norm(dim=1).tolist()[:4]}")
