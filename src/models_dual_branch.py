import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class SequenceBranch(nn.Module):
    """
    Branch A: Temporal Pulse.
    Processes a time-ordered sequence of per-packet features.
    Input: (batch, seq_len=128, input_dim=3) — [size_norm, ipt_norm, direction]
    Output: (batch, d_model=128)
    """
    def __init__(self, input_dim: int = 3, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        if HAS_MAMBA:
            self.layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
                for _ in range(n_layers)
            ])
            self.is_mamba = True
        else:
            self.layers = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.is_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        if self.is_mamba:
            for layer in self.layers:
                x = layer(x)
        else:
            x, _ = self.layers(x)
        return torch.mean(x, dim=1)  # (batch, d_model)


class StatBranch(nn.Module):
    """
    Branch B: Contextual Environment.
    Processes macro flow statistics via MLP with BatchNorm for training stability.
    Input: (batch, input_dim=18)
    Output: (batch, hidden_dim=128)
    """
    def __init__(self, input_dim: int = 18, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DualBranchEncoder(nn.Module):
    """
    Full encoder: fuses Branch A (Mamba sequence) + Branch B (stat MLP)
    into a L2-normalized embedding on the unit hypersphere.

    Tensor contract:
      seq_data:  (batch, 128, 3)  — [size_norm, ipt_norm, direction]
      stat_data: (batch, 18)      — full statistical feature vector
      output:    (batch, embed_dim=256)  — L2-normalized

    ProjectionHead uses no intermediate compression + BatchNorm (SimCLR design).
    """
    def __init__(
        self,
        seq_input_dim: int = 3,
        stat_input_dim: int = 18,
        d_model: int = 128,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.seq_branch = SequenceBranch(input_dim=seq_input_dim, d_model=d_model)
        self.stat_branch = StatBranch(input_dim=stat_input_dim, hidden_dim=d_model)

        fused_dim = d_model * 2  # 256
        self.projection_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, embed_dim),
        )

    def forward(self, seq_data: torch.Tensor, stat_data: torch.Tensor) -> torch.Tensor:
        seq_feat = self.seq_branch(seq_data)    # (batch, 128)
        stat_feat = self.stat_branch(stat_data)  # (batch, 128)
        fused = torch.cat([seq_feat, stat_feat], dim=1)  # (batch, 256)
        embedding = self.projection_head(fused)          # (batch, embed_dim)
        return nn.functional.normalize(embedding, p=2, dim=1)


if __name__ == "__main__":
    model = DualBranchEncoder()
    print(f"DualBranchEncoder initialized. Using Mamba: {HAS_MAMBA}")
    dummy_seq = torch.randn(8, 128, 3)
    dummy_stat = torch.randn(8, 18)
    out = model(dummy_seq, dummy_stat)
    print(f"Output embedding shape: {out.shape}")  # expect (8, 256)
    print(f"L2 norms (expect all 1.0): {out.norm(dim=1)}")
