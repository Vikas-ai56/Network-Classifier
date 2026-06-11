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
    Input: (batch, seq_len=30, input_dim=3) — [size_norm, ipt_norm, direction]
    Output: (batch, d_model=128)

    Mamba path:  returns the final hidden state x[:, -1, :]
    BiLSTM path: concatenates final forward and backward hidden states from the last layer
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
            return x[:, -1, :]  # final hidden state: (batch, d_model)
        else:
            _, (hn, _) = self.layers(x)
            # hn: (num_layers*2, batch, hidden_size); concat fwd+bwd of last layer
            return torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)  # (batch, d_model)


class StatBranch(nn.Module):
    """
    Branch B: Contextual Environment.
    Processes scale-invariant flow statistics + 5-tuple port embeddings via MLP.
    Input: stat (batch, input_dim=16), ports (batch, 2) — [src_port, dst_port] as ints
    Output: (batch, hidden_dim=128)

    LayerNorm replaces BatchNorm to prevent intra-batch leakage during contrastive training.
    """
    def __init__(self, input_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.port_embedding = nn.Embedding(65536, 16)
        mlp_input_dim = input_dim + 2 * 16  # stat(16) + flattened port embeddings(32) = 48
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, ports: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim), ports: (batch, 2) — integer port values [0, 65535]
        port_emb = self.port_embedding(ports)           # (batch, 2, 16)
        port_emb_flat = port_emb.view(x.size(0), -1)   # (batch, 32)
        x_fused = torch.cat([x, port_emb_flat], dim=1) # (batch, 48)
        return self.mlp(x_fused)                        # (batch, hidden_dim)


class DualBranchEncoder(nn.Module):
    """
    Full encoder: fuses Branch A (Mamba/LSTM sequence) + Branch B (stat MLP + port embeddings)
    into a L2-normalized embedding on the unit hypersphere.

    Tensor contract:
      seq_data:   (batch, 30, 3)   — [size_norm, ipt_norm, direction]
      stat_data:  (batch, 16)      — scale-invariant statistical feature vector
      ports_data: (batch, 2)       — [src_port, dst_port] as raw integers [0, 65535]
      output:     (batch, embed_dim=256)  — L2-normalized

    ProjectionHead uses LayerNorm (not BatchNorm) to prevent intra-batch leakage.
    """
    def __init__(
        self,
        seq_input_dim: int = 3,
        stat_input_dim: int = 16,
        d_model: int = 128,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.seq_branch = SequenceBranch(input_dim=seq_input_dim, d_model=d_model)
        self.stat_branch = StatBranch(input_dim=stat_input_dim, hidden_dim=d_model)

        fused_dim = d_model * 2  # 256
        self.projection_head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, embed_dim),
        )

    def forward(
        self,
        seq_data: torch.Tensor,
        stat_data: torch.Tensor,
        ports_data: torch.Tensor,
    ) -> torch.Tensor:
        seq_feat = self.seq_branch(seq_data)                  # (batch, 128)
        stat_feat = self.stat_branch(stat_data, ports_data)   # (batch, 128)
        fused = torch.cat([seq_feat, stat_feat], dim=1)        # (batch, 256)
        embedding = self.projection_head(fused)                # (batch, embed_dim)
        return nn.functional.normalize(embedding, p=2, dim=1)


if __name__ == "__main__":
    model = DualBranchEncoder()
    print(f"DualBranchEncoder initialized. Using Mamba: {HAS_MAMBA}")
    dummy_seq = torch.randn(8, 30, 3)
    dummy_stat = torch.randn(8, 16)
    dummy_ports = torch.randint(0, 65536, (8, 2))
    out = model(dummy_seq, dummy_stat, dummy_ports)
    print(f"Output embedding shape: {out.shape}")   # expect (8, 256)
    print(f"L2 norms (expect all 1.0): {out.norm(dim=1)}")
