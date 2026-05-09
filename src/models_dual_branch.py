import torch
import torch.nn as nn
from typing import Optional, Tuple

# Try to import Mamba, fallback to LSTM if unavailable
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False

class SequenceBranch(nn.Module):
    """
    Branch A: The Sequence
    Processes a sequence of packet features (e.g., Size and IAT).
    """
    def __init__(self, input_dim: int = 2, d_model: int = 128, n_layers: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        if HAS_MAMBA:
            # Using Mamba layers as per the primary plan
            self.layers = nn.ModuleList([
                Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2) 
                for _ in range(n_layers)
            ])
            self.is_mamba = True
        else:
            # Fallback to 2-layer bidirectional LSTM as per the design document
            self.layers = nn.LSTM(
                input_size=d_model, 
                hidden_size=d_model // 2, 
                num_layers=n_layers, 
                batch_first=True, 
                bidirectional=True
            )
            self.is_mamba = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)
        
        if self.is_mamba:
            for layer in self.layers:
                x = layer(x)
        else:
            x, _ = self.layers(x)
            
        # Global Average Pooling over the sequence dimension
        return torch.mean(x, dim=1)

class StatBranch(nn.Module):
    """
    Branch B: The Stats
    Processes static scalars (RTT, Jitter, Total Bytes, etc.)
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class DualBranchEncoder(nn.Module):
    """
    The "Brain": Combines Sequence and Statistical branches.
    Outputs a context-aware embedding.
    """
    def __init__(
        self, 
        seq_input_dim: int = 2, 
        stat_input_dim: int = 4, 
        d_model: int = 128,
        embed_dim: int = 128
    ):
        super().__init__()
        self.seq_branch = SequenceBranch(input_dim=seq_input_dim, d_model=d_model)
        self.stat_branch = StatBranch(input_dim=stat_input_dim, hidden_dim=d_model)
        
        # Fusion and Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, embed_dim)
        )
        
    def forward(self, seq_data: torch.Tensor, stat_data: torch.Tensor) -> torch.Tensor:
        # seq_data: (batch, 128, 2)
        # stat_data: (batch, 4)
        
        seq_feat = self.seq_branch(seq_data)
        stat_feat = self.stat_branch(stat_data)
        
        # Concatenate features
        fused = torch.cat([seq_feat, stat_feat], dim=1)
        
        # Project to embedding space
        embedding = self.projection_head(fused)
        
        # L2 Normalize for Cosine Similarity during SupCon
        return nn.functional.normalize(embedding, p=2, dim=1)

if __name__ == "__main__":
    # Quick sanity check
    model = DualBranchEncoder()
    print(f"Model initialized. Using Mamba: {HAS_MAMBA}")
    
    dummy_seq = torch.randn(8, 128, 2)
    dummy_stat = torch.randn(8, 4)
    
    out = model(dummy_seq, dummy_stat)
    print(f"Output embedding shape: {out.shape}")
