# Installation Instructions

The project runs on Linux with an NVIDIA GPU (recommended) or on CPU (slower; BiLSTM path).
Python **3.10–3.12**.

## Option A — Cloud GPU (vast.ai), the path we used

This is the exact, tested setup for an RTX 4090 instance.

### 1. Rent an instance
- Template: **PyTorch** (CUDA 12.x, torch preinstalled)
- GPU: **RTX 4090**
- Disk: **≥ 40 GB**
- Attach your SSH public key (vast.ai → Account → SSH Keys)

### 2. Connect
```bash
ssh -i ~/.ssh/vastai_key -p <PORT> root@<IP>
# or use the vast.ai in-browser Jupyter terminal
```

### 3. Set up the code (one paste)
```bash
cd /workspace && \
git clone https://github.com/Vikas-ai56/Samsung-ennovatex.git && \
cd Samsung-ennovatex && \
git checkout feat/kpi-improvements && \
pip install -q cesnet-datazoo faiss-gpu && \
([ -f requirements.txt ] && pip install -q -r requirements.txt || true) && \
echo "=== SETUP DONE ===" && \
git log --oneline -1 && \
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `=== SETUP DONE ===`, the latest commit, and `CUDA: True NVIDIA GeForce RTX 4090`.

> **`causal-conv1d` build error is expected and harmless.** It is a Mamba build
> dependency that needs torch present at build time and may fail. The model
> **auto-falls back to BiLSTM**, so training proceeds normally. This is why the install
> wraps that step in `|| true`.

### 4. Activate the venv before running anything
The vast.ai PyTorch template keeps Python in a virtualenv. A fresh shell (e.g. a new
`tmux` session) does **not** inherit it:
```bash
source /venv/main/bin/activate
# or call it directly: /venv/main/bin/python main.py ...
```

## Option B — Local machine

```bash
git clone https://github.com/Vikas-ai56/Samsung-ennovatex.git
cd Samsung-ennovatex
git checkout feat/kpi-improvements

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install PyTorch matching your CUDA (see https://pytorch.org/get-started/locally/)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1

pip install numpy pandas pyarrow scikit-learn tqdm matplotlib
pip install cesnet-datazoo          # for CESNET streaming
pip install faiss-cpu               # or faiss-gpu on CUDA
# Optional (GPU only, may need build tools): pip install mamba-ssm causal-conv1d
```

CPU-only works for inference, evaluation, and small experiments. Full streaming training
is intended for GPU.

## Dependency notes

| Package | Required? | Notes |
|---|---|---|
| `torch` 2.1.1 | Yes | Match CUDA version to your GPU/driver |
| `cesnet-datazoo` | For training/eval on CESNET | Streams the dataset; first run downloads metadata |
| `faiss-gpu` / `faiss-cpu` | Recommended | NumPy fallback exists if absent |
| `scikit-learn` | Yes | KPI measurement |
| `mamba-ssm` + `causal-conv1d` | Optional | BiLSTM fallback used automatically if missing |
| `wandb` | Optional | Runs in offline mode; safe to omit |

## Verify the install
```bash
python -c "
import torch, numpy, pandas, sklearn, cesnet_datazoo, faiss
from src.models_dual_branch import HAS_MAMBA, DualBranchEncoder
import torch as T
m = DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=256, embed_dim=256)
out = m(T.randn(4,30,3), T.randn(4,16))
print('OK | Mamba:', HAS_MAMBA, '| out', tuple(out.shape), '| L2', round(out.norm(dim=1).mean().item(),3))
"
```
Expected: `OK | Mamba: False | out (4, 256) | L2 1.0` (Mamba may be `True` if the CUDA build succeeded).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `python: command not found` | venv not activated in a new shell | `source /venv/main/bin/activate` |
| `causal-conv1d` build fails | Mamba CUDA build needs torch first | Ignore — BiLSTM fallback runs |
| `Permission denied (publickey)` (ssh) | Key not on account before renting | Add key, then destroy + re-rent |
| Training ends after 1 epoch | `persistent_workers=True` on IterableDataset | Already set to `False` in repo |
| `RuntimeError: shape mismatch` loading checkpoint | Wrong `stat_input_dim` | Must be **16** (not 18) |

<!-- SCREENSHOT: successful `=== SETUP DONE ===` + CUDA True output -->
