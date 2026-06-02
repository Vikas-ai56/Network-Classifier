# Training Guide — QoS-Aware Encrypted Traffic Classifier
### Team THETA | Samsung EnnovateX Hackathon

---

## 1. Architecture at a Glance

One model. One training script. No Path B.

```
Flow (CESNET parquet / ISCXVPN2016 JSON / NFStream PCAP)
        │
        ▼  dataset_unified.py / streaming_dataset.py
        │
        ├──► Branch A — Mamba SSM (or BiLSTM fallback)
        │    Input:  (batch, 128, 3)  →  [size_norm, ipt_norm, direction]
        │    Output: (batch, 128)
        │
        ├──► Branch B — MLP + BatchNorm
        │    Input:  (batch, 18)      →  18-feature statistical vector
        │    Output: (batch, 128)
        │
        └──► Fusion concat (batch, 256)
                  └──► ProjectionHead: Linear→BatchNorm→ReLU→Linear
                  └──► L2-normalize
                  └──► embedding (batch, 256) on unit hypersphere
                               │
              ┌────────────────┴────────────────┐
         TRAINING                          EVALUATION
         MarginBasedSupConLoss             1. Geometric Validation Layer
         λ_pos=0.7, λ_neg=0.3                (pairwise cosine sim → KPIs)
                                          2. ProtoNet (zero-day episodes)
                                          3. k-NN k=5 (final accuracy)
```

**Tensor contract (all paths must produce these shapes):**

| Tensor | Shape | Content |
|---|---|---|
| seq | `(batch, 128, 3)` | `[log1p(size)/log1p(1500), log1p(ipt)/log1p(5000), direction]` |
| stat | `(batch, 18)` | bytes, packets, duration, mean/std size+IPT, 8×PHIST, flow_len |
| embedding | `(batch, 256)` | L2-normalised, unit norm |

---

## 2. GPU Setup (Vast.ai)

### 2.1 Rent the instance
- **GPU:** RTX 4090 (interruptible, ~$0.25–0.35/hr)
- **Disk:** 50 GB minimum (200 GB if downloading CESNET locally)
- **Image:** `pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel`
- **Payment:** BitPay/USDT if Indian card fails (no KYC)

```bash
vastai search offers \
  --type interruptible \
  --order score- \
  --query "gpu_name=RTX_4090 num_gpus=1 cuda_vers>=12.1 disk_space>=50"
```

### 2.2 First-time setup on the instance
```bash
# SSH in, then:
cd /workspace
git clone https://github.com/YOUR_REPO/Netwok-Classifier.git
cd Netwok-Classifier

# Install dependencies
pip install causal-conv1d==1.1.0 numpy==1.26.2 scikit-learn==1.3.2 \
  pandas==2.1.3 einops==0.7.0 transformers==4.35.2 timm==0.4.12 \
  nfstream wandb cesnet-datazoo pyarrow

# Install vendored Mamba from source (critical — must be from source, not PyPI)
cd mamba-1p1p1 && pip install -e . && cd ..

# Verify Mamba CUDA kernels loaded
python3 -c "from mamba_ssm import Mamba; m = Mamba(128).cuda(); print('Mamba CUDA: OK')"
```

### 2.3 Verify pipeline before starting training
```bash
python3 -c "
import torch
from src.models_dual_branch import DualBranchEncoder, HAS_MAMBA
from src.train_supcon import MarginBasedSupConLoss
m = DualBranchEncoder().cuda()
out = m(torch.randn(4,128,3).cuda(), torch.randn(4,18).cuda())
print(f'Mamba: {HAS_MAMBA}  |  Output: {out.shape}  |  Norms: {out.norm(dim=1).tolist()}')
# Expect: Output: torch.Size([4, 256])  |  Norms: [1.0, 1.0, 1.0, 1.0]
"
```

---

## 3. Datasets — What Goes Where

| Dataset | Role | Storage needed |
|---|---|---|
| **ISCXVPN2016** | Primary training (local, labeled) | ~500 MB (already in repo) |
| **CESNET-QUIC22** | Secondary training (streamed, no full download) | ~50 MB metadata only |
| **5G Kaggle** | Generalization eval only — **never in training** | ~1 GB |
| **MAWI** | Post-submission robustness check only | unlabeled, skip for now |

### 3.1 ISCXVPN2016 layout (already in repo)
```
datasets/netmamba/ISCXVPN2016/images_sampled_new/
  youtube/
    flow_001.json    ← {"lengths": [64, 128, ...], "intervals": [12.5, 45.0, ...]}
  netflix/
  gaming/
  voip/
  ...
```

### 3.2 5G Kaggle (download directly to instance)
```bash
pip install kaggle
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USER","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d kimdaegyeom/5g-traffic-datasets \
  -p /workspace/datasets/5g_kaggle/ --unzip
```

### 3.3 CESNET streaming — no download required
CESNET-QUIC22 (150M flows) is pulled batch-by-batch from the CESNET Data Zoo API during training. Only ~50 MB of metadata is stored locally.

```bash
# cesnet-datazoo is already installed from §2.2
# The --streaming flag in main.py handles everything
```

---

## 4. Training

### 4.1 Mode 1 — Local ISCXVPN2016 (start here to verify pipeline)

```bash
python3 main.py \
  --data_dir datasets/netmamba/ISCXVPN2016/images_sampled_new \
  --epochs 50 \
  --batch_size 128
```

**When to use:** First run on a new GPU instance. Verifies the full pipeline end-to-end in minutes before committing to a long CESNET run.

### 4.2 Mode 2 — CESNET Streaming (main training run)

```bash
# 1M flows per epoch — recommended starting point
python3 main.py \
  --streaming \
  --streaming_root /workspace/.cesnet_cache \
  --streaming_size S \
  --batch_size 128 \
  --epochs 10

# 10M flows per epoch — use after pipeline is verified
python3 main.py \
  --streaming \
  --streaming_root /workspace/.cesnet_cache \
  --streaming_size M \
  --batch_size 128 \
  --epochs 5
```

**Streaming sizes:**

| `--streaming_size` | Flows per epoch | Approx time/epoch (RTX 4090) | Cost |
|---|---|---|---|
| `XS` | 100K | ~5 min | ~$0.03 |
| `S` | 1M | ~30 min | ~$0.15 |
| `M` | 10M | ~3 hr | ~$0.90 |

### 4.3 Recommended training sequence

```
Step 1: Verify locally        python3 main.py --data_dir ... --epochs 5 --batch_size 128
Step 2: Validate metrics      Check stdout: loss decreasing, intra_sim rising, inter_sim falling
Step 3: Stream CESNET (S)     python3 main.py --streaming --streaming_size S --epochs 10
Step 4: Scale to M            python3 main.py --streaming --streaming_size M --epochs 5
Step 5: Final eval on 5G      See §6
```

---

## 5. Config Reference

| Parameter | Default | Recommended | Notes |
|---|---|---|---|
| `epochs` | 10 | **50–100** (local), **10–20** (streaming S) | MarginBasedSupConLoss converges faster than standard SupCon |
| `batch_size` | 128 | **128–256** | Larger = more positive/negative pairs per gradient step |
| `seq_len` | 128 | 128 | Fixed — matches NFStream `splt_analysis` and ISCXVPN2016 flows |
| `seq_input_dim` | 3 | 3 | size_norm + ipt_norm + direction (hardcoded) |
| `stat_input_dim` | 18 | 18 | Full feature vector (hardcoded) |
| `embed_dim` | 256 | 256 | 256-dim unit hypersphere |
| `d_model` | 128 | 128 | Mamba hidden dim per branch |
| `lr` | 1e-3 | 1e-3 | AdamW |
| `weight_decay` | 1e-4 | 1e-4 | |
| `grad_clip` | 1.0 | 1.0 | `clip_grad_norm_` — mandatory with Mamba |
| `lambda_pos` | 0.7 | 0.7 | MarginLoss: intra-class cosine sim target |
| `lambda_neg` | 0.3 | 0.3 | MarginLoss: inter-class cosine sim target |
| `n_way` | 5 | 5 | ProtoNet episode classes |
| `k_shot` | 5 | 5 | Support examples per class per episode |
| `k_query` | 15 | 15 | Query examples per class per episode |
| `streaming_size` | S | S → M | XS for smoke test, S for validation, M for production |

---

## 6. Metrics — What to Watch and What They Mean

### 6.1 Live stdout output (every epoch)
```
Epoch 5 | Loss: 0.2341 | ProtoNet Acc: 0.7200 | Intra-Sim: 0.6512 (>0.7) | Inter-Sim: 0.1843 (<0.3)
```

| Metric | KPI Target | Healthy progression | Fail signal |
|---|---|---|---|
| `Loss` | Decreasing | Halves every 10–15 epochs | Flat after epoch 5 → batch too small or LR wrong |
| `ProtoNet Acc` | **≥ 0.85** | > 0.5 by epoch 20 | < 0.25 after 30 epochs → Mamba not on CUDA (BiLSTM fallback) |
| `Intra-Sim` | **≥ 0.70** | Rising from ~0.2 to ≥0.7 | Stuck below 0.4 at epoch 50 → increase epochs/batch |
| `Inter-Sim` | **≤ 0.30** | Falling from ~0.5 to ≤0.3 | Rising = collapse, classes merging |
| `k-NN Acc` | **≥ 0.90** | Printed once after all epochs | < 0.70 = embedding space not separating classes |

### 6.2 W&B dashboard

Logs are written offline and synced manually (safe for venue WiFi):
```bash
# During/after training:
wandb sync --sync-all    # push all cached runs to cloud

# Or view offline:
wandb offline            # already set by main.py
tensorboard --logdir output/ --port 6006
```

**Key W&B plots:**

| Plot name | What to look for |
|---|---|
| `train/loss` | Smooth monotonic decrease. Spikes = LR too high. |
| `eval/intra_sim` | Should cross 0.7 and stay there. |
| `eval/inter_sim` | Should cross below 0.3 and stay there. |
| `eval/proto_acc` | Should exceed 0.85 before you stop training. |
| `final/knn_acc` | Single value at end — your 90% KPI number. |

### 6.3 KPI verification (geometric validation layer)

Run anytime on a checkpoint to get the official KPI numbers:

```python
import torch
from src.models_dual_branch import DualBranchEncoder
from src.train_supcon import execute_validation_layer
from src.dataset_unified import build_dataloaders

device = torch.device("cuda")
model = DualBranchEncoder().to(device)
model.load_state_dict(torch.load("model/best_model.pth")["model_state_dict"])

_, val_loader, _, _ = build_dataloaders("datasets/netmamba/ISCXVPN2016/images_sampled_new")
avg_intra, avg_inter = execute_validation_layer(model, val_loader, device)

print(f"Intra-class cosine sim: {avg_intra:.4f}  (KPI: > 0.7)")
print(f"Inter-class cosine sim: {avg_inter:.4f}  (KPI: < 0.3)")
```

### 6.4 t-SNE embedding plot (demo artifact)

```python
import torch, numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model.eval()
all_emb, all_lbl = [], []
with torch.no_grad():
    for seq, stat, labels in val_loader:
        emb = model(seq.to(device), stat.to(device)).cpu().numpy()
        all_emb.append(emb); all_lbl.extend(labels.numpy())

coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(np.concatenate(all_emb))
CLASS_NAMES = {0:"video", 1:"audio", 2:"gaming", 3:"social", 4:"file_xfer", 5:"browsing", 6:"comms", 7:"vpn"}
plt.figure(figsize=(10, 8))
for cls_id, cls_name in CLASS_NAMES.items():
    mask = np.array(all_lbl) == cls_id
    if mask.any():
        plt.scatter(coords[mask,0], coords[mask,1], label=cls_name, s=10, alpha=0.7)
plt.legend(); plt.title("Flow Embedding Space (t-SNE)")
plt.savefig("output/tsne_embeddings.png", dpi=150)
print("Saved: output/tsne_embeddings.png")
```

### 6.5 Confusion matrix (k-NN predictions)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Extract embeddings (train + val)
def get_embeddings(loader):
    embs, lbls = [], []
    model.eval()
    with torch.no_grad():
        for seq, stat, labels in loader:
            embs.append(model(seq.to(device), stat.to(device)).cpu().numpy())
            lbls.extend(labels.numpy())
    return np.concatenate(embs), np.array(lbls)

train_loader, val_loader, _, _ = build_dataloaders("datasets/...")
X_train, y_train = get_embeddings(train_loader)
X_val, y_val = get_embeddings(val_loader)

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn.fit(X_train, y_train)
y_pred = knn.predict(X_val)

print(classification_report(y_val, y_pred, target_names=list(CLASS_NAMES.values())))

cm = confusion_matrix(y_val, y_pred)
ConfusionMatrixDisplay(cm, display_labels=list(CLASS_NAMES.values())).plot(xticks_rotation=45)
plt.savefig("output/confusion_matrix.png", dpi=150)
```

### 6.6 Fail point identification

| Symptom | Root cause | Fix |
|---|---|---|
| `Intra-Sim` stuck at 0.2 after 30 epochs | Mamba running as BiLSTM (CUDA not found) | `python3 -c "from mamba_ssm import Mamba; Mamba(128).cuda(); print('OK')"` |
| `Loss` = `nan` on epoch 1 | Gradient explosion | Verify `clip_grad_norm_` is active in training loop |
| `Inter-Sim` rising instead of falling | Class collapse — all embeddings converging | Increase `lambda_neg`, check `WeightedRandomSampler` is on |
| `ProtoNet Acc` = 0.20 exactly | Random baseline — model not learning | Check batch contains multiple classes (n_way=5 requires ≥5 classes in dataset) |
| `k-NN Acc` low but `ProtoNet Acc` high | ProtoNet uses 5 prototypes, k-NN uses all train; imbalanced data | Run `report.per_class_counts` from `UnifiedFlowDataset` init log |
| Streaming stalls / stops | CESNET Data Zoo rate limit or network drop | Add `--streaming_size XS` to test; check `/workspace/.cesnet_cache` for partial files |
| OOM on RTX 4090 | Batch too large | Halve `batch_size`; model is ~4M params so OOM is rare |

---

## 7. Checkpoints and Recovery

### 7.1 Checkpoint format (main.py)

Saves on every ProtoNet acc improvement:
```python
{
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_acc": best_acc,
}
# Saved to: model/best_model.pth
```

### 7.2 Resume from checkpoint

```python
checkpoint = torch.load("model/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"] + 1
```

### 7.3 Backup before instance is reclaimed

**Always run this before stopping a Vast.ai interruptible instance:**
```bash
# From your local machine:
rsync -avz -e "ssh -p PORT" root@HOST_IP:/workspace/Netwok-Classifier/model/ ./local_model_backup/
rsync -avz -e "ssh -p PORT" root@HOST_IP:/workspace/Netwok-Classifier/output/ ./local_output_backup/

# Or push to HuggingFace Hub (free 10 GB):
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='model/', repo_id='YOUR_HF_USER/netwok-ckpts', repo_type='dataset')
"
```

---

## 8. Final Evaluation on 5G Kaggle

Run this only after training is complete. 5G Kaggle is the held-out generalization test.

```python
from src.dataset_unified import build_dataloaders
from src.train_supcon import execute_validation_layer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np, torch

device = torch.device("cuda")
model = DualBranchEncoder().to(device)
model.load_state_dict(torch.load("model/best_model.pth")["model_state_dict"])
model.eval()

# Build loaders for ISCXVPN2016 (train embeddings) and 5G Kaggle (test)
train_loader, _, _, _ = build_dataloaders("datasets/netmamba/ISCXVPN2016/")
_, _, test_loader, _ = build_dataloaders("datasets/5g_kaggle/",
                                          source_hint="5g",
                                          min_samples_per_class=1)

def embed(loader):
    embs, lbls = [], []
    with torch.no_grad():
        for seq, stat, lbl in loader:
            embs.append(model(seq.to(device), stat.to(device)).cpu().numpy())
            lbls.extend(lbl.numpy())
    return np.concatenate(embs), np.array(lbls)

X_train, y_train = embed(train_loader)
X_test,  y_test  = embed(test_loader)

knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
knn.fit(X_train, y_train)
acc = knn.score(X_test, y_test)
print(f"5G Kaggle Generalization Accuracy: {acc:.4f}  (KPI: ≥ 0.85)")

intra, inter = execute_validation_layer(model, test_loader, device)
print(f"Intra-sim on test: {intra:.4f}  (KPI: > 0.7)")
print(f"Inter-sim on test: {inter:.4f}  (KPI: < 0.3)")
```

---

## 9. Real-time Inference Latency (< 100ms KPI)

```python
import time, torch
from src.nfstream_extractor import NFStreamExtractor

# Simulate single-flow inference
extractor = NFStreamExtractor()
model.eval()

# Benchmark CPU (worst case for deployment)
dummy_seq = torch.randn(1, 128, 3)
dummy_stat = torch.randn(1, 18)

with torch.no_grad():
    # Warmup
    for _ in range(10):
        model(dummy_seq, dummy_stat)

    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        model(dummy_seq, dummy_stat)
        times.append((time.perf_counter() - t0) * 1000)

print(f"CPU latency: {sum(times)/len(times):.2f} ms avg  |  max: {max(times):.2f} ms")
print(f"KPI (<100ms): {'PASS' if max(times) < 100 else 'FAIL'}")
```

---

## 10. Project Status

### Complete
- [x] `DualBranchEncoder` with Mamba SSM / BiLSTM fallback (`src/models_dual_branch.py`)
- [x] `MarginBasedSupConLoss` — directly optimises for intra>0.7 / inter<0.3 KPIs
- [x] `execute_validation_layer` — pairwise cosine sim matrix for live KPI tracking
- [x] k-NN (k=5) downstream classifier — runs after training
- [x] `EpisodicSampler` + ProtoNet zero-day evaluation
- [x] `UnifiedFlowDataset` — handles CESNET, ISCXVPN2016, 5G Kaggle
- [x] `CESNETStreamingDataset` — streams from Data Zoo with no full download
- [x] Feature engineering: 128-step sequences, 18 stat features, log1p normalization
- [x] Data validation: 8 hard rejection rules + 6 soft warnings
- [x] `WeightedRandomSampler` for class imbalance
- [x] Gradient clipping (`max_norm=1.0`)
- [x] W&B offline logging
- [x] Per-epoch checkpoint with optimizer state
- [x] `NFStreamExtractor` aligned — outputs `(batch, 128, 3)` matching training tensors

### Still needed before submission

| Gap | Impact | Effort |
|---|---|---|
| t-SNE plot script | Most impactful demo artifact | Use §6.4 snippet |
| `models_dual_branch.py` sanity check on CUDA | Confirm Mamba is running, not BiLSTM | One assert at training start |
| LR cosine annealing in `main.py` | 3–8 pp accuracy improvement | `torch.optim.lr_scheduler.CosineAnnealingLR` |
| Cross-dataset eval on 5G Kaggle | Demonstrates 85% generalization KPI | §8 script above |

---

## 11. Learning Resources

| Topic | Resource | What to read |
|---|---|---|
| Why Margin-Based SupCon beats standard | [SupCon paper](https://arxiv.org/abs/2004.11362) | §3 loss formulation; compare to our margin variant |
| Why BatchNorm in projection head matters | [SimCLR paper](https://arxiv.org/abs/2002.05709) | §3 ablation — BatchNorm gives +5pp on all benchmarks |
| Reading t-SNE correctly | [Wattenberg et al.](https://distill.pub/2016/misread-tsne/) | Read fully before presenting t-SNE to judges |
| Diagnosing training failures | [Karpathy recipe](https://karpathy.github.io/2019/04/25/recipe/) | §3 "overfit one batch" trick |
| Mamba selective scan | [Mamba paper](https://arxiv.org/abs/2312.00752) | §3.1 for the SSM mechanism |
| CESNET-QUIC22 dataset spec | [CESNET paper](https://arxiv.org/abs/2209.04461) | §3 for PPI field layout and CATEGORY labels |
| k-NN for metric learning | [k-NN on hypersphere](https://arxiv.org/abs/2104.09864) | §4 — why cosine k-NN beats Euclidean for L2-normalised embeddings |
