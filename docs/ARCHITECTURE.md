# Technical Architecture

## 1. Problem & approach

Deep Packet Inspection fails on encrypted traffic because the payload is opaque. Our
solution classifies a flow purely from its **observable behavior** — the timing, sizes,
and direction of packets — and never from its **identity** (IP addresses, ports, the
"5-tuple"). This is what enables **zero-day generalization**: a model that learns
*behavioral fingerprints* can recognize an application category it was never trained on,
whereas a model that memorizes ports cannot.

Instead of a softmax classifier (which cannot handle unseen classes), we train a
**contrastive metric-learning encoder**. It maps every flow onto a 256-dimensional unit
hypersphere such that same-application flows cluster tightly and different applications
are pushed apart. Classification is then a simple **k-nearest-neighbour** lookup, and a
brand-new class can be recognized from only a few labeled examples.

## 2. End-to-end data flow

```
        CESNET-QUIC22 (streamed)  /  ISCXVPN2016 JSON  /  PCAP via NFStream
                                  │
              ┌───────────────────┴────────────────────┐
              ▼                                         ▼
   streaming_dataset.py                        dataset_unified.py
   (IterableDataset, chunked)                  (map-style, local files)
              │                                         │
              └───────────────────┬─────────────────────┘
                                  ▼
                     feature_engineering.py
        ┌─────────────────────────┴──────────────────────────┐
        ▼                                                     ▼
  Branch A input                                       Branch B input
  (30, 3) per-packet seq                               (16,) flow statistics
  [size_norm, ipt_norm, dir]                           ratios, jitter, histogram, …
        │                                                     │
        ▼                                                     ▼
  SequenceBranch                                       StatBranch
  BiLSTM / Mamba → final                               3-layer MLP
  hidden state → LayerNorm                             (LayerNorm + GELU)
  → (batch, 256)                                       → (batch, 256)
        └──────────────────────┬──────────────────────────────┘
                               ▼
                   CrossAttentionFusion
            (seq ⇄ stat bidirectional attention)
                  → concat → (batch, 512)
                               ▼
                   Projection head (residual)
              Linear → LayerNorm → GELU → Linear (+ skip)
                               ▼
                   L2-normalize → (batch, 256)
                               │
         ┌─────────────────────┴──────────────────────┐
         ▼                                             ▼
     TRAINING                                     EVALUATION
  MarginBasedSupConLoss                    FAISS k-NN (cosine, k=5)
  (pull positives ≥0.7,                    + geometric validation
   push negatives ≤0.3)                    (intra/inter cosine sim)
```

## 3. The model — `src/models_dual_branch.py`

### Branch A — Sequence (temporal behavior)

- **Input:** first 30 packets of the flow as `(30, 3)` — normalized packet size,
  normalized inter-packet time, and direction (`+1`/`−1`, or `0` if unknown).
- **Encoder:** 2-layer **BiLSTM** (`hidden = d_model/2`, bidirectional so the
  concatenated final state = `d_model`). If `mamba-ssm` is importable, a stack of Mamba
  layers is used instead and the final position's hidden state is taken.
- **Output:** the **final hidden state** (not mean-pooling) → `LayerNorm` → `(batch, 256)`.
  Using the final recurrent state gives a stable, order-aware summary of the flow.

### Branch B — Statistics (flow context)

- **Input:** 16 **scale-invariant** flow statistics (see [IMPLEMENTATION.md](IMPLEMENTATION.md)).
- **Encoder:** 3-layer MLP, each block `Linear → LayerNorm → GELU → Dropout(0.1)`.
- **No 5-tuple:** ports/IPs are deliberately **excluded** — they are a leakage shortcut
  that memorizes servers instead of learning behavior, and for QUIC they carry almost no
  signal (dst port ≈ 443, src port ephemeral).

### Fusion — `CrossAttentionFusion`

Bidirectional multi-head cross-attention: the stat vector attends to the sequence summary
and vice-versa, each with a residual `LayerNorm`. Outputs are concatenated to `(batch, 512)`.
This lets the statistical context reweight which part of the temporal pattern matters.

### Projection head

`Linear(512→512) → LayerNorm → GELU → Linear(512→256)` **plus a residual shortcut**
`Linear(512→256)`. Output is **L2-normalized** so every embedding lives on the unit sphere.

> **LayerNorm, not BatchNorm.** BatchNorm in a contrastive projection head leaks
> intra-batch label statistics and inflates metrics dishonestly; LayerNorm is per-sample
> and avoids this.

## 4. Training objective — `MarginBasedSupConLoss`

Defined in `src/train_supcon.py`. It directly optimizes the geometric KPIs:

```
pos_loss = mean over same-class pairs   of  max(0, λ_pos − cos_sim)     # pull together, λ_pos = 0.7
neg_loss = mean over diff-class pairs   of  max(0, cos_sim − λ_neg)     # push apart,   λ_neg = 0.3
loss     = pos_loss + neg_loss
```

Because the margins **are** the KPI thresholds (0.7 / 0.3), the model optimizes the exact
quantities it is scored on. A stronger variant, `HardNegativeMarginLoss` (top-k hardest
negatives + auxiliary SupCon term), is implemented for future runs to improve class
separation.

**Training mechanics:** AdamW, cosine LR schedule with warmup, gradient clipping,
mixed-precision (`torch.amp`), light flow augmentation (Gaussian noise + random packet
dropout), checkpoint auto-resume.

## 5. Evaluation

| KPI | How measured | Script |
|---|---|---|
| Classification accuracy | k-NN (cosine, k=5) + Logistic Regression on 80/20 split | `eval_cesnet.py`, `classify_knn_svm.py` |
| Zero-day generalization | balanced k-shot k-NN over held-out classes, averaged over trials | `eval_cesnet.py`, `zero_day_test.py` |
| Intra-class cosine sim | mean cosine of same-class pairs | `eval_cesnet.py`, `execute_validation_layer` |
| Inter-class cosine sim | mean cosine of different-class pairs | same |
| Latency | single-flow forward pass, mean + p99 | `latency_benchmark.py` |

During training, `main.py` runs a **FAISS** `IndexFlatIP` k-NN every epoch (inner product
on L2-normalized vectors = cosine) as the live accuracy signal, with a NumPy fallback if
FAISS is unavailable.

## 6. Zero-day protocol

Classes **music/audio (1)** and **gaming (2)** are filtered out of the *training* split in
`streaming_dataset.py` (`_ZERO_DAY_CLASSES`) but kept in validation/test. This makes the
zero-day metric a genuine test of generalization to categories the encoder never saw
during training — not just few-shot performance on familiar classes.

<!-- SCREENSHOT: architecture diagram (can replace the ASCII flow above with a rendered figure) -->
<!-- SCREENSHOT: assets/NetMamba.png or a custom architecture render -->
