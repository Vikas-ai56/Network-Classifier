# Implementation Details

## Repository layout

```
Samsung-ennovatex/
├── main.py                     # Training entrypoint (streaming + FAISS eval loop)
├── eval_cesnet.py              # Full KPI evaluation on CESNET val split
├── classify_knn_svm.py         # Classification KPI (k-NN + SVM + LR)
├── zero_day_test.py            # Zero-day KPI (balanced k-shot k-NN)
├── latency_benchmark.py        # Latency KPI (single-flow + batch throughput)
├── finetune_classifier.py      # Optional supervised fine-tune head
├── requirements.txt            # Pinned dependencies
├── KPI_RESULTS.md              # Recorded KPI results for the shipped model
├── src/
│   ├── models_dual_branch.py   # DualBranchEncoder (the model)
│   ├── feature_engineering.py  # Stateless feature extraction & normalization
│   ├── streaming_dataset.py    # CESNET-QUIC22 streaming IterableDataset
│   ├── dataset_unified.py      # Map-style loader (CESNET / ISCXVPN2016 / 5G CSV)
│   ├── data_validator.py       # Flow quality / rejection rules
│   ├── train_supcon.py         # Losses, sampler, schedulers, validation layer
│   ├── nfstream_extractor.py   # PCAP → flow features (live-capture path)
│   ├── models_mamba.py         # Mamba blocks (NetMamba lineage)
│   ├── models_net_mamba.py     # NetMamba-style image pipeline (reference)
│   └── util/                   # LR schedules, pos-embed, misc helpers
├── scripts/                    # Stand-alone analysis / sanity-check scripts
├── dataset/                    # PCAP preprocessing utilities
└── docs/                       # ← this documentation
```

## Feature engineering — `src/feature_engineering.py`

Pure, stateless, side-effect-free functions. All normalizers are computed inline from
module-level constants. `SEQ_LEN = 30`, `SEQ_INPUT_DIM = 3`, `STAT_INPUT_DIM = 16`.

### Branch A — sequential features `(30, 3)`

Extracted from the CESNET **PPI** field (Per-Packet Information: inter-packet times,
directions, sizes) via `extract_seq_features`:

| Slot | Feature | Normalization |
|---|---|---|
| 0 | packet size | `log1p(min(size, 1500)) / log1p(1500)` |
| 1 | inter-packet time | `log1p(min(ipt, 5000ms)) / log1p(5000)` (first IPT forced to 0) |
| 2 | direction | `+1` / `−1` raw (`0` = unknown, e.g. ISCXVPN2016) |

Sequences are head-truncated or tail-zero-padded to exactly 30 packets.

### Branch B — statistical features `(16,)`

Extracted via `extract_stat_features`. **All features are scale-invariant ratios or
log-normalized scalars** — deliberately chosen so the model cannot key on raw volume:

| Idx | Feature | Notes |
|---|---|---|
| 0 | `bytes_ratio` | fwd / total bytes (directional balance) |
| 1 | `packets_total_norm` | log-normalized packet count |
| 2 | `packets_ratio` | fwd / total packets |
| 3 | `mean_pkt_size_norm` | mean packet size |
| 4 | `std_pkt_size_norm` | packet-size variance |
| 5 | `mean_ipt_norm` | mean inter-packet time |
| 6 | `jitter_norm` | **std of IPT ≈ jitter** (QoS signal) |
| 7–14 | `phist_src_norm[0..7]` | 8-bin source packet-size histogram (normalized) |
| 15 | `ppi_len_norm` | number of packets in PPI / SEQ_LEN |

> **Deliberately removed:** `bytes_total` and `duration_ms`. They are *volume/time
> shortcuts* that correlate with how big a class's flows happen to be, not with what the
> application is — keeping them inflates training accuracy but destroys zero-day
> generalization. The 5-tuple ports/IPs are removed for the same reason.

## The streaming dataset — `src/streaming_dataset.py`

- Wraps **cesnet-datazoo**'s `CESNET_QUIC22` and yields `(seq, stat, label)` tensors.
- Streams the dataframe in **8192-row chunks** so a 10 M-flow dataset never has to be
  fully materialized — essential for ephemeral cloud GPU instances.
- Maps CESNET app categories → 7 unified classes (`_CESNET_CAT_MAP`).
- `_ZERO_DAY_CLASSES = {1, 2}` (music, gaming) are excluded **only from the train split**.
- Per-flow quality gates via `FlowValidator` (empty PPI, zero-variance, NaN/Inf, etc.).
- `persistent_workers=False` and `drop_last=True` are required for correctness with an
  `IterableDataset` (documented inline — persistent workers silently end training after
  epoch 1; a trailing size-1 batch breaks normalization).

## Unified loader — `src/dataset_unified.py`

Map-style `Dataset` for local files. Auto-detects source type (parquet → CESNET, JSON →
ISCXVPN2016, CSV → 5G Kaggle), applies an 8-class unified taxonomy, stratified
train/val/test splits, optional class-balanced `WeightedRandomSampler`, and the same
`FlowValidator` rejection rules. Yields `(seq, stat, label)`.

## Training loop — `main.py`

1. Build streaming train/val loaders.
2. Instantiate `DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=256, embed_dim=256)`.
3. `MarginBasedSupConLoss(λ_pos=0.7, λ_neg=0.3)`, AdamW, cosine schedule, AMP.
4. Each epoch: train → **FAISS k-NN accuracy** on val → geometric intra/inter sim →
   checkpoint (`checkpoint_latest.pth`), and save `best_model.pth` on new best accuracy.
5. Auto-resumes from `checkpoint_latest.pth` if interrupted.

## Key design decisions (and why)

| Decision | Rationale |
|---|---|
| Remove 5-tuple (ports/IPs) | Prevent shortcut leakage; preserve zero-day validity; near-zero signal for QUIC |
| Remove `bytes_total`/`duration` | Same anti-shortcut reasoning for volume features |
| LayerNorm in projection head | BatchNorm leaks batch-label stats in contrastive training |
| Final hidden state (not attention pool) | More stable, order-aware sequence summary |
| Margin loss with KPI-valued margins | Optimize the exact geometric quantities being scored |
| FAISS inner-product k-NN | Exact cosine on unit-sphere embeddings; no extra trained classifier head |
| Streaming dataset | Train on 10M flows without a multi-GB download on ephemeral GPUs |
| BiLSTM fallback for Mamba | Guarantees the code runs without a CUDA-specific kernel build |

## Reproducibility notes

- The **shipped model used the BiLSTM path** — on the training instance `causal-conv1d`
  (a Mamba build dependency) failed to compile, so `HAS_MAMBA` was `False`. This is by
  design: the fallback is a first-class path, not a degraded one.
- Fixed seeds are used in evaluation splits (`random_state=42`) for repeatable KPI numbers.
- Trained model: ~1.98 M parameters, ~8 MB checkpoint.

<!-- SCREENSHOT: terminal showing `=== Training complete === Best ProtoNet accuracy : 0.9270` -->
<!-- SCREENSHOT: phist_visualization.png (packet-size histogram feature) -->
