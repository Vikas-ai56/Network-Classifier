# Technical Stack

## Overview

A deep-learning pipeline that classifies **encrypted** network traffic (QUIC/TLS) into
application categories using only **behavioral** flow features — no payload inspection,
no IP/port identity. The model is a contrastive metric-learning encoder that maps each
flow to a 256-dimensional unit-sphere embedding, which is then classified with a
lightweight k-NN.

## Languages

| Language | Use |
|---|---|
| **Python 3.10–3.12** | All training, evaluation, feature engineering, and data pipelines |
| Shell (bash) | Environment setup, orchestration on cloud GPU instances |

## Core frameworks

| Component | Technology |
|---|---|
| Deep-learning framework | **PyTorch 2.1.1** (CUDA 12.1) |
| Sequence encoder | **Mamba SSM** (`mamba-ssm`) with automatic **BiLSTM** fallback |
| Nearest-neighbour index | **FAISS** (`faiss-gpu`) with NumPy fallback |
| Classical ML / metrics | **scikit-learn 1.3.2** (k-NN, Logistic Regression, SVM, reports) |
| Dataset streaming | **cesnet-datazoo** (streams CESNET-QUIC22 without full download) |
| Numerics / tabular | **NumPy 1.26**, **pandas 2.1**, **PyArrow 14** (parquet) |
| Experiment tracking | **Weights & Biases** (offline mode) — optional |

## Model summary

| Property | Value |
|---|---|
| Architecture | DualBranchEncoder (sequence branch + statistics branch + cross-attention fusion) |
| Sequence encoder (shipped) | 2-layer BiLSTM (Mamba used when CUDA build available) |
| Embedding dimension | 256, L2-normalized |
| Trainable parameters | ~1.98 M |
| Training objective | `MarginBasedSupConLoss` (margin-based supervised contrastive) |
| Input — Branch A | `(batch, 30, 3)` per-packet sequence: `[size_norm, ipt_norm, direction]` |
| Input — Branch B | `(batch, 16)` scale-invariant flow statistics |

## Hardware / runtime

| Stage | Platform |
|---|---|
| Training | Single **NVIDIA RTX 4090** (24 GB), rented on **vast.ai** |
| Mixed precision | `torch.amp` autocast + GradScaler on CUDA |
| Inference | CPU or GPU; single-flow latency **1.36 ms** on RTX 4090 |
| Data | CESNET-QUIC22 streamed (`XS` ≈ 10 M raw flows) — no local dataset download required |

## Why this stack

- **PyTorch + Mamba/BiLSTM** — Mamba gives linear-time `O(N)` sequence modelling for
  real-time packet processing; the BiLSTM fallback guarantees the code runs anywhere
  (no CUDA-specific build needed), which is what was actually used for the shipped model.
- **FAISS** — inner-product search over L2-normalized embeddings is exact cosine
  similarity, so classification is a fast k-NN lookup with no extra trained head.
- **cesnet-datazoo** — lets us train on a 10M-flow dataset by streaming chunks, avoiding
  a multi-GB download on ephemeral cloud instances.
- **scikit-learn** — standard, reviewer-trusted implementations for all KPI measurements.

<!-- SCREENSHOT: `nvidia-smi` showing RTX 4090 on the training instance -->
<!-- SCREENSHOT: `python -c "import torch; print(torch.cuda.get_device_name(0))"` output -->
