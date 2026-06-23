# Technical Documentation — Index

**Project:** QoS-Aware Encrypted Network Traffic Classifier
**Event:** Samsung EnnovateX 2025 Hackathon
**Model:** DualBranchEncoder (contrastive metric-learning encoder for encrypted QUIC traffic)

This folder contains the complete technical documentation for the project. Start here.

> ⚠️ **Note on legacy docs:** `docs/README.md`, `docs/TRAINING_GUIDE.md`, and
> `docs/ARCHITECTURE_CHANGE_CHECKLIST.md` describe an **earlier** iteration of the
> architecture (128-dim embeddings, BatchNorm projection head, 18 statistical
> features, 5-tuple port features, `SEQ_LEN=128`). The documents below describe the
> **current, shipped** architecture (256-dim, LayerNorm, 16 behavioral features,
> ports removed, `SEQ_LEN=30`). Where they conflict, **the documents below are authoritative.**

## Contents

| Document | What it covers |
|---|---|
| [TECHNICAL_STACK.md](TECHNICAL_STACK.md) | Languages, frameworks, hardware, runtime environment |
| [OSS_LIBRARIES.md](OSS_LIBRARIES.md) | Every open-source library/project used, with links and purpose |
| [ARCHITECTURE.md](ARCHITECTURE.md) | End-to-end solution architecture, data flow, model design, loss, evaluation |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | File-by-file implementation details, feature engineering, key design decisions |
| [INSTALLATION.md](INSTALLATION.md) | Setup on local + cloud GPU (vast.ai), dependency notes, troubleshooting |
| [USER_GUIDE.md](USER_GUIDE.md) | How to train, evaluate, and benchmark; command reference |
| [FEATURES.md](FEATURES.md) | Salient features and what makes the solution novel |
| [MODELS_AND_DATASETS.md](MODELS_AND_DATASETS.md) | Models used/published, datasets used/published, licenses, HF links |
| [ax.md](ax.md) | **How we used agentic AI / open-weight tooling to build this** (required deliverable) |

## Results at a glance

Trained and evaluated on **CESNET-QUIC22** (`XS` streaming split) on an RTX 4090.

| KPI | Result | Target | Status |
|---|---|---|---|
| Classification accuracy | **90.90%** | ≥ 90% | ✅ |
| Intra-class cosine similarity | **0.7283** | > 0.7 | ✅ |
| Inference latency | **1.36 ms / flow** | < 100 ms | ✅ |
| Zero-day generalization | **84.84%** | ≥ 85% | ❌ (−0.16) |
| Inter-class cosine similarity | **0.3833** | < 0.3 | ❌ (+0.083) |

**3 / 5 KPIs met.** Full breakdown in [../KPI_RESULTS.md](../KPI_RESULTS.md).

