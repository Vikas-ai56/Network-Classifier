# QoS-Aware Encrypted Network Traffic Classifier

- **Problem Statement Number** - 2
- **Problem Statement Title** - Context-Aware Flow Embeddings for Adaptive AI based Network Traffic Classification
- **Team name** - THETA
- **Team members (Names)** - K Vikas, Dhruv Singhal
- **Institute/College Name** - R V College of Engineering, Bengaluru, Karnataka
- **Final Presentation Google Drive Link** - https://drive.google.com/drive/folders/1JLqSfcOGjeKQsiLomRr8SEDWZJ2hRqAJ?usp=drive_link
- **Full Submission Demo Video Link** - *Coming soon*
- **Setup & Result Reproducibility Video Link** - https://youtu.be/SlJP2cqzuV0?feature=shared

---

### Project Artefacts

- **Technical Documentation** - Available in the [`docs/`](docs/) folder:
  - [docs/TECHNICAL_STACK.md](docs/TECHNICAL_STACK.md) — Languages, frameworks, hardware, runtime
  - [docs/OSS_LIBRARIES.md](docs/OSS_LIBRARIES.md) — All OSS libraries with links and licenses
  - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — End-to-end architecture, data flow, model design, loss, evaluation
  - [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) — File-by-file details, feature engineering, key design decisions
  - [docs/INSTALLATION.md](docs/INSTALLATION.md) — Setup on local machine + cloud GPU (vast.ai), troubleshooting
  - [docs/USER_GUIDE.md](docs/USER_GUIDE.md) — Train, evaluate all KPIs, benchmark; full command reference
  - [docs/FEATURES.md](docs/FEATURES.md) — Salient features, novelty, KPI results table
  - [docs/MODELS_AND_DATASETS.md](docs/MODELS_AND_DATASETS.md) — Models and datasets used/published, licenses, HF links

- **[Important]** [`docs/ax.md`](docs/ax.md) — Full retrospective on how open-weight models and agentic AI tools (Claude Code, Mamba SSM, tool chaining, memory, subagents) were used to build this solution, including what worked and what did not.

- **Source Code** - All source code is in [`src/`](src/) and root-level scripts:

  | File | Purpose |
  |---|---|
  | `main.py` | Training entrypoint — streams CESNET-QUIC22, trains encoder, evaluates per epoch |
  | `eval_cesnet.py` | Full KPI evaluation on CESNET-QUIC22 validation split |
  | `classify_knn_svm.py` | Classification KPI: k-NN + SVM + Logistic Regression |
  | `zero_day_test.py` | Zero-day generalization KPI: balanced k-shot k-NN |
  | `latency_benchmark.py` | Latency KPI: single-flow forward pass, mean + p99 |
  | `finetune_classifier.py` | Optional supervised fine-tune head on encoder |
  | `src/models_dual_branch.py` | `DualBranchEncoder`, `SequenceBranch`, `StatBranch`, `CrossAttentionFusion` |
  | `src/feature_engineering.py` | Stateless feature extraction and normalization |
  | `src/streaming_dataset.py` | CESNET-QUIC22 streaming `IterableDataset` |
  | `src/dataset_unified.py` | Map-style loader for CESNET / ISCXVPN2016 / 5G CSV |
  | `src/data_validator.py` | Flow quality gates (`FlowValidator`) |
  | `src/train_supcon.py` | `MarginBasedSupConLoss`, `HardNegativeMarginLoss`, `EpisodicSampler` |
  | `src/nfstream_extractor.py` | PCAP → flow features via NFStream (live-capture path) |

- **Models Used** — The `DualBranchEncoder` is trained **from scratch** on CESNET-QUIC22. No pretrained HuggingFace model weights are used. The open-weight **architectures and libraries** used are:
  - **Mamba SSM** (`mamba-ssm`, Apache-2.0) — sequence encoder backbone → https://github.com/state-spaces/mamba
  - **NetMamba** (reference architecture for traffic encoding) → https://github.com/wangtz19/NetMamba
  - **SupContrast** (BSD-2-Clause, loss methodology basis) → https://github.com/HobbitLong/SupContrast

- **Models Published** — The trained **DualBranchEncoder** checkpoint (~1.98 M parameters) developed as part of this solution is published on HuggingFace under the **Apache-2.0** license (`best_model.pth` + `prototypes.pth` + model card).
  - HuggingFace link: **https://huggingface.co/dhruvsinghal1387/dualbranch-quic-encoder**

- **Datasets Used**
  - **CESNET-QUIC22** — ~10 M+ real QUIC flows from the CESNET ISP backbone; per-packet info, flow stats, 7 application categories. Primary training and evaluation dataset. Accessed via `cesnet-datazoo` (no manual download required). License: **Creative Commons**. → https://github.com/CESNET/cesnet-datazoo · DOI: https://doi.org/10.1016/j.dib.2023.108888
  - **ISCXVPN2016** — VPN / non-VPN encrypted traffic captures; used for reference and cross-dataset checks only. License: Academic/research (UNB). → https://www.unb.ca/cic/datasets/vpn.html

- **Datasets Published** — No new dataset was created or published for this project. Training uses the publicly available CESNET-QUIC22 under its Creative Commons license. Derived feature tensors are computed on the fly and not stored or redistributed.
  - Published dataset link: *N/A*

---

#### Final Presentation

**Slides:** [Samsung EnnovateX — Team THETA (Google Drive)](https://drive.google.com/drive/folders/1JLqSfcOGjeKQsiLomRr8SEDWZJ2hRqAJ?usp=drive_link)

The presentation covers: problem statement, solution innovation and novelty, architecture overview, open datasets/models used and developed, KPI results, agentic AI development approach, and final deliverable details.

---

#### Full Submission Demo Video

*Coming soon.*

The demo video will show the solution working end-to-end — traffic capture, classification output, and how it addresses the problem statement.

---

#### Setup & Result Reproducibility Video

**Video:** https://youtu.be/SlJP2cqzuV0?feature=shared

The reproducibility video demonstrates:

1. Step-by-step project installation (clone repo, create venv, install dependencies)
2. Data download steps (`cesnet-datazoo` first-run metadata download)
3. Model training: `python main.py --streaming --streaming_size XS`
4. Evaluation to reproduce all KPIs: `python eval_cesnet.py --model_path model/best_model.pth --n_samples 5000`

---

### Attribution

This project builds on the following open-source projects:

| Project | Original Link | New features developed |
|---|---|---|
| **NetMamba** | https://github.com/wangtz19/NetMamba | Replaced image-based encoding with behavioral flow feature vectors; added dual-branch cross-attention fusion; added CESNET-QUIC22 streaming pipeline; added contrastive metric-learning training with `MarginBasedSupConLoss`; added zero-day evaluation protocol |
| **Mamba SSM** | https://github.com/state-spaces/mamba | Integrated as the sequence encoder backend with automatic BiLSTM fallback for non-CUDA environments |
| **SupContrast** | https://github.com/HobbitLong/SupContrast | Adapted into `MarginBasedSupConLoss` with KPI-matched margins (0.7 / 0.3) and extended with `HardNegativeMarginLoss` for hard-negative mining |
