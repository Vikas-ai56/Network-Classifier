# Network Traffic Classifier: End-to-End Codebase Documentation

This document explains the repository architecture, entrypoints, logic flow, folder responsibilities, and where to modify the system for future work.

## 1) Quick Orientation

This repository currently contains **two related pipelines**:

1. **Dual-Branch SupCon + ProtoNet pipeline** (main hackathon workflow)
   - Entrypoint: `main.py`
   - Uses sequence + stats encoder from `src/models_dual_branch.py`
   - Uses JSON sequence dataset loader from `src/dataset_netmamba.py`
   - Uses contrastive/prototypical training logic from `src/train_supcon.py`

2. **Image-style NetMamba pretrain/finetune pipeline** (upstream-style training code)
   - Entrypoints:
     - `src/pre-train.py`
     - `src/fine-tune.py`
     - `src/eval.py`
   - Uses model architecture in `src/models_net_mamba.py` and low-level blocks in `src/models_mamba.py`
   - Uses epoch loops in `src/engine.py`

If you are continuing the flow described in root `README.md`, start with **`main.py` pipeline first**.

---

## 2) Repository Structure and Folder Purpose

### Root-level

- `README.md`
  - High-level project narrative and intended architecture.
- `main.py`
  - Simplified end-to-end training script for dual-branch embedding model.
- `requirements.txt`
  - Python dependency lock list.
- `assets/`
  - Visual assets (`NetMamba.png`).
- `dataset/`
  - Dataset preprocessing scripts (pcap splitting, conversion, sampling, merge/split).
- `src/`
  - Model/training/evaluation source code.
- `mamba-1p1p1/`
  - Vendored upstream Mamba implementation and tests.

### `/dataset`

- `dataset_common.py`
  - Shared helpers for packet parsing and conversion to fixed-size flow arrays.
- `dataset_all.py`
  - End-to-end preprocessing orchestration:
    - sample pcaps
    - convert pcap -> array images + stats json
    - split train/valid/test
    - merge datasets for pretraining
- `dataset_cic_iot2022.py`, `dataset_cross_platform.py`, `dataset_iscx_*.py`, `dataset_ustc_tfc2016.py`
  - Dataset-specific flow extraction wrappers around external splitter/editcap/mergecap tools.
- `dataset/netmamba/`
  - Expected location for dataset/model artifacts used by main pipeline.

### `/src`

- `models_dual_branch.py`
  - Sequence branch (Mamba fallback to LSTM), stat branch, fusion/projection encoder.
- `dataset_netmamba.py`
  - JSON loader for dual-branch pipeline.
- `train_supcon.py`
  - SupCon loss, episodic sampling, prototype + query scoring.
- `nfstream_extractor.py`
  - Runtime feature extraction helper from pcap/live traffic via NFStream.
- `models_mamba.py`
  - Core Mamba block factory + embedding block utilities.
- `models_net_mamba.py`
  - Full NetMamba model for pretraining/classification modes.
- `engine.py`
  - Generic train/eval epoch loops and metric tracking.
- `pre-train.py`
  - MAE-like pretraining CLI flow.
- `fine-tune.py`
  - Supervised classification fine-tuning CLI flow.
- `eval.py`
  - Evaluation and optional speed benchmark CLI flow.
- `util/`
  - Distributed setup, checkpointing, LR schedules/decay, positional embeddings, transforms.

---

## 3) Entrypoints and Logic Flow

## A) Main pipeline (`main.py`) — recommended starting point

### Entry
Run from repository root:

```bash
python3 main.py
```

### Runtime flow

1. Validates expected dataset path:
   - `dataset/netmamba/ISCXVPN2016/images_sampled_new`
2. Initializes `NetMambaDataset` (`src/dataset_netmamba.py`)
   - loads `**/*.json`
   - converts each sample into:
     - sequence tensor `(128, 2)` = `[length, interval]`
     - stat tensor `(4,)` from simple moments
     - integer label
3. Builds `DualBranchEncoder` (`src/models_dual_branch.py`)
4. Trains with `SupConLoss` (`src/train_supcon.py`)
5. Performs episodic ProtoNet-style evaluation each epoch using:
   - `EpisodicSampler`
   - `compute_prototypes`
   - `prototypical_loss`
6. Saves best checkpoint to:
   - `model/best_model.pth`

### Main-flow core files

- Orchestrator: `main.py`
- Data: `src/dataset_netmamba.py`
- Model: `src/models_dual_branch.py`
- Loss/sampling/eval math: `src/train_supcon.py`
- Inference-support extraction: `src/nfstream_extractor.py`

## B) NetMamba image pipeline (`src/pre-train.py` + `src/fine-tune.py`)

### Pre-train flow

1. `src/pre-train.py` parses args and initializes distributed env.
2. Loads `ImageFolder(data_path/train)` grayscale transforms.
3. Instantiates `models_net_mamba.<model_name>` in pretrain mode.
4. Uses `engine.pretrain_one_epoch` loop.
5. Saves checkpoints/logs under output directory.

### Fine-tune flow

1. `src/fine-tune.py` builds train/valid/test datasets using `ImageFolder`.
2. Loads classifier model from `models_net_mamba`.
3. Optionally loads pretrain checkpoint and interpolates pos embeddings.
4. Trains via `engine.train_one_epoch`.
5. Evaluates with `engine.evaluate`, saves best checkpoint + final test stats.

### Eval-only flow

- `src/eval.py` loads checkpoint and runs:
  - standard evaluation (`engine.evaluate`) or
  - throughput benchmark (`engine.evaluate_speed_test`)

---

## 4) File Responsibility by Flow Stage

### Stage 1: Raw packet preparation
- `dataset/dataset_*.py`, `dataset/dataset_all.py`, `dataset/dataset_common.py`

### Stage 2: Model-ready data formatting
- Main pipeline: `src/dataset_netmamba.py` (JSON -> tensors)
- Image pipeline: generated PNG datasets consumed by `ImageFolder`

### Stage 3: Model definition
- Main pipeline: `src/models_dual_branch.py`
- Image pipeline: `src/models_net_mamba.py`, `src/models_mamba.py`

### Stage 4: Training criteria and loops
- Main pipeline: `src/train_supcon.py`, `main.py`
- Image pipeline: `src/engine.py`, `src/pre-train.py`, `src/fine-tune.py`

### Stage 5: Evaluation and checkpointing
- Main pipeline: episodic eval in `main.py`
- Image pipeline: `src/engine.py::evaluate`, `src/eval.py`
- Shared support: `src/util/misc.py` (save/load/checkpoint/distributed helpers)

### Stage 6: Inference feature extraction support
- `src/nfstream_extractor.py`

---

## 5) Auxiliary/Supporting Files

- `src/util/misc.py`
  - logging meters, distributed setup, save/load, mixed precision helpers.
- `src/util/lr_sched.py`
  - per-iteration LR scheduling used in engine loops.
- `src/util/lr_decay.py`
  - layer-wise LR decay parameter grouping.
- `src/util/pos_embed.py`
  - positional embedding generation/interpolation.
- `src/util/crop.py`, `src/util/lars.py`
  - transformation and optimizer utility code.

---

## 6) Where to Start for New Tasks

1. Decide target pipeline:
   - If task is SupCon + dual-branch embedding: start in `main.py` path.
   - If task is pretrain/finetune image workflow: start in `src/pre-train.py` / `src/fine-tune.py`.
2. Trace entrypoint imports to locate true implementation files.
3. Validate data format assumptions first (JSON vs ImageFolder pipeline).
4. Change model and data contracts together (input shape, label mapping, output head dimensions).

---

## 7) Architecture Change Playbook (What to edit when requirements change)

### A) Change sequence feature design (packet-level)
Edit:
- `src/dataset_netmamba.py` (feature extraction + shape)
- `src/models_dual_branch.py` (`SequenceBranch` input dimensions/projection)
- `main.py` model construction args

### B) Change static/flow-level features
Edit:
- `src/dataset_netmamba.py` stat vector construction
- `src/models_dual_branch.py` (`StatBranch` input_dim)
- `main.py` encoder initialization `stat_input_dim`

### C) Replace fusion strategy or embedding size
Edit:
- `src/models_dual_branch.py` projection/fusion head
- `main.py` where encoder is instantiated
- `src/train_supcon.py` only if loss assumptions on normalized embedding change

### D) Switch training objective (SupCon/ProtoNet changes)
Edit:
- `src/train_supcon.py`
- training/eval sections in `main.py`

### E) Add new datasets / preprocessing pipelines
Edit:
- add dataset script under `dataset/`
- wire orchestration in `dataset/dataset_all.py`
- ensure output format matches consuming pipeline

### F) Change NetMamba encoder depth/width/byte length (image pipeline)
Edit:
- `src/models_net_mamba.py` factory functions and model init params
- CLI args in `src/pre-train.py`, `src/fine-tune.py`, `src/eval.py`

### G) Change distributed/checkpoint behavior
Edit:
- `src/util/misc.py`
- call sites in `src/pre-train.py`, `src/fine-tune.py`, `src/engine.py`

---

## 8) Practical Notes and Known Alignment Points

- The root README describes the intended dual-branch workflow; the repository also includes a broader NetMamba training stack.
- Some scripts assume specific absolute dataset roots (especially under `dataset/`), so path cleanup is typically a first refactor for portability.
- The project currently has multiple training paths; keep one path as your target and avoid mixing assumptions between them.
