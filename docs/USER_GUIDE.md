# User Guide

This guide covers training, evaluation, and benchmarking. Make sure you've completed
[INSTALLATION.md](INSTALLATION.md) and (on vast.ai) activated the venv with
`source /venv/main/bin/activate`.

## 1. Train the model

```bash
python main.py --streaming --streaming_size XS
```

| Flag | Default | Meaning |
|---|---|---|
| `--streaming` | off | Stream CESNET-QUIC22 (recommended) instead of loading local files |
| `--streaming_size` | `XS` | `XS` (~10M flows) · `S` (~25M) · `M` (~50M) |
| `--epochs` | 35 | Training epochs |
| `--batch_size` | 256 | Batch size |
| `--lr` | 1e-3 | Learning rate (cosine decay) |
| `--warmup_epochs` | 0 | Linear warmup epochs |

**Outputs** (written to `model/`):
- `best_model.pth` — best validation accuracy checkpoint
- `checkpoint_latest.pth` — most recent epoch (used for auto-resume)

**Run it so it survives disconnects** (cloud GPU):
```bash
tmux new -s train
source /venv/main/bin/activate
python main.py --streaming --streaming_size XS
# detach: Ctrl-b then d        reattach: tmux attach -t train
```

Training prints per-epoch lines like:
```
--- Epoch 35 | Loss: 0.0437 | ProtoNet: 0.9270 ✓ | Intra: 0.7283 | Inter: 0.3833 ✗ ---
=== Training complete ===
Best ProtoNet accuracy : 0.9270  (✓ ≥90% KPI MET)
```

## 2. Evaluate all KPIs

```bash
python eval_cesnet.py --model_path model/best_model.pth --n_samples 5000
```

| Flag | Default | Meaning |
|---|---|---|
| `--model_path` | `model/best_model.pth` | Checkpoint to evaluate |
| `--n_samples` | 5000 | Validation samples to stream |
| `--k_shot` | 50 | Shots per class for the zero-day gallery |
| `--n_trials` | 30 | Zero-day random trials to average |

Prints a `FINAL KPI SUMMARY` with pass/fail for classification, zero-day, intra/inter
similarity, and latency.

## 3. Individual KPI scripts

```bash
# Classification (k-NN + SVM + Logistic Regression)
python classify_knn_svm.py --model_path model/best_model.pth

# Zero-day generalization (balanced k-shot k-NN, averaged over trials)
python zero_day_test.py --model_path model/best_model.pth --k_shot 50 --n_trials 50

# Inference latency (single-flow + batch throughput)
python latency_benchmark.py --model_path model/best_model.pth --n_runs 500
```

## 4. (Optional) Supervised fine-tune head

```bash
python finetune_classifier.py --model_path model/best_model.pth
```
Adds a classification head on top of the encoder. Use only if you want a softmax
classifier in addition to the metric-learning embeddings.

## 5. Save your model before destroying a cloud instance ⚠️

The trained model lives **only on the instance** until you copy it off.

```bash
# make a named backup on the instance
cp model/best_model.pth model/best_model_v1.pth
```
Then either:
- **Jupyter:** file browser → `model/` → right-click `best_model.pth` → **Download**, or
- **scp from your local machine:**
  ```bash
  scp -i ~/.ssh/vastai_key -P <PORT> \
    root@<IP>:/workspace/Samsung-ennovatex/model/best_model.pth ./best_model.pth
  ```
Verify locally (`ls -lh best_model.pth`) **before** you destroy the instance.

## Typical end-to-end session

```bash
tmux new -s train
source /venv/main/bin/activate
cd /workspace/Samsung-ennovatex
python main.py --streaming --streaming_size XS        # train  (Ctrl-b d to detach)
python eval_cesnet.py --model_path model/best_model.pth --n_samples 5000   # evaluate
cp model/best_model.pth model/best_model_v1.pth        # back up, then download
```

<!-- SCREENSHOT: full eval_cesnet.py FINAL KPI SUMMARY -->
<!-- SCREENSHOT: latency_benchmark.py single-flow latency result -->
