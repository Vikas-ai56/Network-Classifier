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

## 5. Live demo on real captured traffic 🎥

`live_demo.py` classifies **real** network flows captured off your own machine — proving
the model generalizes beyond the CESNET-QUIC22 training set. It reuses the exact training
feature pipeline (`src/feature_engineering.py` → 30×3 sequence + 16 behavioral stats), so
the tensors always match the trained model. It is **5-tuple-pure**: NFStream runs with
`n_dissections=0`, and IPs/ports are printed for display only — never fed to the model.
Each flow is classified by its nearest class **prototype** (`model/prototypes.pth`) using
cosine similarity, with no retraining at demo time.

**Prerequisites:** `model/best_model.pth`, `model/prototypes.pth`, and `pip install nfstream`.

### Reproduce the results (two-step: capture then classify — most reliable)

```bash
# 1. Capture real QUIC traffic (needs sudo). Ctrl-C to stop after ~20s.
#    While it runs, open YouTube / Spotify / a few websites to generate traffic.
sudo tcpdump -i en0 -w demo.pcap 'udp port 443'

# 2. Classify the captured flows
python3 live_demo.py --pcap demo.pcap
```

### Or classify live, straight off the interface

```bash
sudo python3 live_demo.py --interface en0 --max-flows 20
```

| Flag | Default | Meaning |
|---|---|---|
| `--pcap` | — | Capture file to classify (mutually exclusive with `--interface`) |
| `--interface` | — | Live capture interface, e.g. `en0` (needs sudo) |
| `--model_path` | `model/best_model.pth` | Encoder checkpoint |
| `--prototypes` | `model/prototypes.pth` | Class-prototype gallery |
| `--min-packets` | 10 | Skip flows with fewer bidirectional packets |
| `--max-flows` | 50 | Stop after this many flows (useful for live capture) |
| `--all-protocols` | off | Classify every flow, not just QUIC (UDP/443) |

**Representative real-data output** (captured from a browsing/streaming session — note the
high-packet streaming flows classified with high confidence):

```
Device: cpu | encoder epoch 28
Classifier gallery (6 classes): video_streaming, gaming, social_media, file_transfer, browsing, communication
Source: demo.pcap   (QUIC-only; --all-protocols to widen)

  #  src->dst                pkts  proto/port  PREDICTION         conf
----------------------------------------------------------------------
  1  2401:4900:c947:35ec:89   108    QUIC/443  video_streaming   70.1%
 17  2401:4900:c947:35ec:89  8765    QUIC/443  video_streaming   68.8%
 19  2401:4900:c947:35ec:89   924    QUIC/443  video_streaming   67.5%
  ...
Classified 35 flow(s). Predictions are nearest-prototype (cosine) over the 6-class gallery.
```

> **Honest reading of the output:** the large flows (hundreds–thousands of packets) are the
> real streaming sessions and classify confidently as `video_streaming`. Short flows
> (10–30 packets) carry little behavioral signal yet, so their predictions are lower
> confidence and noisier — this is expected and reported transparently rather than hidden.
> The confidence column is a softmax over cosine similarities; treat the *distribution*
> across flows as the evidence, not any single row as a certainty.

<!-- SCREENSHOT: live_demo.py output table on a fresh real-traffic capture -->

## 6. Save your model before destroying a cloud instance ⚠️

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
