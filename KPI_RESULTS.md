# KPI Results — DualBranchEncoder v1

Trained and evaluated on **CESNET-QUIC22** (streaming, size `XS`) on an RTX 4090.
Commit: `d490c3be` (5-tuple/port features removed to prevent shortcut leakage).

## Summary

| KPI | Result | Target | Status |
|---|---|---|---|
| Classification accuracy | **90.90%** | ≥ 90% | ✅ |
| Intra-class cosine sim | **0.7283** | > 0.7 | ✅ |
| Inference latency | **1.36 ms/flow** | < 100 ms | ✅ |
| Zero-day generalization | **84.84% ± 0.024** | ≥ 85% | ❌ (−0.16) |
| Inter-class cosine sim | **0.3833** | < 0.3 | ❌ (+0.083) |

**3 / 5 KPIs met.** During training the best in-loop k-NN accuracy reached **92.70%**.

## Setup

- **Dataset:** CESNET-QUIC22, streamed via `cesnet-datazoo` (size `XS`), 7 semantic classes.
- **Zero-day classes:** music/audio (1) and gaming (2) held out of the training split; present only in val/test.
- **Model:** DualBranchEncoder — BiLSTM sequence branch (final hidden state) + MLP stat branch (16 behavioral features), cross-attention fusion, residual projection head, 256-dim L2-normalized embedding. ~1.98M params.
- **Loss:** `MarginBasedSupConLoss` (λ_pos=0.7, λ_neg=0.3).
- **Features:** behavioral only — packet sizes, inter-packet times, direction, jitter, size histogram, PPI length. **5-tuple identity (ports/IPs) deliberately excluded** as a shortcut/leakage feature.

## Evaluation (CESNET-QUIC22 val split, 5000 samples)

- **Classification:** k-NN (cosine, k=5) and Logistic Regression on an 80/20 stratified split. Best = 90.90% (LR 90.80%).
- **Zero-day:** 50-shot balanced k-NN, 30 trials.
- **Geometric:** mean pairwise cosine similarity, same-class vs different-class.
- **Latency:** single-flow forward pass, RTX 4090.

## Open items (for the next run)

The two misses are both class-separation problems and move together. Planned fix:
- Switch to `HardNegativeMarginLoss` (mines hardest negatives) to push overlapping classes apart.
- Tighten `λ_neg` 0.3 → 0.2.
- Consider 35 → 50 epochs and/or `size S` for more data.

## Reproduce

```bash
python main.py --streaming --streaming_size XS
python eval_cesnet.py --model_path model/best_model.pth --n_samples 5000
```
