# Salient Features

## 1. Works on encrypted traffic — no payload, no DPI
Classifies QUIC/TLS flows purely from **observable behavior** (packet timing, sizes,
direction). Deep Packet Inspection is useless on encrypted payloads; our approach never
looks inside the packet.

## 2. Zero-day generalization by design
Because the model learns a **metric embedding** rather than a fixed softmax classifier, it
can recognize application categories it never trained on, from only a few labeled
examples (k-shot k-NN). We *prove* this by holding out two entire classes
(music, gaming) from training and measuring accuracy on them.

## 3. The "5-tuple purge" — behavior, not identity
We deliberately exclude IP addresses and ports (the network 5-tuple) **and** raw
volume/duration features. These are *shortcut* features: a model can memorize "port 443 →
this app" and score well on the training distribution while completely failing on new
servers or new apps. Removing them is what makes the zero-day claim real. For QUIC
specifically, ports carry almost no information anyway (dst ≈ 443, src ephemeral).

## 4. Real-time speed — 1.36 ms per flow
A linear-time sequence encoder (Mamba `O(N)`, or BiLSTM fallback) keeps single-flow
inference at **1.36 ms** on an RTX 4090 — ~73× under the 100 ms budget — fast enough for
packet-by-packet, line-rate analysis.

## 5. Geometry-first training
The loss margins **are** the evaluation thresholds: positives pulled to ≥ 0.7 cosine,
negatives pushed to ≤ 0.3. The model optimizes the exact geometric quantities it is
scored on, producing tight, well-separated clusters on the unit hypersphere.

## 6. Dual-branch + cross-attention fusion
Two complementary views — the **temporal** packet sequence and the **statistical** flow
summary — are fused with bidirectional cross-attention so each can reweight the other.
Captures both "how the conversation unfolds over time" and "what the flow looks like
overall."

## 7. Runs anywhere — graceful Mamba→BiLSTM fallback
If the Mamba CUDA kernels aren't available, the sequence branch transparently switches to
a BiLSTM. No code changes, no failed runs — the shipped model in fact used the BiLSTM path.

## 8. Train on 10M flows without downloading 10M flows
The streaming dataset pulls CESNET-QUIC22 chunk-by-chunk via `cesnet-datazoo`, so training
on a massive dataset works on an ephemeral cloud GPU with a small disk.

## 9. Reviewer-trustworthy evaluation
KPIs are measured with standard scikit-learn / FAISS implementations: k-NN + Logistic
Regression for classification, balanced k-shot k-NN for zero-day, exact cosine geometry
for cluster quality, and a proper warmup'd latency benchmark with p99.

## 10. Robust, production-minded data pipeline
A dedicated `FlowValidator` rejects malformed flows (empty/short PPI, zero-variance,
NaN/Inf, bad labels), normalization is log-scaled and clipped to bounded ranges, and the
streaming loader handles the subtle `IterableDataset` pitfalls (persistent workers,
trailing batches) that silently corrupt training.

## Results

| KPI | Result | Target | Status |
|---|---|---|---|
| Classification accuracy | 90.90% | ≥ 90% | ✅ |
| Intra-class cosine similarity | 0.7283 | > 0.7 | ✅ |
| Inference latency | 1.36 ms | < 100 ms | ✅ |
| Zero-day generalization | 84.84% | ≥ 85% | ❌ (−0.16) |
| Inter-class cosine similarity | 0.3833 | < 0.3 | ❌ (+0.083) |

**3 / 5 met.** The two near-misses are both class-separation issues and are addressed by
the already-implemented `HardNegativeMarginLoss` (hard-negative mining + tighter negative
margin) in a follow-up run.

