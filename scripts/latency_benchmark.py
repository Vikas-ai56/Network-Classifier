"""
KPI: Real-time inference latency < 100ms per flow.

Usage:
  python scripts/latency_benchmark.py
  python scripts/latency_benchmark.py --checkpoint model/best_model.pth
"""

import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models_dual_branch import DualBranchEncoder

WARMUP = 100
RUNS   = 1000


def load_model(path, device):
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=18, d_model=256, embed_dim=256)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    return model.to(device).eval()


def bench(model, device, batch_size=1):
    seq  = torch.randn(batch_size, 30, 3,  device=device)
    stat = torch.randn(batch_size, 18,     device=device)
    with torch.no_grad():
        for _ in range(WARMUP):
            model(seq, stat)
        if device.type == "cuda":
            torch.cuda.synchronize()
        lats = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(seq, stat)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    return np.array(lats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model/best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    print(f"\n[ Single-flow latency ]  ({WARMUP} warmup + {RUNS} timed runs)")
    lats = bench(model, device, batch_size=1)

    mean, med = lats.mean(), np.median(lats)
    p95, p99  = np.percentile(lats, 95), np.percentile(lats, 99)
    passed    = p99 < 100.0

    print(f"  Mean      : {mean:.3f} ms")
    print(f"  Median    : {med:.3f} ms")
    print(f"  p95       : {p95:.3f} ms")
    print(f"  p99       : {p99:.3f} ms")
    print(f"  Throughput: {1000/mean:,.0f} flows/sec")

    print(f"\n[ Batch throughput ]")
    print(f"  {'Batch':>6}  {'flows/sec':>12}  {'ms/flow':>10}")
    print(f"  {'-'*34}")
    for bs in [1, 8, 32, 128, 256]:
        l = bench(model, device, batch_size=bs)
        fps = bs * 1000 / l.mean()
        print(f"  {bs:>6}  {fps:>12,.0f}  {1000/fps*bs:>10.3f} ms")

    print(f"\n{'='*45}")
    print(f"  KPI: < 100 ms per flow (p99)")
    print(f"  p99 = {p99:.3f} ms  →  {'✓  KPI MET' if passed else '✗  KPI FAILED'}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()
