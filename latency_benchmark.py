"""
KPI: Inference Latency < 100ms per flow
Benchmarks single-flow and batch inference on the trained DualBranchEncoder.
Usage: python latency_benchmark.py [--model_path ...]
"""

import argparse
import time
import numpy as np
import torch

from src.models_dual_branch import DualBranchEncoder


def benchmark(model, device, batch_size: int, n_runs: int = 500) -> float:
    """Returns mean latency in ms for one forward pass of `batch_size` flows."""
    seq  = torch.randn(batch_size, 30, 3, device=device)
    stat = torch.randn(batch_size, 16,    device=device)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            model(seq, stat)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(seq, stat)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.percentile(times, 99))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="model/best_model.pth")
    parser.add_argument("--n_runs",     type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=256, embed_dim=256)
    ckpt  = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 0) + 1}\n")

    # Single-flow latency (primary KPI measurement)
    mean_1, p99_1 = benchmark(model, device, batch_size=1, n_runs=args.n_runs)
    kpi_ok = mean_1 < 100
    print(f"=== Single-flow latency ({args.n_runs} runs) ===")
    print(f"Mean : {mean_1:.3f} ms   {'✓ <100ms KPI MET' if kpi_ok else '✗ above 100ms KPI'}")
    print(f"p99  : {p99_1:.3f} ms")

    # Batch throughput (informational)
    print(f"\n=== Batch throughput ===")
    for bs in [32, 128, 256]:
        mean_b, _ = benchmark(model, device, batch_size=bs, n_runs=200)
        per_flow  = mean_b / bs
        print(f"Batch {bs:3d}: {mean_b:7.2f} ms total → {per_flow:.3f} ms/flow")

    print(f"\n=== KPI Result ===")
    print(f"Single-flow mean latency: {mean_1:.3f} ms  {'✓ KPI MET' if kpi_ok else '✗ KPI NOT MET'}")


if __name__ == "__main__":
    main()
