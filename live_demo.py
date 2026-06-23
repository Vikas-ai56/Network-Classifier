"""
live_demo.py — Real-data inference demo for the DualBranchEncoder.

Reads real network traffic (a .pcap/.pcapng file OR a live interface) with
NFStream, turns each flow into the EXACT feature tensors the model was trained
on (via src/feature_engineering.py — single source of truth, so shapes can
never drift), encodes it, and classifies it by nearest class prototype
(model/prototypes.pth) using cosine similarity.

This matches the training distribution best on QUIC traffic (the model was
trained on CESNET-QUIC22), so by default only QUIC flows (UDP, port 443) are
classified. Pass --all-protocols to classify every flow.

Examples
--------
  # 1. Capture some QUIC traffic to a file, then classify it:
  #    (open YouTube / Spotify / a website while this runs ~20s)
  sudo tcpdump -i en0 -w demo.pcap 'udp port 443'      # Ctrl-C to stop
  python live_demo.py --pcap demo.pcap

  # 2. Classify directly off a live interface (needs sudo on macOS/Linux):
  sudo python live_demo.py --interface en0 --max-flows 20

Usage:
  python live_demo.py --pcap <file>            [--all-protocols] [--min-packets 10]
  python live_demo.py --interface en0          [--max-flows 20]
"""

import argparse
import logging
import sys

import numpy as np
import torch

logging.getLogger("src.feature_engineering").setLevel(logging.ERROR)
logging.getLogger("src.data_validator").setLevel(logging.ERROR)

from src.feature_engineering import (
    SEQ_LEN,
    PHIST_BIN_EDGES,
    extract_seq_features,
    extract_stat_features,
)
from src.models_dual_branch import DualBranchEncoder
from src.dataset_unified import UNIFIED_CLASS_NAMES

try:
    from nfstream import NFStreamer
except ImportError:
    sys.exit("nfstream not installed. Run: pip install nfstream")


def _phist_src_counts(sizes):
    """8-bin raw histogram of source-direction packet sizes (CESNET PHIST layout)."""
    counts = np.zeros(8, dtype=np.float32)
    for s in sizes:
        for i in range(8):
            if PHIST_BIN_EDGES[i] <= s < PHIST_BIN_EDGES[i + 1]:
                counts[i] += 1.0
                break
    return counts


def flow_to_tensors(flow):
    """
    Convert one NFStream flow into (seq_data, stat_data) matching the training
    pipeline exactly, or return None if the flow is too short / malformed.

    NFStream SPLT fields (length = splt_analysis, padded with -1):
      splt_direction : 0 = src->dst (client->server), 1 = dst->src
      splt_ps        : per-packet size (bytes)
      splt_piat_ms   : per-packet inter-arrival time (ms)
    """
    dirs_raw = list(getattr(flow, "splt_direction", []) or [])
    ps_raw = list(getattr(flow, "splt_ps", []) or [])
    iat_raw = list(getattr(flow, "splt_piat_ms", []) or [])

    # Keep only real (non-padding) packets; nfstream pads direction with -1.
    sizes, ipts, dirs = [], [], []
    for d, p, t in zip(dirs_raw, ps_raw, iat_raw):
        if d not in (0, 1):
            continue
        sizes.append(float(p))
        ipts.append(float(t))
        dirs.append(1.0 if d == 0 else -1.0)   # client->server = +1, server->client = -1

    if len(sizes) < 3:
        return None

    # CESNET PPI layout: [inter_packet_times, directions, sizes]
    ppi = [ipts, dirs, sizes]

    # Source-direction (client->server) sizes feed the PHIST, like CESNET PHIST_SRC_SIZES.
    src_sizes = [s for s, d in zip(sizes, dirs) if d > 0]

    row = {
        "BYTES":           float(getattr(flow, "src2dst_bytes", 0)),
        "BYTES_REV":       float(getattr(flow, "dst2src_bytes", 0)),
        "PACKETS":         float(getattr(flow, "src2dst_packets", 0)),
        "PACKETS_REV":     float(getattr(flow, "dst2src_packets", 0)),
        "DURATION":        float(getattr(flow, "bidirectional_duration_ms", 0)),
        "PPI":             ppi,
        "PPI_LEN":         len(sizes),
        "PHIST_SRC_SIZES": _phist_src_counts(src_sizes),
    }

    try:
        seq_data = extract_seq_features(ppi, SEQ_LEN)
        stat_data = extract_stat_features(row)
    except ValueError:
        return None

    if not (np.all(np.isfinite(seq_data)) and np.all(np.isfinite(stat_data))):
        return None
    return seq_data, stat_data


def main():
    ap = argparse.ArgumentParser(description="Live real-data classification demo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pcap", help="path to a .pcap / .pcapng capture file")
    src.add_argument("--interface", help="live capture interface, e.g. en0 (needs sudo)")
    ap.add_argument("--model_path", default="model/best_model.pth")
    ap.add_argument("--prototypes", default="model/prototypes.pth")
    ap.add_argument("--min-packets", type=int, default=10,
                    help="skip flows with fewer bidirectional packets")
    ap.add_argument("--max-flows", type=int, default=50,
                    help="stop after classifying this many flows (useful for live)")
    ap.add_argument("--all-protocols", action="store_true",
                    help="classify every flow, not just QUIC (UDP/443)")
    ap.add_argument("--temp", type=float, default=0.07,
                    help="softmax temperature for the confidence score")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- model + prototype gallery ----
    model = DualBranchEncoder(seq_input_dim=3, stat_input_dim=16, d_model=256, embed_dim=256)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    key = "encoder_state_dict" if "encoder_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.to(device).eval()

    protos = torch.load(args.prototypes, map_location=device, weights_only=False)
    proto_ids = sorted(protos.keys())
    proto_mat = torch.stack([protos[c] for c in proto_ids]).to(device)
    proto_mat = torch.nn.functional.normalize(proto_mat, p=2, dim=1)   # (C, 256)
    proto_names = [UNIFIED_CLASS_NAMES.get(c, f"class_{c}") for c in proto_ids]

    print(f"Device: {device} | encoder epoch {ckpt.get('epoch', 0) + 1}")
    print(f"Classifier gallery ({len(proto_ids)} classes): {', '.join(proto_names)}")

    source = args.pcap or args.interface
    print(f"Source: {source}"
          + ("" if args.all_protocols else "   (QUIC-only; --all-protocols to widen)") + "\n")

    streamer = NFStreamer(
        source=source,
        splt_analysis=SEQ_LEN,        # capture first 30 packets per flow
        n_dissections=0,              # no DPI: behavior only, 5-tuple-purge friendly
        statistical_analysis=True,
        accounting_mode=0,
    )

    header = f"{'#':>3}  {'src->dst':<22} {'pkts':>5} {'proto/port':>11}  {'PREDICTION':<16} {'conf':>6}"
    print(header)
    print("-" * len(header))

    shown = 0
    for flow in streamer:
        if flow.bidirectional_packets < args.min_packets:
            continue
        is_quic = (flow.protocol == 17 and (flow.dst_port == 443 or flow.src_port == 443))
        if not args.all_protocols and not is_quic:
            continue

        feats = flow_to_tensors(flow)
        if feats is None:
            continue
        seq_data, stat_data = feats

        with torch.no_grad():
            emb = model(
                torch.from_numpy(seq_data).unsqueeze(0).to(device),
                torch.from_numpy(stat_data).unsqueeze(0).to(device),
            )                                            # (1, 256), L2-normalized
            sims = (emb @ proto_mat.T).squeeze(0)        # cosine similarity to each prototype
            probs = torch.softmax(sims / args.temp, dim=0)
            best = int(torch.argmax(sims))

        label = proto_names[best]
        conf = float(probs[best])
        endpoint = f"{flow.src_ip}->{flow.dst_ip}"
        proto = "QUIC" if is_quic else ("UDP" if flow.protocol == 17 else "TCP")
        print(f"{shown + 1:>3}  {endpoint:<22.22} {flow.bidirectional_packets:>5} "
              f"{proto + '/' + str(flow.dst_port):>11}  {label:<16} {conf:>6.1%}")

        shown += 1
        if shown >= args.max_flows:
            break

    if shown == 0:
        print("No matching flows found. Try a longer capture, lower --min-packets, "
              "or --all-protocols.")
    else:
        print(f"\nClassified {shown} flow(s). Predictions are nearest-prototype "
              f"(cosine) over the {len(proto_ids)}-class gallery.")


if __name__ == "__main__":
    main()
