"""
Karpathy Recipe — Step 1 + Step 2 on ISCXVPN2016 dataset (JSON files).
SEQ_LEN=30 (as set in feature_engineering.py).

Run: python3 scripts/karpathy_checks_iscxvpn.py
"""

import json
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from pathlib import Path

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, "/home/vikas/Netwok-Classifier")

DATA_ROOT = Path("/home/vikas/Netwok-Classifier/dataset/netmamba/ISCXVPN2016/images_sampled_new")
ALL_CLASSES = ["voip", "browsing", "chat", "ftp", "p2p", "streaming", "email"]


def compute_intra_inter(emb_norm, labels):
    sim = emb_norm @ emb_norm.T
    n = len(labels)
    same = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i != j and labels[i] == labels[j]:
                same[i, j] = True
    diff = ~same & ~torch.eye(n, dtype=torch.bool)
    intra = sim[same].mean().item() if same.any() else float("nan")
    inter = sim[diff].mean().item() if diff.any() else float("nan")
    return intra, inter


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Become One with the Data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("STEP 1: DATA INSPECTION (ISCXVPN2016 JSON files)")
print("=" * 65)

# 1a. File counts per class
print("\n── 1a. File counts per class ──")
all_files = {}
for cls in ALL_CLASSES:
    files = list((DATA_ROOT / cls).glob("*.json"))
    all_files[cls] = files
    print(f"  {cls:<12}: {len(files):>5} files")
total_files = sum(len(v) for v in all_files.values())
print(f"  {'TOTAL':<12}: {total_files:>5} files")

# 1b. LABEL_MAP coverage
print("\n── 1b. LABEL_MAP coverage ──")
from src.dataset_unified import LABEL_MAP, UNIFIED_CLASS_NAMES
in_map, out_map = [], []
for cls in ALL_CLASSES:
    if cls in LABEL_MAP:
        in_map.append(cls)
        print(f"  {cls:<12} → class {LABEL_MAP[cls]} ({UNIFIED_CLASS_NAMES[LABEL_MAP[cls]]})")
    else:
        out_map.append(cls)
        print(f"  {cls:<12} → ❌ NOT IN LABEL_MAP (will be rejected)")

in_map_files  = sum(len(all_files[c]) for c in in_map)
out_map_files = sum(len(all_files[c]) for c in out_map)
print(f"\n  Loadable  : {in_map_files:>5} files ({100*in_map_files/total_files:.1f}%)")
print(f"  Rejected  : {out_map_files:>5} files ({100*out_map_files/total_files:.1f}%)")

# 1c. Inspect raw JSON structure from each class
print("\n── 1c. Raw JSON structure — 1 sample per class ──")
print(f"  {'class':<12}  {'n_lengths':>9}  {'n_intervals':>11}  "
      f"{'nonzero_len':>11}  {'nonzero_ipt':>11}  {'max_size':>8}  {'max_ipt':>8}")
for cls in ALL_CLASSES:
    files = all_files[cls]
    if not files:
        continue
    with open(files[0]) as f:
        d = json.load(f)
    lengths   = d.get("lengths", [])
    intervals = d.get("intervals", [])
    nz_len = sum(1 for x in lengths if x != 0)
    nz_ipt = sum(1 for x in intervals if x != 0)
    max_sz = max(lengths) if lengths else 0
    max_ipt = max(intervals) if intervals else 0
    print(f"  {cls:<12}  {len(lengths):>9}  {len(intervals):>11}  "
          f"{nz_len:>11}  {nz_ipt:>11}  {max_sz:>8.0f}  {max_ipt:>8.0f}")

# 1d. Distribution of non-zero packets across all loadable files
print("\n── 1d. Non-zero packet count distribution (loadable classes) ──")
nz_counts = []
all_sizes, all_ipts = [], []
for cls in in_map:
    for fpath in all_files[cls]:
        with open(fpath) as f:
            d = json.load(f)
        lengths   = d.get("lengths", [])
        intervals = d.get("intervals", [])
        nz = sum(1 for x in lengths if x != 0)
        nz_counts.append(nz)
        all_sizes.extend([x for x in lengths if x > 0])
        all_ipts.extend([x for x in intervals if x > 0])

nz_arr = np.array(nz_counts)
print(f"  Non-zero packets per flow: min={nz_arr.min()}  mean={nz_arr.mean():.2f}  "
      f"median={np.median(nz_arr):.0f}  max={nz_arr.max()}")
for k in [1, 2, 3, 4, 5]:
    pct = 100 * (nz_arr == k).sum() / len(nz_arr)
    bar = "█" * int(pct / 2)
    print(f"  {k} non-zero pkts: {(nz_arr==k).sum():>5} flows ({pct:5.1f}%)  {bar}")

# 1e. Packet size and IPT distributions
print("\n── 1e. Feature value distributions (loadable classes) ──")
if all_sizes:
    s = np.array(all_sizes)
    print(f"  Packet sizes (B) : min={s.min():.0f}  median={np.median(s):.0f}  "
          f"max={s.max():.0f}  >1500B: {(s>1500).sum()}")
if all_ipts:
    t = np.array(all_ipts)
    print(f"  IPT values       : min={t.min():.0f}  median={np.median(t):.0f}  "
          f"max={t.max():.0f}  >5000ms: {(t>5000).sum()}")

# 1f. 5 raw samples printed
print("\n── 1f. 5 sample flows (raw JSON values) ──")
sample_files = all_files["voip"][:3] + all_files["browsing"][:2]
for i, fpath in enumerate(sample_files):
    with open(fpath) as f:
        d = json.load(f)
    lengths   = d.get("lengths", [])
    intervals = d.get("intervals", [])
    cls = fpath.parent.name
    nz = sum(1 for x in lengths if x != 0)
    print(f"  [{i}] {cls:<10} | file: {fpath.name[:40]}")
    print(f"       lengths  : {lengths}")
    print(f"       intervals: {intervals}")
    print(f"       non-zero packets: {nz}/5")

# 1g. Validator rejection audit
print("\n── 1g. Validator rejection audit (all loadable files) ──")
from src.data_validator import FlowValidator
validator = FlowValidator()

valid_count, reject_counts = 0, {}
for cls in in_map:
    for fpath in all_files[cls]:
        with open(fpath) as f:
            d = json.load(f)
        lengths   = d.get("lengths", [])
        intervals = d.get("intervals", [])
        ok, reason = validator.validate_iscxvpn_sequence(lengths, intervals, source_path=str(fpath))
        if ok:
            valid_count += 1
        else:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

total_loadable = in_map_files
rejected_total = sum(reject_counts.values())
print(f"  Loadable files  : {total_loadable}")
print(f"  Valid           : {valid_count} ({100*valid_count/total_loadable:.1f}%)")
print(f"  Rejected        : {rejected_total} ({100*rejected_total/total_loadable:.1f}%)")
for reason, cnt in sorted(reject_counts.items(), key=lambda x: -x[1]):
    print(f"    {reason}: {cnt}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Skeleton + Baselines
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2: SKELETON + BASELINE CHECKS  (SEQ_LEN=30)")
print("=" * 65)

from src.feature_engineering import (
    extract_seq_from_iscxvpn, extract_stat_from_iscxvpn,
    SEQ_LEN, STAT_INPUT_DIM,
)
print(f"\nSEQ_LEN in use: {SEQ_LEN}")
assert SEQ_LEN == 30

# Build 64-sample batch from loadable classes
print("\nBuilding 64-sample batch...")
samples = []
for cls in in_map:
    for fpath in all_files[cls]:
        with open(fpath) as f:
            d = json.load(f)
        lengths   = d.get("lengths", [])
        intervals = d.get("intervals", [])
        ok, _ = validator.validate_iscxvpn_sequence(lengths, intervals, source_path=str(fpath))
        if not ok:
            continue
        try:
            seq  = extract_seq_from_iscxvpn(lengths, intervals, SEQ_LEN)
            stat = extract_stat_from_iscxvpn(lengths, intervals)
        except ValueError:
            continue
        if not (np.all(np.isfinite(seq)) and np.all(np.isfinite(stat))):
            continue
        label = LABEL_MAP[cls]
        samples.append((seq, stat, label))
        if len(samples) >= 64:
            break
    if len(samples) >= 64:
        break

print(f"  Valid samples built: {len(samples)}")
if len(samples) < 8:
    print("  FATAL: not enough valid samples")
    sys.exit(1)

seqs   = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32)
stats_ = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32)
labels = torch.tensor([s[2] for s in samples], dtype=torch.long)
print(f"  Label distribution in batch: {dict(zip(*np.unique(labels.numpy(), return_counts=True)))}")

# 2a. Tensor sanity check
print("\n── 2a. Tensor sanity check ──")
print(f"  seq  shape : {list(seqs.shape)}   (batch × SEQ_LEN × 3)")
print(f"  stat shape : {list(stats_.shape)}  (batch × STAT_INPUT_DIM)")
print(f"  NaN in seq : {torch.isnan(seqs).any().item()}")
print(f"  NaN in stat: {torch.isnan(stats_).any().item()}")
print(f"  seq  range : [{seqs.min():.3f}, {seqs.max():.3f}]  (expect [-1,1])")
print(f"  stat range : [{stats_.min():.3f}, {stats_.max():.3f}]  (expect [0,1])")

zero_rows = (seqs.abs().sum(dim=-1) == 0).float().mean().item()
print(f"  Zero-padded seq rows : {100*zero_rows:.1f}%")

print(f"\n  seq[0] all 30 timesteps  (size_norm | ipt_norm | dir=0 sentinel):")
print(f"    {'t':>3}  {'size_norm':>10}  {'ipt_norm':>9}  {'dir':>5}")
for t in range(SEQ_LEN):
    sz, ipt, dr = seqs[0, t].tolist()
    print(f"    {t:>3}  {sz:>10.4f}  {ipt:>9.4f}  {dr:>5.1f}")

print(f"\n  stat[0] ({STAT_INPUT_DIM} features):\n  {stats_[0].numpy()}")

# 2b. Loss at initialisation
print("\n── 2b. Loss at initialisation (random weights) ──")
from src.models_dual_branch import DualBranchEncoder
from src.train_supcon import MarginBasedSupConLoss

torch.manual_seed(42)
model = DualBranchEncoder()
loss_fn = MarginBasedSupConLoss()
model.eval()

with torch.no_grad():
    emb = model(seqs, stats_)
    emb_norm = torch.nn.functional.normalize(emb, dim=-1)
    intra_init, inter_init = compute_intra_inter(emb_norm, labels)
    loss_init = loss_fn(emb, labels).item()

print(f"  Embedding dim  : {emb.shape[1]}")
print(f"  Init intra-sim : {intra_init:.4f}  (expected ~0.0)")
print(f"  Init inter-sim : {inter_init:.4f}  (expected ~0.0)")
print(f"  Init loss      : {loss_init:.4f}  (expected ≈ 0.7)")
if abs(loss_init - 0.7) < 0.15:
    print("  ✅ PASS: init loss in expected range")
else:
    print("  ❌ FLAG: init loss outside expected range")
if intra_init > 0.95:
    print("  ⚠️  WARNING: intra-sim collapsed at init (BatchNorm + zero-padding)")
else:
    print("  ✅ PASS: init similarities look reasonable")

# 2c. Zero-input baseline
print("\n── 2c. Zero-input baseline ──")
with torch.no_grad():
    emb_zero = model(torch.zeros_like(seqs), torch.zeros_like(stats_))
    loss_zero = loss_fn(emb_zero, labels).item()
print(f"  Real input loss : {loss_init:.4f}")
print(f"  Zero input loss : {loss_zero:.4f}")
if loss_zero > loss_init:
    print("  ✅ PASS: model responds to real input signal")
else:
    print("  ❌ FAIL: zero input not worse — check normalization")

# 2d. Single-batch overfit
print("\n── 2d. Single-batch overfit (32 samples, 150 steps, lr=3e-4) ──")
print(f"  {'step':>5}  {'loss':>8}  {'intra-sim':>10}  {'inter-sim':>10}")

torch.manual_seed(42)
model = DualBranchEncoder()
model.train()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

batch_seq    = seqs[:32]
batch_stat   = stats_[:32]
batch_labels = labels[:32]

final_loss = None
for step in range(150):
    opt.zero_grad()
    emb = model(batch_seq, batch_stat)
    loss = loss_fn(emb, batch_labels)
    loss.backward()
    opt.step()

    if step % 25 == 0 or step == 149:
        with torch.no_grad():
            e_n = torch.nn.functional.normalize(emb.detach(), dim=-1)
            intra, inter = compute_intra_inter(e_n, batch_labels)
        print(f"  {step:>5}  {loss.item():>8.4f}  {intra:>10.4f}  {inter:>10.4f}")
        final_loss = loss.item()

print()
if final_loss < 0.05:
    print("  ✅ PASS: clean overfit (loss < 0.05)")
elif final_loss < 0.15:
    print("  ✅ PASS: overfit achieved (loss < 0.15)")
else:
    print(f"  ❌ FAIL/SLOW: loss={final_loss:.4f} after 150 steps")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  SEQ_LEN              : {SEQ_LEN}")
print(f"  Total files          : {total_files}")
print(f"  Loadable (in LABEL_MAP): {in_map_files} ({100*in_map_files/total_files:.1f}%)")
print(f"  Rejected (label)     : {out_map_files} ({100*out_map_files/total_files:.1f}%)")
print(f"  Zero-padding         : {100*zero_rows:.1f}%")
print(f"  Init loss            : {loss_init:.4f}")
print(f"  Init intra/inter-sim : {intra_init:.4f} / {inter_init:.4f}")
print(f"  Overfit final loss   : {final_loss:.4f}")
print()
