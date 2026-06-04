"""
Karpathy Recipe — Step 1 + Step 2 on CESNET sample CSV.
SEQ_LEN=30 (matches CESNET's max PPI_LEN).

Run: python3 scripts/karpathy_checks_cesnet.py
"""

import ast
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# Fix global seed before any model or data work
torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, "/home/vikas/Netwok-Classifier")

CSV = "/home/vikas/Netwok-Classifier/cesnet_sample.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_ppi_string(raw):
    """Parse numpy-printed array strings like '[[ 0.  1.]\n [ 2.  3.]]'."""
    if not isinstance(raw, str):
        return np.array(raw)
    rows = re.findall(r'\[([^\[\]]+)\]', raw)
    parsed = []
    for r in rows:
        nums = list(map(float, re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', r)))
        if nums:
            parsed.append(nums)
    return np.array(parsed)


def compute_intra_inter(emb_norm, labels):
    """Return mean cosine similarity for same-label and different-label pairs."""
    sim = emb_norm @ emb_norm.T
    n = len(labels)
    same = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(n):
            if i != j and labels[i] == labels[j]:
                same[i, j] = True
    diff = ~same & ~torch.eye(n, dtype=torch.bool)
    intra = sim[same].mean().item() if same.any() else float('nan')
    inter = sim[diff].mean().item() if diff.any() else float('nan')
    return intra, inter


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Become One with the Data
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("STEP 1: DATA INSPECTION (cesnet_sample.csv, every-500th-row sample)")
print("=" * 65)

df = pd.read_csv(CSV)
print(f"\nLoaded {len(df):,} rows × {len(df.columns)} columns")

# 1a. Label distribution
print("\n── 1a. APP class distribution ──")
vc = df["APP"].value_counts()
print(f"  Unique classes     : {df['APP'].nunique()}")
print(f"  Total samples      : {len(df):,}")
print(f"  Most common class  : APP={vc.idxmax()}  ({vc.max()} samples)")
print(f"  Rarest class       : APP={vc.idxmin()}  ({vc.min()} samples)")
print(f"  Imbalance ratio    : {vc.max()/vc.min():.0f}x")
print(f"\n  Top 10 classes:\n{vc.head(10).to_string()}")
print(f"\n  NOTE: APP is cesnet-datazoo numeric ID. Streaming API returns text")
print(f"        CATEGORY which maps into LABEL_MAP. CSV inspection only.")

# 1b. Duplicates
print("\n── 1b. Duplicate / corrupt rows ──")
dup_id    = df.duplicated(subset=["ID"]).sum()
dup_tuple = df.duplicated(subset=["SRC_IP","DST_IP","SRC_PORT","DST_PORT"]).sum()
print(f"  Duplicate flow IDs      : {dup_id}")
print(f"  Duplicate 5-tuples      : {dup_tuple}")

# 1c. PPI quality — the most important field
print("\n── 1c. PPI / sequence quality ──")
desc = df["PPI_LEN"].describe()
print(f"  PPI_LEN  mean={desc['mean']:.1f}  std={desc['std']:.1f}  "
      f"min={desc['min']:.0f}  median={desc['50%']:.0f}  max={desc['max']:.0f}")
short  = (df["PPI_LEN"] <= 3).sum()
full30 = (df["PPI_LEN"] >= 30).sum()
print(f"  ≤3 packets (near-unusable): {short} ({100*short/len(df):.1f}%)")
print(f"  =30 packets (full PPI)    : {full30} ({100*full30/len(df):.1f}%)")
# zero-padding fraction at SEQ_LEN=30
avg_real = df["PPI_LEN"].clip(upper=30).mean()
print(f"  With SEQ_LEN=30: avg real rows = {avg_real:.1f} / 30  "
      f"→ {100*(1 - avg_real/30):.1f}% padding  (was 81.5% at SEQ_LEN=128)")

# 1d. Flow statistics distributions
print("\n── 1d. Flow stat distributions ──")
print(f"  {'Column':<18}  {'min':>10}  {'median':>12}  {'max':>14}  {'zeros':>6}")
for col in ["BYTES","BYTES_REV","PACKETS","PACKETS_REV","DURATION"]:
    s = df[col].describe()
    z = (df[col] == 0).sum()
    print(f"  {col:<18}  {s['min']:>10.1f}  {s['50%']:>12.1f}  {s['max']:>14.1f}  {z:>6}")

# 1e. Missing values
print("\n── 1e. Missing values ──")
miss = df.isnull().sum()
miss_nonzero = miss[miss > 0]
if miss_nonzero.empty:
    print("  No missing numeric values.")
else:
    for col, cnt in miss_nonzero.items():
        print(f"  {col}: {cnt} missing ({100*cnt/len(df):.1f}%)")

# 1f. PPI raw values — 5 samples
print("\n── 1f. 5 sample PPI arrays (first 5 timesteps: IPT | dir | size) ──")
for i in range(5):
    raw = df["PPI"].iloc[i]
    try:
        arr = parse_ppi_string(raw)
        n = int(df["PPI_LEN"].iloc[i])
        print(f"  row {i} | APP={df['APP'].iloc[i]} | ppi_len={n} | "
              f"BYTES={int(df['BYTES'].iloc[i])} | DUR={df['DURATION'].iloc[i]:.3f}s")
        print(f"    IPT(ms)  : {arr[0][:5]}")
        print(f"    direction: {arr[1][:5]}")
        print(f"    size(B)  : {arr[2][:5]}")
    except Exception as e:
        print(f"  row {i}: PARSE ERROR — {e}")

# 1g. Validator rejection audit
print("\n── 1g. Validator rejection audit (first 200 rows) ──")
from src.data_validator import FlowValidator
validator = FlowValidator()

reject_counts = {}
valid_count = 0
for _, row in df.head(200).iterrows():
    raw = row["PPI"]
    try:
        arr = parse_ppi_string(raw)
        ppi = [list(arr[0]), list(arr[1]), list(arr[2])]
    except Exception:
        reject_counts["PPI_PARSE_ERROR"] = reject_counts.get("PPI_PARSE_ERROR", 0) + 1
        continue
    ok, reason = validator.validate_ppi(ppi, flow_endreason_active=int(row.get("FLOW_ENDREASON_ACTIVE", 0)))
    if not ok:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
        continue
    stats_dict = {
        "BYTES":               float(row.get("BYTES", 0)),
        "BYTES_REV":           float(row.get("BYTES_REV", 0)),
        "PACKETS":             float(row.get("PACKETS", 0)),
        "PACKETS_REV":         float(row.get("PACKETS_REV", 0)),
        "DURATION":            float(row.get("DURATION", 0)),
        "FLOW_ENDREASON_IDLE": int(row.get("FLOW_ENDREASON_IDLE", 0)),
        "FLOW_ENDREASON_ACTIVE": int(row.get("FLOW_ENDREASON_ACTIVE", 0)),
    }
    ok, reason = validator.validate_stats(stats_dict, is_cesnet=True)
    if not ok:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
        continue
    valid_count += 1

total_checked = 200
rejected_total = sum(reject_counts.values())
print(f"  Checked : {total_checked} rows")
print(f"  Valid   : {valid_count} ({100*valid_count/total_checked:.1f}%)")
print(f"  Rejected: {rejected_total} ({100*rejected_total/total_checked:.1f}%)")
for reason, cnt in sorted(reject_counts.items(), key=lambda x: -x[1]):
    print(f"    {reason}: {cnt}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Skeleton + Baselines
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("STEP 2: SKELETON + BASELINE CHECKS  (SEQ_LEN=30)")
print("=" * 65)

from src.feature_engineering import extract_seq_features, extract_stat_features, SEQ_LEN, STAT_INPUT_DIM

print(f"\nSEQ_LEN in use: {SEQ_LEN}  (confirm = 30)")
assert SEQ_LEN == 30, f"Expected SEQ_LEN=30, got {SEQ_LEN}"

# Build 64-sample batch from CSV
def csv_row_to_sample(row):
    raw = row["PPI"]
    try:
        arr = parse_ppi_string(raw)
    except Exception:
        return None
    if arr.shape[0] < 3:
        return None
    ppi = [list(arr[0]), list(arr[1]), list(arr[2])]
    endreason_active = int(row.get("FLOW_ENDREASON_ACTIVE", 0))
    ok, _ = validator.validate_ppi(ppi, flow_endreason_active=endreason_active)
    if not ok:
        return None
    stats_dict = {
        "BYTES":               float(row.get("BYTES", 0)),
        "BYTES_REV":           float(row.get("BYTES_REV", 0)),
        "PACKETS":             float(row.get("PACKETS", 0)),
        "PACKETS_REV":         float(row.get("PACKETS_REV", 0)),
        "DURATION":            float(row.get("DURATION", 0)),
        "FLOW_ENDREASON_IDLE": int(row.get("FLOW_ENDREASON_IDLE", 0)),
        "FLOW_ENDREASON_ACTIVE": endreason_active,
    }
    ok, _ = validator.validate_stats(stats_dict, is_cesnet=True)
    if not ok:
        return None
    fe_row = {**stats_dict, "PPI": ppi,
              "PPI_LEN": int(row.get("PPI_LEN", len(ppi[0]))),
              "PHIST_SRC_SIZES": [0]*8}
    try:
        seq  = extract_seq_features(ppi, SEQ_LEN)
        stat = extract_stat_features(fe_row)
    except Exception:
        return None
    if not (np.all(np.isfinite(seq)) and np.all(np.isfinite(stat))):
        return None
    label = int(row["APP"]) % 8
    return seq, stat, label

print("\nBuilding 64-sample batch...")
samples, attempts = [], 0
for _, row in df.iterrows():
    attempts += 1
    s = csv_row_to_sample(row)
    if s:
        samples.append(s)
    if len(samples) >= 64:
        break

print(f"  Valid samples : {len(samples)} / {attempts} rows scanned")

if len(samples) < 16:
    print("  FATAL: not enough samples — check validator thresholds")
    sys.exit(1)

seqs   = torch.tensor(np.stack([s[0] for s in samples]), dtype=torch.float32)
stats_ = torch.tensor(np.stack([s[1] for s in samples]), dtype=torch.float32)
labels = torch.tensor([s[2] for s in samples], dtype=torch.long)

# 2a. Visualize tensors right before model input
print("\n── 2a. Tensor sanity check ──")
print(f"  seq  shape : {list(seqs.shape)}   (batch × SEQ_LEN × 3)")
print(f"  stat shape : {list(stats_.shape)}  (batch × STAT_INPUT_DIM)")
print(f"  NaN in seq : {torch.isnan(seqs).any().item()}")
print(f"  NaN in stat: {torch.isnan(stats_).any().item()}")
print(f"  seq  range : [{seqs.min():.3f}, {seqs.max():.3f}]  (expect [-1, 1])")
print(f"  stat range : [{stats_.min():.3f}, {stats_.max():.3f}]  (expect [0, 1])")

zero_rows = (seqs.abs().sum(dim=-1) == 0).float().mean().item()
print(f"  Zero-padded seq rows : {100*zero_rows:.1f}%  (was 81.5% at SEQ_LEN=128)")

print(f"\n  seq[0] all 30 timesteps:")
print(f"    {'t':>3}  {'IPT_norm':>10}  {'dir':>6}  {'size_norm':>10}")
for t in range(int(seqs.shape[1])):
    ipt, d, sz = seqs[0, t].tolist()
    marker = " ← padding start" if t == int(df["PPI_LEN"].iloc[0]) else ""
    print(f"    {t:>3}  {ipt:>10.4f}  {d:>6.1f}  {sz:>10.4f}{marker}")

print(f"\n  stat[0] ({STAT_INPUT_DIM} features):")
print(f"  {stats_[0].numpy()}")

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
print(f"  Init intra-sim : {intra_init:.4f}  (expected ~0.0 for random weights)")
print(f"  Init inter-sim : {inter_init:.4f}  (expected ~0.0 for random weights)")
print(f"  Init loss      : {loss_init:.4f}  (expected ≈ λ_pos ≈ 0.7)")
if abs(loss_init - 0.7) < 0.15:
    print("  ✅ PASS: init loss in expected range")
else:
    print("  ❌ FLAG: init loss outside expected range — check λ_pos setting")

if intra_init > 0.95:
    print("  ⚠️  WARNING: intra-sim very high at init → BatchNorm collapse on padding")
else:
    print("  ✅ PASS: init similarities look random-ish")

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
    print("  ❌ FAIL: zero input not worse — normalization or model issue")

# 2d. Single-batch overfit
print("\n── 2d. Single-batch overfit (32 samples, 150 steps, lr=3e-4) ──")
print(f"  {'step':>5}  {'loss':>8}  {'intra-sim':>10}  {'inter-sim':>10}")

torch.manual_seed(42)
model = DualBranchEncoder()
model.train()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

batch_seq   = seqs[:32]
batch_stat  = stats_[:32]
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
    print("  ✅ PASS: clean overfit (loss < 0.05) — model + loss correctly wired")
elif final_loss < 0.15:
    print("  ✅ PASS: overfit achieved (loss < 0.15)")
else:
    print(f"  ❌ FAIL: loss={final_loss:.4f} after 150 steps — investigate architecture or loss")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  SEQ_LEN              : {SEQ_LEN} (was 128)")
print(f"  Zero-padding         : {100*zero_rows:.1f}% (was 81.5% → now matches real PPI length)")
print(f"  Init loss            : {loss_init:.4f}")
print(f"  Init intra/inter-sim : {intra_init:.4f} / {inter_init:.4f}")
print(f"  Overfit final loss   : {final_loss:.4f}")
print(f"  Valid sample yield   : {len(samples)}/{attempts} from CSV")
print()
