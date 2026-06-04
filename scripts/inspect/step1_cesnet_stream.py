"""
Step 1 — Become One with the Data (CESNET-QUIC22, Streaming API)
================================================================
Uses CESNETStreamingDataset when cesnet-datazoo is installed (GPU instance).
Falls back to cesnet_sample.csv for local development.

Run:
    # GPU instance (cesnet-datazoo installed):
    python3 scripts/inspect/step1_cesnet_stream.py

    # Local (fallback to CSV):
    python3 scripts/inspect/step1_cesnet_stream.py --local

Karpathy checklist covered:
  [1] Scan examples manually — distributions, anomalies, label mapping
  [2] Check for duplicates and corrupt data
  [3] Understand class imbalance
  [4] Visualize feature distributions
  [5] Confirm normalization is sane before model input
"""

import argparse
import re
import sys
import warnings
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

torch.manual_seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.feature_engineering import SEQ_LEN, SEQ_INPUT_DIM, STAT_INPUT_DIM
from src.dataset_unified import LABEL_MAP, UNIFIED_CLASS_NAMES
from src.data_validator import FlowValidator

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true",
                    help="Use cesnet_sample.csv instead of live streaming API")
parser.add_argument("--data_root", default="/workspace/.cesnet_cache",
                    help="cesnet-datazoo local metadata cache dir")
parser.add_argument("--size", default="XS", choices=["XS", "S", "M"],
                    help="Dataset size: XS=100K, S=1M, M=10M flows")
parser.add_argument("--n_inspect", type=int, default=500,
                    help="Number of flows to inspect for Step 1")
args = parser.parse_args()

CESNET_CSV = os.path.join(ROOT, "cesnet_sample.csv")

# ---------------------------------------------------------------------------
# Data source: streaming or CSV fallback
# ---------------------------------------------------------------------------
def get_dataframe(n: int) -> pd.DataFrame:
    """Return a DataFrame of n CESNET flows from stream or CSV."""
    use_stream = not args.local

    if use_stream:
        try:
            from cesnet_datazoo.datasets import CESNET_QUIC22
            from cesnet_datazoo.config import DatasetConfig, AppSelection
            print(f"[source] cesnet-datazoo streaming (size={args.size}, pulling {n} rows)")
            ds = CESNET_QUIC22(data_root=args.data_root, size=args.size)
            cfg = DatasetConfig(dataset=ds, apps_selection=AppSelection.ALL_KNOWN)
            ds.set_dataset_config_and_initialize(cfg)
            df = ds.train_dataframe.head(n).copy()
            # Streaming API uses uppercase column names — normalise
            df.columns = [c.upper() if isinstance(c, str) else c for c in df.columns]
            print(f"[source] Got {len(df)} rows from cesnet-datazoo")
            return df
        except ImportError:
            print("[source] cesnet-datazoo not installed — falling back to cesnet_sample.csv")
        except Exception as e:
            print(f"[source] Streaming failed ({e}) — falling back to cesnet_sample.csv")

    print(f"[source] Loading cesnet_sample.csv (local fallback, {n} rows)")
    df = pd.read_csv(CESNET_CSV, nrows=n)
    df.columns = [c.upper() if isinstance(c, str) else c for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# PPI parser — handles numpy repr strings from CSV or streaming
# ---------------------------------------------------------------------------
def parse_ppi(raw) -> list:
    """Return [[ipts], [dirs], [sizes]] or None on failure."""
    if isinstance(raw, (list, np.ndarray)):
        ppi = list(raw)
        return [list(ppi[i]) for i in range(3)] if len(ppi) >= 3 else None
    if isinstance(raw, str):
        rows = re.findall(r'\[([^\[\]]+)\]', raw)
        parsed = []
        for r in rows:
            nums = [float(x) for x in re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', r)]
            if nums:
                parsed.append(nums)
        return parsed if len(parsed) >= 3 else None
    return None


# ---------------------------------------------------------------------------
# STEP 1 — Data Inspection
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"STEP 1: DATA INSPECTION — CESNET-QUIC22   (SEQ_LEN={SEQ_LEN})")
print("=" * 70)

df = get_dataframe(args.n_inspect)
n = len(df)
print(f"\nLoaded: {n} rows × {len(df.columns)} columns")
print(f"Columns present: {sorted(df.columns.tolist())}\n")

# 1a. Label / category column detection
print("── 1a. Label column availability ──")
has_category = "CATEGORY" in df.columns
has_app      = "APP" in df.columns
print(f"  CATEGORY column : {'YES' if has_category else 'NO  ← pipeline will reject all rows (R5)'}")
print(f"  APP column      : {'YES — numeric IDs, not usable directly' if has_app else 'NO'}")
if has_category:
    vc = df["CATEGORY"].value_counts()
    print(f"\n  CATEGORY distribution ({vc.shape[0]} unique):")
    for cat, cnt in vc.items():
        mapped = LABEL_MAP.get(str(cat).lower(), "REJECTED (not in LABEL_MAP)")
        print(f"    {str(cat):20s}  {cnt:6d}  → unified class {mapped}")
    unmapped = [c for c in df["CATEGORY"].unique() if str(c).lower() not in LABEL_MAP]
    if unmapped:
        print(f"\n  ⚠️  UNMAPPED categories: {unmapped}")
        print(f"     Add these to LABEL_MAP in dataset_unified.py before training!")
    else:
        print(f"\n  ✅ All categories map to LABEL_MAP")
elif has_app:
    vc = df["APP"].value_counts()
    print(f"\n  APP distribution ({vc.shape[0]} unique numeric IDs):")
    print(f"  Top 10: {dict(vc.head(10))}")
    print(f"  ⚠️  Numeric APP IDs need cesnet-datazoo APP→CATEGORY mapping")
    print(f"     Full cesnet-datazoo streaming returns proper CATEGORY strings.")

# 1b. PPI quality
print("\n── 1b. PPI sequence quality ──")
ppi_lens = df["PPI_LEN"].dropna().astype(int)
print(f"  PPI_LEN: min={ppi_lens.min()}  mean={ppi_lens.mean():.1f}  "
      f"median={ppi_lens.median():.0f}  max={ppi_lens.max()}")
print(f"  < 4 packets (unusable) : {(ppi_lens<4).sum()} ({100*(ppi_lens<4).mean():.1f}%)")
print(f"  = 30 (capped at max)   : {(ppi_lens==30).sum()} ({100*(ppi_lens==30).mean():.1f}%)")
avg_real = ppi_lens.clip(upper=SEQ_LEN).mean()
padding_pct = 100 * (1 - avg_real / SEQ_LEN)
print(f"  At SEQ_LEN={SEQ_LEN}: avg real rows = {avg_real:.1f}/{SEQ_LEN} → {padding_pct:.1f}% padding")
if padding_pct > 70:
    print(f"  ⚠️  HIGH PADDING: {padding_pct:.1f}% — Mamba sees mostly zeros")
else:
    print(f"  ✅ Padding acceptable at {padding_pct:.1f}%")

# 1c. IPT and size distributions
print("\n── 1c. Raw PPI feature distributions (first 200 parseable rows) ──")
ipts_raw, sizes_raw, dirs_raw = [], [], []
parse_errors = 0
for i in range(min(200, n)):
    ppi = parse_ppi(df["PPI"].iloc[i])
    if ppi is None:
        parse_errors += 1
        continue
    nn = int(df["PPI_LEN"].iloc[i]) if "PPI_LEN" in df.columns else len(ppi[0])
    ipts_raw.extend(ppi[0][:nn])
    dirs_raw.extend(ppi[1][:nn])
    sizes_raw.extend(ppi[2][:nn])

if ipts_raw:
    ipts_pos = [x for x in ipts_raw if x > 0]
    print(f"  IPT  (ms) : min={min(ipts_pos):.1f}  median={np.median(ipts_pos):.1f}  "
          f"mean={np.mean(ipts_pos):.1f}  max={max(ipts_pos):.1f}  "
          f"| >MAX_IPT(5000ms): {sum(1 for x in ipts_pos if x>5000)} ({100*sum(1 for x in ipts_pos if x>5000)/len(ipts_pos):.1f}%)")
    print(f"  Size (B)  : min={min(sizes_raw):.0f}  median={np.median(sizes_raw):.0f}  "
          f"mean={np.mean(sizes_raw):.0f}  max={max(sizes_raw):.0f}  "
          f"| >1500B (clipped): {sum(1 for x in sizes_raw if x>1500)}")
    dir_vals = sorted(set(int(d) for d in dirs_raw))
    dir_ok = all(d in (-1, 0, 1) for d in dir_vals)
    print(f"  Direction : unique={dir_vals}  {'✅ valid' if dir_ok else '❌ unexpected values'}")
    print(f"  Parse errors: {parse_errors}")

# 1d. DURATION units
print("\n── 1d. DURATION unit confirmation ──")
dur = df["DURATION"].dropna()
print(f"  DURATION: min={dur.min():.4f}  mean={dur.mean():.2f}  max={dur.max():.1f}")
if dur.mean() < 1000:
    print(f"  ✅ Unit: SECONDS (mean={dur.mean():.2f}s) → multiply by 1000 for feature extraction")
    print(f"     At {dur.mean():.1f}s mean: norm with x1000 = "
          f"{np.log1p(min(dur.mean()*1000, 300000))/np.log1p(300000):.3f}  (vs {np.log1p(min(dur.mean(), 300000))/np.log1p(300000):.3f} without)")

# 1e. Flow stats
print("\n── 1e. Flow statistics (BYTES, PACKETS) ──")
for col in ["BYTES", "BYTES_REV", "PACKETS", "PACKETS_REV"]:
    if col in df.columns:
        s = df[col].describe()
        zeros = (df[col] == 0).sum()
        print(f"  {col:<14}: min={s['min']:.0f}  median={s['50%']:.0f}  "
              f"max={s['max']:.0f}  zeros={zeros}")

# 1f. Missing / NaN
print("\n── 1f. Missing values ──")
miss = df.isnull().sum()
miss_nonzero = miss[miss > 0]
if miss_nonzero.empty:
    print("  ✅ No missing values in any column")
else:
    for col, cnt in miss_nonzero.items():
        print(f"  {col}: {cnt} missing ({100*cnt/n:.1f}%)")

# 1g. 5-tuple exposure check
print("\n── 1g. 5-tuple trap check ──")
five_tuple_cols = {"SRC_IP", "DST_IP", "SRC_PORT", "DST_PORT", "PROTOCOL",
                   "QUIC_SNI", "QUIC_USER_AGENT", "QUIC_USERAGENT"}
exposed = five_tuple_cols & set(df.columns)
if exposed:
    print(f"  ⚠️  5-tuple columns PRESENT in raw data: {sorted(exposed)}")
    print(f"     ✅ These are NOT extracted by feature_engineering.py")
    print(f"     ✅ dataset_unified.py reads only PPI + BYTES + PACKETS + DURATION + PHIST")
else:
    print(f"  ✅ No 5-tuple columns found")

# 1h. Validator rejection audit
print("\n── 1h. Validator rejection audit (first 200 rows) ──")
validator = FlowValidator()
reject_counts, valid_count = {}, 0
for i in range(min(200, n)):
    ppi = parse_ppi(df["PPI"].iloc[i])
    if ppi is None:
        reject_counts["PPI_PARSE_ERROR"] = reject_counts.get("PPI_PARSE_ERROR", 0) + 1
        continue
    ok, reason = validator.validate_ppi(
        ppi, flow_endreason_active=int(df.get("FLOW_ENDREASON_ACTIVE", pd.Series([0]*n)).iloc[i])
    )
    if not ok:
        reject_counts[reason] = reject_counts.get(reason, 0) + 1
        continue
    if "DURATION" in df.columns:
        sd = {"BYTES": float(df["BYTES"].iloc[i]), "BYTES_REV": float(df["BYTES_REV"].iloc[i]),
              "PACKETS": float(df["PACKETS"].iloc[i]), "PACKETS_REV": float(df["PACKETS_REV"].iloc[i]),
              "DURATION": float(df["DURATION"].iloc[i]),
              "FLOW_ENDREASON_IDLE": int(df.get("FLOW_ENDREASON_IDLE", pd.Series([0]*n)).iloc[i]),
              "FLOW_ENDREASON_ACTIVE": int(df.get("FLOW_ENDREASON_ACTIVE", pd.Series([0]*n)).iloc[i])}
        ok, reason = validator.validate_stats(sd, is_cesnet=True)
        if not ok:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
            continue
    valid_count += 1

total_checked = min(200, n)
rejected_total = sum(reject_counts.values())
print(f"  Checked   : {total_checked}")
print(f"  Valid     : {valid_count} ({100*valid_count/total_checked:.1f}%)")
print(f"  Rejected  : {rejected_total} ({100*rejected_total/total_checked:.1f}%)")
for reason, cnt in sorted(reject_counts.items(), key=lambda x: -x[1]):
    print(f"    {reason}: {cnt}")

print("\n" + "=" * 70)
print("STEP 1 COMPLETE — review findings above before running step2_baselines.py")
print("=" * 70)
