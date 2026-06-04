"""
export_cesnet_sample.py

Downloads CESNET-QUIC22 (XS = ~100K flows) and saves every Nth row as a CSV
for manual inspection. Default: every 500th row → ~200 rows.

Usage:
    python scripts/export_cesnet_sample.py [--nth 500] [--split train] \
        [--out cesnet_sample.csv] [--cache /tmp/cesnet_cache]
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Export a CESNET-QUIC22 sample to CSV")
    parser.add_argument("--nth", type=int, default=500,
                        help="Take every Nth row (default: 500)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many rows (before nth sampling)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--out", default="cesnet_sample.csv",
                        help="Output CSV path")
    parser.add_argument("--cache", default="/tmp/cesnet_cache",
                        help="Local cache dir for cesnet-datazoo metadata")
    parser.add_argument("--size", default="XS", choices=["XS", "S", "M"],
                        help="Dataset size (XS=100K, S=1M, M=10M). Default: XS")
    args = parser.parse_args()

    try:
        from cesnet_datazoo.datasets import CESNET_QUIC22
        from cesnet_datazoo.config import DatasetConfig, AppSelection
    except ImportError:
        sys.exit("cesnet-datazoo not installed. Run: pip install cesnet-datazoo")

    print(f"Initialising CESNET-QUIC22 (size={args.size}) …")
    os.makedirs(args.cache, exist_ok=True)

    dataset = CESNET_QUIC22(data_root=args.cache, size=args.size, silent=False)

    config = DatasetConfig(
        dataset=dataset,
        apps_selection=AppSelection.ALL_KNOWN,
        need_train_set=(args.split == "train"),
        need_val_set=(args.split == "val"),
        need_test_set=(args.split == "test"),
        train_workers=0,
        val_workers=0,
        test_workers=0,
        return_other_fields=True,
    )
    dataset.set_dataset_config_and_initialize(config)

    print(f"Loading {args.split} dataframe …")
    if args.split == "train":
        df = dataset.get_train_df()
    elif args.split == "val":
        df = dataset.get_val_df()
    else:
        df = dataset.get_test_df()

    print(f"Full {args.split} set: {len(df):,} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    if args.limit:
        df = df.iloc[:args.limit]
        print(f"Limited to first {args.limit} rows")

    sample = df.iloc[::args.nth].copy()
    print(f"Sampled every {args.nth}th row → {len(sample):,} rows")

    out_path = os.path.abspath(args.out)
    sample.to_csv(out_path, index=True)
    print(f"\nSaved to: {out_path}")
    print(sample.head(5).to_string())


if __name__ == "__main__":
    main()
