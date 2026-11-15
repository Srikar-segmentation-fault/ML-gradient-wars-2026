"""
build_features_half.py - density-matched features (features_v2_half.parquet).

Why: October (test) has ~0.5x the rows of a training month, and a random 50% row-thinning of
September reproduces October's entity statistics (density_check.py: median |log ratio| 0.010 vs
0.264 for raw September). Window counts / link features computed on full-density train therefore
mean something different on test. Fix: compute the features on half-density train.

How (reuses build_features_v2.py unchanged, no new feature code):
  run 0: build_features_v2 on (train rows with half==0) + test  -> train-half-0 rows AND test rows
  run 1: build_features_v2 on (train rows with half==1) + test  -> train-half-1 rows
  merge: every train row gets features computed among its own half only; test rows come from run 0
         (test is already thinned, and its look-back into September is then also half density).
Categorical codes (pl.Categorical physical codes depend on the run), timestamp, label and is_test
are taken from features_v2.parquet so encodings stay identical. Output row order == features_v2.parquet
row order (asserted), so blend.py / val_labels.npy keep working.

The split: parity of the hex number in txn_id (deterministic, effectively random).

Usage:
    python build_features_half.py            # needs features_v2.parquet + the 4 CSVs, ~3-5 min
Outputs: features_v2_half.parquet (refuses to overwrite), _half_run0.parquet, _half_run1.parquet
"""
import os
import subprocess
import sys
import time

import polars as pl

SRC = "build_features_v2.py"
# --bit B picks which bit of the hex txn_id decides the half (default 0 = original split).
# A different bit gives an independent random thinning -> a second, decorrelated feature set.
BIT = int(sys.argv[sys.argv.index("--bit") + 1]) if "--bit" in sys.argv else 0
SUFFIX = "" if BIT == 0 else f"_b{BIT}"
OUT = f"features_v2_half{SUFFIX}.parquet"
DIV = 2 ** BIT
CAT_COLS = [
    "receiver_type", "txn_type", "merchant_category", "device_city", "app",
    "sender_home_city", "sender_bank", "sender_kyc", "sender_os",
    "receiver_home_city", "receiver_bank", "receiver_kyc", "receiver_os",
    "merchant_cat_master", "merchant_city",
]
KEEP_FROM_ORIG = set(CAT_COLS) | {"txn_id", "timestamp", "label", "is_test"}


def half_expr(k):
    return ((pl.col("txn_id").str.slice(2).str.to_integer(base=16) // DIV) % 2) == k


def tmp_name(k):
    return f"_half{SUFFIX}_run{k}.parquet"


def run_half(k):
    """Executes build_features_v2.py's main() with train filtered to half k and a different output."""
    out = tmp_name(k)
    if os.path.exists(out):
        print(f"[skip] {out} exists")
        return out
    src = open(SRC, encoding="utf-8").read()
    a = 'train_pl = pl.read_csv("train.csv")'
    b = 'out_file = "features_v2.parquet"'
    assert src.count(a) == 1 and src.count(b) == 1, "build_features_v2.py changed - patch points not found"
    src = src.replace(a, a + f'\n    train_pl = train_pl.filter(((pl.col("txn_id").str.slice(2)'
                             f'.str.to_integer(base=16) // {DIV}) % 2) == {k})\n'
                             f'    print("BIT {BIT} HALF {k}: train rows", train_pl.height)')
    src = src.replace(b, f'out_file = "{out}"')
    t = time.time()
    g = {"__name__": "__half__"}
    exec(compile(src, f"{SRC}[half{k}]", "exec"), g)
    g["main"]()
    print(f"half {k} built in {time.time() - t:.0f}s")
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--half":
        run_half(int(sys.argv[2]))
        return
    if os.path.exists(OUT):
        sys.exit(f"{OUT} already exists - rename or delete it first (never overwritten).")
    t0 = time.time()

    # sanity: every txn_id parses as hex, halves are balanced
    ids = pl.read_csv("train.csv", columns=["txn_id", "label"])
    h = ids.with_columns(((pl.col("txn_id").str.slice(2).str.to_integer(base=16) // DIV) % 2).alias("h"))
    assert h["h"].null_count() == 0, "some txn_id are not TX<hex>"
    print(f"bit {BIT} -> {OUT}")
    print(h.group_by("h").agg(pl.len(), (pl.col("label") != "legit").mean().alias("fraud_rate")).sort("h"))

    # each half in its own process (keeps peak memory at one build)
    for k in (0, 1):
        subprocess.run([sys.executable, __file__, "--half", str(k), "--bit", str(BIT)], check=True)

    orig = pl.read_parquet("features_v2.parquet")
    feat_cols = [c for c in orig.columns if c not in KEEP_FROM_ORIG]
    r0 = pl.read_parquet(tmp_name(0))
    r1 = pl.read_parquet(tmp_name(1))
    missing = [c for c in feat_cols if c not in r0.columns]
    assert not missing, f"half build lacks columns: {missing}"

    part0 = r0.filter((pl.col("is_test") == 1) | half_expr(0)).select(["txn_id"] + feat_cols)
    part1 = r1.filter((pl.col("is_test") == 0) & half_expr(1)).select(["txn_id"] + feat_cols)
    half = pl.concat([part0, part1.select(part0.columns)], how="vertical_relaxed")
    assert half["txn_id"].is_unique().all() and half.height == orig.height, (half.height, orig.height)

    merged = orig.select(["txn_id"] + [c for c in orig.columns if c in KEEP_FROM_ORIG and c != "txn_id"]) \
                 .with_row_index("_ord") \
                 .join(half, on="txn_id", how="left").sort("_ord").drop("_ord")
    merged = merged.select(orig.columns)
    for c in feat_cols:  # keep dtypes identical to the original parquet
        if merged[c].dtype != orig[c].dtype:
            merged = merged.with_columns(pl.col(c).cast(orig[c].dtype))
    assert merged["txn_id"].equals(orig["txn_id"]), "row order differs from features_v2.parquet"

    # report how much the density-sensitive columns moved (train rows only)
    tr = orig["is_test"] == 0
    print("\nMean on train rows, full-density vs half-density (test mean for reference):")
    for c in ["receiver_inflow_count_1h", "receiver_inflow_count_24h", "receiver_unique_senders_24h",
              "sender_unique_receivers_24h", "device_unique_senders_24h", "has_inflow_24h",
              "has_outflow_24h", "sender_prev_txn_sec"]:
        if c in orig.columns:
            print(f"  {c:32s} full {orig.filter(tr)[c].mean():12.4f} | half {merged.filter(tr)[c].mean():12.4f}"
                  f" | test {merged.filter(~tr)[c].mean():12.4f}")

    merged.write_parquet(OUT)
    print(f"\nWrote {OUT} {merged.shape} in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
