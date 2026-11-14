"""
reproduce.py - rebuilds the best public upload (blend `ens5`, public 0.4946) from the raw CSVs.

Steps (each is a separate script, called with the exact arguments used for the scored upload):
  1. build_features_v2.py                        -> features_v2.parquet
  2. train_v3.py        --name v3_full_new_noexp  (drops the 9 expanding-history columns)
  3. train_v3.py        --name v3_safe            (drops 17 history / density-sensitive columns)
  4. train_v3_noip.py   --name v3_noip            (its own built-in drop list + every ip_risk column)
  5. train_member.py    --name m_extra            (no ip_risk, extra_trees, seed 11, 300 rounds)
  6. train_member.py    --name m_down             (legit downsampled to 30%, 63 leaves, seed 21)
  7. blend.py --runs <the five above> --tag ens5  (equal weights, multipliers re-tuned on the blend)
  8. copy submission_blend_ens5.csv -> submission.csv and check it against test.csv

Usage (Python 3.12, from the repo root, with train.csv / test.csv / accounts.csv / merchants.csv
in the same folder):
    python reproduce.py                 # full run, about 1 hour on an 8-core laptop
    python reproduce.py --skip-existing # resume: skip steps whose outputs already exist
    python reproduce.py --smoke         # quick check: features + one 20%-sample member, 50 rounds

Run in a clean folder. The training scripts never overwrite outputs (they add a _run2 suffix),
so stale val_probs_*/test_probs_* files from an older run would be picked up by blend.py.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

PY = sys.executable
THREADS = "8"  # thread count used for the scored m_extra / m_down runs (physical cores)

V3_EXPANDING = [
    "sender_past_txn_count", "sender_past_amount_sum", "sender_sec_since_first_seen",
    "receiver_past_txn_count", "receiver_past_amount_sum", "receiver_sec_since_first_seen",
    "device_past_txn_count", "device_past_amount_sum", "device_sec_since_first_seen",
]
V3_SAFE_DROP = V3_EXPANDING + [
    "receiver_unique_senders_1h", "receiver_unique_senders_24h",
    "sender_unique_receivers_1h", "sender_unique_receivers_24h",
    "device_unique_senders_1h", "device_unique_senders_24h",
    "receiver_ip_risk_mean_24h", "receiver_new_payee_share_24h",
]
assert len(V3_EXPANDING) == 9 and len(V3_SAFE_DROP) == 17

MEMBERS = ["v3_safe", "v3_noip", "v3_full_new_noexp", "m_extra", "m_down"]

STEPS = [
    # (name, outputs that mark the step done, argv)
    ("features", ["features_v2.parquet"], [PY, "build_features_v2.py"]),
    ("v3_full_new_noexp", None,
     [PY, "train_v3.py", "--name", "v3_full_new_noexp", "--drop", ",".join(V3_EXPANDING)]),
    ("v3_safe", None,
     [PY, "train_v3.py", "--name", "v3_safe", "--drop", ",".join(V3_SAFE_DROP)]),
    ("v3_noip", None, [PY, "train_v3_noip.py", "--name", "v3_noip"]),
    ("m_extra", None,
     [PY, "train_member.py", "--name", "m_extra", "--no-ip", "--seed", "11", "--rounds", "300",
      "--threads", THREADS,
      "--params", '{"extra_trees": true, "feature_fraction": 0.4, "min_child_samples": 200}']),
    ("m_down", None,
     [PY, "train_member.py", "--name", "m_down", "--legit_frac", "0.3", "--seed", "21",
      "--threads", THREADS, "--params", '{"num_leaves": 63}']),
    ("blend", ["submission_blend_ens5.csv"],
     [PY, "blend.py", "--runs", ",".join(MEMBERS), "--tag", "ens5"]),
]


def outputs_for(name, explicit):
    if explicit is not None:
        return explicit
    return [f"val_probs_{name}.npy", f"test_probs_{name}.npy"]


def run(argv):
    print("\n" + "=" * 100)
    print("RUN:", " ".join(argv[1:]))
    print("=" * 100, flush=True)
    t = time.time()
    subprocess.run(argv, check=True)
    print(f"--> finished in {time.time() - t:.0f}s", flush=True)


def check_submission(path):
    import pandas as pd
    sub = pd.read_csv(path)
    test_ids = pd.read_csv("test.csv", usecols=["txn_id"])["txn_id"]
    assert list(sub.columns) == ["id", "label"], sub.columns
    assert len(sub) == len(test_ids), (len(sub), len(test_ids))
    assert sub["id"].is_unique and set(sub["id"]) == set(test_ids), "ids do not match test.csv"
    print(f"\n{path}: {len(sub):,} rows, ids match test.csv. Label counts:")
    print(sub["label"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a step if its output files already exist")
    ap.add_argument("--smoke", action="store_true",
                    help="features + one 20%%-sample train_member run with 50 rounds")
    args = ap.parse_args()

    for f in ["train.csv", "test.csv", "accounts.csv", "merchants.csv"]:
        if not os.path.exists(f):
            sys.exit(f"Missing {f}: place the competition CSVs in the repo root.")

    t0 = time.time()
    if args.smoke:
        if not os.path.exists("features_v2.parquet"):
            run(STEPS[0][2])
        run([PY, "train_member.py", "--name", "smoke", "--sample", "--rounds", "50",
             "--threads", THREADS])
        print(f"\nSmoke test OK in {time.time() - t0:.0f}s")
        return

    for name, explicit, argv in STEPS:
        outs = outputs_for(name, explicit)
        if args.skip_existing and all(os.path.exists(o) for o in outs):
            print(f"[skip] {name}: {outs} already exist")
            continue
        run(argv)

    shutil.copyfile("submission_blend_ens5.csv", "submission.csv")
    check_submission("submission.csv")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> submission.csv")


if __name__ == "__main__":
    main()
