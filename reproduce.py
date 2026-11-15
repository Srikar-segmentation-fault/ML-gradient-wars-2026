"""
reproduce.py - rebuilds the top public uploads from the raw CSVs.

Supported presets (--preset):
  - ens7h        (default, public LB 0.4958): 5 full-density + 2 half-density members
  - ens8hh       (3 full-density + 5 half-density members across bit-0 and bit-7 splits)
  - ens10h       (5 full-density + 5 half-density members)
  - ens11h       (ens10h + m_half_xt)
  - ens16_master (all 16 full-density and multi-split half-density members)

Usage (Python 3.12, from the repo root, with train.csv / test.csv / accounts.csv / merchants.csv
in the same folder):
    python reproduce.py                         # default: ens7h -> submission.csv
    python reproduce.py --preset ens11h         # 11-model blend -> submission.csv
    python reproduce.py --preset ens16_master   # 16-model blend -> submission.csv
    python reproduce.py --skip-existing         # resume: skip steps whose outputs already exist
    python reproduce.py --smoke                 # quick check: features + one 20%-sample member, 50 rounds

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
THREADS = "8"

V3_EXPANDING = [
    "sender_past_txn_count", "sender_past_amount_sum", "sender_sec_since_first_seen",
    "receiver_past_txn_count", "receiver_past_amount_sum", "receiver_sec_since_first_seen",
    "device_past_txn_count", "device_past_amount_sum", "device_sec_since_first_seen",
]
UD6 = [
    "receiver_unique_senders_1h", "receiver_unique_senders_24h",
    "sender_unique_receivers_1h", "sender_unique_receivers_24h",
    "device_unique_senders_1h", "device_unique_senders_24h",
]
UD7 = UD6 + ["receiver_new_payee_share_24h"]
UD8 = UD7 + ["receiver_ip_risk_mean_24h"]
V3_SAFE_DROP = V3_EXPANDING + UD8
assert len(V3_EXPANDING) == 9 and len(V3_SAFE_DROP) == 17

ALL_STEPS = {
    "features": (["features_v2.parquet"], [PY, "build_features_v2.py"]),
    "features_half": (["features_v2_half.parquet"], [PY, "build_features_half.py"]),
    "features_half_b7": (["features_v2_half_b7.parquet"], [PY, "build_features_half.py", "--bit", "7"]),
    "v3_full_new_noexp": (None, [PY, "train_v3.py", "--name", "v3_full_new_noexp", "--drop", ",".join(V3_EXPANDING)]),
    "v3_safe": (None, [PY, "train_v3.py", "--name", "v3_safe", "--drop", ",".join(V3_SAFE_DROP)]),
    "v3_noip": (None, [PY, "train_v3_noip.py", "--name", "v3_noip"]),
    "m_extra": (None, [PY, "train_member.py", "--name", "m_extra", "--no-ip", "--seed", "11", "--rounds", "300",
                       "--threads", THREADS, "--params", '{"extra_trees": true, "feature_fraction": 0.4, "min_child_samples": 200}']),
    "m_down": (None, [PY, "train_member.py", "--name", "m_down", "--legit_frac", "0.3", "--seed", "21",
                      "--threads", THREADS, "--params", '{"num_leaves": 63}']),
    "m_down_half": (None, [PY, "train_member.py", "--name", "m_down_half", "--features", "features_v2_half.parquet",
                           "--legit_frac", "0.3", "--seed", "21", "--threads", THREADS, "--params", '{"num_leaves": 63}']),
    "m_down_half_ud": (None, [PY, "train_member.py", "--name", "m_down_half_ud", "--features", "features_v2_half.parquet",
                              "--legit_frac", "0.3", "--seed", "21", "--threads", THREADS, "--params", '{"num_leaves": 63}',
                              "--undrop", ",".join(UD6)]),
    "m_half_b7": (None, [PY, "train_member.py", "--name", "m_half_b7", "--features", "features_v2_half_b7.parquet",
                         "--legit_frac", "0.3", "--seed", "31", "--threads", THREADS, "--params", '{"num_leaves": 63}',
                         "--undrop", ",".join(UD6)]),
    "m_half_s22": (None, [PY, "train_member.py", "--name", "m_half_s22", "--features", "features_v2_half.parquet",
                          "--legit_frac", "0.3", "--seed", "22", "--threads", "2", "--params", '{"num_leaves": 63}',
                          "--undrop", ",".join(UD6)]),
    "m_half_b7_noip": (None, [PY, "train_member.py", "--name", "m_half_b7_noip", "--features", "features_v2_half_b7.parquet",
                              "--legit_frac", "0.3", "--seed", "41", "--threads", THREADS, "--params", '{"num_leaves": 63}',
                              "--undrop", ",".join(UD6), "--no-ip"]),
    "m_half_xt": (None, [PY, "train_member.py", "--name", "m_half_xt", "--features", "features_v2_half.parquet",
                         "--legit_frac", "0.3", "--seed", "51", "--threads", "2",
                         "--params", '{"extra_trees": true, "feature_fraction": 0.4, "min_child_samples": 200}',
                         "--undrop", ",".join(UD6)]),
    "m_half_L95": (None, [PY, "train_member.py", "--name", "m_half_L95", "--features", "features_v2_half.parquet",
                          "--legit_frac", "0.3", "--seed", "61", "--rounds", "175", "--threads", THREADS,
                          "--params", '{"num_leaves": 95, "min_child_samples": 180, "feature_fraction": 0.5}',
                          "--undrop", ",".join(UD8)]),
    "m_half_col35": (None, [PY, "train_member.py", "--name", "m_half_col35", "--features", "features_v2_half.parquet",
                            "--legit_frac", "0.3", "--seed", "71", "--rounds", "200", "--threads", THREADS,
                            "--params", '{"num_leaves": 63, "feature_fraction": 0.35, "bagging_fraction": 0.85, "lambda_l1": 2.0, "lambda_l2": 5.0}',
                            "--undrop", ",".join(UD7)]),
    "m_half_d22": (None, [PY, "train_member.py", "--name", "m_half_d22", "--features", "features_v2_half.parquet",
                          "--legit_frac", "0.22", "--seed", "81", "--rounds", "175", "--threads", THREADS,
                          "--params", '{"num_leaves": 63, "min_child_samples": 250}',
                          "--undrop", ",".join(UD7)]),
    "m_half_d45_noip": (None, [PY, "train_member.py", "--name", "m_half_d45_noip", "--features", "features_v2_half.parquet",
                               "--legit_frac", "0.45", "--seed", "91", "--rounds", "175", "--threads", THREADS, "--no-ip",
                               "--params", '{"num_leaves": 75, "min_child_samples": 250}',
                               "--undrop", ",".join(UD7)]),
    "m_half_b7_L95": (None, [PY, "train_member.py", "--name", "m_half_b7_L95", "--features", "features_v2_half_b7.parquet",
                             "--legit_frac", "0.3", "--seed", "62", "--rounds", "175", "--threads", THREADS,
                             "--params", '{"num_leaves": 95, "min_child_samples": 180}',
                             "--undrop", ",".join(UD8)]),
}

PRESETS = {
    "ens7h": ["v3_safe", "v3_noip", "v3_full_new_noexp", "m_extra", "m_down", "m_down_half", "m_down_half_ud"],
    "ens8hh": ["v3_safe", "m_extra", "m_down", "m_down_half", "m_down_half_ud", "m_half_b7", "m_half_s22", "m_half_b7_noip"],
    "ens10h": ["v3_safe", "v3_noip", "v3_full_new_noexp", "m_extra", "m_down", "m_down_half", "m_down_half_ud", "m_half_b7", "m_half_s22", "m_half_b7_noip"],
    "ens11h": ["v3_safe", "v3_noip", "v3_full_new_noexp", "m_extra", "m_down", "m_down_half", "m_down_half_ud", "m_half_b7", "m_half_s22", "m_half_b7_noip", "m_half_xt"],
    "ens16_master": ["v3_safe", "v3_noip", "v3_full_new_noexp", "m_extra", "m_down", "m_down_half", "m_down_half_ud",
                     "m_half_b7", "m_half_s22", "m_half_b7_noip", "m_half_xt", "m_half_L95", "m_half_col35",
                     "m_half_d22", "m_half_d45_noip", "m_half_b7_L95"],
}


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
    ap.add_argument("--preset", default="ens7h", choices=list(PRESETS.keys()),
                    help="which ensemble submission to reproduce (default: ens7h)")
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
            run(ALL_STEPS["features"][1])
        run([PY, "train_member.py", "--name", "smoke", "--sample", "--rounds", "50",
             "--threads", THREADS])
        print(f"\nSmoke test OK in {time.time() - t0:.0f}s")
        return

    members = PRESETS[args.preset]
    needed_features = ["features", "features_half"]
    if any("b7" in m for m in members):
        needed_features.append("features_half_b7")

    for step_name in needed_features + members:
        explicit, argv = ALL_STEPS[step_name]
        outs = outputs_for(step_name, explicit)
        if args.skip_existing and all(os.path.exists(o) for o in outs):
            print(f"[skip] {step_name}: {outs} already exist")
            continue
        run(argv)

    sub_blend = f"submission_blend_{args.preset}.csv"
    run([PY, "blend.py", "--runs", ",".join(members), "--tag", args.preset])
    shutil.copyfile(sub_blend, "submission.csv")
    check_submission("submission.csv")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> submission.csv")


if __name__ == "__main__":
    main()
