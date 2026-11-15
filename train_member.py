"""
train_member.py - one LightGBM ensemble member, same features/split/tuner as train_v3.py (v3_safe).

Speed design (why this differs from the old version):
  * The training call has NO valid_sets and NO callbacks, so nothing calls predict() while training.
  * Validation Macro-F1 is evaluated AFTER training at every --eval_every rounds, using
    cumulative raw scores (predict on trees [prev, k) only), so total predict cost = one full predict.
  * The SAME thread count is passed to training and to every predict() call (no thread flip-flopping).

Saved: val_probs_<name>.npy, test_probs_<name>.npy (float32, features_v2.parquet row order,
       same shapes as v3_safe), plus a row in results.csv (backed up first).
"""
import os
import sys
import time
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    import psutil
except ImportError:
    psutil = None

CLASSES = [
    "account_takeover", "fake_cashback", "layering_chain", "legit",
    "loan_app_extortion", "merchant_collusion", "mule_account",
    "phishing_collect", "qr_swap", "refund_abuse", "sim_swap", "velocity_bot",
]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}
LEGIT_IDX = LABEL_TO_ID["legit"]
NUM_CLASSES = 12

# Exactly train_v3.py's EXCLUDED_COLS (18).
EXCLUDED_COLS = {
    "txn_id", "sender_id", "receiver_id", "device_id", "is_test", "timestamp",
    "label", "target", "app_version", "opened_on", "sender_opened_on",
    "receiver_opened_on", "onboarded_on", "merchant_onboarded_on",
    "inflow_ts_key", "outflow_ts_key", "rev_bwd_key", "rev_fwd_key",
}

# Exactly the 17 columns of the v3_safe --drop list (copied from the logged run).
V3_SAFE_DROP = [
    "sender_past_txn_count", "sender_past_amount_sum", "sender_sec_since_first_seen",
    "receiver_past_txn_count", "receiver_past_amount_sum", "receiver_sec_since_first_seen",
    "device_past_txn_count", "device_past_amount_sum", "device_sec_since_first_seen",
    "receiver_unique_senders_1h", "receiver_unique_senders_24h",
    "sender_unique_receivers_1h", "sender_unique_receivers_24h",
    "device_unique_senders_1h", "device_unique_senders_24h",
    "receiver_ip_risk_mean_24h", "receiver_new_payee_share_24h",
]
assert len(V3_SAFE_DROP) == 17


class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def get_safe_path(path):
    """Never overwrite an existing file."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(f"{base}_run{k}{ext}"):
        k += 1
    return f"{base}_run{k}{ext}"


def fast_confusion_f1(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.bincount(y_true * num_classes + y_pred,
                     minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1e-12), 0.0)
    return float(f1.mean()), f1


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def adjust_legit(P, frac):
    """Undo legit downsampling: multiply the legit column by frac and renormalize rows."""
    if frac >= 1.0:
        return P
    P = P.copy()
    P[:, LEGIT_IDX] *= frac
    P /= P.sum(axis=1, keepdims=True)
    return P


def legit_keep_idx(y, frac, seed):
    """Indices to keep when downsampling legit rows to `frac` (all fraud rows kept), sorted."""
    idx_legit = np.where(y == LEGIT_IDX)[0]
    idx_fraud = np.where(y != LEGIT_IDX)[0]
    rng = np.random.default_rng(seed)
    kept = rng.choice(idx_legit, size=int(len(idx_legit) * frac), replace=False)
    return np.sort(np.concatenate([idx_fraud, kept]))


def optimize_multipliers_3pass(P, y):
    """Same tuner as train_v3.py: legit fixed at 1.0, 2 coarse log-grid passes + 1 fine pass."""
    w = np.ones(NUM_CLASSES, dtype=np.float64)
    best, _ = fast_confusion_f1(y, np.argmax(P, axis=1))
    coarse = np.sort(np.append(np.geomspace(0.3, 30.0, 93), 1.0))
    for _ in range(2):
        for c in range(NUM_CLASSES):
            if c == LEGIT_IDX:
                continue
            best_w = w[c]
            for cand in coarse:
                w[c] = cand
                s, _ = fast_confusion_f1(y, np.argmax(P * w, axis=1))
                if s > best + 1e-7:
                    best, best_w = s, cand
            w[c] = best_w
    fine_global = np.geomspace(0.3, 30.0, 201)
    for c in range(NUM_CLASSES):
        if c == LEGIT_IDX:
            continue
        ctr = w[c]
        local = np.geomspace(max(0.3, ctr * 0.65), min(30.0, ctr * 1.50), 121)
        grid = np.unique(np.clip(np.concatenate([fine_global, local, [1.0, ctr]]), 0.3, 30.0))
        best_w = w[c]
        for cand in grid:
            w[c] = cand
            s, _ = fast_confusion_f1(y, np.argmax(P * w, axis=1))
            if s > best + 1e-7:
                best, best_w = s, cand
        w[c] = best_w
    w[LEGIT_IDX] = 1.0
    return w


def append_to_results_csv(run_name, is_sample, features_path, drop_list, best_iter,
                          raw_f1, tune_f1, eval_f1, full_f1, per_class_f1, params, w):
    if os.path.exists("results.csv"):
        shutil.copy("results.csv", get_safe_path("results_backup.csv"))
    row = {
        "timestamp_ist": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": run_name, "sample_mode": is_sample, "features_file": features_path,
        "dropped_cols": ",".join(drop_list) if drop_list else "NONE",
        "best_iteration": int(best_iter),
        "val_raw_macro_f1": round(raw_f1, 5),
        "val_tune_half_macro_f1": round(tune_f1, 5),
        "val_eval_half_macro_f1": round(eval_f1, 5),
        "val_full_tuned_macro_f1": round(full_f1, 5),
    }
    for i, c in enumerate(CLASSES):
        row[f"f1_{c}"] = round(float(per_class_f1[i]), 5)
    row["multipliers"] = json.dumps({CLASSES[i]: round(float(w[i]), 4) for i in range(NUM_CLASSES)})
    p = dict(params)
    p["best_f1_iteration"] = int(best_iter)
    row["params"] = json.dumps(p)
    new = pd.DataFrame([row])
    if os.path.exists("results.csv"):
        new = pd.concat([pd.read_csv("results.csv"), new], ignore_index=True)
    new.to_csv("results.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="features_v2.parquet")
    ap.add_argument("--drop", default="", help="comma-separated extra columns to drop (on top of v3_safe)")
    ap.add_argument("--no-ip", action="store_true", help="also drop every column containing 'ip_risk'")
    ap.add_argument("--params", default="{}", help="JSON of LightGBM param overrides")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--legit_frac", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--threads", type=int, default=None,
                    help="default: physical cores. 0 = let LightGBM decide (use for both train and predict)")
    ap.add_argument("--sample", action="store_true", help="20%% of train rows (smoke test)")
    ap.add_argument("--undrop", default="",
                    help="comma-separated v3_safe-dropped columns to put back (e.g. on density-matched features)")
    ap.add_argument("--val_features", default="",
                    help="diagnostic: take validation rows' features from this parquet (same row order)")
    args = ap.parse_args()

    tag = args.name
    if args.sample and not tag.endswith("_sample"):
        tag += "_sample"
    logger = TeeLogger(get_safe_path(f"diagnostics_{tag}.txt"))
    sys.stdout = logger
    try:
        t_total = time.time()
        timings = {}

        if args.threads is not None:
            n_threads = args.threads
        elif psutil is not None and psutil.cpu_count(logical=False):
            n_threads = psutil.cpu_count(logical=False)
        else:
            n_threads = max(1, (os.cpu_count() or 2) // 2)
        print("=" * 100)
        print(f"TRAIN_MEMBER.PY RUN: {tag} | seed {args.seed} | legit_frac {args.legit_frac} | "
              f"rounds {args.rounds} | threads {n_threads} (used for train AND predict)")
        print("=" * 100)

        # ---- features -------------------------------------------------------
        t0 = time.time()
        df = pd.read_parquet(args.features)
        print(f"Loaded {args.features}: {len(df):,} rows x {df.shape[1]} cols in {time.time() - t0:.1f}s")

        extra = [c.strip() for c in args.drop.replace("\\", "").split(",") if c.strip()]
        if args.no_ip:
            extra += [c for c in df.columns if "ip_risk" in c.lower()]
        undrop = {c.strip() for c in args.undrop.split(",") if c.strip()}
        bad_undrop = sorted(undrop - set(V3_SAFE_DROP))
        assert not bad_undrop, f"--undrop columns are not in the v3_safe drop list: {bad_undrop}"
        safe_drop = [c for c in V3_SAFE_DROP if c not in undrop]
        if undrop:
            print(f"Re-enabled (--undrop) {len(undrop)} v3_safe columns: {sorted(undrop)}")
        extra = sorted(set(extra) - set(safe_drop))
        unknown = [c for c in extra if c not in df.columns]
        if unknown:
            print(f"[WARN] extra drop columns not in parquet: {unknown}")
        drop_all = set(safe_drop) | set(extra)

        def keep(c):
            return (c not in EXCLUDED_COLS and c not in drop_all
                    and not c.endswith(("_opened_on", "_onboarded_on", "_key"))
                    and "datetime" not in str(df[c].dtype))
        feature_cols = [c for c in df.columns if keep(c)]
        base_cols = [c for c in df.columns
                     if c not in EXCLUDED_COLS and c not in V3_SAFE_DROP
                     and not c.endswith(("_opened_on", "_onboarded_on", "_key"))
                     and "datetime" not in str(df[c].dtype)]
        leak = [c for c in feature_cols
                if c.endswith("_tgt") or c in ("target", "label", "txn_id", "sender_id", "receiver_id", "device_id")]
        assert not leak, f"LEAKAGE: forbidden columns in features: {leak}"
        print(f"v3_safe base features: {len(base_cols)} (expected 108)")
        print(f"Extra drops beyond v3_safe ({len(extra)}): {extra if extra else 'NONE'}")
        print(f"Final feature count: {len(feature_cols)} | ip_risk columns still present: "
              f"{[c for c in feature_cols if 'ip_risk' in c.lower()] or 'NONE'}")
        if len(base_cols) != 108:
            print("[WARN] v3_safe base feature count is not 108 - check the parquet / drop list!")

        # ---- data -----------------------------------------------------------
        train_df = df[df["is_test"] == 0].copy()
        test_df = df[df["is_test"] == 1].copy()
        del df
        train_df["target"] = train_df["label"].map(LABEL_TO_ID).astype(np.int32)
        if args.sample:
            print("[--sample] using 20% of train rows (fixed sampling seed 42; full test set kept)")
            train_df = (train_df.sample(frac=0.20, random_state=42)
                        .sort_values(["timestamp", "txn_id"]).reset_index(drop=True))
        val_mask = (train_df["timestamp"] >= pd.Timestamp("2026-09-01")).to_numpy()

        X_all_full = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_all_full = train_df["target"].to_numpy(dtype=np.int32)
        X_te = test_df[feature_cols].to_numpy(dtype=np.float32)
        X_va, y_va = X_all_full[val_mask], y_all_full[val_mask]
        X_tr, y_tr = X_all_full[~val_mask], y_all_full[~val_mask]
        if args.val_features:
            # cross-density check: train on --features, validate on the same rows from --val_features
            alt = pd.read_parquet(args.val_features, columns=["txn_id", "is_test"] + feature_cols)
            alt = alt[alt["is_test"] == 0]
            assert not args.sample, "--val_features is not supported with --sample"
            assert np.array_equal(alt["txn_id"].to_numpy(), train_df["txn_id"].to_numpy()), \
                "row order of --val_features differs from --features"
            X_va = alt[feature_cols].to_numpy(dtype=np.float32)[val_mask]
            del alt
            print(f"[--val_features] validation X taken from {args.val_features} (test X unchanged)")
        if args.legit_frac < 1.0:
            k = legit_keep_idx(y_tr, args.legit_frac, args.seed)
            print(f"Downsampled legit in TRAIN part to {args.legit_frac}: {len(y_tr):,} -> {len(k):,} rows")
            X_tr, y_tr = X_tr[k], y_tr[k]
        print(f"Split -> train (Jul-Aug) {len(X_tr):,} | validation (Sep+) {len(X_va):,} | test {len(X_te):,}")

        # ---- params ---------------------------------------------------------
        params = {
            "objective": "multiclass", "num_class": NUM_CLASSES, "metric": "multi_logloss",
            "boosting_type": "gbdt", "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 300, "feature_fraction": 0.5, "bagging_fraction": 0.7,
            "bagging_freq": 1, "lambda_l2": 10.0,
        }
        params.update(json.loads(args.params.replace('\\"', '"')))
        params.update({"num_threads": n_threads, "seed": args.seed, "verbose": -1})
        print("LightGBM params:\n" + json.dumps(params, indent=2))

        # ---- Phase 1: train validation model (no valid_sets, no callbacks) ---
        t0 = time.time()
        booster = lgb.train(params, lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols),
                            num_boost_round=args.rounds)
        timings["Phase 1 train (val model)"] = time.time() - t0
        print(f"\nPhase 1: trained {args.rounds} rounds in {timings['Phase 1 train (val model)']:.1f}s "
              f"({timings['Phase 1 train (val model)'] / args.rounds * 25:.1f}s per 25 rounds)")

        # ---- Phase 2: checkpoint evaluation via cumulative raw scores --------
        t0 = time.time()
        ckpts = list(range(args.eval_every, args.rounds + 1, args.eval_every))
        if ckpts[-1] != args.rounds:
            ckpts.append(args.rounds)
        raw = np.zeros((len(X_va), NUM_CLASSES), dtype=np.float64)
        prev, best_f1, best_iter, P_best = 0, -1.0, ckpts[0], None
        for k in ckpts:
            tc = time.time()
            raw += booster.predict(X_va, start_iteration=prev, num_iteration=k - prev,
                                   raw_score=True, num_threads=n_threads)
            prev = k
            P_k = adjust_legit(softmax(raw), args.legit_frac)
            f1, _ = fast_confusion_f1(y_va, np.argmax(P_k, axis=1))
            star = ""
            if f1 > best_f1:
                best_f1, best_iter, P_best, star = f1, k, P_k, "  *BEST*"
            print(f"  Iter {k:3d} | val_raw_macro_f1 {f1:.5f} | eval {time.time() - tc:5.1f}s{star}")
        if args.sample:  # smoke-test the cumulative trick against a direct predict
            P_dir = booster.predict(X_va, num_iteration=args.rounds, num_threads=n_threads)
            assert np.allclose(P_dir, softmax(raw), atol=1e-4), "cumulative raw-score eval mismatch!"
            print("  [check] cumulative-score eval matches direct predict")
        timings["Phase 2 checkpoint eval"] = time.time() - t0

        # ---- Phase 3: multipliers (legit fixed at 1.0) ----------------------
        t0 = time.time()
        P_va = P_best
        perm = np.random.default_rng(42).permutation(len(y_va))
        idx_t, idx_e = perm[:len(y_va) // 2], perm[len(y_va) // 2:]
        w = optimize_multipliers_3pass(P_va[idx_t], y_va[idx_t])
        raw_f1, _ = fast_confusion_f1(y_va, np.argmax(P_va, axis=1))
        tune_f1, _ = fast_confusion_f1(y_va[idx_t], np.argmax(P_va[idx_t] * w, axis=1))
        eval_f1, eval_pc = fast_confusion_f1(y_va[idx_e], np.argmax(P_va[idx_e] * w, axis=1))
        full_f1, full_pc = fast_confusion_f1(y_va, np.argmax(P_va * w, axis=1))
        timings["Phase 3 multiplier tuning"] = time.time() - t0
        print(f"\nBest iteration {best_iter} | raw {raw_f1:.5f} | tune-half {tune_f1:.5f} | "
              f"EVAL-HALF {eval_f1:.5f} | FULL tuned {full_f1:.5f}")
        print("Multipliers: " + ", ".join(f"{CLASSES[i]}={w[i]:.2f}" for i in range(NUM_CLASSES) if i != LEGIT_IDX))
        print("Per-class tuned F1 (full val): " + ", ".join(f"{CLASSES[i]}={full_pc[i]:.3f}" for i in range(NUM_CLASSES)))

        # ---- Phase 4: final retrain on train+val, predict test --------------
        t0 = time.time()
        final_trees = max(1, int(round(best_iter * 1.2)))
        X_fit, y_fit = X_all_full, y_all_full
        if args.legit_frac < 1.0:
            k = legit_keep_idx(y_fit, args.legit_frac, args.seed + 1)
            X_fit, y_fit = X_fit[k], y_fit[k]
        print(f"\nPhase 4: final model on train+val ({len(y_fit):,} rows), {final_trees} trees...")
        final = lgb.train(params, lgb.Dataset(X_fit, label=y_fit, feature_name=feature_cols),
                          num_boost_round=final_trees)
        P_test = adjust_legit(final.predict(X_te, num_iteration=final_trees, num_threads=n_threads),
                              args.legit_frac)
        if args.legit_frac < 1.0:
            print(f"[legit_frac] multiplied legit column by {args.legit_frac} and renormalized rows "
                  f"(validation AND test probabilities)")
        timings["Phase 4 final retrain+predict"] = time.time() - t0

        # ---- save -----------------------------------------------------------
        if not args.sample:
            if os.path.exists("val_labels.npy"):
                assert np.array_equal(np.load("val_labels.npy"), y_va), \
                    "validation labels differ from val_labels.npy -> row order mismatch, do NOT blend"
            if os.path.exists("val_probs_v3_safe.npy"):
                assert np.load("val_probs_v3_safe.npy", mmap_mode="r").shape == P_va.shape
        vf = get_safe_path(f"val_probs_{tag}.npy")
        tf = get_safe_path(f"test_probs_{tag}.npy")
        np.save(vf, P_va.astype(np.float32))
        np.save(tf, P_test.astype(np.float32))
        print(f"Saved {vf} {P_va.shape} and {tf} {P_test.shape}")
        append_to_results_csv(tag, args.sample, args.features, sorted(drop_all), best_iter,
                              raw_f1, tune_f1, eval_f1, full_f1, eval_pc, params, w)

        print("\n" + "=" * 60 + "\nPER-PHASE TIMING\n" + "=" * 60)
        for k_, v_ in timings.items():
            print(f"  {k_:<32s}: {v_:8.1f}s")
        print(f"  {'TOTAL':<32s}: {time.time() - t_total:8.1f}s")
    finally:
        sys.stdout = sys.__stdout__
        logger.close()


if __name__ == "__main__":
    main()