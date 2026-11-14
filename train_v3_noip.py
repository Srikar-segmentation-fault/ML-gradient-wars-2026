import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

CLASSES = [
    "account_takeover", "fake_cashback", "layering_chain", "legit",
    "loan_app_extortion", "merchant_collusion", "mule_account",
    "phishing_collect", "qr_swap", "refund_abuse", "sim_swap", "velocity_bot"
]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}
LEGIT_IDX = LABEL_TO_ID["legit"]

EXCLUDED_COLS = {
    "txn_id", "sender_id", "receiver_id", "device_id", "is_test", "copy",
    "timestamp", "label", "target", "app_version", "opened_on",
    "sender_opened_on", "receiver_opened_on", "onboarded_on",
    "merchant_onboarded_on", "inflow_ts_key", "outflow_ts_key",
    "rev_bwd_key", "rev_fwd_key", "val_idx", "samp_idx"
}

V3_SAFE_DROP_LIST = [
    "sender_past_txn_count", "sender_past_amount_sum", "sender_sec_since_first_seen",
    "sender_amount_expanding_zscore", "sender_past_hour_sin_mean", "sender_past_hour_cos_mean",
    "sender_hour_sin_cos_dist", "sender_city_past_share", "receiver_past_txn_count",
    "receiver_past_amount_sum", "receiver_sec_since_first_seen", "device_past_txn_count",
    "device_past_amount_sum", "device_sec_since_first_seen",
    "day_of_month", "time_since_sender_last_inflow_sec", "sender_prev_txn_sec"
]

class TeeLogger:
    def __init__(self, filename: str):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

def get_safe_path(filepath: str) -> str:
    if not os.path.exists(filepath): return filepath
    base, ext = os.path.splitext(filepath)
    k = 2
    while os.path.exists(f"{base}_run{k}{ext}"):
        k += 1
    return f"{base}_run{k}{ext}"

def fast_confusion_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 12):
    cm = np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 2.0 * tp + fp + fn
        f1_per_class = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    return float(np.mean(f1_per_class)), f1_per_class, cm

def fast_multi_logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))

def optimize_multipliers_3pass(P_tune: np.ndarray, y_tune: np.ndarray, num_classes: int = 12):
    w = np.ones(num_classes, dtype=np.float64)
    best_score, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune, axis=1), num_classes)
    coarse_log_grid = np.geomspace(0.3, 30.0, 93)
    if 1.0 not in coarse_log_grid: coarse_log_grid = np.sort(np.append(coarse_log_grid, 1.0))

    for pass_idx in [1, 2]:
        for c in range(num_classes):
            if c == LEGIT_IDX: continue
            best_c_w = w[c]
            for cand in coarse_log_grid:
                w[c] = cand
                score, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune * w, axis=1), num_classes)
                if score > best_score + 1e-7:
                    best_score = score
                    best_c_w = cand
            w[c] = best_c_w

    fine_global_grid = np.geomspace(0.3, 30.0, 201)
    for c in range(num_classes):
        if c == LEGIT_IDX: continue
        center = w[c]
        low, high = max(0.3, center * 0.65), min(30.0, center * 1.50)
        local_fine = np.geomspace(low, high, 121)
        cand_grid = np.unique(np.clip(np.concatenate([fine_global_grid, local_fine, [1.0, center]]), 0.3, 30.0))
        best_c_w = w[c]
        for cand in cand_grid:
            w[c] = cand
            score, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune * w, axis=1), num_classes)
            if score > best_score + 1e-7:
                best_score = score
                best_c_w = cand
        w[c] = best_c_w
    return w

def append_run_to_results_csv(run_name, is_sample, features_path, drop_list, best_iter, val_raw_f1, val_tune_f1, val_eval_f1, val_full_f1, per_class_f1, params, multipliers):
    row = {
        "timestamp_ist": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": run_name,
        "sample_mode": is_sample,
        "features_file": features_path,
        "dropped_cols": ",".join(drop_list) if drop_list else "NONE",
        "best_iteration": int(best_iter),
        "val_raw_macro_f1": round(val_raw_f1, 5),
        "val_tune_half_macro_f1": round(val_tune_f1, 5),
        "val_eval_half_macro_f1": round(val_eval_f1, 5),
        "val_full_tuned_macro_f1": round(val_full_f1, 5),
    }
    for i, c in enumerate(CLASSES): row[f"f1_{c}"] = round(float(per_class_f1[i]), 5)
    row["multipliers"] = json.dumps({CLASSES[i]: round(float(multipliers[i]), 4) for i in range(len(CLASSES))})
    params_logged = dict(params)
    params_logged["best_f1_iteration"] = int(best_iter)
    params_logged["features_file"] = features_path
    params_logged["dropped_cols"] = drop_list
    row["params"] = json.dumps(params_logged)

    res_df = pd.DataFrame([row])
    if os.path.exists("results.csv"):
        existing_df = pd.read_csv("results.csv")
        combined_df = pd.concat([existing_df, res_df], ignore_index=True)
        combined_df.to_csv("results.csv", index=False)
    else:
        res_df.to_csv("results.csv", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="features_v2.parquet")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--name", type=str, default="v3_noip")
    args = parser.parse_args()

    run_tag = f"{args.name}{'_sample' if args.sample else ''}"
    diag_file = get_safe_path(f"diagnostics_{run_tag}.txt")
    logger = TeeLogger(diag_file)
    sys.stdout = logger

    try:
        t_total = time.time()
        print("=" * 105)
        print(f"TRAIN_V3_NOIP.PY RUN: {run_tag}")
        print(f"Features File : {args.features}")
        print("=" * 105)

        t_stage = time.time()
        df = pd.read_parquet(args.features)
        
        # 1. Identify IP Risk columns dynamically
        ip_risk_cols = [c for c in df.columns if "ip_risk" in c.lower()]
        print("\nIdentified columns containing 'ip_risk':")
        for c in ip_risk_cols:
            print(f"  - {c}")
            
        target_leak_cols = [c for c in df.columns if c.endswith("_tgt") or c == "target"]

        # 2. Build final drop list
        full_drop_list = set(V3_SAFE_DROP_LIST) | set(ip_risk_cols) | set(target_leak_cols) | EXCLUDED_COLS
        
        feature_cols = [
            c for c in df.columns
            if c not in full_drop_list
            and not c.endswith("_opened_on")
            and not c.endswith("_onboarded_on")
            and not c.endswith("_key")
            and "datetime" not in str(df[c].dtype)
        ]
        
        print(f"\nFinal Drop List contains {len(full_drop_list)} distinct exclusions.")
        print(f"Final active feature count: {len(feature_cols)} features")

        # FIX: Split test set off BEFORE mapping target labels to avoid IntCastingNaNError
        train_df = df[df["is_test"] == 0].copy()
        test_df = df[df["is_test"] == 1].copy()
        del df
        
        train_df["target"] = train_df["label"].map(LABEL_TO_ID).astype(np.int32)

        if args.sample:
            print("\n[--sample MODE ACTIVE] Sampling 20% of train_df...")
            train_df = train_df.sample(frac=0.20, random_state=42).sort_values("timestamp").reset_index(drop=True)

        val_mask = train_df["timestamp"] >= pd.Timestamp("2026-09-01")
        tr_data = train_df[~val_mask]
        va_data = train_df[val_mask]

        X_tr = tr_data[feature_cols].to_numpy(dtype=np.float32)
        y_tr = tr_data["target"].to_numpy(dtype=np.int32)
        X_va = va_data[feature_cols].to_numpy(dtype=np.float32)
        y_va = va_data["target"].to_numpy(dtype=np.int32)

        print(f"Split sizes -> Train (Jul-Aug): {len(X_tr):,} rows | Validation (Sep+): {len(X_va):,} rows")

        # ---------------------------------------------------------------------
        # Train LightGBM
        # ---------------------------------------------------------------------
        total_rounds = 250
        lgb_params = {
            "objective": "multiclass", "num_class": 12, "metric": "multi_logloss",
            "boosting_type": "gbdt", "learning_rate": 0.05, "num_leaves": 31,
            "min_child_samples": 300, "feature_fraction": 0.5, "bagging_fraction": 0.7,
            "bagging_freq": 1, "lambda_l2": 10.0, "seed": 42, "n_jobs": args.threads, "verbose": -1,
        }

        print(f"\nTraining LightGBM ({total_rounds} Rounds)...")
        rng = np.random.default_rng(42)
        sub_size = min(100_000, len(X_tr))
        tr_sub_idx = rng.choice(len(X_tr), size=sub_size, replace=False)
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)

        best_val_f1 = -1.0
        best_f1_iter = 25

        def eval_every_25_cb(env):
            nonlocal best_val_f1, best_f1_iter
            it = env.iteration + 1
            if it % 25 == 0 or it == 1 or it == total_rounds:
                p_val = env.model.predict(X_va, num_iteration=it)
                val_f1, _, _ = fast_confusion_f1(y_va, np.argmax(p_val, axis=1), 12)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_f1_iter = it
                    
                print(f"  Iter {it:3d} | val_raw_macro_f1: {val_f1:.5f} {' *BEST*' if val_f1 == best_val_f1 else ''}")

        booster = lgb.train(lgb_params, dtrain, num_boost_round=total_rounds, valid_sets=[dval], callbacks=[eval_every_25_cb])

        # ---------------------------------------------------------------------
        # Optimize Multipliers
        # ---------------------------------------------------------------------
        print("\nOptimizing multipliers on validation...")
        P_va = booster.predict(X_va, num_iteration=best_f1_iter)
        
        # FIXED: Only unpack 3 variables from fast_confusion_f1
        val_raw_f1, _, _ = fast_confusion_f1(y_va, np.argmax(P_va, axis=1), 12)

        perm = rng.permutation(len(y_va))
        idx_tune, idx_eval = perm[:len(y_va)//2], perm[len(y_va)//2:]

        P_tune, y_tune = P_va[idx_tune], y_va[idx_tune]
        P_eval, y_eval = P_va[idx_eval], y_va[idx_eval]

        w_opt = optimize_multipliers_3pass(P_tune, y_tune)
        
        val_tune_f1, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune * w_opt, axis=1), 12)
        val_eval_f1, eval_per_class_f1, _ = fast_confusion_f1(y_eval, np.argmax(P_eval * w_opt, axis=1), 12)
        val_full_f1, _, _ = fast_confusion_f1(y_va, np.argmax(P_va * w_opt, axis=1), 12)

        print(f"Val Full Macro-F1 (Tuned): {val_full_f1:.5f}")

        # ---------------------------------------------------------------------
        # Retrain & Predict Test
        # ---------------------------------------------------------------------
        final_trees = max(1, int(round(best_f1_iter * 1.2)))
        print(f"\nRetraining Final Model for {final_trees} trees...")
        
        X_all = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_all = train_df["target"].to_numpy(dtype=np.int32)
        dtrain_all = lgb.Dataset(X_all, label=y_all, feature_name=feature_cols)

        final_booster = lgb.train(lgb_params, dtrain_all, num_boost_round=final_trees)
        
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        P_test = final_booster.predict(X_test, num_iteration=final_trees)

        # ---------------------------------------------------------------------
        # Save Output
        # ---------------------------------------------------------------------
        val_prob_file = get_safe_path(f"val_probs_{run_tag}.npy")
        test_prob_file = get_safe_path(f"test_probs_{run_tag}.npy")
        sub_file = get_safe_path(f"submission_{run_tag}.csv")
        
        np.save(val_prob_file, P_va.astype(np.float32))
        np.save(test_prob_file, P_test.astype(np.float32))
        
        sub = pd.DataFrame({
            "id": test_df["txn_id"].to_numpy(),
            "label": [ID_TO_LABEL[i] for i in np.argmax(P_test * w_opt, axis=1)]
        })
        sub.to_csv(sub_file, index=False)
        print(f"\nSaved {val_prob_file}, {test_prob_file}, {sub_file}")

        append_run_to_results_csv(
            run_name=args.name, is_sample=args.sample, features_path=args.features,
            drop_list=list(set(ip_risk_cols) | set(V3_SAFE_DROP_LIST)),
            best_iter=best_f1_iter, val_raw_f1=val_raw_f1, val_tune_f1=val_tune_f1,
            val_eval_f1=val_eval_f1, val_full_f1=val_full_f1, per_class_f1=eval_per_class_f1,
            params=lgb_params, multipliers=w_opt
        )

        print(f"\nCompleted train_v3_noip.py in {time.time() - t_total:.2f}s.")
    finally:
        sys.stdout = sys.__stdout__
        logger.close()

if __name__ == "__main__":
    main()