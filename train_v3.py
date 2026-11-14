import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

# All 12 competition classes in deterministic alphabetical order
CLASSES = [
    "account_takeover",
    "fake_cashback",
    "layering_chain",
    "legit",
    "loan_app_extortion",
    "merchant_collusion",
    "mule_account",
    "phishing_collect",
    "qr_swap",
    "refund_abuse",
    "sim_swap",
    "velocity_bot",
]
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}
LEGIT_IDX = LABEL_TO_ID["legit"]

EXCLUDED_COLS = {
    "txn_id",
    "sender_id",
    "receiver_id",
    "device_id",
    "is_test",
    "timestamp",
    "label",
    "target",
    "app_version",
    "opened_on",
    "sender_opened_on",
    "receiver_opened_on",
    "onboarded_on",
    "merchant_onboarded_on",
    "inflow_ts_key",
    "outflow_ts_key",
    "rev_bwd_key",
    "rev_fwd_key",
}


class TeeLogger:
    """Writes all stdout to both console and diagnostics_<name>.txt."""
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
    """Prevents overwriting any existing submission or probability files."""
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    k = 2
    while os.path.exists(f"{base}_run{k}{ext}"):
        k += 1
    return f"{base}_run{k}{ext}"


def fast_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 12):
    """Computes Macro-F1, per-class F1, Precision, Recall, Accuracy, and Confusion Matrix via np.bincount."""
    cm = np.bincount(
        y_true * num_classes + y_pred,
        minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        rec = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        denom = 2.0 * tp + fp + fn
        f1_per_class = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
        acc = float(tp.sum() / cm.sum()) if cm.sum() > 0 else 0.0

    return float(np.mean(f1_per_class)), f1_per_class, prec, rec, acc, cm


def fast_confusion_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 12):
    macro_f1, f1_per_class, _, _, _, cm = fast_confusion_metrics(y_true, y_pred, num_classes)
    return macro_f1, f1_per_class, cm


def fast_multi_logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def optimize_multipliers_3pass(P_tune: np.ndarray, y_tune: np.ndarray, num_classes: int = 12):
    """
    Requirement 3:
    - legit fixed at 1.0
    - other classes searched on a log grid [0.3, 30.0]
    - 3 passes with a finer local log grid on the 3rd pass
    - reports any non-legit multiplier left at exactly 1.0
    """
    w = np.ones(num_classes, dtype=np.float64)
    best_score, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune, axis=1), num_classes)

    coarse_log_grid = np.geomspace(0.3, 30.0, 93)
    if 1.0 not in coarse_log_grid:
        coarse_log_grid = np.sort(np.append(coarse_log_grid, 1.0))

    # Passes 1 & 2: Full log grid [0.3, 30.0]
    for pass_idx in [1, 2]:
        t_p = time.time()
        for c in range(num_classes):
            if c == LEGIT_IDX:
                w[c] = 1.0
                continue
            best_c_w = w[c]
            for cand in coarse_log_grid:
                w[c] = cand
                preds = np.argmax(P_tune * w, axis=1)
                score, _, _ = fast_confusion_f1(y_tune, preds, num_classes)
                if score > best_score + 1e-7:
                    best_score = score
                    best_c_w = cand
            w[c] = best_c_w
        print(f"  Multiplier Pass {pass_idx}/3 (coarse log grid) -> Tune Macro-F1: {best_score:.5f} ({time.time() - t_p:.2f}s)")

    # Pass 3: Finer log grid (201 points globally + dense local neighborhood around current w[c])
    t_p3 = time.time()
    fine_global_grid = np.geomspace(0.3, 30.0, 201)
    for c in range(num_classes):
        if c == LEGIT_IDX:
            w[c] = 1.0
            continue
        center = w[c]
        low = max(0.3, center * 0.65)
        high = min(30.0, center * 1.50)
        local_fine = np.geomspace(low, high, 121)
        cand_grid = np.unique(np.clip(np.concatenate([fine_global_grid, local_fine, [1.0, center]]), 0.3, 30.0))

        best_c_w = w[c]
        for cand in cand_grid:
            w[c] = cand
            preds = np.argmax(P_tune * w, axis=1)
            score, _, _ = fast_confusion_f1(y_tune, preds, num_classes)
            if score > best_score + 1e-7:
                best_score = score
                best_c_w = cand
        w[c] = best_c_w
    print(f"  Multiplier Pass 3/3 (fine log grid)   -> Tune Macro-F1: {best_score:.5f} ({time.time() - t_p3:.2f}s)")

    left_at_one = [
        CLASSES[c] for c in range(num_classes)
        if c != LEGIT_IDX and np.isclose(w[c], 1.0, atol=1e-6)
    ]
    if left_at_one:
        print(f"  [REPORT] Non-legit multipliers left at exactly 1.0: {left_at_one}")
    else:
        print("  [REPORT] Non-legit multipliers left at exactly 1.0: None (all 11 fraud classes moved off 1.0)")

    return w, left_at_one


def append_run_to_results_csv(
    run_name, is_sample, features_path, drop_list, best_iter,
    val_raw_f1, val_tune_f1, val_eval_f1, val_full_f1, per_class_f1, params, multipliers
):
    """Logs run to results.csv (including feature file and drop list) without overwriting existing rows."""
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
    for i, c in enumerate(CLASSES):
        row[f"f1_{c}"] = round(float(per_class_f1[i]), 5)
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
    print(f"\nAppended run '{run_name}' (features={features_path}, dropped={len(drop_list)}) to results.csv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="features_v2.parquet", help="Path to features parquet file")
    parser.add_argument("--drop", type=str, default="", help="Comma-separated list of columns to drop")
    parser.add_argument("--threads", type=int, default=-1, help="Number of LightGBM threads (default -1)")
    parser.add_argument("--sample", action="store_true", help="Use 20% of train")
    parser.add_argument("--name", type=str, default="lgbm_v3", help="Run name for logs and saved files")
    args = parser.parse_args()

    run_tag = f"{args.name}{'_sample' if args.sample else ''}"
    diag_file = get_safe_path(f"diagnostics_{run_tag}.txt")
    logger = TeeLogger(diag_file)
    sys.stdout = logger

    try:
        t_total = time.time()
        print("=" * 90)
        print(f"TRAIN_V3.PY RUN: {run_tag}")
        print(f"Features File : {args.features}")
        print(f"Threads       : {args.threads} | Sample Mode: {args.sample}")
        print(f"Log File      : {diag_file}")
        print("=" * 90)

        # ---------------------------------------------------------------------
        # STAGE 1: Load Parquet & Apply Exclusions + --drop List
        # ---------------------------------------------------------------------
        t_stage = time.time()
        print(f"\n[Stage 1] Reading {args.features}...")
        df = pd.read_parquet(args.features)
        print(f"Loaded {len(df):,} rows x {df.shape[1]} cols in {time.time() - t_stage:.2f}s")

        user_drop_cols = [c.strip() for c in args.drop.split(",") if c.strip()] if args.drop else []
        user_drop_set = set(user_drop_cols)

        feature_cols = [
            c for c in df.columns
            if c not in EXCLUDED_COLS
            and c not in user_drop_set
            and not c.endswith("_opened_on")
            and not c.endswith("_onboarded_on")
            and not c.endswith("_key")
            and "datetime" not in str(df[c].dtype)
        ]
        assert not any("datetime" in str(df[c].dtype) for c in feature_cols), "Datetime column leaked into feature_cols!"

        print(f"User --drop list ({len(user_drop_cols)} requested): {user_drop_cols if user_drop_cols else 'None'}")
        print(f"Final active feature count: {len(feature_cols)} features")

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
        val_counts = np.bincount(y_va, minlength=len(CLASSES))

        # ---------------------------------------------------------------------
        # STAGE 2: Train LightGBM (250 Rounds at lr=0.05, Eval Every 25 Rounds)
        # ---------------------------------------------------------------------
        t_stage = time.time()
        total_rounds = 250
        lgb_params = {
            "objective": "multiclass",
            "num_class": 12,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 300,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,
            "lambda_l2": 10.0,
            "seed": 42,
            "n_jobs": args.threads,
            "verbose": -1,
        }

        print("\n" + "=" * 90)
        print(f"[Stage 2] Training LightGBM ({total_rounds} Rounds, lr=0.05, Evaluating Every 25 Rounds)")
        print("=" * 90)

        rng = np.random.default_rng(42)
        sub_size = min(100_000, len(X_tr))
        tr_sub_idx = rng.choice(len(X_tr), size=sub_size, replace=False)
        X_tr_sub = X_tr[tr_sub_idx]
        y_tr_sub = y_tr[tr_sub_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, free_raw_data=False)

        history_rows = []
        best_val_f1 = -1.0
        best_f1_iter = 25

        def eval_every_25_cb(env):
            nonlocal best_val_f1, best_f1_iter
            it = env.iteration + 1
            if it % 25 == 0 or it == 1 or it == total_rounds:
                val_ll = np.nan
                if env.evaluation_result_list:
                    for item in env.evaluation_result_list:
                        if item[0] == "valid" and item[1] == "multi_logloss":
                            val_ll = float(item[2])

                p_val = env.model.predict(X_va, num_iteration=it)
                if np.isnan(val_ll):
                    val_ll = fast_multi_logloss(y_va, p_val)
                val_f1, _, _ = fast_confusion_f1(y_va, np.argmax(p_val, axis=1), 12)

                p_tr = env.model.predict(X_tr_sub, num_iteration=it)
                tr_f1, _, _ = fast_confusion_f1(y_tr_sub, np.argmax(p_tr, axis=1), 12)

                marker = ""
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_f1_iter = it
                    marker = " *BEST*"

                print(
                    f"  Iter {it:3d} | val_logloss: {val_ll:.6f} | "
                    f"val_raw_macro_f1: {val_f1:.5f} | train_100k_macro_f1: {tr_f1:.5f}{marker}"
                )
                history_rows.append({
                    "iteration": it,
                    "val_logloss": round(val_ll, 6),
                    "val_raw_macro_f1": round(val_f1, 5),
                    "train_100k_macro_f1": round(tr_f1, 5),
                })

        booster = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=total_rounds,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[eval_every_25_cb]
        )
        print(f"\nCompleted Stage 2 Validation Training in {time.time() - t_stage:.2f}s")
        print(f"Best Validation Raw-Argmax Macro-F1: {best_val_f1:.5f} at Iteration {best_f1_iter}")

        # ---------------------------------------------------------------------
        # STAGE 3: 3-Pass Log-Grid Multiplier Optimization & Per-Class Metrics
        # ---------------------------------------------------------------------
        t_stage = time.time()
        print("\n" + "=" * 90)
        print("[Stage 3] Multiplier Optimization (legit=1.0, 3 Passes on Log Grid [0.3, 30.0])")
        print("=" * 90)

        P_va = booster.predict(X_va, num_iteration=best_f1_iter)
        raw_preds = np.argmax(P_va, axis=1)
        val_raw_f1, raw_per_class_f1, raw_prec, raw_rec, val_raw_acc, _ = fast_confusion_metrics(y_va, raw_preds)

        perm = rng.permutation(len(y_va))
        half = len(y_va) // 2
        idx_tune, idx_eval = perm[:half], perm[half:]

        P_tune, y_tune = P_va[idx_tune], y_va[idx_tune]
        P_eval, y_eval = P_va[idx_eval], y_va[idx_eval]

        raw_tune_f1, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune, axis=1))
        raw_eval_f1, _, _ = fast_confusion_f1(y_eval, np.argmax(P_eval, axis=1))

        best_w, left_at_one = optimize_multipliers_3pass(P_tune, y_tune)

        tuned_tune_preds = np.argmax(P_tune * best_w, axis=1)
        tuned_eval_preds = np.argmax(P_eval * best_w, axis=1)
        tuned_full_preds = np.argmax(P_va * best_w, axis=1)

        val_tune_f1, _, _ = fast_confusion_f1(y_tune, tuned_tune_preds)
        val_eval_f1, eval_per_class_f1, eval_prec, eval_rec, eval_acc, _ = fast_confusion_metrics(y_eval, tuned_eval_preds)
        val_full_f1, full_per_class_f1, full_prec, full_rec, full_acc, _ = fast_confusion_metrics(y_va, tuned_full_preds)

        print(f"\nVal Tune Half (50%) Macro-F1 : {raw_tune_f1:.5f} (raw) -> {val_tune_f1:.5f} (tuned)")
        print(f"Val Eval Half (50%) Macro-F1 : {raw_eval_f1:.5f} (raw) -> {val_eval_f1:.5f} (OUT-OF-SAMPLE TUNED)")
        print(f"Val Full (100%)     Macro-F1 : {val_raw_f1:.5f} (raw) -> {val_full_f1:.5f} (tuned) | Acc: {full_acc:.5f}")
        print(f"Stage 3 Multiplier Optimization runtime: {time.time() - t_stage:.2f}s")

        # Requirement 5: Per-Class Precision, Recall, F1 Table
        print("\n" + "=" * 98)
        print("PER-CLASS PRECISION, RECALL & F1 TABLE (VALIDATION AT BEST ITERATION)")
        print("=" * 98)
        print(
            f"{'Class':<20s} | {'Support':>7s} | {'Mult':>6s} | {'Raw F1':>7s} | "
            f"{'FullPrec':>8s} | {'FullRec':>8s} | {'FullF1':>7s} | {'EvalHalfF1':>10s}"
        )
        print("-" * 98)
        for i, c in enumerate(CLASSES):
            print(
                f"{c:<20s} | {val_counts[i]:7,d} | {best_w[i]:6.2f} | {raw_per_class_f1[i]:7.4f} | "
                f"{full_prec[i]:8.4f} | {full_rec[i]:8.4f} | {full_per_class_f1[i]:7.4f} | {eval_per_class_f1[i]:10.4f}"
            )
        print("-" * 98)
        print(
            f"{'MACRO AVERAGE':<20s} | {len(y_va):7,d} | {'-':>6s} | {val_raw_f1:7.4f} | "
            f"{np.mean(full_prec):8.4f} | {np.mean(full_rec):8.4f} | {val_full_f1:7.4f} | {val_eval_f1:10.4f}"
        )

        # Requirement 5: Feature Importance Table
        gain_imp = booster.feature_importance(importance_type="gain", iteration=best_f1_iter)
        split_imp = booster.feature_importance(importance_type="split", iteration=best_f1_iter)
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "gain": gain_imp,
            "gain_pct": 100.0 * gain_imp / max(1e-9, np.sum(gain_imp)),
            "splits": split_imp,
        }).sort_values("gain", ascending=False).reset_index(drop=True)
        imp_df.index = imp_df.index + 1

        print("\n" + "=" * 90)
        print("FEATURE IMPORTANCE TABLE (TOP 45 BY GAIN AT BEST ITERATION)")
        print("=" * 90)
        print(imp_df.head(45).to_string())

        # ---------------------------------------------------------------------
        # STAGE 4: Final Retrain on 100% Train (best_f1_iter x 1.2) & Save Outputs
        # ---------------------------------------------------------------------
        t_stage = time.time()
        final_trees = max(1, int(round(best_f1_iter * 1.2)))
        print("\n" + "=" * 90)
        print(f"[Stage 4] Retraining Final Model on 100% Train ({len(train_df):,} rows) for {final_trees} trees...")
        print("=" * 90)

        X_all = train_df[feature_cols].to_numpy(dtype=np.float32)
        y_all = train_df["target"].to_numpy(dtype=np.int32)
        dtrain_all = lgb.Dataset(X_all, label=y_all, feature_name=feature_cols)

        final_booster = lgb.train(
            lgb_params,
            dtrain_all,
            num_boost_round=final_trees
        )
        print(f"Final model retrained in {time.time() - t_stage:.2f}s")

        t_pred = time.time()
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
        P_test = final_booster.predict(X_test, num_iteration=final_trees)
        test_pred_ids = np.argmax(P_test * best_w, axis=1)
        print(f"Predicted on test ({len(X_test):,} rows) in {time.time() - t_pred:.2f}s")

        print("\n" + "=" * 90)
        print("PREDICTED CLASS DISTRIBUTION (VALIDATION VS TEST)")
        print("=" * 90)
        val_pred_counts = np.bincount(tuned_full_preds, minlength=len(CLASSES))
        test_pred_counts = np.bincount(test_pred_ids, minlength=len(CLASSES))
        print(f"{'Class':<22s} | {'Val True %':>10s} | {'Val Pred %':>10s} | {'Test Pred %':>11s} | {'Test Count':>10s}")
        print("-" * 90)
        for i, c in enumerate(CLASSES):
            v_true_pct = 100.0 * val_counts[i] / len(y_va)
            v_pred_pct = 100.0 * val_pred_counts[i] / len(y_va)
            t_pred_pct = 100.0 * test_pred_counts[i] / len(test_pred_ids)
            print(f"{c:<22s} | {v_true_pct:9.4f}% | {v_pred_pct:9.4f}% | {t_pred_pct:10.4f}% | {test_pred_counts[i]:10,d}")

        # Requirement 4: Save val_probs_<name>.npy, test_probs_<name>.npy, val_labels.npy & submission
        sub_file = get_safe_path(f"submission_{run_tag}.csv")
        val_prob_file = get_safe_path(f"val_probs_{run_tag}.npy")
        test_prob_file = get_safe_path(f"test_probs_{run_tag}.npy")
        val_labels_file = "val_labels_sample.npy" if args.sample else "val_labels.npy"

        sub = pd.DataFrame({
            "id": test_df["txn_id"].to_numpy(),
            "label": [ID_TO_LABEL[i] for i in test_pred_ids]
        })
        sub.to_csv(sub_file, index=False)
        np.save(val_prob_file, P_va.astype(np.float32))
        np.save(test_prob_file, P_test.astype(np.float32))
        np.save(val_labels_file, y_va.astype(np.int32))

        print(f"\nSaved files:")
        print(f"  - Submission      : {sub_file} ({len(sub):,} rows, NaNs: {sub.isna().sum().sum()})")
        print(f"  - Val Probs       : {val_prob_file} {P_va.shape}")
        print(f"  - Test Probs      : {test_prob_file} {P_test.shape}")
        print(f"  - Val True Labels : {val_labels_file} {y_va.shape}")

        # Requirement 6: Log to results.csv including --drop list and feature file
        append_run_to_results_csv(
            run_name=args.name,
            is_sample=args.sample,
            features_path=args.features,
            drop_list=user_drop_cols,
            best_iter=best_f1_iter,
            val_raw_f1=val_raw_f1,
            val_tune_f1=val_tune_f1,
            val_eval_f1=val_eval_f1,
            val_full_f1=val_full_f1,
            per_class_f1=eval_per_class_f1,
            params=lgb_params,
            multipliers=best_w
        )
        print(f"\nTotal train_v3.py runtime: {time.time() - t_total:.2f}s")

    finally:
        sys.stdout = sys.__stdout__
        logger.close()


if __name__ == "__main__":
    main()