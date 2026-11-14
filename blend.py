import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

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


def get_safe_path(filepath: str) -> str:
    """Prevents overwriting any existing submission files."""
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    k = 2
    while os.path.exists(f"{base}_run{k}{ext}"):
        k += 1
    return f"{base}_run{k}{ext}"


def resolve_npy_file(prefix: str, run_name: str) -> str:
    """Finds `<prefix>_<run_name>.npy` with convenient prefix fallbacks."""
    candidates = [
        f"{prefix}_{run_name}.npy",
        f"{prefix}_lgbm_{run_name}.npy",
        run_name if run_name.endswith(".npy") else "",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find '{prefix}_{run_name}.npy' (also checked '{prefix}_lgbm_{run_name}.npy')."
    )


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


def optimize_multipliers_3pass(P_tune: np.ndarray, y_tune: np.ndarray, num_classes: int = 12):
    """
    Identical 3-pass multiplier search as train_v3.py:
    - legit fixed at 1.0
    - other 11 classes searched on a log grid [0.3, 30.0]
    - 3 passes with a finer local + global log grid on Pass 3
    """
    w = np.ones(num_classes, dtype=np.float64)
    best_score, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune, axis=1), num_classes)

    coarse_log_grid = np.geomspace(0.3, 30.0, 93)
    if 1.0 not in coarse_log_grid:
        coarse_log_grid = np.sort(np.append(coarse_log_grid, 1.0))

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


def load_val_labels_and_test_ids():
    """Loads validation labels and chronologically sorted test txn_ids matching features_v2.parquet / features.parquet."""
    pq_file = "features_v2.parquet" if os.path.exists("features_v2.parquet") else "features.parquet"
    df_meta = pd.read_parquet(pq_file, columns=["txn_id", "is_test", "timestamp", "label"])

    test_ids = df_meta.loc[df_meta["is_test"] == 1, "txn_id"].to_numpy()
    if os.path.exists("val_labels.npy"):
        y_va = np.load("val_labels.npy").astype(np.int32)
    else:
        train_meta = df_meta[df_meta["is_test"] == 0]
        val_meta = train_meta[train_meta["timestamp"] >= pd.Timestamp("2026-09-01")]
        y_va = val_meta["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int32)
    return y_va, test_ids


def append_blend_to_results_csv(tag, run_names, norm_weights, val_raw_f1, val_tune_f1, val_eval_f1, val_full_f1, per_class_f1, multipliers):
    """Appends blend run to results.csv without overwriting existing entries."""
    row = {
        "timestamp_ist": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": f"blend_{tag}",
        "sample_mode": False,
        "features_file": "BLEND",
        "dropped_cols": "NONE",
        "best_iteration": 0,
        "val_raw_macro_f1": round(val_raw_f1, 5),
        "val_tune_half_macro_f1": round(val_tune_f1, 5),
        "val_eval_half_macro_f1": round(val_eval_f1, 5),
        "val_full_tuned_macro_f1": round(val_full_f1, 5),
    }
    for i, c in enumerate(CLASSES):
        row[f"f1_{c}"] = round(float(per_class_f1[i]), 5)
    row["multipliers"] = json.dumps({CLASSES[i]: round(float(multipliers[i]), 4) for i in range(len(CLASSES))})
    row["params"] = json.dumps({
        "runs": run_names,
        "weights": [round(float(w), 5) for w in norm_weights],
    })

    res_df = pd.DataFrame([row])
    if os.path.exists("results.csv"):
        existing_df = pd.read_csv("results.csv")
        combined_df = pd.concat([existing_df, res_df], ignore_index=True)
        combined_df.to_csv("results.csv", index=False)
    else:
        res_df.to_csv("results.csv", index=False)
    print(f"Appended blend run 'blend_{tag}' to results.csv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, required=True, help="Comma-separated list of run names (e.g. lgbm_v2_full,v3_full_new_noexp)")
    parser.add_argument("--weights", type=str, default="", help="Comma-separated list of weights (e.g. 0.4,0.6). Defaults to equal weights.")
    parser.add_argument("--tag", type=str, default="", help="Tag for submission_blend_<tag>.csv")
    args = parser.parse_args()

    t0 = time.time()
    run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    if not run_names:
        raise ValueError("Please provide at least one run name via --runs.")

    if args.weights.strip():
        raw_weights = [float(w.strip()) for w in args.weights.split(",") if w.strip()]
        if len(raw_weights) != len(run_names):
            raise ValueError(f"--weights count ({len(raw_weights)}) must match --runs count ({len(run_names)})!")
    else:
        raw_weights = [1.0] * len(run_names)

    weights = np.array(raw_weights, dtype=np.float64)
    if np.any(weights < 0) or np.sum(weights) <= 0:
        raise ValueError("All weights must be non-negative and sum to > 0.")
    norm_weights = weights / np.sum(weights)

    tag = args.tag.strip() if args.tag.strip() else "_".join(run_names)

    print("=" * 95)
    print(f"BLEND.PY — PROBABILITY ENSEMBLING & 3-PASS MULTIPLIER TUNING (tag='{tag}')")
    print("=" * 95)

    y_va, test_ids = load_val_labels_and_test_ids()
    val_counts = np.bincount(y_va, minlength=len(CLASSES))

    P_va_blend = None
    P_test_blend = None

    print("\nLoading & evaluating individual component runs:")
    for idx, (r_name, w_norm) in enumerate(zip(run_names, norm_weights), 1):
        val_file = resolve_npy_file("val_probs", r_name)
        test_file = resolve_npy_file("test_probs", r_name)

        p_va_i = np.load(val_file).astype(np.float64)
        p_te_i = np.load(test_file).astype(np.float64)

        assert p_va_i.shape == (len(y_va), 12), f"Shape mismatch in {val_file}: {p_va_i.shape} vs {(len(y_va), 12)}"
        assert p_te_i.shape == (260582, 12), f"Shape mismatch in {test_file}: {p_te_i.shape} vs (260582, 12)"

        indiv_raw_f1, _, _ = fast_confusion_f1(y_va, np.argmax(p_va_i, axis=1), 12)
        print(
            f"  [{idx}/{len(run_names)}] Run '{r_name}' | Weight: {w_norm:.4f} | "
            f"Val File: {val_file} | Test File: {test_file} | Individual Raw Val Macro-F1: {indiv_raw_f1:.5f}"
        )

        if P_va_blend is None:
            P_va_blend = w_norm * p_va_i
            P_test_blend = w_norm * p_te_i
        else:
            P_va_blend += w_norm * p_va_i
            P_test_blend += w_norm * p_te_i

    # Evaluate blended probabilities BEFORE multiplier tuning (w = 1.0)
    raw_blend_preds = np.argmax(P_va_blend, axis=1)
    val_raw_f1, raw_per_class_f1, raw_prec, raw_rec, val_raw_acc, _ = fast_confusion_metrics(y_va, raw_blend_preds)

    # Identical Tune-Half / Eval-Half split as train_v3.py (seed=42)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y_va))
    half = len(y_va) // 2
    idx_tune, idx_eval = perm[:half], perm[half:]

    P_tune, y_tune = P_va_blend[idx_tune], y_va[idx_tune]
    P_eval, y_eval = P_va_blend[idx_eval], y_va[idx_eval]

    raw_tune_f1, _, _ = fast_confusion_f1(y_tune, np.argmax(P_tune, axis=1))
    raw_eval_f1, _, _ = fast_confusion_f1(y_eval, np.argmax(P_eval, axis=1))

    print("\n" + "=" * 95)
    print("RE-TUNING 11 FRAUD MULTIPLIERS ON BLENDED PROBABILITIES (TUNE HALF -> EVAL HALF)")
    print("=" * 95)
    best_w, left_at_one = optimize_multipliers_3pass(P_tune, y_tune)

    tuned_tune_preds = np.argmax(P_tune * best_w, axis=1)
    tuned_eval_preds = np.argmax(P_eval * best_w, axis=1)
    tuned_full_preds = np.argmax(P_va_blend * best_w, axis=1)

    val_tune_f1, _, _ = fast_confusion_f1(y_tune, tuned_tune_preds)
    val_eval_f1, eval_per_class_f1, eval_prec, eval_rec, eval_acc, _ = fast_confusion_metrics(y_eval, tuned_eval_preds)
    val_full_f1, full_per_class_f1, full_prec, full_rec, full_acc, _ = fast_confusion_metrics(y_va, tuned_full_preds)

    print("\n--- Blended Validation Macro-F1 Summary (Before vs After Multiplier Tuning) ---")
    print(f"Blend Val Tune Half (50%) Macro-F1 : {raw_tune_f1:.5f} (before) -> {val_tune_f1:.5f} (after)")
    print(f"Blend Val Eval Half (50%) Macro-F1 : {raw_eval_f1:.5f} (before) -> {val_eval_f1:.5f} (OUT-OF-SAMPLE TUNED)")
    print(f"Blend Val Full (100%)     Macro-F1 : {val_raw_f1:.5f} (before) -> {val_full_f1:.5f} (after) | Acc: {full_acc:.5f}")

    print("\n" + "=" * 98)
    print("PER-CLASS PRECISION, RECALL & F1 TABLE ON BLEND")
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

    # Apply tuned multipliers to blended test probabilities
    test_pred_ids = np.argmax(P_test_blend * best_w, axis=1)

    print("\n" + "=" * 95)
    print("PREDICTED CLASS DISTRIBUTION ON TEST VS VALIDATION")
    print("=" * 95)
    val_pred_counts = np.bincount(tuned_full_preds, minlength=len(CLASSES))
    test_pred_counts = np.bincount(test_pred_ids, minlength=len(CLASSES))
    print(f"{'Class':<22s} | {'Val True %':>10s} | {'Val Pred %':>10s} | {'Test Pred %':>11s} | {'Test Count':>10s}")
    print("-" * 95)
    for i, c in enumerate(CLASSES):
        v_true_pct = 100.0 * val_counts[i] / len(y_va)
        v_pred_pct = 100.0 * val_pred_counts[i] / len(y_va)
        t_pred_pct = 100.0 * test_pred_counts[i] / len(test_pred_ids)
        print(f"{c:<22s} | {v_true_pct:9.4f}% | {v_pred_pct:9.4f}% | {t_pred_pct:10.4f}% | {test_pred_counts[i]:10,d}")

    sub_path = get_safe_path(f"submission_blend_{tag}.csv")
    sub = pd.DataFrame({
        "id": test_ids,
        "label": [ID_TO_LABEL[i] for i in test_pred_ids]
    })

    # Verify strict competition constraints
    assert len(sub) == 260582, f"Expected 260,582 rows, got {len(sub)}!"
    assert int(sub.isna().sum().sum()) == 0, "Submission contains NaNs!"
    assert sub["id"].nunique() == 260582, "Duplicate txn_ids found in submission!"

    sub.to_csv(sub_path, index=False)
    print(f"\n[VERIFIED] Wrote {sub_path}: {len(sub):,} rows, { int(sub.isna().sum().sum()) } NaNs, 100% unique IDs.")

    append_blend_to_results_csv(
        tag=tag,
        run_names=run_names,
        norm_weights=norm_weights,
        val_raw_f1=val_raw_f1,
        val_tune_f1=val_tune_f1,
        val_eval_f1=val_eval_f1,
        val_full_f1=val_full_f1,
        per_class_f1=eval_per_class_f1,
        multipliers=best_w
    )
    print(f"Completed blend.py in {time.time() - t0:.2f}s.")


if __name__ == "__main__":
    main()