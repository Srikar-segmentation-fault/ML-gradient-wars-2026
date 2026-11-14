import time
import numpy as np
import pandas as pd
import polars as pl

# Requirement 5: All expanding-history and entity-norm deviation columns in one list for --drop
EXPANDING_COLS = [
    "sender_past_txn_count",
    "sender_past_amount_sum",
    "sender_sec_since_first_seen",
    "sender_amount_expanding_zscore",
    "sender_past_hour_sin_mean",
    "sender_past_hour_cos_mean",
    "sender_hour_sin_cos_dist",
    "sender_city_past_share",
    "receiver_past_txn_count",
    "receiver_past_amount_sum",
    "receiver_sec_since_first_seen",
    "device_past_txn_count",
    "device_past_amount_sum",
    "device_sec_since_first_seen",
]


def run_windowed_n_unique(df: pl.DataFrame, group_col: str, target_col: str, period: str, out_col: str) -> pl.Series:
    """Computes trailing rolling n_unique for (group_col, target_col) over `period`."""
    sub = df.select(["row_id", "timestamp", group_col, target_col]).sort([group_col, "timestamp", "row_id"])
    rolled = sub.rolling(index_column="timestamp", period=period, group_by=group_col).agg(
        pl.col(target_col).n_unique().cast(pl.Int32).alias(out_col)
    )
    res = sub.select("row_id").with_columns(rolled[out_col]).sort("row_id")
    return res[out_col]


def main():
    t_start = time.time()
    print("=" * 90)
    print("BUILD_FEATURES_V2.PY — V1 FEATURES + CAUSAL EXPANDING, 2-HOP CHAIN & ROLLING DISTINCTS")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # STAGE 0: Load Datasets & Build All V1 Features Unchanged
    # -------------------------------------------------------------------------
    t_stage = time.time()
    print("\n[Stage 0] Loading datasets with Polars...")
    train_pl = pl.read_csv("train.csv")
    test_pl = pl.read_csv("test.csv")
    accounts_pl = pl.read_csv("accounts.csv")
    merchants_pl = pl.read_csv("merchants.csv")
    print(f"Loaded in {time.time() - t_stage:.2f}s | Train: {train_pl.shape}, Test: {test_pl.shape}")

    t_stage = time.time()
    print("\n[Stage 1] Building all V1 baseline, master join, as-of & rolling features...")
    if "app_version" in train_pl.columns:
        train_pl = train_pl.drop("app_version")
    if "app_version" in test_pl.columns:
        test_pl = test_pl.drop("app_version")

    train_pl = train_pl.with_columns(pl.lit(0).cast(pl.Int8).alias("is_test"))
    test_pl = test_pl.with_columns([
        pl.lit("UNKNOWN").alias("label"),
        pl.lit(1).cast(pl.Int8).alias("is_test")
    ])

    test_pl = test_pl.select(train_pl.columns)
    df = pl.concat([train_pl, test_pl], how="vertical")

    df = df.with_columns(
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S")
    ).sort(["timestamp", "txn_id"]).with_row_index("row_id")

    df = df.with_columns(
        ((pl.col("receiver_type") == "P2P") & pl.col("merchant_category").is_not_null())
        .cast(pl.Int8)
        .alias("p2p_with_merchant_cat")
    )

    accounts_pl = accounts_pl.with_columns(pl.col("opened_on").str.to_datetime("%Y-%m-%d"))
    merchants_pl = merchants_pl.with_columns(pl.col("onboarded_on").str.to_datetime("%Y-%m-%d"))

    sender_acc = accounts_pl.rename({
        "account_id": "sender_id",
        "home_city": "sender_home_city",
        "bank_code": "sender_bank",
        "kyc_level": "sender_kyc",
        "opened_on": "sender_opened_on",
        "os": "sender_os"
    })
    receiver_acc = accounts_pl.rename({
        "account_id": "receiver_id",
        "home_city": "receiver_home_city",
        "bank_code": "receiver_bank",
        "kyc_level": "receiver_kyc",
        "opened_on": "receiver_opened_on",
        "os": "receiver_os"
    })
    merch_ren = merchants_pl.rename({
        "merchant_id": "receiver_id",
        "category": "merchant_cat_master",
        "city": "merchant_city",
        "onboarded_on": "merchant_onboarded_on"
    })

    df = df.join(sender_acc, on="sender_id", how="left")
    df = df.join(receiver_acc, on="receiver_id", how="left")
    df = df.join(merch_ren, on="receiver_id", how="left")
    df = df.sort("row_id")

    df = df.with_columns([
        ((pl.col("timestamp") - pl.col("sender_opened_on")).dt.total_seconds() / 86400.0).alias("sender_acct_age_calc"),
        ((pl.col("timestamp") - pl.col("receiver_opened_on")).dt.total_seconds() / 86400.0).alias("receiver_p2p_age_days"),
        ((pl.col("timestamp") - pl.col("merchant_onboarded_on")).dt.total_seconds() / 86400.0).alias("merchant_age_days"),
    ])
    df = df.with_columns([
        (pl.col("sender_account_age_days") - pl.col("sender_acct_age_calc")).alias("acct_age_discrepancy"),
        pl.coalesce([pl.col("receiver_p2p_age_days"), pl.col("merchant_age_days")]).alias("receiver_entity_age_days"),
    ]).drop(["sender_opened_on", "receiver_opened_on", "merchant_onboarded_on"])

    df = df.with_columns([
        (pl.col("device_city") != pl.col("sender_home_city")).cast(pl.Int8).alias("sender_away_from_home"),
        pl.when(pl.col("receiver_type") == "P2P")
          .then((pl.col("device_city") != pl.col("receiver_home_city")).cast(pl.Int8))
          .otherwise((pl.col("device_city") != pl.col("merchant_city")).cast(pl.Int8))
          .alias("receiver_away_from_device_city"),
        (pl.col("sender_bank") == pl.col("receiver_bank")).fill_null(False).cast(pl.Int8).alias("same_bank"),
        (pl.col("amount") / (pl.col("balance_before") + 1.0)).alias("amount_to_balance_ratio"),
        (pl.col("balance_before") - pl.col("amount")).alias("balance_after"),
        ((pl.col("balance_before") - pl.col("amount")) < 100.0).cast(pl.Int8).alias("is_near_zero_balance_after"),
        (pl.col("amount") % 100 == 0).cast(pl.Int8).alias("is_round_100"),
        (pl.col("amount") % 500 == 0).cast(pl.Int8).alias("is_round_500"),
        (
            (pl.col("amount") % 1000 >= 900) |
            (pl.col("amount") % 500 >= 480) |
            ((pl.col("amount") >= 1900) & (pl.col("amount") < 2000)) |
            ((pl.col("amount") >= 4900) & (pl.col("amount") < 5000)) |
            ((pl.col("amount") >= 9900) & (pl.col("amount") < 10000))
        ).cast(pl.Int8).alias("just_under_limit"),
        ((pl.col("amount") >= 1.0) & (pl.col("amount") <= 10.0)).cast(pl.Int8).alias("is_micro_bait"),
        pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("timestamp").dt.weekday().cast(pl.Int8).alias("weekday"),
        pl.col("timestamp").dt.day().cast(pl.Int8).alias("day_of_month"),
    ])
    df = df.with_columns([
        ((pl.col("hour") >= 22) | (pl.col("hour") <= 5)).cast(pl.Int8).alias("is_late_night"),
        (pl.col("amount") / (pl.col("amount").median().over("sender_id") + 1.0)).alias("amount_vs_sender_median"),
        (pl.col("amount") / (pl.col("amount").median().over("receiver_id") + 1.0)).alias("amount_vs_receiver_median"),
        (pl.col("receiver_type") == "P2P").cast(pl.Int8).alias("_is_p2p_code"),
    ])
    print(f"Completed Stage 1 in {time.time() - t_stage:.2f}s")

    # -------------------------------------------------------------------------
    # V1 + V2 ADDITION 1 & 2: As-Of Joins Carrying Linked Attributes + 2-Hop Chain
    # -------------------------------------------------------------------------
    t_asof = time.time()
    print("\n[Stage 2] Running As-Of Joins (V1 + Addition 1 Linked Attributes + Addition 2 2-Hop Chain)...")

    inflow_lookup = df.select([
        pl.col("receiver_id").alias("sender_id"),
        (pl.col("timestamp") + pl.duration(microseconds=1)).alias("inflow_ts_key"),
        pl.col("timestamp").alias("last_inflow_ts"),
        pl.col("amount").alias("last_inflow_amount"),
        pl.col("hour").alias("prev_inflow_hour"),
        pl.col("ip_risk_score").alias("prev_inflow_ip_risk_score"),
        pl.col("device_age_days").alias("prev_inflow_device_age_days"),
        pl.col("_is_p2p_code").alias("prev_inflow_receiver_type"),
    ]).sort(["sender_id", "inflow_ts_key"])

    df_sorted_sender = df.sort(["sender_id", "timestamp", "row_id"])
    joined_bwd = df_sorted_sender.join_asof(
        inflow_lookup,
        left_on="timestamp",
        right_on="inflow_ts_key",
        by="sender_id",
        strategy="backward"
    ).with_columns(
        (pl.col("timestamp") - pl.col("last_inflow_ts")).dt.total_seconds().alias("time_since_sender_last_inflow_sec")
    )

    outflow_lookup = df.select([
        pl.col("sender_id").alias("receiver_id"),
        (pl.col("timestamp") - pl.duration(microseconds=1)).alias("outflow_ts_key"),
        pl.col("timestamp").alias("next_outflow_ts"),
        pl.col("amount").alias("next_outflow_amount"),
        pl.col("hour").alias("next_outflow_hour"),
        pl.col("ip_risk_score").alias("next_outflow_ip_risk_score"),
        pl.col("device_age_days").alias("next_outflow_device_age_days"),
        pl.col("_is_p2p_code").alias("next_outflow_receiver_type"),
        pl.col("row_id").alias("_hop1_row_id"),
    ]).sort(["receiver_id", "outflow_ts_key"])

    df_sorted_receiver = joined_bwd.sort(["receiver_id", "timestamp", "row_id"])
    joined_fwd = df_sorted_receiver.join_asof(
        outflow_lookup,
        left_on="timestamp",
        right_on="outflow_ts_key",
        by="receiver_id",
        strategy="forward"
    ).with_columns(
        (pl.col("next_outflow_ts") - pl.col("timestamp")).dt.total_seconds().alias("time_until_receiver_next_outflow_sec")
    )

    TOL_24H = 86400.0
    df = joined_fwd.sort("row_id").with_columns([
        (pl.col("amount") / (pl.col("last_inflow_amount") + 1.0)).alias("sender_inflow_amount_ratio"),
        (pl.col("next_outflow_amount") / (pl.col("amount") + 1.0)).alias("receiver_outflow_amount_ratio"),
        (pl.col("time_since_sender_last_inflow_sec") <= TOL_24H).fill_null(False).cast(pl.Int8).alias("has_inflow_24h"),
        (pl.col("time_until_receiver_next_outflow_sec") <= TOL_24H).fill_null(False).cast(pl.Int8).alias("has_outflow_24h"),
    ]).drop([
        "inflow_ts_key", "last_inflow_ts",
        "outflow_ts_key", "next_outflow_ts", "_is_p2p_code"
    ])

    hop2_lookup = df.select([
        pl.col("row_id").alias("_hop1_row_id"),
        pl.col("time_until_receiver_next_outflow_sec").alias("hop2_outflow_gap_sec"),
        pl.col("receiver_outflow_amount_ratio").alias("hop2_outflow_amount_ratio"),
    ])
    df = df.join(hop2_lookup, on="_hop1_row_id", how="left").sort("row_id").drop("_hop1_row_id")

    rev_lookup_bwd = df.select([
        pl.col("receiver_id").alias("sender_id"),
        pl.col("sender_id").alias("receiver_id"),
        (pl.col("timestamp") + pl.duration(microseconds=1)).alias("rev_bwd_key"),
        pl.col("timestamp").alias("prior_rev_ts"),
        pl.col("amount").alias("prior_reverse_amount"),
    ]).sort(["sender_id", "receiver_id", "rev_bwd_key"])

    rev_lookup_fwd = df.select([
        pl.col("receiver_id").alias("sender_id"),
        pl.col("sender_id").alias("receiver_id"),
        (pl.col("timestamp") - pl.duration(microseconds=1)).alias("rev_fwd_key"),
        pl.col("timestamp").alias("next_rev_ts"),
        pl.col("amount").alias("next_reverse_amount"),
    ]).sort(["sender_id", "receiver_id", "rev_fwd_key"])

    df_pair_sorted = df.sort(["sender_id", "receiver_id", "timestamp", "row_id"])
    df_pair_sorted = df_pair_sorted.join_asof(
        rev_lookup_bwd,
        left_on="timestamp",
        right_on="rev_bwd_key",
        by=["sender_id", "receiver_id"],
        strategy="backward"
    ).join_asof(
        rev_lookup_fwd,
        left_on="timestamp",
        right_on="rev_fwd_key",
        by=["sender_id", "receiver_id"],
        strategy="forward"
    )

    df = df_pair_sorted.sort("row_id").with_columns([
        (pl.col("timestamp") - pl.col("prior_rev_ts")).dt.total_seconds().alias("time_since_reverse_sec"),
        (pl.col("amount") / (pl.col("prior_reverse_amount") + 0.1)).alias("reverse_amount_ratio"),
        (
            (pl.col("prior_reverse_amount") >= 1.0) &
            (pl.col("prior_reverse_amount") <= 10.0) &
            (pl.col("amount") >= 100.0)
        ).fill_null(False).cast(pl.Int8).alias("is_cashback_trap"),
        (pl.col("next_rev_ts") - pl.col("timestamp")).dt.total_seconds().alias("time_until_next_reverse_sec"),
        (
            (pl.col("amount") >= 1.0) &
            (pl.col("amount") <= 10.0) &
            (pl.col("next_reverse_amount") >= 100.0)
        ).fill_null(False).cast(pl.Int8).alias("is_micro_bait_followed_by_large"),
    ]).with_columns([
        (pl.col("time_since_reverse_sec") <= TOL_24H).fill_null(False).cast(pl.Int8).alias("has_rev_bwd_24h"),
        (pl.col("time_until_next_reverse_sec") <= TOL_24H).fill_null(False).cast(pl.Int8).alias("has_rev_fwd_24h"),
        ((pl.col("last_inflow_amount") >= 1.0) & (pl.col("last_inflow_amount") <= 10.0)).fill_null(False).cast(pl.Int8).alias("sender_last_inflow_is_micro"),
        ((pl.col("last_inflow_amount") >= 1.0) & (pl.col("last_inflow_amount") <= 10.0) & (pl.col("amount") >= 100.0)).fill_null(False).cast(pl.Int8).alias("micro_inflow_then_large_outflow"),
    ]).drop([
        "rev_bwd_key", "prior_rev_ts",
        "rev_fwd_key", "next_rev_ts"
    ])
    print(f"Completed Stage 2 As-Of & 2-Hop Chain Joins in {time.time() - t_asof:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 3: V1 Whole-Period, Lead/Lag & Rolling Count/Sum Features (Unchanged)
    # -------------------------------------------------------------------------
    t_v1_roll = time.time()
    print("\n[Stage 3] Building V1 graph degrees, lead/lag gaps & rolling counts/sums...")
    df = df.with_columns([
        pl.col("sender_id").n_unique().over("device_id").cast(pl.Int32).alias("device_unique_senders"),
        pl.col("receiver_id").n_unique().over("device_id").cast(pl.Int32).alias("device_unique_receivers"),
        pl.col("device_id").n_unique().over("sender_id").cast(pl.Int32).alias("sender_unique_devices"),
        pl.col("sender_id").n_unique().over("receiver_id").cast(pl.Int32).alias("receiver_unique_senders"),
        pl.col("receiver_id").n_unique().over("sender_id").cast(pl.Int32).alias("sender_unique_receivers"),
        pl.len().over(["sender_id", "receiver_id"]).cast(pl.Int32).alias("pair_txn_count"),
        pl.col("amount").std().over(["sender_id", "receiver_id"]).fill_null(0.0).alias("pair_amount_std"),
        pl.col("is_round_100").mean().over(["sender_id", "receiver_id"]).alias("pair_round_100_ratio"),
        (pl.col("txn_type") == "REFUND").cast(pl.Int32).sum().over("sender_id").alias("sender_refund_count"),
        (pl.col("txn_type") == "REFUND").cast(pl.Float32).mean().over("sender_id").alias("sender_refund_ratio"),
        (pl.col("txn_type") == "COLLECT").cast(pl.Int32).sum().over("sender_id").alias("sender_collect_count"),
        (pl.col("txn_type") == "COLLECT").cast(pl.Int32).sum().over("receiver_id").alias("receiver_collect_count"),
    ])

    df = df.sort(["sender_id", "timestamp", "row_id"]).with_columns([
        (pl.col("timestamp") - pl.col("timestamp").shift(1).over("sender_id")).dt.total_seconds().alias("sender_prev_txn_sec"),
        (pl.col("timestamp").shift(-1).over("sender_id") - pl.col("timestamp")).dt.total_seconds().alias("sender_next_txn_sec"),
    ])
    df = df.sort(["device_id", "timestamp", "row_id"]).with_columns([
        (pl.col("timestamp") - pl.col("timestamp").shift(1).over("device_id")).dt.total_seconds().alias("device_prev_txn_sec"),
        (pl.col("timestamp").shift(-1).over("device_id") - pl.col("timestamp")).dt.total_seconds().alias("device_next_txn_sec"),
    ])
    df = df.sort(["receiver_id", "timestamp", "row_id"]).with_columns([
        (pl.col("timestamp") - pl.col("timestamp").shift(1).over("receiver_id")).dt.total_seconds().alias("receiver_prev_txn_sec"),
        (pl.col("timestamp").shift(-1).over("receiver_id") - pl.col("timestamp")).dt.total_seconds().alias("receiver_next_txn_sec"),
    ])

    df = df.sort(["device_id", "timestamp", "row_id"])
    dev_10m = df.rolling(index_column="timestamp", period="10m", group_by="device_id").agg(
        pl.len().cast(pl.Int32).alias("device_txn_count_10m")
    )
    dev_1h = df.rolling(index_column="timestamp", period="1h", group_by="device_id").agg([
        pl.len().cast(pl.Int32).alias("device_txn_count_1h"),
        pl.col("amount").sum().alias("device_amount_sum_1h"),
    ])
    df = df.with_columns([
        dev_10m["device_txn_count_10m"],
        dev_1h["device_txn_count_1h"],
        dev_1h["device_amount_sum_1h"],
    ])

    df = df.sort(["receiver_id", "timestamp", "row_id"])
    rec_1h = df.rolling(index_column="timestamp", period="1h", group_by="receiver_id").agg([
        pl.len().cast(pl.Int32).alias("receiver_inflow_count_1h"),
        pl.col("amount").sum().alias("receiver_inflow_sum_1h"),
    ])
    rec_24h = df.rolling(index_column="timestamp", period="24h", group_by="receiver_id").agg([
        pl.len().cast(pl.Int32).alias("receiver_inflow_count_24h"),
        pl.col("ip_risk_score").mean().alias("receiver_ip_risk_mean_24h"),
        pl.col("new_payee").mean().alias("receiver_new_payee_share_24h"),
    ])
    df = df.with_columns([
        rec_1h["receiver_inflow_count_1h"],
        rec_1h["receiver_inflow_sum_1h"],
        rec_24h["receiver_inflow_count_24h"],
        rec_24h["receiver_ip_risk_mean_24h"],
        rec_24h["receiver_new_payee_share_24h"],
    ])

    df = df.sort(["sender_id", "timestamp", "row_id"])
    snd_1h = df.rolling(index_column="timestamp", period="1h", group_by="sender_id").agg([
        pl.len().cast(pl.Int32).alias("sender_outflow_count_1h"),
        pl.col("amount").sum().alias("sender_outflow_sum_1h"),
    ])
    snd_24h = df.rolling(index_column="timestamp", period="24h", group_by="sender_id").agg(
        pl.len().cast(pl.Int32).alias("sender_outflow_count_24h")
    )
    df = df.with_columns([
        snd_1h["sender_outflow_count_1h"],
        snd_1h["sender_outflow_sum_1h"],
        snd_24h["sender_outflow_count_24h"],
    ])
    print(f"Completed Stage 3 (including Addition 6 trailing-24h receiver aggregates) in {time.time() - t_v1_roll:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 4 (FIXED): Expanding History & Entity-Norm Deviations (Past Rows Only)
    # -------------------------------------------------------------------------
    t_exp = time.time()
    print("\n[Stage 4] Building Additions 3 & 4: Expanding History & Entity-Norm Deviations (Past Rows Only)...")

    # Step 4.1: Create plain helper columns first (no .over() inside arithmetic)
    two_pi_over_24 = 2.0 * np.pi / 24.0
    df = df.with_columns([
        pl.col("amount").cast(pl.Float64).alias("_amt_f64"),
        (pl.col("amount").cast(pl.Float64) * pl.col("amount").cast(pl.Float64)).alias("_amt_sq_f64"),
        (pl.col("hour").cast(pl.Float64) * two_pi_over_24).sin().alias("_hour_sin"),
        (pl.col("hour").cast(pl.Float64) * two_pi_over_24).cos().alias("_hour_cos"),
    ])

    # Step 4.2: Sort by sender_id chronologically and run each .over() on a plain column
    df = df.sort(["sender_id", "timestamp", "row_id"])
    df = df.with_columns([
        pl.col("_amt_f64").cum_count().over("sender_id").cast(pl.Int64).alias("_snd_cum_cnt"),
        pl.col("_amt_f64").cum_sum().over("sender_id").alias("_snd_cum_sum"),
        pl.col("_amt_sq_f64").cum_sum().over("sender_id").alias("_snd_cum_sq_sum"),
        pl.col("timestamp").first().over("sender_id").alias("_snd_first_ts"),
        pl.col("_hour_sin").cum_sum().over("sender_id").alias("_snd_cum_hsin"),
        pl.col("_hour_cos").cum_sum().over("sender_id").alias("_snd_cum_hcos"),
        pl.col("device_city").cum_count().over(["sender_id", "device_city"]).cast(pl.Int64).alias("_snd_city_cum_cnt"),
    ])

    # Step 4.3: Subtract current row to get strict past counts/totals, and make features NaN (None) when no history exists
    df = df.with_columns([
        (pl.col("_snd_cum_cnt") - 1).alias("_snd_past_n"),
        (pl.col("_snd_cum_sum") - pl.col("_amt_f64")).alias("_snd_past_sum"),
        (pl.col("_snd_cum_sq_sum") - pl.col("_amt_sq_f64")).alias("_snd_past_sq_sum"),
        (pl.col("_snd_cum_hsin") - pl.col("_hour_sin")).alias("_snd_past_hsin_sum"),
        (pl.col("_snd_cum_hcos") - pl.col("_hour_cos")).alias("_snd_past_hcos_sum"),
        (pl.col("_snd_city_cum_cnt") - 1).alias("_snd_city_past_n"),
        (pl.col("timestamp") - pl.col("_snd_first_ts")).dt.total_seconds().alias("_snd_raw_sec_first"),
    ])

    df = df.with_columns([
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(pl.col("_snd_past_n").cast(pl.Float64))
          .otherwise(None)
          .alias("sender_past_txn_count"),
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(pl.col("_snd_past_sum"))
          .otherwise(None)
          .alias("sender_past_amount_sum"),
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(pl.col("_snd_raw_sec_first"))
          .otherwise(None)
          .alias("sender_sec_since_first_seen"),
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(pl.col("_snd_past_sum") / pl.col("_snd_past_n").cast(pl.Float64))
          .otherwise(None)
          .alias("_snd_past_mean"),
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(pl.col("_snd_past_hsin_sum") / pl.col("_snd_past_n").cast(pl.Float64))
          .otherwise(None)
          .alias("sender_past_hour_sin_mean"),
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(pl.col("_snd_past_hcos_sum") / pl.col("_snd_past_n").cast(pl.Float64))
          .otherwise(None)
          .alias("sender_past_hour_cos_mean"),
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(pl.col("_snd_city_past_n").cast(pl.Float64) / pl.col("_snd_past_n").cast(pl.Float64))
          .otherwise(None)
          .alias("sender_city_past_share"),
    ])

    # Step 4.4: Compute sender expanding std, z-score (requires >= 2 past rows), and sin/cos hour Euclidean distance
    df = df.with_columns([
        pl.when(pl.col("_snd_past_n") >= 2)
          .then(
              (
                  (pl.col("_snd_past_sq_sum") - (pl.col("_snd_past_sum") * pl.col("_snd_past_sum") / pl.col("_snd_past_n").cast(pl.Float64)))
                  / (pl.col("_snd_past_n").cast(pl.Float64) - 1.0)
              ).clip(lower_bound=0.0).sqrt()
          )
          .otherwise(None)
          .alias("_snd_past_std"),
    ])

    df = df.with_columns([
        pl.when(pl.col("_snd_past_n") >= 2)
          .then((pl.col("_amt_f64") - pl.col("_snd_past_mean")) / (pl.col("_snd_past_std") + 1.0))
          .otherwise(None)
          .alias("sender_amount_expanding_zscore"),
        pl.when(pl.col("_snd_past_n") >= 1)
          .then(
              (
                  (pl.col("_hour_sin") - pl.col("sender_past_hour_sin_mean")).pow(2) +
                  (pl.col("_hour_cos") - pl.col("sender_past_hour_cos_mean")).pow(2)
              ).sqrt()
          )
          .otherwise(None)
          .alias("sender_hour_sin_cos_dist"),
    ]).drop([
        "_snd_cum_cnt", "_snd_cum_sum", "_snd_cum_sq_sum", "_snd_first_ts",
        "_snd_cum_hsin", "_snd_cum_hcos", "_snd_city_cum_cnt",
        "_snd_past_n", "_snd_past_sum", "_snd_past_sq_sum",
        "_snd_past_hsin_sum", "_snd_past_hcos_sum", "_snd_city_past_n",
        "_snd_raw_sec_first", "_snd_past_mean", "_snd_past_std",
        "_amt_sq_f64", "_hour_sin", "_hour_cos"
    ])

    # Step 4.5: Receiver expanding history (past rows only; NaN when no prior history)
    df = df.sort(["receiver_id", "timestamp", "row_id"])
    df = df.with_columns([
        pl.col("_amt_f64").cum_count().over("receiver_id").cast(pl.Int64).alias("_rec_cum_cnt"),
        pl.col("_amt_f64").cum_sum().over("receiver_id").alias("_rec_cum_sum"),
        pl.col("timestamp").first().over("receiver_id").alias("_rec_first_ts"),
    ])
    df = df.with_columns([
        (pl.col("_rec_cum_cnt") - 1).alias("_rec_past_n"),
        (pl.col("_rec_cum_sum") - pl.col("_amt_f64")).alias("_rec_past_sum"),
        (pl.col("timestamp") - pl.col("_rec_first_ts")).dt.total_seconds().alias("_rec_raw_sec_first"),
    ])
    df = df.with_columns([
        pl.when(pl.col("_rec_past_n") >= 1)
          .then(pl.col("_rec_past_n").cast(pl.Float64))
          .otherwise(None)
          .alias("receiver_past_txn_count"),
        pl.when(pl.col("_rec_past_n") >= 1)
          .then(pl.col("_rec_past_sum"))
          .otherwise(None)
          .alias("receiver_past_amount_sum"),
        pl.when(pl.col("_rec_past_n") >= 1)
          .then(pl.col("_rec_raw_sec_first"))
          .otherwise(None)
          .alias("receiver_sec_since_first_seen"),
    ]).drop(["_rec_cum_cnt", "_rec_cum_sum", "_rec_first_ts", "_rec_past_n", "_rec_past_sum", "_rec_raw_sec_first"])

    # Step 4.6: Device expanding history (past rows only; NaN when no prior history)
    df = df.sort(["device_id", "timestamp", "row_id"])
    df = df.with_columns([
        pl.col("_amt_f64").cum_count().over("device_id").cast(pl.Int64).alias("_dev_cum_cnt"),
        pl.col("_amt_f64").cum_sum().over("device_id").alias("_dev_cum_sum"),
        pl.col("timestamp").first().over("device_id").alias("_dev_first_ts"),
    ])
    df = df.with_columns([
        (pl.col("_dev_cum_cnt") - 1).alias("_dev_past_n"),
        (pl.col("_dev_cum_sum") - pl.col("_amt_f64")).alias("_dev_past_sum"),
        (pl.col("timestamp") - pl.col("_dev_first_ts")).dt.total_seconds().alias("_dev_raw_sec_first"),
    ])
    df = df.with_columns([
        pl.when(pl.col("_dev_past_n") >= 1)
          .then(pl.col("_dev_past_n").cast(pl.Float64))
          .otherwise(None)
          .alias("device_past_txn_count"),
        pl.when(pl.col("_dev_past_n") >= 1)
          .then(pl.col("_dev_past_sum"))
          .otherwise(None)
          .alias("device_past_amount_sum"),
        pl.when(pl.col("_dev_past_n") >= 1)
          .then(pl.col("_dev_raw_sec_first"))
          .otherwise(None)
          .alias("device_sec_since_first_seen"),
    ]).drop(["_dev_cum_cnt", "_dev_cum_sum", "_dev_first_ts", "_dev_past_n", "_dev_past_sum", "_dev_raw_sec_first", "_amt_f64"])

    df = df.sort("row_id")
    print(f"Completed Stage 4 Expanding History & Deviations in {time.time() - t_exp:.2f}s")

    # -------------------------------------------------------------------------
    # STAGE 5: Windowed Distinct Counts (1h & 24h) with 5% Benchmark First
    # -------------------------------------------------------------------------
    t_win = time.time()
    print("\n[Stage 5] Benchmarking Addition 5 Windowed Distinct Counts (1h & 24h) on 5% of rows...")

    distinct_tasks = [
        ("receiver_id", "sender_id", "1h", "receiver_unique_senders_1h"),
        ("receiver_id", "sender_id", "24h", "receiver_unique_senders_24h"),
        ("sender_id", "receiver_id", "1h", "sender_unique_receivers_1h"),
        ("sender_id", "receiver_id", "24h", "sender_unique_receivers_24h"),
        ("device_id", "sender_id", "1h", "device_unique_senders_1h"),
        ("device_id", "sender_id", "24h", "device_unique_senders_24h"),
    ]

    n_sample_5pct = int(df.height * 0.05)
    df_bench = df.head(n_sample_5pct)
    added_distinct_cols = []

    for grp_col, tgt_col, period, col_name in distinct_tasks:
        tb0 = time.time()
        _ = run_windowed_n_unique(df_bench, grp_col, tgt_col, period, col_name)
        dt_5pct = time.time() - tb0
        projected_full_sec = dt_5pct * 20.0
        print(
            f"  Benchmark {col_name:<30s}: 5% took {dt_5pct:5.2f}s -> "
            f"Projected full runtime: {projected_full_sec:6.1f}s ({projected_full_sec / 60.0:.2f} min)"
        )
        if projected_full_sec > 600.0:
            print(f"  [SKIP] {col_name} projected runtime ({projected_full_sec:.1f}s) exceeds 10-minute limit!")
        else:
            tf0 = time.time()
            series_out = run_windowed_n_unique(df, grp_col, tgt_col, period, col_name)
            df = df.with_columns(series_out)
            added_distinct_cols.append(col_name)
            print(f"  [DONE] Computed full {col_name:<30s} in {time.time() - tf0:5.2f}s")

    del df_bench
    print(f"Completed Stage 5 Windowed Distinct Counts in {time.time() - t_win:.2f}s")

    # -------------------------------------------------------------------------
    # Requirement 6: Leakage & Causality Check on 3 Senders (First 5 Rows Each)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("LEAKAGE & CAUSALITY CHECK: FIRST 5 ROWS FOR 3 SENDERS (CONFIRM ROW 1 IS NaN & ROW k USES < k)")
    print("=" * 95)

    # Pick 3 senders that have at least 5 transactions
    sender_counts = df.group_by("sender_id").len().filter(pl.col("len") >= 5).sort("sender_id")
    sample_senders = sender_counts["sender_id"].head(3).to_list()

    check_cols = [
        "sender_id", "timestamp", "amount", "hour", "device_city",
        "sender_past_txn_count", "sender_past_amount_sum", "sender_sec_since_first_seen",
        "sender_amount_expanding_zscore", "sender_past_hour_sin_mean",
        "sender_past_hour_cos_mean", "sender_hour_sin_cos_dist", "sender_city_past_share"
    ]

    for s_id in sample_senders:
        s_df = (
            df.filter(pl.col("sender_id") == s_id)
              .sort(["timestamp", "row_id"])
              .select(check_cols)
              .head(5)
              .to_pandas()
        )
        # Programmatic assertions on row 1 (index 0) and row k (index 1..4)
        assert pd.isna(s_df.loc[0, "sender_past_txn_count"]), f"Leakage: Row 1 sender_past_txn_count is not NaN for {s_id}!"
        assert pd.isna(s_df.loc[0, "sender_past_amount_sum"]), f"Leakage: Row 1 sender_past_amount_sum is not NaN for {s_id}!"
        assert pd.isna(s_df.loc[0, "sender_sec_since_first_seen"]), f"Leakage: Row 1 sender_sec_since_first_seen is not NaN for {s_id}!"
        assert pd.isna(s_df.loc[0, "sender_city_past_share"]), f"Leakage: Row 1 sender_city_past_share is not NaN for {s_id}!"
        assert pd.isna(s_df.loc[0, "sender_amount_expanding_zscore"]), f"Leakage: Row 1 z-score is not NaN for {s_id}!"
        assert pd.isna(s_df.loc[1, "sender_amount_expanding_zscore"]), f"Leakage: Row 2 z-score is not NaN (only 1 past row) for {s_id}!"

        expected_cum_amt = s_df["amount"].cumsum().shift(1)
        for k in range(1, len(s_df)):
            assert np.isclose(s_df.loc[k, "sender_past_amount_sum"], expected_cum_amt.iloc[k], atol=1e-4), (
                f"Mismatch at row {k+1} for {s_id}!"
            )

        print(f"\nSender: {s_id} (Verified: Row 1 expanding features are all NaN; Row k strictly equals sum/stats of rows 1..k-1)")
        print(s_df.round(4).to_string(index=False))

    print("\n" + "=" * 95)
    print(f"EXPANDING_COLS LIST ({len(EXPANDING_COLS)} cols, for --drop flag in train_v3.py):")
    print(f"--drop {','.join(EXPANDING_COLS)}")
    print("=" * 95)

    # -------------------------------------------------------------------------
    # Encode String Categoricals & Print Summary of New V2 Features
    # -------------------------------------------------------------------------
    t_enc = time.time()
    df = df.sort("row_id").drop("row_id")

    cat_cols = [
        "receiver_type", "txn_type", "merchant_category", "device_city", "app",
        "sender_home_city", "sender_bank", "sender_kyc", "sender_os",
        "receiver_home_city", "receiver_bank", "receiver_kyc", "receiver_os",
        "merchant_cat_master", "merchant_city"
    ]
    for c in cat_cols:
        if c in df.columns:
            df = df.with_columns(
                pl.col(c).fill_null("MISSING").cast(pl.Categorical).to_physical().cast(pl.Int32).alias(c)
            )

    new_v2_features = [
        # 1. Linked-transaction attributes
        "prev_inflow_hour",
        "prev_inflow_ip_risk_score",
        "prev_inflow_device_age_days",
        "prev_inflow_receiver_type",
        "next_outflow_hour",
        "next_outflow_ip_risk_score",
        "next_outflow_device_age_days",
        "next_outflow_receiver_type",
        # 2. 2-hop chain
        "hop2_outflow_gap_sec",
        "hop2_outflow_amount_ratio",
        # 3 & 4. Expanding history & entity-norm deviations (EXPANDING_COLS)
        *EXPANDING_COLS,
        # 5. Windowed distinct counts (1h and 24h)
        *added_distinct_cols,
        # 6. Trailing-24h counterparty aggregates
        "receiver_ip_risk_mean_24h",
        "receiver_new_payee_share_24h",
    ]

    print("\n" + "=" * 90)
    print(f"NEW V2 FEATURES SUMMARY ({len(new_v2_features)} new features added)")
    print("=" * 90)
    print(f"{'#':>3s} | {'Feature Name':<42s} | {'DType':<10s} | {'NaN Count':>10s} | {'NaN Rate %':>10s}")
    print("-" * 90)
    n_rows = df.height
    for idx, col in enumerate(new_v2_features, 1):
        null_cnt = int(df[col].null_count())
        null_pct = 100.0 * null_cnt / n_rows
        print(f"{idx:3d} | {col:<42s} | {str(df[col].dtype):<10s} | {null_cnt:10,d} | {null_pct:9.3f}%")

    print(f"\nCompleted encoding & summary in {time.time() - t_enc:.2f}s")

    t_write = time.time()
    out_file = "features_v2.parquet"
    print(f"\n[Stage 6] Writing {out_file} ({df.shape[0]:,} rows x {df.shape[1]} cols)...")
    df.write_parquet(out_file)
    print(f"Wrote {out_file} in {time.time() - t_write:.2f}s | Total build_features_v2.py runtime: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()