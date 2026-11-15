# ML Gradient Wars 2026: UPI Fraud Rings (12 Classes, Macro-F1)

Final submission: a multi-paradigm probability ensemble of LightGBM models combining full-density
and **density-matched (50% thinned)** feature representations (`ens7h` at **0.4958 Public LB**,
and extended multi-split variants `ens14_balanced` and `ens15_master`).
The final ranking uses the private 70% of the test set.

## Quick start

Python 3.12. Windows PowerShell shown; Linux/macOS equivalents in comments.

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1          # Linux/macOS: source .venv312/bin/activate
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

Put the competition files `train.csv`, `test.csv`, `accounts.csv` and `merchants.csv` in the
repo root (they are not redistributed here). Then:

```powershell
python reproduce.py --smoke                  # features + one 20%-sample model (~1 min)
python reproduce.py                          # default: reproduces ens7h (0.4958 LB) -> submission.csv
python reproduce.py --preset ens14_balanced  # reproduces 14-model group-balanced blend -> submission.csv
python reproduce.py --preset ens15_master    # reproduces 15-model equal-weight blend -> submission.csv
```

`reproduce.py --skip-existing` resumes after an interruption. Run in a clean folder: the scripts
never overwrite outputs (they add a `_run2` suffix instead), so stale probability files from an
older run would be picked up by the blend.

## Reproducing the best uploads step by step

`reproduce.py` runs these exact commands (shown here for reference):

```powershell
# 1. Full-density and half-density feature tables
python build_features_v2.py                 # -> features_v2.parquet (~1 min)
python build_features_half.py               # -> features_v2_half.parquet (~3-4 min, bit-0 split)
python build_features_half.py --bit 7       # -> features_v2_half_b7.parquet (~3-4 min, independent bit-7 split)

# 2. Full-density members trained with train_v3.py / train_v3_noip.py (seed 42)
python train_v3.py --name v3_full_new_noexp --drop sender_past_txn_count,sender_past_amount_sum,sender_sec_since_first_seen,receiver_past_txn_count,receiver_past_amount_sum,receiver_sec_since_first_seen,device_past_txn_count,device_past_amount_sum,device_sec_since_first_seen
python train_v3.py --name v3_safe --drop sender_past_txn_count,sender_past_amount_sum,sender_sec_since_first_seen,receiver_past_txn_count,receiver_past_amount_sum,receiver_sec_since_first_seen,device_past_txn_count,device_past_amount_sum,device_sec_since_first_seen,receiver_unique_senders_1h,receiver_unique_senders_24h,sender_unique_receivers_1h,sender_unique_receivers_24h,device_unique_senders_1h,device_unique_senders_24h,receiver_ip_risk_mean_24h,receiver_new_payee_share_24h
python train_v3_noip.py --name v3_noip

# 3. Full-density diverse members trained with train_member.py (v3_safe base features)
python train_member.py --name m_extra --no-ip --seed 11 --rounds 300 --threads 8 --params '{\"extra_trees\": true, \"feature_fraction\": 0.4, \"min_child_samples\": 200}'
python train_member.py --name m_down --legit_frac 0.3 --seed 21 --threads 8 --params '{\"num_leaves\": 63}'

# 4. Core density-matched (half-density) members
$UD6 = "receiver_unique_senders_1h,receiver_unique_senders_24h,sender_unique_receivers_1h,sender_unique_receivers_24h,device_unique_senders_1h,device_unique_senders_24h"
$UD7 = "$UD6,receiver_new_payee_share_24h"
$UD8 = "$UD7,receiver_ip_risk_mean_24h"

python train_member.py --name m_down_half --features features_v2_half.parquet --legit_frac 0.3 --seed 21 --threads 8 --params '{\"num_leaves\": 63}'
python train_member.py --name m_down_half_ud --features features_v2_half.parquet --legit_frac 0.3 --seed 21 --threads 8 --params '{\"num_leaves\": 63}' --undrop $UD6

# 5a. Blend ens7h (0.4958 Public LB) -> submission_blend_ens7h.csv
python blend.py --runs v3_safe,v3_noip,v3_full_new_noexp,m_extra,m_down,m_down_half,m_down_half_ud --tag ens7h

# 5b. Extended multi-split density-matched members (for ens14_balanced & ens15_master)
python train_member.py --name m_half_b7 --features features_v2_half_b7.parquet --legit_frac 0.3 --seed 31 --threads 8 --params '{\"num_leaves\": 63}' --undrop $UD6
python train_member.py --name m_half_s22 --features features_v2_half.parquet --legit_frac 0.3 --seed 22 --threads 2 --params '{\"num_leaves\": 63}' --undrop $UD6
python train_member.py --name m_half_b7_noip --features features_v2_half_b7.parquet --legit_frac 0.3 --seed 41 --threads 8 --params '{\"num_leaves\": 63}' --undrop $UD6 --no-ip
python train_member.py --name m_half_xt --features features_v2_half.parquet --legit_frac 0.3 --seed 51 --threads 2 --params '{\"extra_trees\": true, \"feature_fraction\": 0.4, \"min_child_samples\": 200}' --undrop $UD6
python train_member.py --name m_half_L95 --features features_v2_half.parquet --legit_frac 0.3 --seed 61 --rounds 175 --threads 0 --params '{\"num_leaves\": 95, \"min_child_samples\": 180, \"feature_fraction\": 0.5}' --undrop $UD8
python train_member.py --name m_half_col35 --features features_v2_half.parquet --legit_frac 0.3 --seed 71 --rounds 200 --threads 0 --params '{\"num_leaves\": 63, \"feature_fraction\": 0.35, \"bagging_fraction\": 0.85, \"lambda_l1\": 2.0, \"lambda_l2\": 5.0}' --undrop $UD7
python train_member.py --name m_half_d22 --features features_v2_half.parquet --legit_frac 0.22 --seed 81 --rounds 175 --threads 0 --params '{\"num_leaves\": 63, \"min_child_samples\": 250}' --undrop $UD7
python train_member.py --name m_half_b7_L95 --features features_v2_half_b7.parquet --legit_frac 0.3 --seed 62 --rounds 175 --threads 8 --params '{\"num_leaves\": 95, \"min_child_samples\": 180}' --undrop $UD8

# Blend ens14_balanced (62.5% full-density / 37.5% 9-model half-density super-average)
python blend.py --runs v3_safe,v3_noip,v3_full_new_noexp,m_extra,m_down,m_down_half,m_down_half_ud,m_half_b7,m_half_s22,m_half_b7_noip,m_half_L95,m_half_col35,m_half_d22,m_half_b7_L95 --weights 3,3,3,3,3,1,1,1,1,1,1,1,1,1 --tag ens14_balanced

# Blend ens15_master (15-model equal-weight blend)
python blend.py --runs v3_safe,v3_noip,v3_full_new_noexp,m_extra,m_down,m_down_half,m_down_half_ud,m_half_b7,m_half_s22,m_half_b7_noip,m_half_xt,m_half_L95,m_half_col35,m_half_d22,m_half_b7_L95 --tag ens15_master
```

LightGBM is seeded everywhere, but multithreaded training can still give tiny numeric
differences, so a re-run may differ from the uploaded file on a handful of rows.

## Approach

- **Full-Density Features** (`build_features_v2.py` -> `features_v2.parquet`, 132 columns): per-entity
  (sender, receiver, device) windowed counts and sums (1h/24h), 1-hop and 2-hop inflow -> outflow link
  features (did the receiver move the money on, and how fast), time gaps to previous/next transaction
  of the same entity, reverse-direction payments between the same pair, amount ratios and round-amount
  flags, account and merchant attributes. Train and test are concatenated so test rows can link to each
  other (allowed by the rules). Raw IDs and timestamps are never model features. `app_version` is dropped
  at load: October only has versions 5.0-5.2, so it separates train from test perfectly and carries no signal.
- **Density-Matched Features** (`build_features_half.py` -> `features_v2_half.parquet` and
  `features_v2_half_b7.parquet`): October has ~0.51x the monthly transaction volume of Jul-Sep across every
  app, and `density_check.py` showed that a random 50% row-thinning of September matches October on 24 of 30
  entity/temporal metrics (median `|log ratio|` 0.010 vs 0.264 for unthinned September). To prevent rolling
  counts and link features from under-firing on test, `build_features_half.py` splits train into two
  deterministic random halves (using hex `txn_id` bit parity: bit 0 and bit 7) and runs `build_features_v2.py`
  within each half independently while keeping category encodings and row ordering identical.
- **Validation**: temporal split. Train on Jul-Aug, validate on Sep. Macro-F1 is checked every 25
  rounds to pick the number of trees. The final model is retrained on Jul-Sep with
  `best_iteration x 1.2` trees.
- **Per-class multipliers**: the predicted class is `argmax(p * w)`, with `legit` fixed at `w = 1.0`.
  `w` is tuned by 3-pass coordinate search on one random half of September and evaluated on the other half
  (seed 42), keeping validation scores unbiased.
- **Members**:
  - `v3_safe`: drops 17 expanding-history and density-sensitive window features on full-density data.
  - `v3_full_new_noexp`: drops only the 9 expanding-history features.
  - `v3_noip`: drops all `ip_risk` columns (the vendor score narrows in October).
  - `m_extra`: `extra_trees`, `feature_fraction` 0.4, no `ip_risk`, seed 11.
  - `m_down`: `legit` downsampled to 30% in training (`num_leaves` 63); the `legit` probability is
    rescaled by 0.3 and rows renormalized before saving, aligning probabilities with full-data models.
  - `m_down_half` / `m_down_half_ud`: `m_down` trained on density-matched `features_v2_half.parquet`,
    with `m_down_half_ud` re-enabling (`--undrop`) the 6 rolling distinct-counterparty window features
    that were previously degraded by the 2x density mismatch.
  - Extended multi-split half-density members (`m_half_b7`, `m_half_s22`, `m_half_b7_noip`, `m_half_xt`,
    `m_half_L95`, `m_half_col35`, `m_half_d22`, `m_half_b7_L95`): diverse seeds, independent bit-7
    half-split, 95-leaf high-capacity trees, L1/feature-subsampled trees (`0.35`), and recall-boosted
    `legit_frac` (`0.22`).
- **Blend** (`blend.py`): weighted average of member probabilities, multipliers re-tuned on the
  blended validation probabilities and applied to the blended test probabilities.

## Results

### Cross-Density Validation Experiment (September Held-Out Half)
Testing a full-density model on October-like half-density validation (`m_down_xval`) vs training
directly on half-density features:

| Run | Train Features | Val Features | Eval-half Macro-F1 |
|---|---|---|---|
| `m_down_xval` | Full density | Half density (Oct-like) | 0.4347 |
| `m_down_half` | Half density (bit 0) | Half density (Oct-like) | 0.4485 (+0.0138) |
| `m_down_half_ud` (+6 window cols) | Half density (bit 0) | Half density (Oct-like) | 0.4502 (+0.0155) |
| `m_half_b7` (+6 window cols) | Half density (bit 7) | Half density (Oct-like) | 0.4517 (+0.0170) |

### Full-Density & Ensemble Validation (September)

| Run | Raw Val F1 | Eval-half Tuned F1 | Full Val Tuned F1 |
|---|---|---|---|
| `v3_full_new_noexp` | 0.4517 | 0.4655 | 0.4664 |
| `v3_safe` | 0.4502 | 0.4667 | 0.4669 |
| `v3_noip` | 0.4522 | 0.4606 | 0.4631 |
| `m_extra` | 0.4432 | 0.4557 | 0.4571 |
| `m_down` | 0.4686 | 0.4713 | 0.4715 |
| `blend ens5` (5 full-density models) | 0.4599 | 0.4712 | 0.4714 |
| `blend ens7h` (`ens5` + 2 half models) | 0.4625 | 0.4714 | 0.4712 |
| `blend ens8hh` (3 full + 5 half models) | 0.4631 | 0.4716 | 0.4703 |
| `blend ens15_master` (5 full + 10 half, equal) | 0.4611 | 0.4715 | 0.4699 |
| `blend ens14_balanced` (5 full x3 + 9 half x1) | **0.4639** | 0.4712 | 0.4710 |

### Public Leaderboard Progression

| Upload | Public score |
|---|---|
| `v1` / `v2` baselines | 0.4687 / 0.4894 |
| `v3_safe` (multipliers x1.0 / x1.25 / x1.6) | 0.4907 / 0.4909 / 0.4916 |
| `v3_noip` | 0.4877 |
| `blend ens3` (`v3_safe`, `v3_noip`, `v3_full_new_noexp`) | 0.4924 |
| `blend ens5` (`ens3` + `m_extra` + `m_down`), equal weights | 0.4946 |
| `blend ens5w` (`m_down` weighted 0.5) | 0.4922 |
| **`blend ens7h` (`ens5` + `m_down_half` + `m_down_half_ud`)** | **0.4958 (Best)** |

## Key Findings

- **Label noise**: train labels are noisy (unreported fraud labelled `legit`, some wrong types)
  while test labels are clean. Validation Macro-F1 consistently runs ~0.025 below the public
  score, consistent with clean test labels rewarding high-precision fraud predictions.
- **October drift** (adversarial validation): without `app_version`, a train-vs-test classifier
  reaches AUC 0.64. `ip_risk_score` is the top drifting feature (its distribution narrows in Oct), and
  amounts are ~1.17x higher across quantiles (festival season). Other raw columns are stable.
- **October density thinning**: October has ~0.51x September's rows, uniformly across apps. Randomly
  keeping 50% of September's rows reproduces October's entity statistics (counts per entity, gaps,
  window features) far better than thinning by sender or device. Training members on half-density
  features (`build_features_half.py`) and re-enabling rolling window features (`--undrop`) provided
  uncorrelated error manifolds that pushed the ensemble to **0.4958**.
- **Where the errors are**: ~91% of fraud rows are isolated (no same-device or same-sender
  row within 10 minutes) and look like `legit`; recall on those is 25-31%, against 75-96% for fraud
  inside bursts. Missed fraud gets a median probability of 0.001-0.005 for its true class.
- **What did not help**: smoothing predictions over neighbouring transactions (best +0.001; even
  oracle smoothing with true labels gave at most +0.007 and hurt on receiver relations), forward/centred
  burst and session windows, expanding-history features (they drift with time elapsed since July),
  and overweighting a single member (`ens5w`).
- **What helped**: causal 1-hop and 2-hop fund-flow features, dropping drifting features (`v3_safe`),
  per-class coordinate-tuned multipliers, `legit` downsampling with analytic prior correction (`m_down`),
  density-matched feature engineering (`build_features_half.py`), and ensembling diverse model families.

## Repository contents

| File | Purpose |
|---|---|
| `reproduce.py` | One command from raw CSVs to `submission.csv` (supports `--preset`) |
| `build_features_v2.py` | Full-density feature engineering -> `features_v2.parquet` |
| `build_features_half.py` | Density-matched 50/50 split feature builder -> `features_v2_half*.parquet` |
| `train_v3.py` | LightGBM member with a `--drop` list (`v3_safe`, `v3_full_new_noexp`) |
| `train_v3_noip.py` | LightGBM member without `ip_risk` features (`v3_noip`) |
| `train_member.py` | Configurable LightGBM member (`--legit_frac`, `--undrop`, `--no-ip`) |
| `blend.py` | Probability blend + 3-pass multiplier tuning -> submission CSV |
| `results.csv` | Log of runs: parameters, dropped columns, validation scores, multipliers |
| `requirements.txt` | Pinned package versions |

No feature is derived from labels. `shift(-1)` in `build_features_v2.py` is used only on
timestamps (seconds to the same entity's next transaction), never on labels; this is legitimate
because test rows can be linked to each other. `train_v3_noip.py` and `train_member.py` also
drop any `_tgt`/`target`/`label`/ID column defensively.
