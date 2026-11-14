# ML Gradient Wars 2026: UPI fraud rings (12 classes, Macro-F1)

Final submission: an equal-weight probability blend of five LightGBM models (tag `ens5`),
**public leaderboard 0.4946**. The final ranking uses the private 70% of the test set.

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
python reproduce.py --smoke     # features + one 20%-sample model, a few minutes
python reproduce.py             # full pipeline -> submission.csv (about 1 hour, 8-core laptop)
```

`reproduce.py --skip-existing` resumes after an interruption. Run in a clean folder: the scripts
never overwrite outputs (they add a `_run2` suffix instead), so stale probability files from an
older run would be picked up by the blend.

## Reproducing the best upload step by step

`reproduce.py` runs exactly these commands (shown here for reference):

```powershell
python build_features_v2.py                                   # -> features_v2.parquet (~1 min)

# members trained with train_v3.py (seed 42, all cores, ~12-13 min each)
python train_v3.py --name v3_full_new_noexp --drop sender_past_txn_count,sender_past_amount_sum,sender_sec_since_first_seen,receiver_past_txn_count,receiver_past_amount_sum,receiver_sec_since_first_seen,device_past_txn_count,device_past_amount_sum,device_sec_since_first_seen
python train_v3.py --name v3_safe --drop sender_past_txn_count,sender_past_amount_sum,sender_sec_since_first_seen,receiver_past_txn_count,receiver_past_amount_sum,receiver_sec_since_first_seen,device_past_txn_count,device_past_amount_sum,device_sec_since_first_seen,receiver_unique_senders_1h,receiver_unique_senders_24h,sender_unique_receivers_1h,sender_unique_receivers_24h,device_unique_senders_1h,device_unique_senders_24h,receiver_ip_risk_mean_24h,receiver_new_payee_share_24h

# v3_noip has its drop list built in (21 columns: 17 history/time columns + every ip_risk column)
python train_v3_noip.py --name v3_noip

# members trained with train_member.py (v3_safe feature set as base, 8 threads)
python train_member.py --name m_extra --no-ip --seed 11 --rounds 300 --threads 8 --params "<json>"
#   <json> = {"extra_trees": true, "feature_fraction": 0.4, "min_child_samples": 200}
python train_member.py --name m_down --legit_frac 0.3 --seed 21 --threads 8 --params "<json>"
#   <json> = {"num_leaves": 63}

# blend (equal weights) -> submission_blend_ens5.csv
python blend.py --runs v3_safe,v3_noip,v3_full_new_noexp,m_extra,m_down --tag ens5
```

JSON quoting differs between shells, which is why `reproduce.py` passes the arguments as a list.
The exact settings, dropped columns, scores and multipliers of every run are logged in
`results.csv` (rows `v3_full_new_noexp`, `v3_safe`, the second `v3_noip` row (the first is a
sample run), `m_extra`, `m_down`, `blend_ens5`).

LightGBM is seeded everywhere, but multithreaded training can still give tiny numeric
differences, so a re-run may differ from the uploaded file on a handful of rows.

## Approach

- **Features** (`build_features_v2.py` -> `features_v2.parquet`, 132 columns): per-entity
  (sender, receiver, device) windowed counts and sums (1h/24h), inflow -> outflow link features
  (did the receiver move the money on, and how fast), time gaps to previous/next transaction of the
  same entity, reverse-direction payments between the same pair, amount ratios and round-amount flags, account and merchant
  attributes. Train and test are concatenated so test rows can link to each other (allowed by the
  rules). Raw IDs and timestamps are never model features. `app_version` is dropped at load:
  October only has versions 5.0-5.2, so it separates train from test perfectly and carries no signal.
- **Validation**: temporal split. Train on Jul-Aug, validate on Sep. Macro-F1 is checked every 25
  rounds to pick the number of trees. The final model is retrained on Jul-Sep with
  best_iteration x 1.2 trees.
- **Per-class multipliers**: the predicted class is argmax(p x w), with legit fixed at w = 1.0.
  w is tuned by coordinate search on one random half of September and checked on the other half
  (seed 42), which keeps the reported score honest.
- **Members**:
  - `v3_safe`: drops 17 expanding-history and density-sensitive window features (see below).
  - `v3_full_new_noexp`: drops only the 9 expanding-history features.
  - `v3_noip`: drops all `ip_risk` columns (the vendor score drifts in October).
  - `m_extra`: extra_trees, feature_fraction 0.4, no `ip_risk`, different seed.
  - `m_down`: legit downsampled to 30% in training; the legit probability is rescaled by 0.3 and
    rows renormalized before saving, so its probabilities are on the same scale as the others.
- **Blend** (`blend.py`): weighted average of member probabilities, multipliers re-tuned on the
  blended validation probabilities and applied to the blended test probabilities.

## Results

Validation (September, tuned multipliers, held-out half):

| Run | Eval-half Macro-F1 |
|---|---|
| v3_full_new_noexp | 0.4655 |
| v3_safe | 0.4667 |
| v3_noip | 0.4606 |
| m_extra | 0.4557 |
| m_down | 0.4713 |
| blend ens5 | 0.4712 |

Public leaderboard:

| Upload | Public score |
|---|---|
| v3_safe (multipliers x1.0 / x1.25 / x1.6) | 0.4907 / 0.4909 / 0.4916 |
| v3_noip | 0.4877 |
| blend ens3 (v3_safe, v3_noip, v3_full_new_noexp) | 0.4924 |
| **blend ens5 (ens3 + m_extra + m_down), equal weights** | **0.4946** |
| blend ens5w (m_down weighted 0.5) | 0.4922 |

Differences below about 0.003 are within noise on the 30% public split. We picked the final
upload as the equal-weight blend, which is also the most stable choice for the private split.

## Findings

- **Label noise**: train labels are noisy (unreported fraud labelled legit, some wrong types)
  while test labels are clean. Validation Macro-F1 is consistently about 0.025 below the public
  score, which fits that.
- **October drift** (adversarial validation): without `app_version`, a train-vs-test classifier
  reaches AUC 0.64. `ip_risk_score` is the top drifting feature (its distribution narrows), and
  amounts are about 1.17x higher across quantiles (festival season). Other raw columns are stable.
- **October is thinner**: it has about 0.51x September's rows, uniformly across apps. Randomly
  keeping 50% of September's rows reproduces October's entity statistics (counts per entity, gaps,
  window features) far better than thinning by sender or device. Count and window features learned
  at full density therefore under-fire on test. Dropping the density-sensitive window features
  (`v3_safe`) gave the best single model on the public board.
- **Where the errors are**: about 91% of fraud rows are isolated (no same-device or same-sender
  row within 10 minutes) and look like legit; recall on those is 25-31%, against 75-96% for fraud
  inside bursts. Missed fraud gets a median probability of 0.001-0.005 for its true class.
- **What did not help**: smoothing predictions over neighbouring transactions (best +0.001; even
  smoothing with true labels gave at most +0.007), forward/centred burst and session windows,
  expanding-history features (they drift with the length of history), weighting the blend towards
  the best validation member (ens5w).
- **What helped**: dropping drifting/density-sensitive features, per-class multipliers, legit
  downsampling (`m_down`, +0.0045 on validation over v3_safe) and blending diverse members.

## Repository contents

| File | Purpose |
|---|---|
| `reproduce.py` | One command from raw CSVs to `submission.csv` |
| `build_features_v2.py` | Feature engineering -> `features_v2.parquet` |
| `train_v3.py` | LightGBM member with a `--drop` list (v3_safe, v3_full_new_noexp) |
| `train_v3_noip.py` | LightGBM member without ip_risk features (v3_noip) |
| `train_member.py` | Configurable LightGBM member (m_extra, m_down) |
| `blend.py` | Probability blend + multiplier tuning -> submission |
| `results.csv` | Log of every run: parameters, dropped columns, scores, multipliers |
| `requirements.txt` | Pinned package versions |

No feature is derived from labels. `shift(-1)` in `build_features_v2.py` is used only on
timestamps (seconds to the same entity's next transaction), never on labels; this is legitimate
because test rows can be linked to each other. `train_v3_noip.py` and `train_member.py` also
drop any `_tgt`/`target`/`label`/ID column defensively. Exploratory and diagnostic scripts used
during the round are not included.
