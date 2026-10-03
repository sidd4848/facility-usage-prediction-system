# Facility Usage Prediction System

Predicts, for a resident who has just made a booking, their **next** booking: **facility, usage weekday,
usage hour and nudge time** (weekday + time of day at which to notify them). Includes a synthetic
dataset generator, leakage-safe feature pipeline, training / evaluation / prediction workflow and a
reviewable Excel output.

## Setup

```bash
pip install -r requirements.txt      # Python 3.10+ ; tested on 3.12
```

## Run (about 2-5 minutes in total, CPU only, deterministic with seed 42)

```bash
python -m fup.cli all        # generate -> train -> predict -> evaluate
```

or step by step:

```bash
python -m fup.cli generate   # data/bookings.csv (+ data/ground_truth/resident_profiles.csv, analysis only)
python -m fup.cli train      # tunes on validation, refits on train+val, saves models/pipeline.joblib
python -m fup.cli predict    # predictions for bookings created after 2025-04-15 -> outputs/predictions.csv
python -m fup.cli evaluate   # compares with actuals -> outputs/predictions_review.xlsx, metrics.json
python -m pytest -q tests    # leakage tests
```

`predict --since "2025-05-01"` restricts the prediction window. The submitted `data/bookings.csv` is the
dataset used; running `generate` again reproduces it exactly.

## Layout

| Path | Content |
|---|---|
| `data/bookings.csv` | Synthetic dataset (booking_id, resident_id, facility, booking_ts, usage_ts, duration) |
| `fup/` | `generate_data`, `features`, `pipeline`, `train`, `evaluate`, `report`, `cli` |
| `models/pipeline.joblib` | Trained model artifact |
| `outputs/predictions_review.xlsx` | **Review workbook**: Summary, Predictions vs Actual, Error analysis, Validation comparison |
| `outputs/predictions_review.csv`, `predictions_vs_actual.csv`, `metrics.json` | Same results in plain files |
| `docs/TECHNICAL_DOCUMENTATION.md` | Data, features, modelling, evaluation, results, limitations |
| `tests/test_leakage.py` | Tests that features cannot see the future / the target row |

## Headline result (2,280 held-out records, chronologically after all training/tuning data)

| Output | Model | Resident-mode baseline |
|---|---|---|
| Facility (top-1) | 56.2% | 51.6% |
| Usage day (exact) | 37.7% | 33.3% |
| Usage hour (exact) | 38.2% | 36.2% |
| Nudge (weekday + within 60 min) | 24.7% | 19.9% |
| **Overall (mean of 4 outputs)** | **39.2%** | 35.3% |
| All 4 correct | 12.8% | 10.1% |

See the documentation for definitions, error analysis and limitations.
