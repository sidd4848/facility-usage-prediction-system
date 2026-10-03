"""Shared constants. Everything that defines the problem lives here."""
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")   # silence joblib core-count warning on Windows

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

BOOKINGS_CSV = DATA_DIR / "bookings.csv"
RESIDENTS_TRUTH_CSV = DATA_DIR / "ground_truth" / "resident_profiles.csv"
MODEL_PATH = MODEL_DIR / "pipeline.joblib"
PREDICTIONS_CSV = OUT_DIR / "predictions.csv"
METRICS_JSON = OUT_DIR / "metrics.json"
REVIEW_XLSX = OUT_DIR / "predictions_review.xlsx"
SELECTION_CSV = OUT_DIR / "model_selection_validation.csv"

SEED = 42

# ---- Facilities ------------------------------------------------------------
FACILITY_IDS = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08"]
FACILITY_NAMES = ["Gym", "Pool", "Tennis", "Badminton", "Basketball", "Clubhouse", "Hall", "Yoga"]
NF = len(FACILITY_IDS)
FAC_IDX = {fid: i for i, fid in enumerate(FACILITY_IDS)}

HOURS = list(range(5, 23))          # bookable start hours 05:00 .. 22:00
NH = len(HOURS)
HOUR_MIN = HOURS[0]

LEADS = np.array([0, 1, 2, 3, 5, 7, 10, 14])   # booking lead (calendar days) classes
NL = len(LEADS)

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---- Dataset window / chronological split (by booking timestamp) -----------
DATA_START = "2024-01-01"
DATA_END = "2025-06-30 23:59:59"
TRAIN_END = "2025-02-28 23:59:59"    # train : booking_ts <= this
VAL_END = "2025-04-15 23:59:59"      # val   : (TRAIN_END, VAL_END]  (tuning only)
# test  : booking_ts > VAL_END  (never used for fitting or tuning)

# ---- Feature-window parameters ----------------------------------------------
HL_RES_SHORT_DAYS = 45.0     # recency half-life, resident-level stats
HL_GLOB_SHORT_DAYS = 30.0    # recency half-life, community-level stats

# ---- Evaluation ---------------------------------------------------------------
NUDGE_TOL_MIN = 60           # nudge counts as correct if same weekday and within +-60 min
