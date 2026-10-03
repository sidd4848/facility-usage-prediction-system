"""Command line entry point:  python -m fup.cli {generate,train,predict,evaluate,all}"""
import argparse

import joblib
import pandas as pd

from . import config as C
from . import evaluate as E
from . import features as F
from . import generate_data, pipeline as P, report, train as T


def load_bookings(path=C.BOOKINGS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["booking_ts", "usage_ts"])


def cmd_generate(a):
    df, prof = generate_data.generate(a.seed)
    C.BOOKINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    C.RESIDENTS_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.BOOKINGS_CSV, index=False)
    prof.to_csv(C.RESIDENTS_TRUTH_CSV, index=False)
    print(f"wrote {len(df):,} bookings -> {C.BOOKINGS_CSV}")


def cmd_train(a):
    T.train(load_bookings())


def cmd_predict(a):
    """Predict the next booking for every booking made after --since (default: start of the test period)."""
    bundle = joblib.load(C.MODEL_PATH)
    since = pd.Timestamp(a.since or C.VAL_END)
    bookings = load_bookings()
    S = F.build_samples(bookings)
    S = S.subset((S.meta.target_booking_ts > since).to_numpy())
    pred = bundle["pipeline"].predict(S)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(C.PREDICTIONS_CSV, index=False)
    for kind in ("repeat_last", "resident_mode"):
        P.baseline_predict(S, kind).to_csv(C.OUT_DIR / f"baseline_{kind}_predictions.csv", index=False)
    print(f"{len(pred):,} predictions (targets after {since.date()}) -> {C.PREDICTIONS_CSV}")


def cmd_evaluate(a):
    bookings = load_bookings()
    pred = pd.read_csv(C.PREDICTIONS_CSV)
    m = E.join_actuals(pred, bookings)
    mt = E.metrics(m)
    base = {}
    for kind, label in (("repeat_last", "Baseline: repeat last booking"), ("resident_mode", "Baseline: resident most-frequent")):
        bp = pd.read_csv(C.OUT_DIR / f"baseline_{kind}_predictions.csv")
        base[label] = E.metrics(E.join_actuals(bp, bookings))
    truth = pd.read_csv(C.RESIDENTS_TRUTH_CSV) if C.RESIDENTS_TRUTH_CSV.exists() else None
    errs = E.error_analysis(m, truth)
    sel = pd.read_csv(C.SELECTION_CSV) if C.SELECTION_CSV.exists() else None
    E.save_metrics({"model": mt, "baselines": base})
    m.to_csv(C.OUT_DIR / "predictions_vs_actual.csv", index=False)
    report.build_workbook(m, mt, base, errs, sel)
    review = report.review_table(m)
    review.to_csv(C.OUT_DIR / "predictions_review.csv", index=False)
    print(f"test records: {mt['n_test_records']:,}")
    for k in ("facility", "usage_day", "usage_hour", "nudge_time", "overall"):
        print(f"  {k}: {mt[k]}")
    for k, v in base.items():
        print(f"  {k}: overall {v['overall']['overall_accuracy_pct']:.1f}%")
    print(f"workbook -> {C.REVIEW_XLSX}")


def main():
    ap = argparse.ArgumentParser(prog="fup")
    ap.add_argument("command", choices=["generate", "train", "predict", "evaluate", "all"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--since", default=None, help="predict only bookings created after this timestamp")
    a = ap.parse_args()
    steps = {"generate": [cmd_generate], "train": [cmd_train], "predict": [cmd_predict], "evaluate": [cmd_evaluate],
             "all": [cmd_generate, cmd_train, cmd_predict, cmd_evaluate]}[a.command]
    for s in steps:
        s(a)


if __name__ == "__main__":
    main()
