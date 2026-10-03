"""Scoring of predictions against the actual bookings, plus error analysis."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from . import config as C
from . import features as F


def join_actuals(pred: pd.DataFrame, bookings: pd.DataFrame) -> pd.DataFrame:
    b = bookings.reset_index(drop=True)
    L = F.label_arrays(b)
    act = pd.DataFrame({
        "target_booking_id": b.booking_id, "act_fac": L["fac"], "act_dow": L["dow"], "act_hour": L["hour"],
        "act_bdow": L["bdow"], "act_lead_days": L["lead_days"],
        "act_tod_min": np.floor(L["tod"]).astype(int),     # clock minute, as shown in the review sheet
        "act_booking_ts": b.booking_ts, "act_usage_ts": b.usage_ts,
    })
    m = pred.merge(act, on="target_booking_id", how="left", validate="one_to_one")
    assert m.act_fac.notna().all(), "prediction refers to a booking id that is not in the dataset"
    m["ok_fac"] = m.pred_fac == m.act_fac
    m["ok_dow"] = m.pred_dow == m.act_dow
    m["ok_hour"] = m.pred_hour == m.act_hour
    m["tod_err_min"] = (m.pred_nudge_tod_min - m.act_tod_min).abs()
    m["ok_nudge"] = (m.pred_nudge_dow == m.act_bdow) & (m.tod_err_min <= C.NUDGE_TOL_MIN)
    m["score"] = m[["ok_fac", "ok_dow", "ok_hour", "ok_nudge"]].sum(axis=1)
    return m


def _circ(a, b, k):
    d = np.abs(a - b) % k
    return np.minimum(d, k - d)


def metrics(m: pd.DataFrame) -> dict:
    n = len(m)
    top3 = (m["pred_fac_top3"].apply(lambda s: s.split(",")) if "pred_fac_top3" in m else None)
    out = {"n_test_records": int(n)}
    fac = {"accuracy": m.ok_fac.mean(),
           "macro_f1": f1_score(m.act_fac, m.pred_fac, average="macro", labels=sorted(m.act_fac.unique()), zero_division=0)}
    if top3 is not None:
        fac["top3_accuracy"] = float(np.mean([str(a) in t for a, t in zip(m.act_fac, top3)]))
    out["facility"] = fac
    out["usage_day"] = {"accuracy": m.ok_dow.mean(),
                        "within_1_day": (_circ(m.pred_dow, m.act_dow, 7) <= 1).mean(),
                        "weekend_vs_weekday_accuracy": ((m.pred_dow >= 5) == (m.act_dow >= 5)).mean()}
    out["usage_hour"] = {"accuracy": m.ok_hour.mean(),
                         "within_1_hour": ((m.pred_hour - m.act_hour).abs() <= 1).mean(),
                         "mae_hours": (m.pred_hour - m.act_hour).abs().mean()}
    out["nudge_time"] = {"accuracy": m.ok_nudge.mean(),
                         "booking_weekday_accuracy": (m.pred_nudge_dow == m.act_bdow).mean(),
                         "lead_days_accuracy": (m.pred_lead_days == m.act_lead_days).mean(),
                         "time_of_day_mae_min": m.tod_err_min.mean(),
                         "time_of_day_median_abs_err_min": m.tod_err_min.median(),
                         "time_of_day_within_60min": (m.tod_err_min <= C.NUDGE_TOL_MIN).mean()}
    out["overall"] = {"mean_outputs_correct_of_4": m.score.mean(),
                      "overall_accuracy_pct": m.score.mean() / 4 * 100,
                      "all_4_correct": (m.score == 4).mean(),
                      "at_least_3_correct": (m.score >= 3).mean(),
                      "at_least_2_correct": (m.score >= 2).mean(),
                      "score_distribution": {int(k): int(v) for k, v in m.score.value_counts().sort_index().items()}}

    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        return round(float(x), 4) if isinstance(x, (float, np.floating)) else x
    return clean(out)


def error_analysis(m: pd.DataFrame, truth: pd.DataFrame | None = None) -> dict:
    """Returns {name: DataFrame}."""
    def agg(g):
        return pd.Series({"records": len(g), "facility": g.ok_fac.mean(), "day": g.ok_dow.mean(),
                          "hour": g.ok_hour.mean(), "nudge": g.ok_nudge.mean(), "avg_of_4": g.score.mean(),
                          "all_4": (g.score == 4).mean()})
    res = {}
    bins = pd.cut(m.n_hist, [0, 4, 14, 39, 10_000], labels=["1-4", "5-14", "15-39", "40+"])
    res["by_history_size"] = m.groupby(bins, observed=True).apply(agg, include_groups=False).reset_index()
    res["by_history_size"].columns = ["bookings_before_origin"] + list(res["by_history_size"].columns[1:])

    g = m.assign(act_facility=m.act_fac.map(dict(enumerate(C.FACILITY_NAMES))))
    res["by_actual_facility"] = g.groupby("act_facility").apply(agg, include_groups=False).reset_index()

    cond = m.assign(facility_predicted=np.where(m.ok_fac, "correct", "wrong"))
    res["day_hour_given_facility_correctness"] = cond.groupby("facility_predicted").apply(
        agg, include_groups=False).reset_index()

    conf = (m[~m.ok_fac].assign(actual=lambda d: d.act_fac.map(dict(enumerate(C.FACILITY_NAMES))),
                                predicted=lambda d: d.pred_fac.map(dict(enumerate(C.FACILITY_NAMES))))
            .groupby(["actual", "predicted"]).size().sort_values(ascending=False).head(10)
            .rename("count").reset_index())
    res["top_facility_confusions"] = conf

    if truth is not None:
        t = truth[["resident_id", "drifts", "lead_archetype"]]
        d = m.merge(t, on="resident_id", how="left")
        d["resident_type"] = np.where(d.drifts, "behaviour changed during history", "stable")
        res["by_resident_drift"] = d.groupby("resident_type").apply(agg, include_groups=False).reset_index()
        res["by_lead_archetype"] = d.groupby("lead_archetype").apply(agg, include_groups=False).reset_index()

    dd = (m.pred_hour - m.act_hour)
    res["hour_error_distribution"] = (dd.clip(-3, 3).value_counts(normalize=True).sort_index()
                                      .rename("share").reset_index().rename(columns={"index": "pred_minus_actual_hour"}))
    return res


def save_metrics(obj, path=C.METRICS_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
