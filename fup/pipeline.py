"""Cascaded prediction pipeline: facility -> usage day -> usage hour -> (lead days, booking time-of-day) -> nudge.

Each stage is a gradient-boosted model (sklearn HistGradientBoosting). Stages are *trained* on true upstream
values (teacher forcing). At inference the day / hour / lead stages are mixed over the facility probabilities
(see FacilityPipeline.predict), so inference only ever uses information available at the origin booking.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import config as C
from . import features as F

DEFAULT_PARAMS = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15)
STAGES = ("fac", "day", "hour", "lead", "tod")
MIN_FAC_PROB = 0.02          # facilities less likely than this are ignored in the mixture


def _clf(p):
    return HistGradientBoostingClassifier(min_samples_leaf=20, l2_regularization=1.0, random_state=C.SEED, **p)


def _reg(p):
    return HistGradientBoostingRegressor(loss="absolute_error", min_samples_leaf=20, l2_regularization=1.0,
                                         random_state=C.SEED, **p)


def stage_data(S: F.Samples, stage: str):
    """Teacher-forced (X, y) for a stage."""
    m = S.meta
    f, dow, hi, lb = m.y_fac.to_numpy(), m.y_dow.to_numpy(), m.y_hi.to_numpy(), m.y_lb.to_numpy()
    if stage == "fac":
        return F.X_fac(S), f
    if stage == "day":
        return F.X_day(S, f), dow
    if stage == "hour":
        return F.X_hour(S, f, dow), hi
    if stage == "lead":
        return F.X_lead(S, f, dow, hi), lb
    if stage == "tod":
        return F.X_tod(S, f, dow, hi, lb), m.y_tod.to_numpy()
    raise ValueError(stage)


def _full_proba(model, X, n_classes):
    """predict_proba on all classes, even those absent from the training labels."""
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return out


class FacilityPipeline:
    def __init__(self, params: dict | None = None):
        self.params = {s: dict(DEFAULT_PARAMS) for s in STAGES}
        for k, v in (params or {}).items():
            self.params[k] = v
        self.models = {}

    def fit(self, S: F.Samples):
        for st in STAGES:
            X, y = stage_data(S, st)
            mdl = _reg(self.params[st]) if st == "tod" else _clf(self.params[st])
            self.models[st] = mdl.fit(X, y)
        return self

    def predict(self, S: F.Samples) -> pd.DataFrame:
        """Facility = argmax. Day / hour / lead are the argmax of the facility-probability-weighted mixture
        sum_f P(f) * P(target | f, ...), so a confidently-wrong facility does not drag the other outputs."""
        M, n = self.models, len(S)
        p_fac = _full_proba(M["fac"], F.X_fac(S), C.NF)
        f_hat = p_fac.argmax(1)
        active = [(f, np.flatnonzero(p_fac[:, f] >= MIN_FAC_PROB)) for f in range(C.NF)]

        def mix(n_cls, build, model, **fixed):
            acc = np.zeros((n, n_cls))
            for f, idx in active:
                if len(idx) == 0:
                    continue
                sub = F.Samples(None, [S.snaps[i] for i in idx])
                X = build(sub, np.full(len(idx), f), *[v[idx] for v in fixed.values()])
                acc[idx] += p_fac[idx, f][:, None] * _full_proba(model, X, n_cls)
            return acc

        dow = mix(7, F.X_day, M["day"]).argmax(1)
        hi = mix(C.NH, F.X_hour, M["hour"], dow=dow).argmax(1)
        lb = mix(C.NL, F.X_lead, M["lead"], dow=dow, hi=hi).argmax(1)
        tod_acc, w_acc = np.zeros(n), np.zeros(n)
        for f, idx in active:
            if len(idx) == 0:
                continue
            sub = F.Samples(None, [S.snaps[i] for i in idx])
            X = F.X_tod(sub, np.full(len(idx), f), dow[idx], hi[idx], lb[idx])
            tod_acc[idx] += p_fac[idx, f] * M["tod"].predict(X)
            w_acc[idx] += p_fac[idx, f]
        return _assemble(S, f_hat, dow, hi, lb, tod_acc / w_acc, p_fac)


def _assemble(S, f, dow, hi, lb, tod, p_fac=None):
    lead = C.LEADS[lb]
    hour = np.asarray(hi) + C.HOUR_MIN
    tod = np.clip(tod, 6 * 60, 23 * 60)
    same_day = lead == 0
    tod = np.where(same_day, np.minimum(tod, hour * 60 - 30), tod)       # cannot book after the slot starts
    tod = np.clip(tod, 0, 24 * 60 - 1)
    out = pd.DataFrame({
        "target_booking_id": S.meta.target_booking_id.to_numpy(),
        "resident_id": S.meta.resident_id.to_numpy(),
        "origin_booking_id": S.meta.origin_booking_id.to_numpy(),
        "n_hist": S.meta.n_hist.to_numpy(),
        "past_1": S.meta.past_1.to_numpy(),
        "past_2": S.meta.past_2.to_numpy(),
        "pred_fac": f, "pred_dow": dow, "pred_hour": hour, "pred_lead_days": lead,
        "pred_nudge_dow": (dow - lead) % 7, "pred_nudge_tod_min": np.round(tod).astype(int),   # nudge is sent on a whole minute
    })
    if p_fac is not None:
        top3 = np.argsort(-p_fac, axis=1)[:, :3]
        out["pred_fac_top3"] = [",".join(map(str, r)) for r in top3]
        out["pred_fac_prob"] = np.round(p_fac.max(1), 4)
    return out


# ---------------------------------------------------------------------------------------------------
# baselines (no learning): used for the validation model comparison and the test-set reference rows
# ---------------------------------------------------------------------------------------------------
def baseline_predict(S: F.Samples, kind: str) -> pd.DataFrame:
    f, dow, hi, lb, tod = [], [], [], [], []
    for s in S.snaps:
        r, g, ex = s.res, s.glob, s.ex
        if kind == "repeat_last":
            fi = ex["last_f"]
            d = ex["last_dow_f"][fi]
            h = ex["last_hr_fd"][fi][1 if d >= 5 else 0]
            h = h if h >= 0 else ex["last_hr"]
            l = ex["last_lb_f"][fi]
            t = ex["last_tod_f"][fi]
        elif kind == "resident_mode":
            fi = int(np.argmax(r.fac[0] + 1e-3 * g.fac[0]))
            d = int(np.argmax(r.dow[0][fi] + 1e-3 * g.dow[0][fi]))
            dt = 1 if d >= 5 else 0
            h = int(np.argmax(r.hr[0][fi][dt] + 1e-3 * g.hr[0][fi][dt]))
            l = int(np.argmax(r.lead[0][fi] + 1e-3 * g.lead[0][fi]))
            w = r.tod[0].sum(0)
            t = w[1] / w[0]
        else:
            raise ValueError(kind)
        f.append(fi); dow.append(int(d)); hi.append(int(h)); lb.append(int(l)); tod.append(t)
    return _assemble(S, np.array(f), np.array(dow), np.array(hi), np.array(lb), np.array(tod, float))
