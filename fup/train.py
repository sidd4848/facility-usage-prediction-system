"""Training: chronological split, validation-based tuning, model comparison, final refit, artifact save.

  train   : target booking_ts <= TRAIN_END
  val     : TRAIN_END < target booking_ts <= VAL_END   -> hyper-parameter choice + model comparison only
  test    : target booking_ts > VAL_END               -> never touched here
The final model is refit on train+val with the chosen hyper-parameters.
"""
import itertools
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as F
from . import pipeline as P

GRID = [dict(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves)
        for (lr, it), leaves in itertools.product([(0.05, 200), (0.1, 100)], [8, 24])]
N_CLASSES = {"fac": C.NF, "day": 7, "hour": C.NH, "lead": C.NL}


def split_masks(samples: F.Samples):
    t = samples.meta.target_booking_ts
    tr = (t <= pd.Timestamp(C.TRAIN_END)).to_numpy()
    va = ((t > pd.Timestamp(C.TRAIN_END)) & (t <= pd.Timestamp(C.VAL_END))).to_numpy()
    te = (t > pd.Timestamp(C.VAL_END)).to_numpy()
    return tr, va, te


def _val_score(stage, params, Xtr, ytr, Xva, yva):
    if stage == "tod":
        m = P._reg(params).fit(Xtr, ytr)
        return float(np.mean(np.abs(m.predict(Xva) - yva))), m
    m = P._clf(params).fit(Xtr, ytr)
    proba = P._full_proba(m, Xva, N_CLASSES[stage])
    return float(log_loss(yva, np.clip(proba, 1e-6, 1), labels=list(range(N_CLASSES[stage])))), m


def _baseline_stage_acc(S: F.Samples, stage: str, kind: str):
    """Stage-wise (teacher-forced) accuracy of rule baselines, for the model-comparison table."""
    m, hit = S.meta, []
    for s, row in zip(S.snaps, m.itertuples()):
        r, g, ex = s.res, s.glob, s.ex
        if stage == "fac":
            pred = ex["last_f"] if kind == "repeat_last" else int(np.argmax(r.fac[0] + 1e-3 * g.fac[0])) \
                if kind == "resident_mode" else int(np.argmax(g.fac[0]))
            hit.append(pred == row.y_fac)
        elif stage == "day":
            f = row.y_fac
            pred = ex["last_dow_f"][f] if kind == "repeat_last" else int(np.argmax(r.dow[0][f] + 1e-3 * g.dow[0][f]))
            hit.append(pred == row.y_dow)
        elif stage == "hour":
            f, dt = row.y_fac, 1 if row.y_dow >= 5 else 0
            pred = ex["last_hr_fd"][f][dt] if kind == "repeat_last" else int(np.argmax(r.hr[0][f][dt] + 1e-3 * g.hr[0][f][dt]))
            hit.append(pred == row.y_hi)
    return float(np.mean(hit))


def train(bookings: pd.DataFrame, verbose=True):
    S = F.build_samples(bookings)
    tr, va, te = split_masks(S)
    Str, Sva = S.subset(tr), S.subset(va)
    if verbose:
        print(f"samples: train {tr.sum():,} | val {va.sum():,} | test {te.sum():,} (test untouched)")

    rows, best = [], {}
    for st in P.STAGES:
        Xtr, ytr = P.stage_data(Str, st)
        Xva, yva = P.stage_data(Sva, st)
        results = []
        for params in GRID:
            sc, _ = _val_score(st, params, Xtr, ytr, Xva, yva)
            results.append((sc, params))
            rows.append(dict(stage=st, model="HistGradientBoosting", config=str(params),
                             metric="MAE_min" if st == "tod" else "log_loss", value=round(sc, 4)))
        best[st] = min(results, key=lambda r: r[0])[1]
        if verbose:
            print(f"  [{st}] best params {best[st]}  val={min(results, key=lambda r: r[0])[0]:.4f}")

        # --- alternatives, validation only -------------------------------------------------------
        mdl = P._reg(best[st]).fit(Xtr, ytr) if st == "tod" else P._clf(best[st]).fit(Xtr, ytr)
        if st == "tod":
            rows.append(dict(stage=st, model="HGB (chosen)", config="", metric="MAE_min",
                             value=round(float(np.mean(np.abs(mdl.predict(Xva) - yva))), 2)))
            w = np.array([s.res.tod[0].sum(0) for s in Sva.snaps])
            rows.append(dict(stage=st, model="resident mean booking time (baseline)", config="", metric="MAE_min",
                             value=round(float(np.mean(np.abs(w[:, 1] / w[:, 0] - yva))), 2)))
            rows.append(dict(stage=st, model="community mean booking time (baseline)", config="", metric="MAE_min",
                             value=round(float(np.mean(np.abs(np.mean(ytr) - yva))), 2)))
            continue
        rows.append(dict(stage=st, model="HGB (chosen)", config="", metric="accuracy",
                         value=round(float((mdl.predict(Xva) == yva).mean()), 4)))
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, C=0.5)).fit(Xtr, ytr)
        rows.append(dict(stage=st, model="Logistic regression", config="C=0.5", metric="accuracy",
                         value=round(float((lr.predict(Xva) == yva).mean()), 4)))
        if st in ("fac", "day", "hour"):
            kinds = ["global_most_popular", "resident_mode", "repeat_last"] if st == "fac" else ["resident_mode", "repeat_last"]
            for k in kinds:
                rows.append(dict(stage=st, model=f"baseline: {k}", config="", metric="accuracy",
                                 value=round(_baseline_stage_acc(Sva, st, k), 4)))
    sel = pd.DataFrame(rows)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    sel.to_csv(C.SELECTION_CSV, index=False)

    final = P.FacilityPipeline(best).fit(S.subset(tr | va))
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "params": best, "train_end": C.TRAIN_END, "val_end": C.VAL_END,
                 "sklearn": sklearn.__version__, "python": platform.python_version(), "seed": C.SEED},
                C.MODEL_PATH, compress=3)
    if verbose:
        print(f"saved {C.MODEL_PATH} | validation comparison -> {C.SELECTION_CSV}")
    return final, sel
