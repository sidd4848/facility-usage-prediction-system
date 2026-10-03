"""Leakage-safe, point-in-time feature construction.

Design
------
One *sample* = "a resident has just made booking k (the *origin*); predict their next booking k+1
(the *target*)". The bookings table is walked once in booking-timestamp order. After booking k is folded
into the running state, a snapshot of that state is frozen as the sample's feature source. Only afterwards
does the walk move on, so by construction a sample's features can only see:
  * the resident's own bookings 1..k   (their scheduled usage time is known at booking time), and
  * community-wide bookings made at or before the origin's booking timestamp.
The target row's facility / usage / booking time are used solely as labels. No feature uses the target's
booking timestamp (that would leak the "nudge time" answer).

State = decayed counts at two horizons: [0] all-history (no decay) and [1] recency-weighted
(exponential half-life 45 d per resident, 30 d community). Sparse resident counts are shrunk towards the
community distribution (Bayesian-style smoothing), which also handles cold-start residents.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C

NF, NH, NL = C.NF, C.NH, C.NL
_T0 = pd.Timestamp(C.DATA_START)


# --------------------------------------------------------------------------------------------------
# running statistics
# --------------------------------------------------------------------------------------------------
class Stats:
    """Count tables at two recency horizons (index 0 = no decay, index 1 = exponential decay)."""

    NAMES = ("fac", "dow", "hr", "lead", "tod", "trans")

    def __init__(self, half_life_days: float):
        self.hl = half_life_days
        self.t = None
        self.fac = np.zeros((2, NF))
        self.dow = np.zeros((2, NF, 7))
        self.hr = np.zeros((2, NF, 2, NH))
        self.lead = np.zeros((2, NF, NL))
        self.tod = np.zeros((2, NF, 3))        # weight, sum, sum of squares (minutes of day)
        self.trans = np.zeros((2, NF, NF))     # previous facility -> next facility

    def decay(self, t_days: float):
        if self.t is not None and t_days > self.t:
            k = 0.5 ** ((t_days - self.t) / self.hl)
            for n in self.NAMES:
                getattr(self, n)[1] *= k
        if self.t is None or t_days > self.t:
            self.t = t_days

    def update(self, f, dow, hi, lb, tod, prev_f):
        dt = 1 if dow >= 5 else 0
        self.fac[:, f] += 1
        self.dow[:, f, dow] += 1
        self.hr[:, f, dt, hi] += 1
        self.lead[:, f, lb] += 1
        self.tod[:, f, 0] += 1
        self.tod[:, f, 1] += tod
        self.tod[:, f, 2] += tod * tod
        if prev_f >= 0:
            self.trans[:, prev_f, f] += 1

    def frozen(self):
        s = Stats.__new__(Stats)
        s.hl, s.t = self.hl, self.t
        for n in self.NAMES:
            setattr(s, n, getattr(self, n).astype(np.float32))
        return s


@dataclass
class ResState:
    stats: Stats
    n: int = 0
    first_t: float = 0.0
    last_f: int = -1
    last_dow: int = -1
    last_hr: int = -1
    last_lb: int = -1
    last_tod: float = -1.0
    last_booking_t: float = 0.0
    prev_booking_gap: float = -1.0
    last_dow_f: np.ndarray = field(default_factory=lambda: np.full(NF, -1.0))
    last_hr_fd: np.ndarray = field(default_factory=lambda: np.full((NF, 2), -1.0))
    last_lb_f: np.ndarray = field(default_factory=lambda: np.full(NF, -1.0))
    last_tod_f: np.ndarray = field(default_factory=lambda: np.full(NF, -1.0))
    last_use_t_f: np.ndarray = field(default_factory=lambda: np.full(NF, np.nan))
    hist: list = field(default_factory=list)       # last two bookings, for human-readable review only


@dataclass
class Snap:
    res: Stats
    glob: Stats
    ex: dict


@dataclass
class Samples:
    meta: pd.DataFrame
    snaps: list

    def __len__(self):
        return len(self.meta)

    def subset(self, mask):
        idx = np.flatnonzero(np.asarray(mask))
        return Samples(self.meta.iloc[idx].reset_index(drop=True), [self.snaps[i] for i in idx])


# --------------------------------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------------------------------
def label_arrays(df: pd.DataFrame):
    """Row-wise label/derived values used both for state updates and as targets."""
    fac = df["facility_id"].map(C.FAC_IDX).to_numpy()
    u, b = df["usage_ts"], df["booking_ts"]
    dow = u.dt.dayofweek.to_numpy()
    hour = u.dt.hour.to_numpy()
    hi = np.clip(hour - C.HOUR_MIN, 0, NH - 1)
    lead_days = (u.dt.normalize() - b.dt.normalize()).dt.days.to_numpy()
    lb = np.abs(lead_days[:, None] - C.LEADS[None, :]).argmin(1)
    tod = (b.dt.hour * 60 + b.dt.minute + b.dt.second / 60.0).to_numpy()
    return dict(fac=fac, dow=dow, hour=hour, hi=hi, lead_days=lead_days, lb=lb, tod=tod,
                bdow=b.dt.dayofweek.to_numpy())


def _fmt_hist(name, u, b):
    return f"{name} / {C.DOW_NAMES[u.dayofweek]} / {u:%H:%M} / {C.DOW_NAMES[b.dayofweek]} {b:%H:%M}"


# --------------------------------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------------------------------
def build_samples(df: pd.DataFrame) -> Samples:
    df = df.sort_values(["booking_ts", "booking_id"], kind="stable").reset_index(drop=True)
    L = label_arrays(df)
    n = len(df)
    res_ids = df["resident_id"].to_numpy()
    nxt = np.full(n, -1)
    last_seen = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last_seen.get(res_ids[i], -1)
        last_seen[res_ids[i]] = i

    bt = ((df["booking_ts"] - _T0).dt.total_seconds() / 86400.0).to_numpy()
    ut = ((df["usage_ts"] - _T0).dt.total_seconds() / 86400.0).to_numpy()
    glob = Stats(C.HL_GLOB_SHORT_DAYS)
    states: dict = {}
    rows, snaps = [], []

    for i in range(n):
        rid, t = res_ids[i], bt[i]
        st = states.get(rid)
        if st is None:
            st = states[rid] = ResState(Stats(C.HL_RES_SHORT_DAYS), first_t=t)
        st.stats.decay(t)
        glob.decay(t)
        f, dow, hi, lb, tod = int(L["fac"][i]), int(L["dow"][i]), int(L["hi"][i]), int(L["lb"][i]), L["tod"][i]
        st.stats.update(f, dow, hi, lb, tod, st.last_f)
        glob.update(f, dow, hi, lb, tod, st.last_f)
        dt = 1 if dow >= 5 else 0
        st.prev_booking_gap = (t - st.last_booking_t) if st.n else -1.0
        st.n += 1
        st.last_f, st.last_dow, st.last_hr, st.last_lb, st.last_tod = f, dow, hi, lb, tod
        st.last_booking_t = t
        st.last_dow_f[f], st.last_hr_fd[f, dt], st.last_lb_f[f], st.last_tod_f[f] = dow, hi, lb, tod
        st.last_use_t_f[f] = ut[i]
        st.hist = (st.hist + [_fmt_hist(C.FACILITY_NAMES[f], df["usage_ts"].iloc[i], df["booking_ts"].iloc[i])])[-2:]

        j = nxt[i]
        if j < 0:
            continue
        ts_i = df["booking_ts"].iloc[i]
        ex = dict(
            n=st.n,
            rate=st.n / max(1.0, (t - st.first_t) / 7.0 + 1.0),
            last_f=f, last_dow=dow, last_hr=hi, last_lb=lb, last_tod=tod,
            origin_dow=ts_i.dayofweek, origin_tod_h=ts_i.hour + ts_i.minute / 60.0,
            origin_doy_sin=np.sin(2 * np.pi * ts_i.dayofyear / 365.25),
            origin_doy_cos=np.cos(2 * np.pi * ts_i.dayofyear / 365.25),
            origin_use_dow=dow, origin_use_hr=hi, origin_gap_days=ut[i] - t,
            prev_gap=st.prev_booking_gap,
            last_dow_f=st.last_dow_f.copy(), last_hr_fd=st.last_hr_fd.copy(), last_lb_f=st.last_lb_f.copy(),
            last_tod_f=st.last_tod_f.copy(),
            days_since_use_f=np.nan_to_num(t - st.last_use_t_f, nan=999.0),
        )
        snaps.append(Snap(st.stats.frozen(), glob.frozen(), ex))
        rows.append(dict(
            resident_id=rid, origin_booking_id=df["booking_id"].iloc[i], origin_ts=ts_i,
            target_booking_id=df["booking_id"].iloc[j], target_booking_ts=df["booking_ts"].iloc[j],
            n_hist=st.n, past_1=st.hist[0] if len(st.hist) == 2 else "", past_2=st.hist[-1],
            y_fac=L["fac"][j], y_dow=L["dow"][j], y_hour=L["hour"][j], y_hi=L["hi"][j],
            y_lead_days=L["lead_days"][j], y_lb=L["lb"][j], y_tod=L["tod"][j], y_bdow=L["bdow"][j],
        ))
    return Samples(pd.DataFrame(rows), snaps)


# --------------------------------------------------------------------------------------------------
# feature builders (pure functions of a Snap + upstream decisions)
# --------------------------------------------------------------------------------------------------
def _norm(x):
    x = np.asarray(x, np.float64)
    return x / x.sum()


def _shrink(counts, prior, a):
    counts = np.asarray(counts, np.float64)
    return (counts + a * prior) / (counts.sum() + a)


def _oh(i, k):
    v = np.zeros(k)
    if i >= 0:
        v[int(i)] = 1.0
    return v


def _ex_scalars(ex):
    return [ex["origin_dow"], ex["origin_tod_h"], ex["origin_doy_sin"], ex["origin_doy_cos"],
            ex["origin_use_dow"], ex["origin_use_hr"], ex["origin_gap_days"]]


def feat_fac(s: Snap) -> np.ndarray:
    r, g, ex = s.res, s.glob, s.ex
    gl, gs = _norm(g.fac[0] + 1), _norm(g.fac[1] + 0.5)
    rl, rs = _shrink(r.fac[0], gl, 3.0), _shrink(r.fac[1], gs, 2.0)
    trans = _norm(g.trans[0][ex["last_f"]] + 0.5)
    sc = [np.log1p(ex["n"]), ex["rate"], r.fac[1].sum(), ex["prev_gap"], ex["last_dow"], ex["last_hr"]] + _ex_scalars(ex)
    return np.concatenate([rl, rs, gs, trans, _oh(ex["last_f"], NF), np.array(sc, np.float64)])


def feat_day(s: Snap, f: int) -> np.ndarray:
    r, g, ex = s.res, s.glob, s.ex
    gl, gs = _norm(g.dow[0][f] + 1), _norm(g.dow[1][f] + 0.5)
    rl, rs = _shrink(r.dow[0][f], gl, 2.0), _shrink(r.dow[1][f], gs, 2.0)
    ra = _shrink(r.dow[0].sum(0), np.full(7, 1 / 7), 2.0)
    sc = [np.log1p(r.fac[0][f]), r.fac[1][f], rl.argmax(), rs.argmax(), rl.max(), ex["last_dow_f"][f],
          ex["last_dow"], ex["days_since_use_f"][f], ex["n"]] + _ex_scalars(ex)
    return np.concatenate([_oh(f, NF), rl, rs, ra, gs, np.array(sc, np.float64)])


def feat_hour(s: Snap, f: int, dow: int) -> np.ndarray:
    r, g, ex = s.res, s.glob, s.ex
    dt = 1 if dow >= 5 else 0
    gl, gs = _norm(g.hr[0][f][dt] + 1), _norm(g.hr[1][f][dt] + 0.5)
    rl, rs = _shrink(r.hr[0][f][dt], gl, 2.0), _shrink(r.hr[1][f][dt], gs, 2.0)
    ra = _shrink(r.hr[0][f].sum(0), _norm(g.hr[0][f].sum(0) + 1), 2.0)
    sc = [dow, dt, np.log1p(r.hr[0][f][dt].sum()), np.log1p(r.fac[0][f]), rl.argmax(), rs.argmax(), ra.argmax(),
          rl.max(), ex["last_hr_fd"][f][dt], ex["last_hr_fd"][f][1 - dt], ex["last_hr"], ex["n"]] + _ex_scalars(ex)
    return np.concatenate([_oh(f, NF), rl, rs, ra, gl, np.array(sc, np.float64)])


def feat_lead(s: Snap, f: int, dow: int, hi: int) -> np.ndarray:
    r, g, ex = s.res, s.glob, s.ex
    gl = _norm(g.lead[0][f] + 1)
    rl, rs = _shrink(r.lead[0][f], gl, 2.0), _shrink(r.lead[1][f], gl, 2.0)
    ra = _shrink(r.lead[0].sum(0), _norm(g.lead[0].sum(0) + 1), 2.0)
    sc = [dow, hi, np.log1p(r.lead[0][f].sum()), rl.argmax(), ex["last_lb_f"][f], ex["last_lb"], ex["n"]]
    return np.concatenate([_oh(f, NF), rl, rs, ra, gl, np.array(sc, np.float64)])


def _tod_stats(arr, prior_mean, k=2.0):
    w, sm, ss = arr
    mean = (sm + k * prior_mean) / (w + k)
    var = max(ss / w - (sm / w) ** 2, 0.0) if w > 0 else 0.0
    return mean, np.sqrt(var), w


def feat_tod(s: Snap, f: int, dow: int, hi: int, lb: int) -> np.ndarray:
    r, g, ex = s.res, s.glob, s.ex
    gw = g.tod[0].sum(0)
    gmean = gw[1] / max(gw[0], 1.0)
    gf = g.tod[0][f]
    gfm = gf[1] / max(gf[0], 1.0) if gf[0] > 0 else gmean
    sc = []
    for h in (0, 1):
        m_all, sd_all, w_all = _tod_stats(r.tod[h].sum(0), gmean, 2.0)
        m_f, sd_f, _ = _tod_stats(r.tod[h][f], m_all, 2.0)
        sc += [m_all, sd_all, w_all, m_f, sd_f]
    sc += [gfm, ex["last_tod"], ex["last_tod_f"][f], C.HOURS[hi] * 60, C.LEADS[lb], lb, dow, ex["n"]]
    return np.concatenate([_oh(f, NF), np.array(sc, np.float64)])


def X_fac(S: Samples):
    return np.vstack([feat_fac(s) for s in S.snaps])


def X_day(S: Samples, f):
    return np.vstack([feat_day(s, int(a)) for s, a in zip(S.snaps, f)])


def X_hour(S: Samples, f, dow):
    return np.vstack([feat_hour(s, int(a), int(b)) for s, a, b in zip(S.snaps, f, dow)])


def X_lead(S: Samples, f, dow, hi):
    return np.vstack([feat_lead(s, int(a), int(b), int(c)) for s, a, b, c in zip(S.snaps, f, dow, hi)])


def X_tod(S: Samples, f, dow, hi, lb):
    return np.vstack([feat_tod(s, int(a), int(b), int(c), int(d)) for s, a, b, c, d in zip(S.snaps, f, dow, hi, lb)])
