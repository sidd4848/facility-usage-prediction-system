"""Synthetic facility-booking data for one residential community.

Generative story (every bullet is something the model is expected to be able to learn):
  * Facility popularity is imbalanced (Gym >> Hall) and seasonal (pool in summer, courts dip in monsoon,
    hall peaks around festivals). Yoga studio opens mid-history (cold-start facility); the pool is
    closed Mondays and has a two-week maintenance closure.
  * Residents differ in activity level (log-normal -> many sparse residents), in facility preferences
    (peaked Dirichlet), and in habits per (resident, facility): preferred weekdays, a weekday hour and a
    weekend hour, and a typical booking lead time.
  * Each resident has a personal booking time-of-day habit (morning / lunch / afternoon / evening / night).
  * Noise: exploration bookings, off-habit days/hours, random-time bookings.
  * Changing behaviour: ~30% of residents hit a change point after which preferences, habit hours and
    booking time-of-day partly change.
  * Staggered join / leave dates -> cold-start and short histories.
Only the four core fields (+ ids, duration) are written to bookings.csv; the hidden generator profile is
written separately to data/ground_truth/ and is used for error analysis only, never as a model input.
"""
import argparse

import numpy as np
import pandas as pd

from . import config as C

N_RESIDENTS = 320

POP = np.array([0.28, 0.14, 0.10, 0.13, 0.07, 0.10, 0.05, 0.13])
POP = POP / POP.sum()

# weekday prior per facility (Mon..Sun)
DAY_PRIOR = np.array([
    [1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8],    # Gym
    [0.5, 0.7, 0.8, 0.9, 1.1, 2.0, 2.0],    # Pool
    [0.8, 0.8, 0.8, 0.9, 1.0, 1.6, 1.6],    # Tennis
    [1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.2],    # Badminton
    [0.8, 0.8, 0.9, 1.0, 1.0, 1.5, 1.5],    # Basketball
    [0.6, 0.6, 0.7, 0.8, 1.4, 2.0, 1.8],    # Clubhouse
    [0.3, 0.3, 0.3, 0.4, 1.0, 2.5, 2.2],    # Hall
    [1.2, 1.0, 1.2, 1.0, 1.0, 1.4, 1.2],    # Yoga
])


def _hour_profile(peaks: dict, lo: int, hi: int) -> np.ndarray:
    w = np.full(C.NH, 0.03)
    for h, v in peaks.items():
        w[h - C.HOUR_MIN] = v
    for i, h in enumerate(C.HOURS):
        if h < lo or h > hi:
            w[i] = 0.0
    return w / w.sum()


# (weekday profile, weekend profile) per facility, restricted to opening hours
HOUR_PROFILE = [
    (_hour_profile({6: 3, 7: 4, 18: 4, 19: 4, 20: 2}, 5, 21), _hour_profile({7: 3, 8: 4, 9: 3, 17: 2}, 5, 21)),
    (_hour_profile({7: 3, 17: 3, 18: 4, 19: 2}, 6, 20), _hour_profile({8: 2, 10: 3, 11: 3, 16: 3, 17: 3}, 6, 20)),
    (_hour_profile({6: 4, 7: 3, 17: 3, 18: 3}, 6, 21), _hour_profile({6: 4, 7: 4, 8: 3, 16: 2, 17: 2}, 6, 21)),
    (_hour_profile({18: 2, 19: 4, 20: 4, 21: 3}, 6, 22), _hour_profile({9: 2, 17: 3, 18: 3, 19: 3, 20: 2}, 6, 22)),
    (_hour_profile({17: 3, 18: 4, 19: 2}, 6, 21), _hour_profile({7: 3, 8: 3, 16: 3, 17: 3}, 6, 21)),
    (_hour_profile({18: 3, 19: 4, 20: 3}, 10, 22), _hour_profile({11: 2, 16: 3, 18: 3, 19: 3, 20: 2}, 10, 22)),
    (_hour_profile({18: 3, 19: 4}, 9, 22), _hour_profile({11: 3, 12: 3, 18: 4, 19: 3}, 9, 22)),
    (_hour_profile({6: 4, 7: 4, 8: 2, 18: 3}, 6, 20), _hour_profile({7: 3, 8: 4, 9: 3, 17: 2}, 6, 20)),
]

LEAD_ARCH = {
    "planner": np.array([0.02, 0.10, 0.25, 0.25, 0.20, 0.12, 0.05, 0.01]),
    "mixed": np.array([0.10, 0.40, 0.25, 0.12, 0.07, 0.04, 0.015, 0.005]),
    "last_minute": np.array([0.55, 0.35, 0.07, 0.02, 0.01, 0.0, 0.0, 0.0]),
}
LEAD_BIG = np.array([0.0, 0.0, 0.0, 0.15, 0.30, 0.30, 0.20, 0.05])      # Hall / Clubhouse
TOD_MODES = [(18 * 60, 0.35), (12.5 * 60, 0.20), (8 * 60, 0.20), (21 * 60, 0.15), (15.5 * 60, 0.10)]
TOD_MU = [m for m, _ in TOD_MODES]
TOD_P = [p for _, p in TOD_MODES]
DURATION = {0: [60], 1: [60, 90], 2: [60, 90], 3: [60], 4: [60, 90], 5: [120, 180], 6: [180, 240], 7: [60, 75]}


def _norm(x):
    x = np.asarray(x, float)
    return x / x.sum()


def _season_matrix():
    S = np.ones((13, C.NF))
    S[:, 1] = [0, 0.4, 0.5, 1.0, 1.6, 1.8, 1.6, 1.0, 1.0, 1.0, 0.9, 0.6, 0.4]
    S[1, 0] = 1.4
    S[2, 0] = 1.25
    for m in (6, 7, 8, 9):
        S[m, 2] = 0.55
        S[m, 4] = 0.55
    for m in (10, 11, 12):
        S[m, 6] = 1.4
    return S


def _draw_habits(rng, f):
    """Per-(resident, facility) habits: weekday distribution, weekday hour, weekend hour."""
    k = int(rng.choice([1, 2], p=[0.55, 0.45]))
    prefd = rng.choice(7, size=k, replace=False, p=_norm(DAY_PRIOR[f]))
    day = 0.3 * _norm(DAY_PRIOR[f])
    day[prefd] += 0.7 / k
    return dict(day=day,
                hr_wd=int(rng.choice(C.HOURS, p=HOUR_PROFILE[f][0])),
                hr_we=int(rng.choice(C.HOURS, p=HOUR_PROFILE[f][1])))


def _draw_lead_mode(rng, f, arch):
    p = LEAD_BIG if f in (5, 6) else LEAD_ARCH[arch]
    return int(rng.choice(C.NL, p=_norm(p)))


def _draw_pref(rng):
    return 0.90 * rng.dirichlet(POP * 1.2) + 0.10 * POP


def generate(seed: int = C.SEED, n_residents: int = N_RESIDENTS):
    rng = np.random.default_rng(seed)
    days = pd.date_range(C.DATA_START, periods=640, freq="D")
    month_of = days.month.values
    start = pd.Timestamp(C.DATA_START)
    end_min = (pd.Timestamp(C.DATA_END) - start).total_seconds() / 60.0
    S = _season_matrix()

    def day_of(ts):
        return (pd.Timestamp(ts) - start).days

    yoga_open = day_of("2024-09-01")
    pool_maint = (day_of("2025-02-03"), day_of("2025-02-16"))

    def is_open(f, d):
        if f == 7 and d < yoga_open:
            return False
        if f == 1 and (d % 7 == 0 or pool_maint[0] <= d <= pool_maint[1]):
            return False
        return True

    rows, profiles = [], []
    for ri in range(n_residents):
        rid = f"R-{100 + ri}"
        rate = float(np.clip(rng.lognormal(np.log(0.45), 0.9), 0.04, 5.0))
        join_d = 0 if rng.random() < 0.75 else int(rng.integers(0, 480))
        leave_d = int(rng.integers(300, 540)) if rng.random() < 0.10 else 10_000
        arch = str(rng.choice(list(LEAD_ARCH), p=[0.3, 0.4, 0.3]))
        pref = _draw_pref(rng)
        habits = [_draw_habits(rng, f) for f in range(C.NF)]
        lead_mode = [_draw_lead_mode(rng, f, arch) for f in range(C.NF)]
        tod_mu = TOD_MU[int(rng.choice(len(TOD_MU), p=TOD_P))]
        tod_sd = rng.uniform(15, 35)

        drifts = rng.random() < 0.30
        cp_d = int(rng.integers(120, 450)) if drifts else 10_000
        pref2, habits2, tod_mu2 = pref, habits, tod_mu
        if drifts:
            pref2 = 0.35 * pref + 0.65 * _draw_pref(rng)
            habits2 = []
            for f in range(C.NF):
                h = dict(habits[f])
                if rng.random() < 0.5:
                    h["hr_wd"] = int(rng.choice(C.HOURS, p=HOUR_PROFILE[f][0]))
                    h["hr_we"] = int(rng.choice(C.HOURS, p=HOUR_PROFILE[f][1]))
                if rng.random() < 0.3:
                    h["day"] = _draw_habits(rng, f)["day"]
                habits2.append(h)
            if rng.random() < 0.4:
                tod_mu2 = TOD_MU[int(rng.choice(len(TOD_MU), p=TOD_P))]
        profiles.append(dict(resident_id=rid, activity_per_week=round(rate, 3), join_day=join_d,
                             leave_day=None if leave_d == 10_000 else leave_d, lead_archetype=arch,
                             drifts=bool(drifts), change_day=cp_d if drifts else None))

        cursor = float(join_d) + rng.exponential(2.0)
        prev_f = -1
        scale = max(0.5, 7.0 / rate - 3.5)
        while True:
            cursor += rng.exponential(scale)
            cd = int(cursor)
            if cd > min(leave_d, 600):
                break
            after = cd >= cp_d
            P, H, TM = (pref2, habits2, tod_mu2) if after else (pref, habits, tod_mu)

            usage_d = f = None
            for _ in range(5):
                w = (POP if rng.random() < 0.06 else P) * S[month_of[cd], :]
                if prev_f >= 0:
                    w = w.copy()
                    w[prev_f] *= 1.8
                w = w * np.array([is_open(k, cd) for k in range(C.NF)], float)
                if w.sum() == 0:
                    break
                f_try = int(rng.choice(C.NF, p=_norm(w)))
                dd = _norm(H[f_try]["day"]) if rng.random() > 0.12 else np.full(7, 1 / 7)
                wd = int(rng.choice(7, p=dd))
                ud = cd + ((wd - cd) % 7)
                if is_open(f_try, ud):
                    usage_d, f = ud, f_try
                    break
            if f is None:
                continue
            dow = usage_d % 7
            hab = H[f]
            base_h = hab["hr_we"] if dow >= 5 else hab["hr_wd"]
            prof = HOUR_PROFILE[f][1 if dow >= 5 else 0]
            nz = np.nonzero(prof > 0)[0]
            lo, hi = int(nz[0]) + C.HOUR_MIN, int(nz[-1]) + C.HOUR_MIN
            u = rng.random()
            if u < 0.72:
                hour = int(base_h)
            elif u < 0.87:
                hour = int(base_h + rng.choice([-1, 1]))
            else:
                hour = int(rng.choice(C.HOURS, p=prof))
            hour = int(np.clip(hour, lo, hi))
            minute = 30 if rng.random() < 0.03 else 0
            usage_min = usage_d * 1440 + hour * 60 + minute

            lead = int(C.LEADS[lead_mode[f]]) if rng.random() < 0.70 else int(C.LEADS[_draw_lead_mode(rng, f, arch)])
            if rng.random() < 0.12:
                tod = rng.uniform(7 * 60, 22.5 * 60)
            else:
                tod = float(np.clip(rng.normal(TM, tod_sd), 6 * 60, 23 * 60))
            booking_min = (usage_d - lead) * 1440 + tod
            while booking_min > usage_min - 30:
                booking_min -= 1440
            booking_min += rng.uniform(0, 1)  # sub-minute jitter

            prev_f = f
            cursor = max(cursor, usage_d)
            if booking_min < join_d * 1440 or booking_min < 0 or booking_min > end_min:
                continue
            rows.append((rid, C.FACILITY_IDS[f], booking_min, usage_min, int(rng.choice(DURATION[f]))))

    df = pd.DataFrame(rows, columns=["resident_id", "facility_id", "_b", "_u", "duration_min"])
    df["booking_ts"] = (start + pd.to_timedelta(df["_b"], unit="m")).dt.floor("s")
    df["usage_ts"] = start + pd.to_timedelta(df["_u"], unit="m")
    df = df.sort_values(["booking_ts", "resident_id"]).reset_index(drop=True)
    df.insert(0, "booking_id", [f"B{i + 1:06d}" for i in range(len(df))])
    df["facility_name"] = df["facility_id"].map(dict(zip(C.FACILITY_IDS, C.FACILITY_NAMES)))
    df = df[["booking_id", "resident_id", "facility_id", "facility_name", "booking_ts", "usage_ts", "duration_min"]]
    assert (df["usage_ts"] > df["booking_ts"]).all()
    return df, pd.DataFrame(profiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--residents", type=int, default=N_RESIDENTS)
    a = ap.parse_args()
    df, prof = generate(a.seed, a.residents)
    C.BOOKINGS_CSV.parent.mkdir(parents=True, exist_ok=True)
    C.RESIDENTS_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.BOOKINGS_CSV, index=False)
    prof.to_csv(C.RESIDENTS_TRUTH_CSV, index=False)
    print(f"wrote {len(df):,} bookings, {df.resident_id.nunique()} residents -> {C.BOOKINGS_CSV}")
    print(df.facility_name.value_counts().to_string())
    per = df.groupby("resident_id").size()
    print("bookings/resident: median", per.median(), "mean", round(per.mean(), 1), "min", per.min(),
          "max", per.max(), "| <=5:", int((per <= 5).sum()))


if __name__ == "__main__":
    main()
