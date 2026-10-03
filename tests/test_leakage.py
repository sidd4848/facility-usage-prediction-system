"""Leakage tests: features of a sample may depend only on bookings made up to its origin."""
import numpy as np
import pandas as pd
import pytest

from fup import config as C
from fup import features as F
from fup import generate_data


@pytest.fixture(scope="module")
def df():
    d, _ = generate_data.generate(seed=7, n_residents=25)
    return d


def _feats(S, i):
    s = S.snaps[i]
    return np.concatenate([F.feat_fac(s), F.feat_day(s, 0), F.feat_hour(s, 0, 2), F.feat_lead(s, 0, 2, 3),
                           F.feat_tod(s, 0, 2, 3, 1)])


def test_origin_precedes_target(df):
    S = F.build_samples(df)
    assert (S.meta.origin_ts <= S.meta.target_booking_ts).all()


def test_future_rows_do_not_change_features(df):
    """Dropping everything after a cut-off must not change features of samples whose target is before it."""
    full = F.build_samples(df)
    cut = df.booking_ts.quantile(0.6)
    part = F.build_samples(df[df.booking_ts <= cut])
    key = dict(zip(full.meta.target_booking_id, range(len(full))))
    checked = 0
    for j, tid in enumerate(part.meta.target_booking_id):
        np.testing.assert_allclose(_feats(part, j), _feats(full, key[tid]), rtol=1e-5, atol=1e-5)
        checked += 1
    assert checked > 100


def test_target_row_values_do_not_change_its_own_features(df):
    full = F.build_samples(df)
    j = len(full) // 2
    tid = full.meta.target_booking_id[j]
    mod = df.copy()
    r = mod.index[mod.booking_id == tid][0]
    mod.loc[r, "facility_id"] = "F08" if mod.loc[r, "facility_id"] != "F08" else "F01"
    mod.loc[r, "usage_ts"] = mod.loc[r, "usage_ts"] + pd.Timedelta(days=3, hours=2)
    S2 = F.build_samples(mod)
    j2 = int(np.flatnonzero(S2.meta.target_booking_id == tid)[0])
    np.testing.assert_allclose(_feats(full, j), _feats(S2, j2), rtol=1e-6, atol=1e-6)
    assert full.meta.y_fac[j] != S2.meta.y_fac[j2]            # the label did change, the features did not


def test_split_is_chronological():
    assert pd.Timestamp(C.TRAIN_END) < pd.Timestamp(C.VAL_END)
