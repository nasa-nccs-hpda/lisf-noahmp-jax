"""Monthly interpolation weights of the Noah-MP 5.0 input LAI (dveg 7-9), as LIS_computeTemporalWeights('monthly')."""
import datetime as dt
import numpy as np
from lisf_noahmp_jax.driver.run_mp import LaiTracker


def test_weights_pairs_and_sum():
    w = LaiTracker.weights
    assert w(dt.datetime(2017, 7, 14, 6))[:2] == (6, 7)
    assert w(dt.datetime(2017, 7, 16, 0))[:2] == (7, 8)
    assert w(dt.datetime(2017, 1, 5))[:2] == (12, 1)
    assert w(dt.datetime(2017, 12, 20))[:2] == (12, 1)
    for d in (dt.datetime(2017, 7, 14, 6), dt.datetime(2017, 1, 5), dt.datetime(2016, 3, 10), dt.datetime(2017, 12, 31)):
        a, b, w1, w2 = w(d)
        assert abs(float(w1) + float(w2) - 1.0) < 1e-6 and 0.0 <= w1 <= 1.0 and 0.0 <= w2 <= 1.0


def test_tlai_interpolates_between_maps():
    L = np.zeros((12, 3)); L[5] = 1.0; L[6] = 2.0                     # June 1, July 2
    t = LaiTracker(L, 900)
    v = t.tlai(dt.datetime(2017, 7, 15, 12))                          # July 15 noon: weights close to July
    assert np.all(v > 1.9) and np.all(v <= 2.0)
    v = t.tlai(dt.datetime(2017, 7, 1, 0))                            # July 1: about halfway between the June and July centres, closer to June
    assert np.all((v > 1.4) & (v < 1.7))
