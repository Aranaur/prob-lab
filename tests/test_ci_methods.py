import numpy as np
import pytest
from ci_methods import compute_ci_mean, compute_ci_proportion, compute_ci_bootstrap

def test_compute_ci_mean():
    np.random.seed(42)
    data = np.random.normal(0, 1, size=(30, 2))
    
    lo_t, hi_t = compute_ci_mean(data, method="t", level=0.95)
    assert lo_t.shape == (2,)
    assert hi_t.shape == (2,)
    assert np.all(lo_t < hi_t)
    
    lo_z, hi_z = compute_ci_mean(data, method="z", level=0.95, sigma=1.0)
    assert lo_z.shape == (2,)
    assert hi_z.shape == (2,)
    assert np.all(lo_z < hi_z)

    with pytest.raises(ValueError):
        compute_ci_mean(data, method="invalid")

def test_compute_ci_proportion():
    k = np.array([5, 10])
    n = 20
    
    lo_w, hi_w = compute_ci_proportion(k, n, method="wald", level=0.95)
    assert lo_w.shape == (2,)
    assert hi_w.shape == (2,)
    
    lo_wi, hi_wi = compute_ci_proportion(k, n, method="wilson", level=0.95)
    assert lo_wi.shape == (2,)
    
    lo_cp, hi_cp = compute_ci_proportion(k, n, method="clopper_pearson", level=0.95)
    assert lo_cp.shape == (2,)

def test_compute_ci_bootstrap():
    np.random.seed(42)
    data = np.random.normal(0, 1, size=(30, 2))
    
    lo, hi = compute_ci_bootstrap(data, level=0.95, statistic="mean", B=100)
    assert lo.shape == (2,)
    assert hi.shape == (2,)
    assert np.all(lo < hi)
