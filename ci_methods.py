# =============================================================================
# ci_methods.py — Confidence-interval methods
#
# Three explicit public functions — no universal API:
#   compute_ci_mean        → t-interval / z-interval for the mean
#   compute_ci_proportion  → Wald / Wilson / Clopper-Pearson for proportions
#   compute_ci_bootstrap   → percentile bootstrap for arbitrary statistics
#
# All functions accept either scalar / 1-D or batched 2-D input and return
# a (lower, upper) tuple of numpy arrays (or floats for scalar inputs).
#
# Used by: CI Explorer (root server.py) and Bayesian Explorer Tab 2.
# =============================================================================

from __future__ import annotations

import numpy as np
from scipy import stats


# ── Mean CI ──────────────────────────────────────────────────────────────────

def compute_ci_mean(
    data: np.ndarray,
    method: str,
    level: float = 0.95,
    sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence interval for the mean.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n,)`` for a single sample, or ``(n, k)`` for ``k`` batched
        samples of size ``n``. Values reduced along axis 0.
    method : {"t", "z"}
        ``"t"`` — Student's t with unknown σ (uses ``s``).
        ``"z"`` — Normal with known σ (``sigma`` required).
    level : float
        Nominal confidence level in ``(0, 1)``.
    sigma : float | None
        Population standard deviation. Required for ``method="z"``.

    Returns
    -------
    (lo, hi) : tuple of np.ndarray
        Lower/upper bounds with shape ``(k,)`` (or scalar if ``data`` is 1-D).
    """
    if method not in ("t", "z"):
        raise ValueError(f"compute_ci_mean: method must be 't' or 'z', got {method!r}")
    if not (0.0 < level < 1.0):
        raise ValueError(f"compute_ci_mean: level must be in (0, 1), got {level}")
    if method == "z" and sigma is None:
        raise ValueError("compute_ci_mean: sigma is required for method='z'")

    arr = np.asarray(data, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(f"compute_ci_mean: data must be 1-D or 2-D, got ndim={arr.ndim}")

    n = arr.shape[0]
    if n < 2:
        raise ValueError("compute_ci_mean: need at least 2 observations")

    estimates = arr.mean(axis=0)
    alpha = 1.0 - level

    if method == "t":
        stds = arr.std(axis=0, ddof=1)
        ses = stds / np.sqrt(n)
        crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    else:  # "z"
        ses = float(sigma) / np.sqrt(n)
        crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

    return estimates - crit * ses, estimates + crit * ses


# ── Proportion CI ────────────────────────────────────────────────────────────

def compute_ci_proportion(
    k: int | np.ndarray,
    n: int,
    method: str,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int | np.ndarray
        Number of successes. Scalar for one experiment, 1-D array for batches.
    n : int
        Total number of Bernoulli trials per experiment (same for all batches).
    method : {"wald", "wilson", "clopper_pearson"}
    level : float
        Nominal confidence level in ``(0, 1)``.

    Returns
    -------
    (lo, hi) : tuple of np.ndarray
        Lower/upper proportion bounds (clipped to ``[0, 1]`` except Wald,
        which is intentionally left unclipped to expose its failure mode).
    """
    if method not in ("wald", "wilson", "clopper_pearson"):
        raise ValueError(
            f"compute_ci_proportion: method must be 'wald' / 'wilson' / 'clopper_pearson', "
            f"got {method!r}"
        )
    if not (0.0 < level < 1.0):
        raise ValueError(f"compute_ci_proportion: level must be in (0, 1), got {level}")
    if n <= 0:
        raise ValueError(f"compute_ci_proportion: n must be positive, got {n}")

    k_arr = np.asarray(k)
    phat = k_arr / n

    alpha = 1.0 - level
    zc = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if method == "wald":
        se = np.sqrt(phat * (1.0 - phat) / n)
        return phat - zc * se, phat + zc * se

    if method == "wilson":
        z2 = zc ** 2
        denom = 1.0 + z2 / n
        ctr = (phat + z2 / (2.0 * n)) / denom
        marg = zc * np.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n ** 2)) / denom
        return np.clip(ctr - marg, 0.0, 1.0), np.clip(ctr + marg, 0.0, 1.0)

    # clopper_pearson
    k_int = np.round(k_arr).astype(int)
    lo = np.where(
        k_int == 0, 0.0,
        stats.beta.ppf(alpha / 2.0, k_int, n - k_int + 1),
    )
    hi = np.where(
        k_int == n, 1.0,
        stats.beta.ppf(1.0 - alpha / 2.0, k_int + 1, n - k_int),
    )
    return lo, hi


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def compute_ci_bootstrap(
    data: np.ndarray,
    level: float = 0.95,
    statistic: str = "mean",
    p_level: int = 25,
    B: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile-bootstrap CI for the given statistic.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n,)`` or ``(n, k)``. Resampling is performed along axis 0.
    level : float
        Nominal confidence level in ``(0, 1)``.
    statistic : {"mean", "median", "variance", "percentile"}
    p_level : int
        Percentile (0–100) used only when ``statistic="percentile"``.
    B : int
        Number of bootstrap resamples.
    rng : np.random.Generator | None
        Optional seeded RNG for reproducibility. Defaults to ``np.random``.

    Returns
    -------
    (lo, hi) : tuple of np.ndarray
        Lower/upper percentile bounds with shape ``(k,)`` (or scalar).
    """
    if statistic not in ("mean", "median", "variance", "percentile"):
        raise ValueError(
            f"compute_ci_bootstrap: statistic must be 'mean' / 'median' / "
            f"'variance' / 'percentile', got {statistic!r}"
        )
    if not (0.0 < level < 1.0):
        raise ValueError(f"compute_ci_bootstrap: level must be in (0, 1), got {level}")
    if B <= 0:
        raise ValueError(f"compute_ci_bootstrap: B must be positive, got {B}")
    if statistic == "percentile" and not (0 <= p_level <= 100):
        raise ValueError(f"compute_ci_bootstrap: p_level must be in [0, 100], got {p_level}")

    arr = np.asarray(data, dtype=float)
    if arr.ndim not in (1, 2):
        raise ValueError(f"compute_ci_bootstrap: data must be 1-D or 2-D, got ndim={arr.ndim}")

    # Normalise to 2-D (n, k) for a single code path, squeeze back at the end.
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    n, k = arr.shape

    randint = (rng.integers if rng is not None else np.random.randint)
    idx = randint(0, n, size=(B, n, k))
    j_idx = np.arange(k)[np.newaxis, np.newaxis, :]
    boot = arr[idx, j_idx]  # (B, n, k)

    if statistic == "mean":
        boot_stats = boot.mean(axis=1)
    elif statistic == "median":
        boot_stats = np.median(boot, axis=1)
    elif statistic == "variance":
        boot_stats = np.var(boot, axis=1, ddof=1)
    else:  # percentile
        boot_stats = np.percentile(boot, p_level, axis=1)

    alpha_pct = (1.0 - level) * 100.0
    lo = np.percentile(boot_stats, alpha_pct / 2.0, axis=0)
    hi = np.percentile(boot_stats, 100.0 - alpha_pct / 2.0, axis=0)

    if squeeze:
        return float(lo), float(hi)
    return lo, hi
