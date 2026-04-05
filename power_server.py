# =============================================================================
# Power Explorer — server logic
# =============================================================================

import numpy as np
from scipy import stats
from scipy.stats import nct as nct_dist
from scipy.optimize import brentq
from shiny import reactive, render, ui

from utils import tip
from power_plots import draw_power_distributions, draw_power_curve

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _fig_to_ui(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)
    return ui.div(ui.HTML(html), class_="plotly-container")


# ── Core power computation ───────────────────────────────────────────────────

def _se_df(test_type: str, n: int, n2: int | None = None):
    """Return (SE, df) for the chosen test structure (σ = 1 standardised)."""
    if test_type in ("one_z", "one_t", "paired_t"):
        se = 1.0 / np.sqrt(n)
        df = n - 1
    else:  # two_t
        _n2 = n2 or n
        se = np.sqrt(1.0 / n + 1.0 / _n2)
        df = n + _n2 - 2
    return float(se), int(max(df, 1))


def _power_value(d: float, n: int, alpha: float, alt: str,
                 test_type: str, n2: int | None = None) -> float:
    """Compute statistical power for |d| ≥ 0."""
    se, df = _se_df(test_type, n, n2)
    lam = abs(d) / se  # noncentrality parameter

    if test_type == "one_z":
        if alt == "two-sided":
            z = stats.norm.ppf(1 - alpha / 2)
            return float(stats.norm.cdf(lam - z) + stats.norm.cdf(-lam - z))
        z = stats.norm.ppf(1 - alpha)
        return float(stats.norm.cdf(lam - z))

    # t-tests (one, two, paired)
    if alt == "two-sided":
        tc = stats.t.ppf(1 - alpha / 2, df)
        return float(1 - nct_dist.cdf(tc, df, lam) + nct_dist.cdf(-tc, df, lam))
    tc = stats.t.ppf(1 - alpha, df)
    return float(1 - nct_dist.cdf(tc, df, lam))


# ── Solve-for helpers (Brent root-finding) ───────────────────────────────────

def _solve_n(d, alpha, target, alt, test_type, n2=None):
    if abs(d) < 1e-9:
        return None

    def f(n_f):
        return _power_value(d, int(max(2, round(n_f))), alpha, alt, test_type, n2) - target

    if f(2) >= 0:
        return 2
    if f(200_000) < 0:
        return None
    try:
        return int(np.ceil(brentq(f, 2, 200_000)))
    except ValueError:
        return None


def _solve_d(n, alpha, target, alt, test_type, n2=None):
    def f(d):
        return _power_value(d, n, alpha, alt, test_type, n2) - target

    if f(0.001) >= 0:
        return 0.001
    if f(5.0) < 0:
        return None
    try:
        return float(brentq(f, 0.001, 5.0))
    except ValueError:
        return None


def _solve_alpha(d, n, target, alt, test_type, n2=None):
    def f(a):
        return _power_value(d, n, a, alt, test_type, n2) - target

    try:
        return float(brentq(f, 1e-10, 0.999))
    except ValueError:
        return None


# ── Server function ──────────────────────────────────────────────────────────

def power_server(input, output, session, is_dark):

    # ── Safe input readers ────────────────────────────────────────────────────
    def _d():
        try:
            v = input.pw_d()
            return float(v) if v is not None else 0.5
        except Exception:
            return 0.5

    def _n():
        try:
            v = input.pw_n()
            return max(int(v), 2) if v is not None else 30
        except Exception:
            return 30

    def _n2():
        try:
            v = input.pw_n2()
            return max(int(v), 2) if v is not None else _n()
        except Exception:
            return _n()

    def _alpha():
        try:
            v = input.pw_alpha()
            return float(v) if v is not None else 0.05
        except Exception:
            return 0.05

    def _power_input():
        try:
            v = input.pw_power()
            return float(v) if v is not None else 0.80
        except Exception:
            return 0.80

    # ── Dynamic params (n₂ for two-sample) ────────────────────────────────────
    @render.ui
    def pw_dynamic_params():
        if input.pw_test_type() == "two_t":
            return ui.div(
                ui.input_numeric(
                    "pw_n2",
                    ui.TagList("n\u2082 (group\u00a02)",
                               tip("Sample size for the second group.")),
                    value=30, min=2, max=5000, step=1, width="100%",
                ),
                class_="slider-row",
            )
        return ui.div()

    # ── Presets ───────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pw_preset)
    def _apply_preset():
        p = input.pw_preset()
        if not p:
            return
        if p == "clinical":
            ui.update_slider("pw_d", value=0.3)
            ui.update_numeric("pw_n", value=100)
            ui.update_slider("pw_alpha", value=0.01)
            ui.update_select("pw_test_type", selected="one_t")
            ui.update_select("pw_alternative", selected="two-sided")
            ui.update_select("pw_solve_for", selected="power")
        elif p == "ab_test":
            ui.update_slider("pw_d", value=0.2)
            ui.update_numeric("pw_n", value=200)
            ui.update_slider("pw_alpha", value=0.05)
            ui.update_select("pw_test_type", selected="two_t")
            ui.update_select("pw_alternative", selected="two-sided")
            ui.update_select("pw_solve_for", selected="power")
        elif p == "psych":
            ui.update_slider("pw_d", value=0.5)
            ui.update_numeric("pw_n", value=50)
            ui.update_slider("pw_alpha", value=0.05)
            ui.update_select("pw_test_type", selected="one_t")
            ui.update_select("pw_alternative", selected="two-sided")
            ui.update_select("pw_solve_for", selected="power")
        elif p == "small":
            ui.update_slider("pw_d", value=0.2)
            ui.update_slider("pw_alpha", value=0.05)
            ui.update_slider("pw_power", value=0.80)
            ui.update_select("pw_test_type", selected="one_t")
            ui.update_select("pw_alternative", selected="two-sided")
            ui.update_select("pw_solve_for", selected="n")

    # ── Core computed parameters ──────────────────────────────────────────────
    @reactive.calc
    def pw_computed():
        """Return (d, n, alpha, power) — one value is computed from the rest."""
        solve = input.pw_solve_for()
        d     = _d()
        n     = _n()
        alpha = _alpha()
        power = _power_input()
        alt   = input.pw_alternative()
        tt    = input.pw_test_type()
        n2    = _n2() if tt == "two_t" else None

        if solve == "power":
            power = _power_value(d, n, alpha, alt, tt, n2)
        elif solve == "n":
            result = _solve_n(d, alpha, power, alt, tt, n2)
            n = result if result is not None else n
        elif solve == "d":
            result = _solve_d(n, alpha, power, alt, tt, n2)
            d = result if result is not None else d
        elif solve == "alpha":
            result = _solve_alpha(d, n, power, alt, tt, n2)
            alpha = result if result is not None else alpha

        # Clamp power to [0, 1]
        power = max(0.0, min(1.0, power))
        return d, n, alpha, power

    # ── Stat card text outputs ────────────────────────────────────────────────
    @render.text
    def pw_stat_d():
        d, *_ = pw_computed()
        return f"{d:.3f}"

    @render.text
    def pw_stat_n():
        _, n, *_ = pw_computed()
        return f"{n:,}"

    @render.text
    def pw_stat_alpha():
        _, _, alpha, _ = pw_computed()
        return f"{alpha:.3f}"

    @render.text
    def pw_stat_power():
        *_, power = pw_computed()
        return f"{power:.3f}"

    # ── Power curve data ──────────────────────────────────────────────────────
    @reactive.calc
    def _curve_data():
        d, n, alpha, _ = pw_computed()
        alt = input.pw_alternative()
        tt  = input.pw_test_type()
        n2  = _n2() if tt == "two_t" else None

        if abs(d) < 1e-9:
            return np.array([]), np.array([])

        max_n = max(300, n * 3)
        ns = np.unique(np.concatenate([
            np.arange(2, min(60, max_n + 1)),
            np.linspace(60, max_n, 150).astype(int),
        ]))

        # Vectorised for z-test
        if tt == "one_z":
            lams = abs(d) * np.sqrt(ns)
            if alt == "two-sided":
                z = stats.norm.ppf(1 - alpha / 2)
                powers = stats.norm.cdf(lams - z) + stats.norm.cdf(-lams - z)
            else:
                z = stats.norm.ppf(1 - alpha)
                powers = stats.norm.cdf(lams - z)
        else:
            powers = np.array([
                _power_value(d, int(ni), alpha, alt, tt, n2) for ni in ns
            ])

        return ns, powers

    # ── Chart renderers ───────────────────────────────────────────────────────
    @render.ui
    def pw_dist_plot():
        d, n, alpha, power = pw_computed()
        tt  = input.pw_test_type()
        n2  = _n2() if tt == "two_t" else None
        se, df = _se_df(tt, n, n2)

        fig = draw_power_distributions(
            d=d, n=n, se=se, alpha=alpha, power=power,
            alternative=input.pw_alternative(),
            test_type=tt, df=df, dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def pw_curve_plot():
        d, n, alpha, power = pw_computed()
        ns, powers = _curve_data()

        fig = draw_power_curve(
            ns=ns, powers=powers,
            current_n=n, current_power=power,
            alpha=alpha, dark=is_dark(),
        )
        return _fig_to_ui(fig)
