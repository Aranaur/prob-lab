# =============================================================================
# Power Explorer — server logic
# =============================================================================

import numpy as np
from scipy import stats
from scipy.stats import nct as nct_dist
from scipy.optimize import brentq
from shiny import reactive, render, ui

from utils import tip
from power_plots import (
    draw_power_distributions, draw_power_curve, draw_cohens_d_overlap,
    draw_prop_distributions, draw_prop_effect,
    draw_ratio_distributions, draw_ratio_effect,
)

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _fig_to_ui(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)
    return ui.div(ui.HTML(html), class_="plotly-container")


# ── Core power computation (continuous) ──────────────────────────────────────

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
        pw = float(1 - nct_dist.cdf(tc, df, lam) + nct_dist.cdf(-tc, df, lam))
    else:
        tc = stats.t.ppf(1 - alpha, df)
        pw = float(1 - nct_dist.cdf(tc, df, lam))
    return 1.0 if np.isnan(pw) else pw


# ── Solve-for helpers — continuous (Brent root-finding) ──────────────────────

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


# ── Proportion power computation ─────────────────────────────────────────────

def _power_prop(p0: float, p1: float, n: int, alpha: float, alt: str) -> float:
    """Two-sample pooled z-test power for proportions (equal groups, n per group)."""
    _n = max(n, 2)
    p_pool = (p0 + p1) / 2
    se_h0 = np.sqrt(p_pool * (1 - p_pool) * 2 / _n)
    se_h1 = np.sqrt(p0 * (1 - p0) / _n + p1 * (1 - p1) / _n)
    if se_h0 < 1e-15 or se_h1 < 1e-15:
        return float(alpha)
    delta = p1 - p0
    if alt == "two-sided":
        z = stats.norm.ppf(1 - alpha / 2)
        pw = stats.norm.cdf((-z * se_h0 - delta) / se_h1) + \
             stats.norm.sf((z * se_h0 - delta) / se_h1)
    elif alt == "greater":
        z = stats.norm.ppf(1 - alpha)
        pw = stats.norm.sf((z * se_h0 - delta) / se_h1)
    else:  # less
        z = stats.norm.ppf(alpha)
        pw = stats.norm.cdf((z * se_h0 - delta) / se_h1)
    return float(np.clip(pw, 0.0, 1.0))


def _solve_n_prop(p0, p1, alpha, target, alt):
    if abs(p1 - p0) < 1e-9:
        return None

    def f(n_f):
        return _power_prop(p0, p1, int(max(2, round(n_f))), alpha, alt) - target

    if f(2) >= 0:
        return 2
    if f(2_000_000) < 0:
        return None
    try:
        return int(np.ceil(brentq(f, 2, 2_000_000)))
    except ValueError:
        return None


def _solve_p1_prop(p0, n, alpha, target, alt):
    """Solve for p₁ > p₀ (two-sided: |p₁ − p₀| gives target power)."""
    def f(p1):
        return _power_prop(p0, p1, n, alpha, alt) - target

    lo = p0 + 1e-6
    hi = 1.0 - 1e-6
    if lo >= hi:
        return None
    if f(lo) >= 0:
        return float(lo)
    if f(hi) < 0:
        return None
    try:
        return float(brentq(f, lo, hi))
    except ValueError:
        return None


def _solve_alpha_prop(p0, p1, n, target, alt):
    def f(a):
        return _power_prop(p0, p1, n, a, alt) - target

    try:
        return float(brentq(f, 1e-10, 0.999))
    except ValueError:
        return None


# ── Ratio power computation (Delta Method) ──────────────────────────────────

def _var_ratio(mu_x: float, var_x: float, mu_y: float, var_y: float,
               cov_xy: float) -> float:
    """Per-observation variance of the ratio R = X/Y via Delta Method.
    Var(R) ≈ (1/μ_y²)[σ²_x + R²·σ²_y − 2R·σ_xy]"""
    if abs(mu_y) < 1e-15:
        return 1e12  # degenerate — denominator mean ≈ 0
    r = mu_x / mu_y
    return (var_x + r ** 2 * var_y - 2 * r * cov_xy) / (mu_y ** 2)


def _power_ratio(var_r: float, r0: float, r1: float, n: int,
                 alpha: float, alt: str) -> float:
    """Two-sample z-test power for ratio metrics (equal groups, n per group)."""
    _n = max(n, 2)
    se = np.sqrt(2 * var_r / _n)
    if se < 1e-15:
        return float(alpha)
    delta = r1 - r0
    if alt == "two-sided":
        z = stats.norm.ppf(1 - alpha / 2)
        pw = stats.norm.cdf((-z * se - delta) / se) + \
             stats.norm.sf((z * se - delta) / se)
    elif alt == "greater":
        z = stats.norm.ppf(1 - alpha)
        pw = stats.norm.sf((z * se - delta) / se)
    else:  # less
        z = stats.norm.ppf(alpha)
        pw = stats.norm.cdf((z * se - delta) / se)
    return float(np.clip(pw, 0.0, 1.0))


def _solve_n_ratio(var_r, r0, r1, alpha, target, alt):
    if abs(r1 - r0) < 1e-12:
        return None

    def f(n_f):
        return _power_ratio(var_r, r0, r1, int(max(2, round(n_f))), alpha, alt) - target

    if f(2) >= 0:
        return 2
    if f(2_000_000) < 0:
        return None
    try:
        return int(np.ceil(brentq(f, 2, 2_000_000)))
    except ValueError:
        return None


def _solve_lift_ratio(var_r, r0, n, alpha, target, alt):
    """Solve for minimum detectable R₁ (> R₀) given target power."""
    def f(r1):
        return _power_ratio(var_r, r0, r1, n, alpha, alt) - target

    lo = r0 + 1e-9
    hi = r0 * 5 if abs(r0) > 1e-9 else 10.0
    if f(lo) >= 0:
        return float(lo)
    if f(hi) < 0:
        return None
    try:
        return float(brentq(f, lo, hi))
    except ValueError:
        return None


def _solve_alpha_ratio(var_r, r0, r1, n, target, alt):
    def f(a):
        return _power_ratio(var_r, r0, r1, n, a, alt) - target

    try:
        return float(brentq(f, 1e-10, 0.999))
    except ValueError:
        return None


# ── Server function ──────────────────────────────────────────────────────────

def power_server(input, output, session, is_dark):

    # ── Helper: metric mode checks ─────────────────────────────────────────
    def _is_prop():
        try:
            return input.pw_metric_type() == "proportion"
        except Exception:
            return False

    def _is_ratio():
        try:
            return input.pw_metric_type() == "ratio"
        except Exception:
            return False

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

    def _p0():
        try:
            v = input.pw_p0()
            return float(v) if v is not None else 0.10
        except Exception:
            return 0.10

    def _p1():
        try:
            v = input.pw_p1()
            return float(v) if v is not None else 0.12
        except Exception:
            return 0.12

    # ── Ratio-specific readers ───────────────────────────────────────────────
    def _mu_x():
        try:
            v = input.pw_mu_x()
            return float(v) if v is not None else 50.0
        except Exception:
            return 50.0

    def _var_x():
        try:
            v = input.pw_var_x()
            return max(float(v), 1e-6) if v is not None else 2500.0
        except Exception:
            return 2500.0

    def _mu_y():
        try:
            v = input.pw_mu_y()
            return float(v) if v is not None else 1000.0
        except Exception:
            return 1000.0

    def _var_y():
        try:
            v = input.pw_var_y()
            return max(float(v), 1e-6) if v is not None else 100000.0
        except Exception:
            return 100000.0

    def _cov_xy():
        try:
            v = input.pw_cov_xy()
            return float(v) if v is not None else 1000.0
        except Exception:
            return 1000.0

    def _lift_pct():
        try:
            v = input.pw_lift_pct()
            return float(v) if v is not None else 5.0
        except Exception:
            return 5.0

    # ── Computed-parameter display helper ─────────────────────────────────────
    def _param_display(label: str, value_str: str, param: str = "") -> ui.Tag:
        """Read-only card shown instead of an input when that param is computed."""
        cls = f"computed-param cp-{param}" if param else "computed-param"
        return ui.div(
            ui.div(
                ui.tags.span(label, class_="cp-label-text"),
                ui.tags.span("computed", class_="cp-badge"),
                class_="cp-label",
            ),
            ui.div(value_str, class_="cp-value"),
            class_=cls,
        )

    # ── Solve-for relabelling on metric_type change ──────────────────────────
    @reactive.effect
    @reactive.event(input.pw_metric_type)
    def _sync_solve_for_choices():
        if _is_ratio():
            ui.update_select("pw_solve_for", choices={
                "power": "Power (1\u2212\u03b2)",
                "n":     "Sample size (n)",
                "d":     "Expected lift (%)",
                "alpha": "Significance level (\u03b1)",
            })
        elif _is_prop():
            ui.update_select("pw_solve_for", choices={
                "power": "Power (1\u2212\u03b2)",
                "n":     "Sample size (n)",
                "d":     "Target rate (p\u2081)",
                "alpha": "Significance level (\u03b1)",
            })
        else:
            ui.update_select("pw_solve_for", choices={
                "power": "Power (1\u2212\u03b2)",
                "n":     "Sample size (n)",
                "d":     "Effect size (d)",
                "alpha": "Significance level (\u03b1)",
            })

    # ── Computed result — displayed right after "Solve for" ──────────────────
    @render.ui
    def pw_computed_result():
        solve = input.pw_solve_for()
        is_prop = _is_prop()
        is_ratio = _is_ratio()
        d_or_p1, n, alpha, power, n_feasible = pw_computed()

        if solve == "power":
            return _param_display("Power (1\u2212\u03b2)", f"{power:.3f}", "power")
        if solve == "n":
            val = f"{n:,}" if n_feasible else "\u2014 (increase effect or \u03b1)"
            if is_prop or is_ratio:
                label = "Sample size (n per group)"
            else:
                tt = input.pw_test_type()
                label = "Sample size (n per group)" if tt == "two_t" else "Sample size (n)"
            return _param_display(label, val, "n")
        if solve == "d":
            if is_ratio:
                # d_or_p1 holds R₁; convert back to lift %
                r0 = _mu_x() / max(abs(_mu_y()), 1e-15)
                lift = (d_or_p1 - r0) / r0 * 100 if abs(r0) > 1e-15 else 0.0
                return _param_display("Expected lift (%)", f"{lift:.2f}%", "d")
            if is_prop:
                return _param_display("Target rate (p\u2081)", f"{d_or_p1:.4f}", "d")
            return _param_display("Cohen\u2019s d", f"{d_or_p1:.3f}", "d")
        if solve == "alpha":
            return _param_display("\u03b1 (significance level)", f"{alpha:.4f}", "alpha")
        return ui.div()

    # ── Test type selector — dynamic choices per metric type ─────────────────
    @render.ui
    def pw_test_type_ui():
        with reactive.isolate():
            _p = _pw_preset_params()

        if _is_ratio():
            return ui.input_select(
                "pw_test_type",
                ui.TagList("Test type",
                           tip("Two-sample z-test for ratio metrics using the "
                               "Delta Method (Taylor linearisation) for variance.")),
                choices={"delta_z": "Delta method Z"},
                selected=_p.get("pw_test_type", "delta_z"), width="100%",
            )

        if _is_prop():
            return ui.input_select(
                "pw_test_type",
                ui.TagList("Test type",
                           tip("Two-sample z-test for the difference in proportions "
                               "(pooled SE under H\u2080).")),
                choices={"two_prop_z": "Two-sample proportion Z"},
                selected=_p.get("pw_test_type", "two_prop_z"), width="100%",
            )

        return ui.input_select(
            "pw_test_type",
            ui.TagList(
                "Test type",
                tip("Z-test uses known \u03c3. "
                    "t-tests estimate \u03c3 from data, giving slightly less power."),
            ),
            choices={
                "one_z":    "One-sample Z",
                "one_t":    "One-sample t",
                "two_t":    "Two-sample t (independent)",
                "paired_t": "Paired t",
            },
            selected=_p.get("pw_test_type", "one_t"), width="100%",
        )

    # ── Conditional inputs — hidden when that param is being solved ──────────
    @render.ui
    def pw_input_d():
        """Cohen's d slider (continuous), p₀/p₁ sliders (proportion),
        or ratio parameter inputs (ratio)."""
        solve = input.pw_solve_for()

        if _is_ratio():
            # When solving for lift, hide the lift slider
            if solve == "d":
                # Still show population parameters, just hide lift %
                with reactive.isolate():
                    _p = _pw_preset_params()
                mu_x = _p.get("pw_mu_x", 50.0)
                var_x = _p.get("pw_var_x", 2500.0)
                mu_y = _p.get("pw_mu_y", 1000.0)
                var_y = _p.get("pw_var_y", 100000.0)
                cov = _p.get("pw_cov_xy", 1000.0)
                r0 = mu_x / mu_y if abs(mu_y) > 1e-15 else 0.0
                vr = _var_ratio(mu_x, var_x, mu_y, var_y, cov)
                return ui.div(
                    ui.div(
                        ui.div(
                            ui.input_numeric("pw_mu_x", ui.TagList("\u03bc\u2093 (num. mean)", tip("Mean of the numerator (e.g. clicks, revenue).")), value=mu_x, step=1, width="100%"),
                            ui.input_numeric("pw_var_x", ui.TagList("\u03c3\u00b2\u2093 (num. var.)", tip("Variance of the numerator.")), value=var_x, min=0, step=100, width="100%"),
                            ui.input_numeric("pw_mu_y", ui.TagList("\u03bc\u1d67 (den. mean)", tip("Mean of the denominator (e.g. views, users).")), value=mu_y, step=10, width="100%"),
                            ui.input_numeric("pw_var_y", ui.TagList("\u03c3\u00b2\u1d67 (den. var.)", tip("Variance of the denominator.")), value=var_y, min=0, step=1000, width="100%"),
                            ui.input_numeric("pw_cov_xy", ui.TagList("Cov(X,Y)", tip("Covariance between numerator and denominator.")), value=cov, step=100, width="100%"),
                        ),
                    ),
                    ui.div(
                        f"R\u2080\u200a=\u200a{r0:.5f}  \u00b7  "
                        f"Var(R)\u200a=\u200a{vr:.6g}",
                        class_="np-preset-hint",
                        style="text-align:center; margin-top:2px;",
                    ),
                )

            with reactive.isolate():
                _p = _pw_preset_params()
            mu_x = _p.get("pw_mu_x", 50.0)
            var_x = _p.get("pw_var_x", 2500.0)
            mu_y = _p.get("pw_mu_y", 1000.0)
            var_y = _p.get("pw_var_y", 100000.0)
            cov = _p.get("pw_cov_xy", 1000.0)
            lift = _p.get("pw_lift_pct", 5.0)
            r0 = mu_x / mu_y if abs(mu_y) > 1e-15 else 0.0
            r1 = r0 * (1 + lift / 100)
            vr = _var_ratio(mu_x, var_x, mu_y, var_y, cov)
            return ui.div(
                ui.div(
                    ui.div(
                        ui.input_numeric("pw_mu_x", ui.TagList("\u03bc\u2093 (num. mean)", tip("Mean of the numerator (e.g. clicks, revenue).")), value=mu_x, step=1, width="100%"),
                        ui.input_numeric("pw_var_x", ui.TagList("\u03c3\u00b2\u2093 (num. var.)", tip("Variance of the numerator.")), value=var_x, min=0, step=100, width="100%"),
                        ui.input_numeric("pw_mu_y", ui.TagList("\u03bc\u1d67 (den. mean)", tip("Mean of the denominator (e.g. views, users).")), value=mu_y, step=10, width="100%"),
                        ui.input_numeric("pw_var_y", ui.TagList("\u03c3\u00b2\u1d67 (den. var.)", tip("Variance of the denominator.")), value=var_y, min=0, step=1000, width="100%"),
                        ui.input_numeric("pw_cov_xy", ui.TagList("Cov(X,Y)", tip("Covariance between numerator and denominator.")), value=cov, step=100, width="100%"),
                    ),
                ),
                ui.input_slider(
                    "pw_lift_pct",
                    ui.TagList("Expected lift (%)",
                               tip("Relative lift of the ratio: R\u2081 = R\u2080 \u00d7 (1 + lift/100).")),
                    min=0.1, max=50, value=lift, step=0.1, width="100%",
                ),
                ui.div(
                    f"R\u2080\u200a=\u200a{r0:.5f}  \u00b7  "
                    f"R\u2081\u200a=\u200a{r1:.5f}  \u00b7  "
                    f"Var(R)\u200a=\u200a{vr:.6g}",
                    class_="np-preset-hint",
                    style="text-align:center; margin-top:2px;",
                ),
            )

        if _is_prop():
            # In proportion mode, "d" slot = effect (p₁).
            # When solving for p₁, hide the sliders.
            if solve == "d":
                return ui.div()
            with reactive.isolate():
                _p = _pw_preset_params()
            p0_val = _p.get("pw_p0", 0.10)
            p1_val = _p.get("pw_p1", 0.12)
            delta = p1_val - p0_val
            rel = delta / p0_val * 100 if p0_val > 1e-9 else 0.0
            return ui.div(
                ui.input_slider(
                    "pw_p0",
                    ui.TagList("Baseline rate (p\u2080)",
                               tip("Control group conversion rate / base rate.")),
                    min=0.001, max=0.50, value=p0_val, step=0.001, width="100%",
                ),
                ui.input_slider(
                    "pw_p1",
                    ui.TagList("Target rate (p\u2081)",
                               tip("Expected rate in the treatment group.")),
                    min=0.002, max=0.99, value=p1_val, step=0.001, width="100%",
                ),
                ui.div(
                    f"\u0394p\u200a=\u200a{delta:+.4f}  \u00b7  "
                    f"rel. lift\u200a=\u200a{rel:+.1f}\u200a%",
                    class_="np-preset-hint",
                    style="text-align:center; margin-top:2px;",
                ),
            )

        # Continuous mode — original Cohen's d slider
        if solve == "d":
            return ui.div()
        with reactive.isolate():
            _p = _pw_preset_params()
        return ui.input_slider(
            "pw_d",
            ui.TagList("Cohen\u2019s d",
                       tip("Standardised effect size: (|\u03bc\u2081\u2212\u03bc\u2080|)/\u03c3. "
                           "Small\u200a\u2248\u200a0.2, medium\u200a\u2248\u200a0.5, large\u200a\u2248\u200a0.8.")),
            min=0, max=2, value=_p.get("pw_d", 0.5), step=0.01, width="100%",
        )

    @render.ui
    def pw_input_n():
        if input.pw_solve_for() == "n":
            return ui.div()
        with reactive.isolate():
            _p = _pw_preset_params()

        if _is_ratio():
            return ui.div(
                ui.input_numeric(
                    "pw_n",
                    ui.TagList("Sample size (n per group)",
                               tip("Number of observations per group (equal groups).")),
                    value=_p.get("pw_n", 5000), min=2, max=2_000_000, step=100, width="100%",
                ),
                class_="slider-row",
            )

        if _is_prop():
            # Proportion mode: always equal groups, single n input
            return ui.div(
                ui.input_numeric(
                    "pw_n",
                    ui.TagList("Sample size (n per group)",
                               tip("Number of observations per group (equal groups).")),
                    value=_p.get("pw_n", 500), min=2, max=2_000_000, step=10, width="100%",
                ),
                class_="slider-row",
            )

        # Continuous mode
        n1_input = ui.div(
            ui.input_numeric(
                "pw_n",
                ui.TagList("Sample size (n\u2081)" if input.pw_test_type() == "two_t" else "Sample size (n)",
                           tip("Number of observations. "
                               "For two-sample tests this is the size of group\u00a01.")),
                value=_p.get("pw_n", 30), min=2, max=5000, step=1, width="100%",
            )
        )

        if input.pw_test_type() == "two_t":
            n2_input = ui.div(
                ui.input_numeric(
                    "pw_n2",
                    ui.TagList("Sample size (n\u2082)",
                               tip("Sample size for the second group.")),
                    value=_p.get("pw_n2", 30), min=2, max=5000, step=1, width="100%",
                )
            )
            return ui.div(ui.div(n1_input, n2_input, class_="group-params-cols"), class_="group-params-block")

        return ui.div(n1_input, class_="slider-row")

    @render.ui
    def pw_input_alpha():
        if input.pw_solve_for() == "alpha":
            return ui.div()
        with reactive.isolate():
            _p = _pw_preset_params()
        return ui.input_slider(
            "pw_alpha",
            ui.TagList("\u03b1 (significance level)",
                       tip("Probability of Type\u00a0I error (false positive).")),
            min=0.005, max=0.2, value=_p.get("pw_alpha", 0.05), step=0.005, width="100%",
        )

    @render.ui
    def pw_input_power():
        if input.pw_solve_for() == "power":
            return ui.div()
        with reactive.isolate():
            _p = _pw_preset_params()
        return ui.input_slider(
            "pw_power",
            ui.TagList("Power (1\u2212\u03b2)",
                       tip("Probability of correctly rejecting a false H\u2080.")),
            min=0.50, max=0.999, value=_p.get("pw_power", 0.80), step=0.005, width="100%",
        )


    # ── Presets ───────────────────────────────────────────────────────────────
    _active_preset    = reactive.value(None)
    _pw_preset_params = reactive.value({})   # param overrides for pw_input_* re-renders

    _PRESET_DESC_CONT = {
        "clinical": (
            "Clinical trial",
            "d\u200a=\u200a0.3, \u03b1\u200a=\u200a0.01, n\u200a=\u200a100, one-sample t. "
            "Conservative \u03b1 protects patients from false positives. "
            "Even a small effect size requires many participants.",
        ),
        "ab_test": (
            "A/B test",
            "d\u200a=\u200a0.2, \u03b1\u200a=\u200a0.05, n\u200a=\u200a200, two-sample t. "
            "Typical web experiment: small effects, large user pools. "
            "Notice how much bigger n must be for the two-sample design.",
        ),
        "psych": (
            "Psychology",
            "d\u200a=\u200a0.5, \u03b1\u200a=\u200a0.05, n\u200a=\u200a50, one-sample t. "
            "Medium effect by Cohen\u2019s convention. "
            "Many classic psychology studies used n\u200a\u2248\u200a20\u201330 \u2014 severely underpowered.",
        ),
        "small": (
            "Small effect \u2192 n",
            "d\u200a=\u200a0.2, \u03b1\u200a=\u200a0.05, target power\u200a=\u200a0.80. "
            "Solve for n: shows how many observations a small effect demands. "
            "Illustrates why underpowered replication studies often fail.",
        ),
    }

    _PRESET_DESC_PROP = {
        "ctr": (
            "CTR +20%",
            "p\u2080\u200a=\u200a5%, p\u2081\u200a=\u200a6%. "
            "Typical web click-through rate test. +20% relative lift but only +1\u200app absolute. "
            "Shows why even small proportional lifts need large samples.",
        ),
        "conversion": (
            "Conversion",
            "p\u2080\u200a=\u200a2%, p\u2081\u200a=\u200a2.5%. "
            "E-commerce checkout conversion. Low base rate amplifies variance \u2014 "
            "n must be very large to detect a +0.5\u200app lift.",
        ),
        "retention": (
            "Retention",
            "p\u2080\u200a=\u200a30%, p\u2081\u200a=\u200a34%. "
            "High base rate means lower relative variance. "
            "The same absolute \u0394p is easier to detect when p\u2080 is closer to 0.5.",
        ),
        "tiny": (
            "Tiny lift",
            "p\u2080\u200a=\u200a10%, p\u2081\u200a=\u200a10.1%. "
            "Platform-scale experiment: 0.1\u200app lift requires enormous n. "
            "Demonstrates why big-tech experiments need millions of users.",
        ),
    }

    _PRESET_DESC_RATIO = {
        "r_ctr": (
            "CTR (clicks/views)",
            "\u03bc\u2093\u200a=\u200a50, \u03bc\u1d67\u200a=\u200a1000, lift\u200a=\u200a5%. "
            "Click-through rate as a ratio metric. "
            "Delta Method accounts for the correlation between clicks and views.",
        ),
        "r_revenue": (
            "Revenue per user",
            "\u03bc\u2093\u200a=\u200a10, \u03bc\u1d67\u200a=\u200a1, lift\u200a=\u200a3%. "
            "Revenue per active user. High variance in numerator "
            "drives large sample sizes even for modest lifts.",
        ),
        "r_aov": (
            "AOV (order value)",
            "\u03bc\u2093\u200a=\u200a75, \u03bc\u1d67\u200a=\u200a1.5, lift\u200a=\u200a5%. "
            "Average order value = total revenue / orders. "
            "Moderate covariance between revenue and order count.",
        ),
        "r_engage": (
            "Time per session",
            "\u03bc\u2093\u200a=\u200a300, \u03bc\u1d67\u200a=\u200a5, lift\u200a=\u200a8%. "
            "Total time / sessions. Large denominator variance "
            "inflates Var(R) and requires more samples.",
        ),
    }

    def _set_preset(d=None, n=None, alpha=None, power=None,
                    test_type="one_t", alternative="two-sided", solve_for="power",
                    p0=None, p1=None,
                    mu_x=None, var_x=None, mu_y=None, var_y=None,
                    cov_xy=None, lift_pct=None):
        # Populate preset params BEFORE ui.update_select so that any re-render
        # triggered by a solve_for / test_type change reads the correct values.
        params = {"pw_test_type": test_type}
        if d        is not None: params["pw_d"]        = d
        if n        is not None: params["pw_n"]        = n; params["pw_n2"] = n
        if alpha    is not None: params["pw_alpha"]    = alpha
        if power    is not None: params["pw_power"]    = power
        if p0       is not None: params["pw_p0"]       = p0
        if p1       is not None: params["pw_p1"]       = p1
        if mu_x     is not None: params["pw_mu_x"]     = mu_x
        if var_x    is not None: params["pw_var_x"]    = var_x
        if mu_y     is not None: params["pw_mu_y"]     = mu_y
        if var_y    is not None: params["pw_var_y"]    = var_y
        if cov_xy   is not None: params["pw_cov_xy"]   = cov_xy
        if lift_pct is not None: params["pw_lift_pct"] = lift_pct
        _pw_preset_params.set(params)

        # Belt-and-suspenders: also send update messages for same-structure cases.
        if d        is not None: ui.update_slider("pw_d",          value=d)
        if n        is not None: ui.update_numeric("pw_n",         value=n)
        if alpha    is not None: ui.update_slider("pw_alpha",      value=alpha)
        if power    is not None: ui.update_slider("pw_power",      value=power)
        if p0       is not None: ui.update_slider("pw_p0",         value=p0)
        if p1       is not None: ui.update_slider("pw_p1",         value=p1)
        if mu_x     is not None: ui.update_numeric("pw_mu_x",     value=mu_x)
        if var_x    is not None: ui.update_numeric("pw_var_x",    value=var_x)
        if mu_y     is not None: ui.update_numeric("pw_mu_y",     value=mu_y)
        if var_y    is not None: ui.update_numeric("pw_var_y",    value=var_y)
        if cov_xy   is not None: ui.update_numeric("pw_cov_xy",   value=cov_xy)
        if lift_pct is not None: ui.update_slider("pw_lift_pct",  value=lift_pct)
        ui.update_select("pw_test_type",   selected=test_type)
        ui.update_select("pw_alternative", selected=alternative)
        ui.update_select("pw_solve_for",   selected=solve_for)

    # Continuous presets
    @reactive.effect
    @reactive.event(input.pw_pre_clinical)
    def _pr_clinical():
        _active_preset.set("clinical")
        _set_preset(d=0.3, n=100, alpha=0.01, test_type="one_t", solve_for="power")

    @reactive.effect
    @reactive.event(input.pw_pre_ab)
    def _pr_ab():
        _active_preset.set("ab_test")
        _set_preset(d=0.2, n=200, alpha=0.05, test_type="two_t", solve_for="power")

    @reactive.effect
    @reactive.event(input.pw_pre_psych)
    def _pr_psych():
        _active_preset.set("psych")
        _set_preset(d=0.5, n=50, alpha=0.05, test_type="one_t", solve_for="power")

    @reactive.effect
    @reactive.event(input.pw_pre_small)
    def _pr_small():
        _active_preset.set("small")
        _set_preset(d=0.2, alpha=0.05, power=0.80, test_type="one_t", solve_for="n")

    # Proportion presets
    @reactive.effect
    @reactive.event(input.pw_pre_ctr)
    def _pr_ctr():
        _active_preset.set("ctr")
        _set_preset(p0=0.05, p1=0.06, alpha=0.05, test_type="two_prop_z", solve_for="n", power=0.80)

    @reactive.effect
    @reactive.event(input.pw_pre_conversion)
    def _pr_conversion():
        _active_preset.set("conversion")
        _set_preset(p0=0.02, p1=0.025, alpha=0.05, test_type="two_prop_z", solve_for="n", power=0.80)

    @reactive.effect
    @reactive.event(input.pw_pre_retention)
    def _pr_retention():
        _active_preset.set("retention")
        _set_preset(p0=0.30, p1=0.34, alpha=0.05, test_type="two_prop_z", solve_for="n", power=0.80)

    @reactive.effect
    @reactive.event(input.pw_pre_tiny)
    def _pr_tiny():
        _active_preset.set("tiny")
        _set_preset(p0=0.10, p1=0.101, alpha=0.05, test_type="two_prop_z", solve_for="n", power=0.80)

    # Ratio presets
    @reactive.effect
    @reactive.event(input.pw_pre_r_ctr)
    def _pr_r_ctr():
        _active_preset.set("r_ctr")
        _set_preset(mu_x=50, var_x=2500, mu_y=1000, var_y=100000,
                    cov_xy=1000, lift_pct=5.0, alpha=0.05,
                    test_type="delta_z", solve_for="n", power=0.80)

    @reactive.effect
    @reactive.event(input.pw_pre_r_revenue)
    def _pr_r_revenue():
        _active_preset.set("r_revenue")
        _set_preset(mu_x=10, var_x=400, mu_y=1, var_y=0.1,
                    cov_xy=2, lift_pct=3.0, alpha=0.05,
                    test_type="delta_z", solve_for="n", power=0.80)

    @reactive.effect
    @reactive.event(input.pw_pre_r_aov)
    def _pr_r_aov():
        _active_preset.set("r_aov")
        _set_preset(mu_x=75, var_x=5000, mu_y=1.5, var_y=1,
                    cov_xy=10, lift_pct=5.0, alpha=0.05,
                    test_type="delta_z", solve_for="n", power=0.80)

    @reactive.effect
    @reactive.event(input.pw_pre_r_engage)
    def _pr_r_engage():
        _active_preset.set("r_engage")
        _set_preset(mu_x=300, var_x=40000, mu_y=5, var_y=4,
                    cov_xy=50, lift_pct=8.0, alpha=0.05,
                    test_type="delta_z", solve_for="n", power=0.80)

    # Presets UI — switches between continuous, proportion, and ratio button sets
    @render.ui
    def pw_presets_ui():
        if _is_ratio():
            return ui.div(
                ui.tags.label(
                    "Scenario presets",
                    style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
                ),
                ui.div(
                    ui.input_action_button("pw_pre_r_ctr",     "CTR",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pw_pre_r_revenue", "Revenue",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pw_pre_r_aov",     "AOV",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pw_pre_r_engage",  "Engagement",
                                           class_="btn-ctrl btn-preset"),
                    class_="np-preset-grid",
                ),
            )

        if _is_prop():
            return ui.div(
                ui.tags.label(
                    "Scenario presets",
                    style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
                ),
                ui.div(
                    ui.input_action_button("pw_pre_ctr",        "CTR +20%",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pw_pre_conversion", "Conversion",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pw_pre_retention",  "Retention",
                                           class_="btn-ctrl btn-preset"),
                    ui.input_action_button("pw_pre_tiny",       "Tiny lift",
                                           class_="btn-ctrl btn-preset"),
                    class_="np-preset-grid",
                ),
            )

        return ui.div(
            ui.tags.label(
                "Scenario presets",
                style="font-weight:500; color:var(--c-text3); font-size:0.82rem; margin-bottom:2px;",
            ),
            ui.div(
                ui.input_action_button("pw_pre_clinical", "Clinical",
                                       class_="btn-ctrl btn-preset"),
                ui.input_action_button("pw_pre_ab",       "A/B Test",
                                       class_="btn-ctrl btn-preset"),
                ui.input_action_button("pw_pre_psych",    "Psych",
                                       class_="btn-ctrl btn-preset"),
                ui.input_action_button("pw_pre_small",    "Small eff.",
                                       class_="btn-ctrl btn-preset"),
                class_="np-preset-grid",
            ),
        )

    @render.ui
    def pw_preset_desc():
        key = _active_preset()
        descs = {**_PRESET_DESC_CONT, **_PRESET_DESC_PROP, **_PRESET_DESC_RATIO}
        if key is None or key not in descs:
            return ui.div(
                "\u2190 Select a preset to see what it demonstrates.",
                class_="np-preset-hint",
            )
        title, body = descs[key]
        return ui.div(
            ui.tags.strong(title + ": "),
            body,
            class_="np-preset-hint np-preset-hint--active",
        )

    # Reset active preset when metric type changes
    @reactive.effect
    @reactive.event(input.pw_metric_type)
    def _reset_preset_on_mode_switch():
        _active_preset.set(None)

    # ── Core computed parameters ──────────────────────────────────────────────
    @reactive.calc
    def pw_computed():
        """Return (effect, n, alpha, power, n_feasible).
        effect = d (continuous), p₁ (proportion), or R₁ (ratio)."""
        solve = input.pw_solve_for()
        alpha = _alpha()
        power = _power_input()
        alt   = input.pw_alternative()
        n_feasible = True

        if _is_ratio():
            mu_x = _mu_x(); var_x = _var_x()
            mu_y = _mu_y(); var_y = _var_y()
            cov  = _cov_xy()
            var_r = _var_ratio(mu_x, var_x, mu_y, var_y, cov)
            r0 = mu_x / mu_y if abs(mu_y) > 1e-15 else 0.0
            lift = _lift_pct()
            r1 = r0 * (1 + lift / 100)
            n  = _n()

            if solve == "power":
                power = _power_ratio(var_r, r0, r1, n, alpha, alt)
            elif solve == "n":
                result = _solve_n_ratio(var_r, r0, r1, alpha, power, alt)
                if result is not None:
                    n = result
                else:
                    n_feasible = False
            elif solve == "d":  # "d" slot = solve for lift → R₁
                result = _solve_lift_ratio(var_r, r0, n, alpha, power, alt)
                r1 = result if result is not None else r1
            elif solve == "alpha":
                result = _solve_alpha_ratio(var_r, r0, r1, n, power, alt)
                alpha = result if result is not None else alpha

            power = max(0.0, min(1.0, power))
            return r1, n, alpha, power, n_feasible

        if _is_prop():
            p0 = _p0()
            p1 = _p1()
            n  = _n()

            if solve == "power":
                power = _power_prop(p0, p1, n, alpha, alt)
            elif solve == "n":
                result = _solve_n_prop(p0, p1, alpha, power, alt)
                if result is not None:
                    n = result
                else:
                    n_feasible = False
            elif solve == "d":  # "d" slot = solve for p₁
                result = _solve_p1_prop(p0, n, alpha, power, alt)
                p1 = result if result is not None else p1
            elif solve == "alpha":
                result = _solve_alpha_prop(p0, p1, n, power, alt)
                alpha = result if result is not None else alpha

            power = max(0.0, min(1.0, power))
            return p1, n, alpha, power, n_feasible

        # Continuous mode
        d  = _d()
        n  = _n()
        tt = input.pw_test_type()
        n2 = _n2() if (tt == "two_t" and solve != "n") else None

        if solve == "power":
            power = _power_value(d, n, alpha, alt, tt, n2)
        elif solve == "n":
            result = _solve_n(d, alpha, power, alt, tt, n2=None)
            if result is not None:
                n = result
            else:
                n_feasible = False
        elif solve == "d":
            result = _solve_d(n, alpha, power, alt, tt, n2)
            d = result if result is not None else d
        elif solve == "alpha":
            result = _solve_alpha(d, n, power, alt, tt, n2)
            alpha = result if result is not None else alpha

        power = max(0.0, min(1.0, power))
        return d, n, alpha, power, n_feasible

    # ── Stats row — dynamic per metric type ──────────────────────────────────
    @render.ui
    def pw_stats_row():
        if _is_ratio():
            return ui.div(
                ui.div(
                    ui.div(
                        "RATIO (R\u2080)\u00a0",
                        tip("Control group ratio \u03bc\u2093/\u03bc\u1d67."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_r0", inline=True), class_="stat-value pw-d"),
                    class_="stat-card",
                ),
                ui.div(
                    ui.div(
                        "LIFT\u00a0",
                        tip("Expected relative lift of the treatment ratio."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_lift", inline=True), class_="stat-value pw-n"),
                    class_="stat-card",
                ),
                ui.div(
                    ui.div(
                        "\u03b1 (TYPE I)\u00a0",
                        tip("Probability of rejecting H\u2080 when it is true."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_alpha", inline=True), class_="stat-value pw-alpha"),
                    class_="stat-card",
                ),
                ui.div(
                    ui.div(
                        "POWER (1\u2212\u03b2)\u00a0",
                        tip("Probability of rejecting H\u2080 when H\u2081 is true."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_power", inline=True), class_="stat-value pw-power"),
                    class_="stat-card",
                ),
                class_="stats-row",
            )

        if _is_prop():
            return ui.div(
                ui.div(
                    ui.div(
                        "BASELINE (p\u2080)\u00a0",
                        tip("Control group proportion / base rate."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_p0", inline=True), class_="stat-value pw-d"),
                    class_="stat-card",
                ),
                ui.div(
                    ui.div(
                        "TARGET (p\u2081)\u00a0",
                        tip("Treatment group proportion / target rate."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_p1", inline=True), class_="stat-value pw-n"),
                    class_="stat-card",
                ),
                ui.div(
                    ui.div(
                        "\u03b1 (TYPE I)\u00a0",
                        tip("Probability of rejecting H\u2080 when it is true."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_alpha", inline=True), class_="stat-value pw-alpha"),
                    class_="stat-card",
                ),
                ui.div(
                    ui.div(
                        "POWER (1\u2212\u03b2)\u00a0",
                        tip("Probability of rejecting H\u2080 when H\u2081 is true."),
                        class_="stat-label",
                    ),
                    ui.div(ui.output_text("pw_stat_power", inline=True), class_="stat-value pw-power"),
                    class_="stat-card",
                ),
                class_="stats-row",
            )

        # Continuous mode
        return ui.div(
            ui.div(
                ui.div(
                    "EFFECT SIZE (d)\u00a0",
                    tip("Cohen\u2019s d: the standardised distance between H\u2080 and H\u2081."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pw_stat_d", inline=True), class_="stat-value pw-d"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(
                    "SAMPLE SIZE (n)\u00a0",
                    tip("Number of observations per group."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pw_stat_n", inline=True), class_="stat-value pw-n"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(
                    "\u03b1 (TYPE I)\u00a0",
                    tip("Probability of rejecting H\u2080 when it is true."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pw_stat_alpha", inline=True), class_="stat-value pw-alpha"),
                class_="stat-card",
            ),
            ui.div(
                ui.div(
                    "POWER (1\u2212\u03b2)\u00a0",
                    tip("Probability of rejecting H\u2080 when H\u2081 is true."),
                    class_="stat-label",
                ),
                ui.div(ui.output_text("pw_stat_power", inline=True), class_="stat-value pw-power"),
                class_="stat-card",
            ),
            class_="stats-row",
        )

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
        _, _, alpha, *_ = pw_computed()
        return f"{alpha:.3f}"

    @render.text
    def pw_stat_power():
        _, _, _, power, _ = pw_computed()
        return f"{power:.3f}"

    @render.text
    def pw_stat_p0():
        return f"{_p0():.4f}"

    @render.text
    def pw_stat_p1():
        p1, *_ = pw_computed()
        return f"{p1:.4f}"

    @render.text
    def pw_stat_r0():
        r0 = _mu_x() / max(abs(_mu_y()), 1e-15)
        return f"{r0:.5f}"

    @render.text
    def pw_stat_lift():
        r1, *_ = pw_computed()
        r0 = _mu_x() / max(abs(_mu_y()), 1e-15)
        lift = (r1 - r0) / r0 * 100 if abs(r0) > 1e-15 else 0.0
        return f"{lift:+.2f}%"

    # ── Power curve data ──────────────────────────────────────────────────────
    @reactive.calc
    def _curve_data():
        d_or_p1, n, alpha, *_ = pw_computed()
        alt = input.pw_alternative()
        solve = input.pw_solve_for()

        if _is_ratio():
            r1 = d_or_p1  # effect slot = R₁
            mu_x = _mu_x(); var_x = _var_x()
            mu_y = _mu_y(); var_y = _var_y()
            cov  = _cov_xy()
            var_r = _var_ratio(mu_x, var_x, mu_y, var_y, cov)
            r0 = mu_x / mu_y if abs(mu_y) > 1e-15 else 0.0
            if abs(r1 - r0) < 1e-12:
                return np.array([]), np.array([])
            max_n = max(5000, int(n * 1.5))
            ns = np.unique(np.concatenate([
                np.arange(2, min(200, max_n + 1)),
                np.linspace(200, max_n, 200).astype(int),
            ]))
            powers = np.array([
                _power_ratio(var_r, r0, r1, int(ni), alpha, alt) for ni in ns
            ])
            return ns, powers

        if _is_prop():
            p0 = _p0()
            p1 = d_or_p1  # effect slot = p₁
            if abs(p1 - p0) < 1e-9:
                return np.array([]), np.array([])
            max_n = max(2000, int(n * 1.5))
            ns = np.unique(np.concatenate([
                np.arange(2, min(100, max_n + 1)),
                np.linspace(100, max_n, 200).astype(int),
            ]))
            powers = np.array([
                _power_prop(p0, p1, int(ni), alpha, alt) for ni in ns
            ])
            return ns, powers

        # Continuous mode
        d = d_or_p1
        tt = input.pw_test_type()
        n2 = _n2() if (tt == "two_t" and solve != "n") else None

        if abs(d) < 1e-9:
            return np.array([]), np.array([])

        max_n = max(300, int(n * 1.5))
        if max_n <= 600:
            ns = np.unique(np.concatenate([
                np.arange(2, min(60, max_n + 1)),
                np.linspace(60, max_n, 150).astype(int),
            ]))
        else:
            ns = np.unique(np.concatenate([
                np.arange(2, 60),
                np.linspace(60, max_n, 200).astype(int),
            ]))

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
        d_or_p1, n, alpha, power, _ = pw_computed()

        if _is_ratio():
            mu_x = _mu_x(); var_x = _var_x()
            mu_y = _mu_y(); var_y = _var_y()
            cov  = _cov_xy()
            var_r = _var_ratio(mu_x, var_x, mu_y, var_y, cov)
            r0 = mu_x / mu_y if abs(mu_y) > 1e-15 else 0.0
            fig = draw_ratio_distributions(
                r0=r0, r1=d_or_p1, n=n, alpha=alpha, power=power,
                alternative=input.pw_alternative(), var_r=var_r, dark=is_dark(),
            )
            return _fig_to_ui(fig)

        if _is_prop():
            fig = draw_prop_distributions(
                p0=_p0(), p1=d_or_p1, n=n, alpha=alpha, power=power,
                alternative=input.pw_alternative(), dark=is_dark(),
            )
            return _fig_to_ui(fig)

        tt    = input.pw_test_type()
        solve = input.pw_solve_for()
        n2    = _n2() if (tt == "two_t" and solve != "n") else None
        se, df = _se_df(tt, n, n2)

        fig = draw_power_distributions(
            d=d_or_p1, n=n, se=se, alpha=alpha, power=power,
            alternative=input.pw_alternative(),
            test_type=tt, df=df, dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def pw_curve_plot():
        _, n, alpha, power, _ = pw_computed()
        ns, powers = _curve_data()

        fig = draw_power_curve(
            ns=ns, powers=powers,
            current_n=n, current_power=power,
            alpha=alpha, dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def pw_overlap_plot():
        d_or_p1, *_ = pw_computed()

        if _is_ratio():
            mu_x = _mu_x(); var_x = _var_x()
            mu_y = _mu_y(); var_y = _var_y()
            cov  = _cov_xy()
            var_r = _var_ratio(mu_x, var_x, mu_y, var_y, cov)
            r0 = mu_x / mu_y if abs(mu_y) > 1e-15 else 0.0
            fig = draw_ratio_effect(r0=r0, r1=d_or_p1, var_r=var_r, dark=is_dark())
            return _fig_to_ui(fig)

        if _is_prop():
            fig = draw_prop_effect(p0=_p0(), p1=d_or_p1, dark=is_dark())
            return _fig_to_ui(fig)

        fig = draw_cohens_d_overlap(d=d_or_p1, dark=is_dark())
        return _fig_to_ui(fig)
