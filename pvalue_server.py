# =============================================================================
# p-value Explorer — server logic
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from scipy.stats import nct as nct_dist
from shiny import reactive, render, ui

from utils import tip
from pvalue_plots import draw_null_dist_plot, draw_pvalue_hist, draw_power_diagram

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _fig_to_ui(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)
    return ui.div(ui.HTML(html), class_="plotly-container")


def pvalue_server(input, output, session, is_dark):

    MAX_DATA = 10_000

    # ── Reactive state ────────────────────────────────────────────────────────
    pv_total    = reactive.value(0)
    pv_rejected = reactive.value(0)
    pv_last_stat: reactive.Value[float | None] = reactive.value(None)
    pv_all_pvalues: reactive.Value[deque]      = reactive.value(deque(maxlen=MAX_DATA))
    pv_is_playing = reactive.value(False)
    pv_speed_ms   = reactive.value(0.5)

    # ── Input helpers (guard against None / 0-as-falsy) ──────────────────────
    def _safe(input_fn, default):
        try:
            v = input_fn()
            return type(default)(v) if v is not None else default
        except Exception:
            return default

    def _get_mu0()     -> float: return _safe(input.pv_mu0,    0.0)
    def _get_mu_true() -> float: return _safe(input.pv_mu_true, 0.5)
    def _get_sigma()   -> float: return max(_safe(input.pv_sigma,  1.0), 1e-6)
    def _get_sigma2()  -> float: return max(_safe(input.pv_sigma2, 1.0), 1e-6)
    def _get_n()       -> int:   return max(_safe(input.pv_n,   10),  2)
    def _get_n2()      -> int:   return max(_safe(input.pv_n2,  10),  2)
    def _get_rho()     -> float:
        r = _safe(input.pv_rho, 0.0)
        return max(-0.999, min(0.999, r))

    # ── Computed SE and df for the current test design ────────────────────────
    @reactive.calc
    def _test_se_df():
        """Returns (se, df) for the chosen test structure."""
        structure = input.pv_test_structure()
        sigma1    = _get_sigma()
        n1        = _get_n()

        if structure == "one":
            se = sigma1 / np.sqrt(n1)
            df = n1 - 1

        elif structure == "two":
            sigma2 = _get_sigma2()
            n2     = _get_n2()
            se     = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
            # Welch df (uses population σ as proxy for theoretical SE)
            v1, v2 = sigma1**2 / n1, sigma2**2 / n2
            df = int((v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1)))

        else:  # paired
            sigma2 = _get_sigma2()
            rho    = _get_rho()
            # SD of differences: σ_d = √(σ₁² + σ₂² − 2ρσ₁σ₂)
            se = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2) / np.sqrt(n1)
            df = n1 - 1

        return float(max(se, 1e-9)), int(max(df, 1))

    # ── Dynamic sidebar params (group 2 σ, n₂, ρ) ────────────────────────────
    @render.ui
    def pv_dynamic_params():
        structure = input.pv_test_structure()

        if structure == "one":
            return ui.div()

        if structure == "two":
            return ui.div(
                ui.input_numeric(
                    "pv_sigma2",
                    ui.TagList("Group\u00a02 \u03c3",
                               tip("Standard deviation of the second population.")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
                ui.input_numeric(
                    "pv_n2",
                    ui.TagList("n\u2082 (group\u00a02)",
                               tip("Sample size for the second group.")),
                    value=10, min=2, max=500, step=1, width="100%",
                ),
                class_="slider-row",
            )

        # paired
        return ui.div(
            ui.input_numeric(
                "pv_sigma2",
                ui.TagList("Group\u00a02 \u03c3",
                           tip("Standard deviation of the second measurement within each pair.")),
                value=1.0, min=0.1, step=0.5, width="100%",
            ),
            ui.input_numeric(
                "pv_rho",
                ui.TagList("\u03c1 (correlation)",
                           tip(
                               "Within-pair Pearson correlation. "
                               "Higher \u03c1 \u2192 smaller SD of differences \u2192 more power."
                           )),
                value=0.5, min=-0.99, max=0.99, step=0.1, width="100%",
            ),
            class_="slider-row",
        )

    # ── Sample size ± ─────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pv_n_minus)
    def _pv_n_minus():
        cur = input.pv_n()
        if cur is not None and cur > 2:
            ui.update_numeric("pv_n", value=cur - 1)

    @reactive.effect
    @reactive.event(input.pv_n_plus)
    def _pv_n_plus():
        cur = input.pv_n()
        if cur is not None and cur < 500:
            ui.update_numeric("pv_n", value=cur + 1)

    # ── Speed ± ───────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pv_speed_minus)
    def _pv_speed_down():
        pv_speed_ms.set(min(pv_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.pv_speed_plus)
    def _pv_speed_up():
        pv_speed_ms.set(max(pv_speed_ms() - 0.05, 0.05))

    # ── Play / Pause ──────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pv_btn_play)
    def _pv_toggle_play():
        pv_is_playing.set(not pv_is_playing())
        label = "Pause" if pv_is_playing() else "Play"
        ui.update_action_button("pv_btn_play", label=label)

    @reactive.effect
    def _pv_auto_draw():
        if pv_is_playing():
            reactive.invalidate_later(pv_speed_ms())
            with reactive.isolate():
                _draw_samples(1)

    # ── Manual buttons ────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pv_btn_sample_1)
    def _pv_s1():   _draw_samples(1)

    @reactive.effect
    @reactive.event(input.pv_btn_sample_50)
    def _pv_s50():  _draw_samples(50)

    @reactive.effect
    @reactive.event(input.pv_btn_sample_100)
    def _pv_s100(): _draw_samples(100)

    # ── Reset ─────────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pv_btn_reset, input.pv_mu0, input.pv_alternative,
                    input.pv_test_method, input.pv_test_structure)
    def _pv_reset():
        pv_total.set(0)
        pv_rejected.set(0)
        pv_last_stat.set(None)
        pv_all_pvalues.set(deque(maxlen=MAX_DATA))
        pv_is_playing.set(False)
        ui.update_action_button("pv_btn_play", label="Play")

    # ── Core sampling ─────────────────────────────────────────────────────────
    def _draw_samples(k: int):
        mu0         = _get_mu0()
        mu_true     = _get_mu_true()
        sigma1      = _get_sigma()
        n1          = _get_n()
        alpha       = input.pv_alpha()
        alternative = input.pv_alternative()
        method      = input.pv_test_method()
        structure   = input.pv_test_structure()

        if structure == "one":
            # ── One-sample ──────────────────────────────────────────────────
            samps = np.random.normal(mu_true, sigma1, size=(n1, k))
            means = samps.mean(axis=0)
            if method == "z":
                ses      = np.full(k, sigma1 / np.sqrt(n1))
                stat_arr = (means - mu0) / ses
                pvals    = _pval(stat_arr, alternative, method, df=n1 - 1)
            else:
                stds     = samps.std(axis=0, ddof=1)
                ses      = stds / np.sqrt(n1)
                stat_arr = (means - mu0) / ses
                pvals    = _pval(stat_arr, alternative, method, df=n1 - 1)

        elif structure == "two":
            # ── Two-sample independent ──────────────────────────────────────
            sigma2 = _get_sigma2()
            n2     = _get_n2()
            # true group means: μ₁ = mu_true + mu0/2, μ₂ = mu_true - mu0/2?
            # Simpler: group 1 mean = mu_true, group 2 mean = 0,
            # so true difference = mu_true - 0 = mu_true.
            # null: difference = mu0.
            s1 = np.random.normal(mu_true, sigma1, size=(n1, k))
            s2 = np.random.normal(0.0,     sigma2, size=(n2, k))
            d  = s1.mean(axis=0) - s2.mean(axis=0)   # observed difference

            if method == "z":
                se_z     = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
                stat_arr = (d - mu0) / se_z
                pvals    = _pval(stat_arr, alternative, "z", df=1)
            else:
                # Welch's t
                var1     = s1.var(axis=0, ddof=1)
                var2     = s2.var(axis=0, ddof=1)
                se_w     = np.sqrt(var1 / n1 + var2 / n2)
                stat_arr = (d - mu0) / se_w
                # Welch df per sample
                v1, v2 = var1 / n1, var2 / n2
                df_w = np.where(
                    (v1 + v2) > 0,
                    (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1)),
                    1.0
                ).astype(float)
                pvals = np.array([
                    _pval_scalar(float(t), alternative, "t", int(max(df, 1)))
                    for t, df in zip(stat_arr, df_w)
                ])

        else:
            # ── Paired ──────────────────────────────────────────────────────
            sigma2 = _get_sigma2()
            rho    = _get_rho()
            # Correlated bivariate normal
            cov    = rho * sigma1 * sigma2
            cov_mx = np.array([[sigma1**2, cov], [cov, sigma2**2]])
            means_pair = [mu_true, 0.0]
            # shape (n1, k, 2)
            pairs = np.random.multivariate_normal(means_pair, cov_mx, size=(n1, k))
            diffs = pairs[:, :, 0] - pairs[:, :, 1]   # (n1, k)
            d_bar = diffs.mean(axis=0)

            if method == "z":
                sigma_d  = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
                se_z     = sigma_d / np.sqrt(n1)
                stat_arr = (d_bar - mu0) / se_z
                pvals    = _pval(stat_arr, alternative, "z", df=n1 - 1)
            else:
                sd_d     = diffs.std(axis=0, ddof=1)
                se_t     = sd_d / np.sqrt(n1)
                stat_arr = (d_bar - mu0) / se_t
                pvals    = _pval(stat_arr, alternative, "t", df=n1 - 1)

        new_rejected = int((pvals < alpha).sum())
        pv_total.set(pv_total() + k)
        pv_rejected.set(pv_rejected() + new_rejected)
        pv_last_stat.set(float(stat_arr[-1]))

        pv = deque(pv_all_pvalues(), maxlen=MAX_DATA)
        pv.extend(float(p) for p in pvals)
        pv_all_pvalues.set(pv)

    # ── p-value computation helpers ───────────────────────────────────────────
    def _pval(stat_arr: np.ndarray, alt: str, method: str, df: int) -> np.ndarray:
        d = stats.norm if method == "z" else stats.t
        kw = {} if method == "z" else {"df": df}
        if alt == "two-sided":
            return 2.0 * d.cdf(-np.abs(stat_arr), **kw)
        elif alt == "greater":
            return 1.0 - d.cdf(stat_arr, **kw)
        else:
            return d.cdf(stat_arr, **kw)

    def _pval_scalar(stat: float, alt: str, method: str, df: int) -> float:
        d = stats.norm if method == "z" else stats.t
        kw = {} if method == "z" else {"df": df}
        if alt == "two-sided":
            return float(2.0 * d.cdf(-abs(stat), **kw))
        elif alt == "greater":
            return float(1.0 - d.cdf(stat, **kw))
        else:
            return float(d.cdf(stat, **kw))

    # ── Theoretical power ─────────────────────────────────────────────────────
    @render.text
    def pv_theo_power():
        mu0     = _get_mu0()
        mu_true = _get_mu_true()
        alpha   = input.pv_alpha()
        alt     = input.pv_alternative()
        method  = input.pv_test_method()
        se, df  = _test_se_df()

        if method == "z":
            if alt == "two-sided":
                z = stats.norm.ppf(1 - alpha / 2)
                p = stats.norm.cdf(-z + (mu_true - mu0) / se) + stats.norm.cdf(-z - (mu_true - mu0) / se)
            elif alt == "greater":
                z = stats.norm.ppf(1 - alpha)
                p = stats.norm.cdf((mu_true - mu0) / se - z)
            else:
                z = stats.norm.ppf(1 - alpha)
                p = stats.norm.cdf(-(mu_true - mu0) / se - z)
        else:
            ncp = (mu_true - mu0) / se
            if alt == "two-sided":
                tc = stats.t.ppf(1 - alpha / 2, df)
                p = nct_dist.cdf(-tc, df, ncp) + 1 - nct_dist.cdf(tc, df, ncp)
            elif alt == "greater":
                tc = stats.t.ppf(1 - alpha, df)
                p = 1 - nct_dist.cdf(tc, df, ncp)
            else:
                tc = stats.t.ppf(1 - alpha, df)
                p = nct_dist.cdf(-tc, df, ncp)
            p = 1.0 if np.isnan(p) else p
        return f"{float(p):.3f}"

    # ── Text outputs ──────────────────────────────────────────────────────────
    @render.text
    def pv_current_pvalue():
        stat = pv_last_stat()
        if stat is None:
            return "\u2014"
        alt    = input.pv_alternative()
        method = input.pv_test_method()
        _, df  = _test_se_df()
        p = _pval_scalar(stat, alt, method, df)
        return f"{p:.4f}" if p >= 0.0001 else "<0.0001"

    @render.text
    def pv_reject_rate():
        td = pv_total()
        if td == 0:
            return "\u2014"
        return f"{100 * pv_rejected() / td:.1f}%"

    @render.text
    def pv_total_tests():
        return f"{pv_total():,}"

    # ── Chart renderers ───────────────────────────────────────────────────────
    @render.ui
    def pv_null_dist_plot():
        _, df = _test_se_df()
        fig = draw_null_dist_plot(
            last_stat=pv_last_stat(),
            df=df,
            alpha=input.pv_alpha(),
            alternative=input.pv_alternative(),
            method=input.pv_test_method(),
            dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def pv_hist_plot():
        fig = draw_pvalue_hist(list(pv_all_pvalues()), alpha=input.pv_alpha(),
                               dark=is_dark())
        return _fig_to_ui(fig)

    @render.ui
    def pv_power_plot():
        se, df = _test_se_df()
        emp    = pv_rejected() / pv_total() if pv_total() > 0 else None
        fig = draw_power_diagram(
            mu0=_get_mu0(),
            mu_true=_get_mu_true(),
            se_val=se,
            df=df,
            alpha=input.pv_alpha(),
            alternative=input.pv_alternative(),
            empirical_rate=emp,
            method=input.pv_test_method(),
            dark=is_dark(),
        )
        return _fig_to_ui(fig)
