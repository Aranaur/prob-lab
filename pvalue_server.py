# =============================================================================
# p-value Explorer — server logic
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from scipy.stats import nct as nct_dist
from shiny import reactive, render, ui

from utils import tip
from pvalue_plots import (
    draw_null_dist_plot,
    draw_pvalue_hist,
    draw_power_diagram,
    draw_effect_scatter,
)
from theme import fig_to_ui




def pvalue_server(input, output, session, is_dark):

    MAX_DATA = 10_000

    # ── Reactive state ────────────────────────────────────────────────────────
    pv_total    = reactive.value(0)
    pv_rejected = reactive.value(0)
    pv_last_stat: reactive.Value[float | None] = reactive.value(None)
    pv_all_pvalues: reactive.Value[deque]      = reactive.value(deque(maxlen=MAX_DATA))
    pv_all_effects: reactive.Value[deque]      = reactive.value(deque(maxlen=MAX_DATA))
    pv_is_playing = reactive.value(False)
    pv_speed_ms   = reactive.value(0.5)

    # Wilcoxon comparison state
    pv_wilcoxon_rejected = reactive.value(0)
    pv_wilcoxon_pvalues: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))

    # Pipeline-mode state (FPR tracking)
    pv_is_null_true: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    pv_null_count       = reactive.value(0)   # experiments drawn from H₀
    pv_false_positives  = reactive.value(0)   # reject ∩ null-true
    pv_true_positives   = reactive.value(0)   # reject ∩ null-false

    # Preset state
    _pv_active_preset = reactive.value(None)
    _pv_preset_params = reactive.value({})   # group-param overrides for pv_group_params

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
    def _get_n()       -> int:
        try:
            structure = input.pv_test_structure()
        except Exception:
            structure = "one"
        if structure == "two":
            return max(_safe(input.pv_n1, 10), 2)
        return max(_safe(input.pv_n, 10), 2)
    def _get_n2()      -> int:   return max(_safe(input.pv_n2,  10),  2)
    def _get_rho()     -> float:
        r = _safe(input.pv_rho, 0.0)
        return max(-0.999, min(0.999, r))
    def _get_mode()    -> str:
        try:
            m = input.pv_mode()
            return m if m in ("single", "pipeline") else "single"
        except Exception:
            return "single"
    def _get_pi()      -> float:
        p = _safe(input.pv_pi, 0.8)
        return max(0.0, min(1.0, p))

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

    # ── Group parameters block ────────────────────────────────────────────────
    @render.ui
    def pv_group_params():
        structure = input.pv_test_structure()
        with reactive.isolate():
            _p = _pv_preset_params()

        def pval(key, default):
            return _p.get(key, default)

        if structure == "one":
            sigma_col = ui.div(
                ui.input_numeric(
                    "pv_sigma",
                    ui.TagList(
                        "Population \u03c3",
                        tip(
                            "Standard deviation of the population. "
                            "Used for data generation and for the z-test statistic."
                        ),
                    ),
                    value=pval("pv_sigma", 1.0), min=0.1, step=0.5, width="100%",
                )
            )
            n_col = ui.div(
                ui.input_numeric(
                    "pv_n",
                    ui.TagList("Sample size (n)", tip("Number of observations in the sample.")),
                    value=pval("pv_n", 10), min=2, max=500, step=1, width="100%",
                )
            )
            return ui.div(
                ui.div(sigma_col, n_col, class_="group-params-cols"),
                class_="group-params-block"
            )

        # ── Shared two-column header ──────────────────────────────────────────
        header = ui.div(
            ui.div(ui.tags.span("Group A", class_="group-col-label")),
            ui.div(ui.tags.span("Group B", class_="group-col-label")),
            class_="group-params-cols",
        )

        # ── σ row (shared by two-sample and paired) ───────────────────────────
        sigma_row = ui.div(
            ui.div(
                ui.input_numeric(
                    "pv_sigma",
                    ui.TagList("\u03c3\u00a0", tip("Standard deviation of Group A.")),
                    value=pval("pv_sigma", 1.0), min=0.1, step=0.5, width="100%",
                ),
            ),
            ui.div(
                ui.input_numeric(
                    "pv_sigma2",
                    ui.TagList("\u03c3\u00a0", tip("Standard deviation of Group B.")),
                    value=pval("pv_sigma2", 1.0), min=0.1, step=0.5, width="100%",
                ),
            ),
            class_="group-params-cols",
        )

        if structure == "two":
            n_row = ui.div(
                ui.div(
                    ui.input_numeric(
                        "pv_n1",
                        ui.TagList("n\u00a0", tip("Sample size of Group A.")),
                        value=pval("pv_n1", 10), min=2, max=500, step=1, width="100%",
                    ),
                ),
                ui.div(
                    ui.input_numeric(
                        "pv_n2",
                        ui.TagList("n\u00a0", tip("Sample size of Group B.")),
                        value=pval("pv_n2", 10), min=2, max=500, step=1, width="100%",
                    ),
                ),
                class_="group-params-cols",
            )
            return ui.div(header, sigma_row, n_row, class_="group-params-block")

        # paired
        advanced_row = ui.div(
            ui.div(
                ui.input_numeric(
                    "pv_rho",
                    ui.TagList(
                        "\u03c1 (correlation)\u00a0",
                        tip(
                            "Within-pair Pearson correlation. "
                            "Higher \u03c1 \u2192 smaller SD of differences \u2192 more power."
                        ),
                    ),
                    value=pval("pv_rho", 0.5), min=-0.99, max=0.99, step=0.1, width="100%",
                ),
            ),
            ui.div(
                ui.input_numeric(
                    "pv_n",
                    ui.TagList("Pairs (n)", tip("Number of paired observations.")),
                    value=pval("pv_n", 10), min=2, max=500, step=1, width="100%",
                ),
            ),
            class_="group-params-cols",
        )
        return ui.div(header, sigma_row, advanced_row, class_="group-params-block")

    @render.ui
    def pv_n_control():
        return ui.div()

    @render.ui
    def pv_pi_control():
        if _get_mode() != "pipeline":
            return ui.div()
        with reactive.isolate():
            cur = _get_pi()
        return ui.input_slider(
            "pv_pi",
            ui.TagList(
                "Prior null rate (\u03c0)",
                tip(
                    "Share of simulated experiments that have no real effect "
                    "(drawn from H\u2080). "
                    "In real A/B testing typically 70\u201395\u202f%. "
                    "Higher \u03c0 \u2192 higher False Positive Risk for the same \u03b1 and power."
                ),
            ),
            min=0.0, max=1.0, value=cur, step=0.05, width="100%",
        )

    # ── Scenario presets ──────────────────────────────────────────────────────
    _PV_PRESET_DESC = {
        "h0": (
            "H\u2080 true (Type\u00a0I error)",
            "One-sample, \u03bc_true\u200a=\u200a\u03bc\u2080\u200a=\u200a0, \u03c3\u200a=\u200a1, n\u200a=\u200a30. "
            "p-values are uniform on [0,\u00a01] \u2014 exactly \u03b1 of tests reject H\u2080. "
            "The textbook definition of Type\u00a0I error rate. "
            "Look at the p-value distribution: after ~500 samples it should flatten into a uniform rectangle, "
            "and the reject rate should hover around \u03b1.",
        ),
        "under": (
            "Low power study",
            "One-sample, \u03bc_true\u200a=\u200a0.2, \u03c3\u200a=\u200a1, n\u200a=\u200a10. "
            "Real effect exists but power \u2248\u200a12\u202f%. "
            "Most p-values exceed 0.05 \u2014 shows why small studies fail to replicate. "
            "Look at the Winner\u2019s Curse chart: the handful of significant runs sit far above the true-effect line "
            "\u2014 published small studies systematically overstate effects.",
        ),
        "largen": (
            "Large n inflation",
            "One-sample, \u03bc_true\u200a=\u200a0.1, \u03c3\u200a=\u200a1, n\u200a=\u200a300. "
            "Tiny effect (d\u200a=\u200a0.1) but power \u2248\u200a95\u202f% \u2014 almost all p < 0.05. "
            "Illustrates statistical vs practical significance. "
            "Look at the current p-value vs the observed effect: you reject H\u2080 almost every time, "
            "yet the effect (0.1\u00a0\u03c3) may be too small to matter in practice.",
        ),
        "outlier": (
            "Outlier disruption + Wilcoxon",
            "One-sample, \u03bc_true\u200a=\u200a0.5, \u03c3\u200a=\u200a1, n\u200a=\u200a20, outlier on, Wilcoxon on. "
            "A single opposing outlier pushes t-test p-values toward non-significance. "
            "Wilcoxon signed-rank resists the contamination. "
            "Look at the two overlapping histograms: the orange Wilcoxon curve concentrates near zero, "
            "while the blue t-test spreads toward 1 \u2014 the reject-rate gap is the cost of one bad point.",
        ),
        "paired": (
            "Paired design efficiency",
            "Paired, \u03bc_true\u200a=\u200a0.3, \u03c3\u200a=\u200a1, \u03c1\u200a=\u200a0.7, n\u200a=\u200a20. "
            "High within-pair correlation shrinks SD of differences. "
            "Power is much greater than an independent two-sample design with the same n. "
            "Look at the Power Diagram: \u03c1\u200a=\u200a0.7 collapses the H\u2081 distribution away from the critical value, "
            "so theoretical power is close to 90\u202f% \u2014 switch Test structure to Two-sample to see it drop sharply.",
        ),
    }

    def _pv_set_preset(structure, mu0, mu_true, alpha, method,
                       outlier=False, wilcoxon=False, **params):
        _pv_preset_params.set(params)
        ui.update_select("pv_test_structure", selected=structure)
        ui.update_numeric("pv_mu0",           value=mu0)
        ui.update_numeric("pv_mu_true",       value=mu_true)
        ui.update_slider("pv_alpha",          value=alpha)
        ui.update_select("pv_test_method",    selected=method)
        ui.update_checkbox("pv_outlier_on",   value=outlier)
        ui.update_checkbox("pv_wilcoxon_on",  value=wilcoxon)
        # Belt-and-suspenders for same-structure case (inputs already in DOM)
        for param_id, value in params.items():
            ui.update_numeric(param_id, value=value)

    @reactive.effect
    @reactive.event(input.pv_pre_h0)
    def _pv_pr_h0():
        _pv_active_preset.set("h0")
        _pv_set_preset("one", mu0=0.0, mu_true=0.0, alpha=0.05, method="t",
                       pv_sigma=1.0, pv_n=30)

    @reactive.effect
    @reactive.event(input.pv_pre_under)
    def _pv_pr_under():
        _pv_active_preset.set("under")
        _pv_set_preset("one", mu0=0.0, mu_true=0.2, alpha=0.05, method="t",
                       pv_sigma=1.0, pv_n=10)

    @reactive.effect
    @reactive.event(input.pv_pre_largen)
    def _pv_pr_largen():
        _pv_active_preset.set("largen")
        _pv_set_preset("one", mu0=0.0, mu_true=0.1, alpha=0.05, method="t",
                       pv_sigma=1.0, pv_n=300)

    @reactive.effect
    @reactive.event(input.pv_pre_outlier)
    def _pv_pr_outlier():
        _pv_active_preset.set("outlier")
        _pv_set_preset("one", mu0=0.0, mu_true=0.5, alpha=0.05, method="t",
                       outlier=True, wilcoxon=True,
                       pv_sigma=1.0, pv_n=20)

    @reactive.effect
    @reactive.event(input.pv_pre_paired)
    def _pv_pr_paired():
        _pv_active_preset.set("paired")
        _pv_set_preset("paired", mu0=0.0, mu_true=0.3, alpha=0.05, method="t",
                       pv_sigma=1.0, pv_sigma2=1.0, pv_rho=0.7, pv_n=20)

    @render.ui
    def pv_preset_desc():
        key = _pv_active_preset()
        if key is None:
            return ui.div(
                "\u2190 Select a preset to see what it demonstrates.",
                class_="np-preset-hint",
            )
        title, body = _PV_PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(title + ": "),
            body,
            class_="np-preset-hint np-preset-hint--active",
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

    # ── Outlier slider (shown only when checkbox is on) ───────────────────────
    @render.ui
    def pv_outlier_slider():
        try:
            on = input.pv_outlier_on()
        except Exception:
            return ui.div()
        if not on:
            return ui.div()
        return ui.input_slider(
            "pv_outlier_mag",
            ui.TagList(
                "Magnitude (\u00d7\u03c3)\u00a0",
                tip("How many standard deviations from the true mean the injected outlier is placed."),
            ),
            min=2, max=15, value=5, step=0.5, width="100%",
        )

    def _get_outlier_mag() -> float:
        try:
            return float(input.pv_outlier_mag())
        except Exception:
            return 5.0

    # ── Reset ─────────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.pv_btn_reset, input.pv_mu0, input.pv_alternative,
                    input.pv_test_method, input.pv_test_structure,
                    input.pv_outlier_on, input.pv_wilcoxon_on,
                    input.pv_mode)
    def _pv_reset():
        pv_total.set(0)
        pv_rejected.set(0)
        pv_last_stat.set(None)
        pv_all_pvalues.set(deque(maxlen=MAX_DATA))
        pv_all_effects.set(deque(maxlen=MAX_DATA))
        pv_wilcoxon_rejected.set(0)
        pv_wilcoxon_pvalues.set(deque(maxlen=MAX_DATA))
        pv_is_null_true.set(deque(maxlen=MAX_DATA))
        pv_null_count.set(0)
        pv_false_positives.set(0)
        pv_true_positives.set(0)
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
        mode        = _get_mode()

        try:
            outlier_on = bool(input.pv_outlier_on())
        except Exception:
            outlier_on = False
        outlier_mag = _get_outlier_mag() if outlier_on else 0.0

        # Direction sign: outlier always opposes the true effect so it
        # reduces power for any magnitude (larger mag → more broken test).
        _eff_sign = float(np.sign(mu_true - mu0)) if abs(mu_true - mu0) > 1e-9 else 1.0

        # Pipeline mode: per-experiment hypothesis indicator.
        # True  → drawn from H₀ (mean = mu0 everywhere).
        # False → drawn from H₁ (mean = mu_true for group 1 / paired component).
        if mode == "pipeline":
            pi = _get_pi()
            is_null_vec = np.random.rand(k) < pi  # (k,) bool
        else:
            is_null_vec = np.zeros(k, dtype=bool)  # all drawn from H₁

        # Per-experiment true mean for group 1 / one-sample / paired component A
        mu_exp = np.where(is_null_vec, mu0, mu_true).astype(float)  # (k,)

        if structure == "one":
            # ── One-sample ──────────────────────────────────────────────────
            noise = np.random.normal(0.0, sigma1, size=(n1, k))
            samps = noise + mu_exp                                     # (n1, k)
            if outlier_on:
                samps[0, :] = mu_exp - _eff_sign * outlier_mag * sigma1
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
            effects_arr = means - mu0

        elif structure == "two":
            # ── Two-sample independent ──────────────────────────────────────
            sigma2 = _get_sigma2()
            n2     = _get_n2()
            # Group 1 mean = mu_exp (mu_true under H₁, mu0 under H₀).
            # Group 2 mean is fixed at 0, so the true difference equals
            # mu_exp in H₁ experiments and mu0 in H₀ experiments.
            s1 = np.random.normal(0.0, sigma1, size=(n1, k)) + mu_exp
            s2 = np.random.normal(0.0, sigma2, size=(n2, k))
            if outlier_on:
                s1[0, :] = mu_exp - _eff_sign * outlier_mag * sigma1
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
            effects_arr = d - mu0

        else:
            # ── Paired ──────────────────────────────────────────────────────
            sigma2 = _get_sigma2()
            rho    = _get_rho()
            # Correlated bivariate normal with zero means, then shift
            # component A by mu_exp per experiment so that the mean of
            # differences equals mu_exp (mu_true under H₁, mu0 under H₀).
            cov    = rho * sigma1 * sigma2
            cov_mx = np.array([[sigma1**2, cov], [cov, sigma2**2]])
            # shape (n1, k, 2)
            pairs = np.random.multivariate_normal([0.0, 0.0], cov_mx, size=(n1, k))
            pairs[:, :, 0] = pairs[:, :, 0] + mu_exp                       # (n1, k)
            diffs = pairs[:, :, 0] - pairs[:, :, 1]                        # (n1, k)
            if outlier_on:
                sigma_d = np.sqrt(sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2)
                diffs[0, :] = mu_exp - _eff_sign * outlier_mag * sigma_d
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
            effects_arr = d_bar - mu0

        reject_mask  = pvals < alpha
        new_rejected = int(reject_mask.sum())
        pv_total.set(pv_total() + k)
        pv_rejected.set(pv_rejected() + new_rejected)
        pv_last_stat.set(float(stat_arr[-1]))

        pv = deque(pv_all_pvalues(), maxlen=MAX_DATA)
        pv.extend(float(p) for p in pvals)
        pv_all_pvalues.set(pv)

        ev = deque(pv_all_effects(), maxlen=MAX_DATA)
        ev.extend(float(e) for e in effects_arr)
        pv_all_effects.set(ev)

        # Pipeline-mode bookkeeping
        nv = deque(pv_is_null_true(), maxlen=MAX_DATA)
        nv.extend(bool(b) for b in is_null_vec)
        pv_is_null_true.set(nv)

        if mode == "pipeline":
            pv_null_count.set(pv_null_count() + int(is_null_vec.sum()))
            pv_false_positives.set(
                pv_false_positives() + int((reject_mask & is_null_vec).sum())
            )
            pv_true_positives.set(
                pv_true_positives() + int((reject_mask & ~is_null_vec).sum())
            )

        # ── Wilcoxon / Mann-Whitney comparison ───────────────────────────
        try:
            wilcoxon_on = bool(input.pv_wilcoxon_on())
        except Exception:
            wilcoxon_on = False

        if wilcoxon_on:
            # Map alternative for scipy
            alt_map = {"two-sided": "two-sided", "greater": "greater", "less": "less"}
            scipy_alt = alt_map.get(alternative, "two-sided")

            wil_pvals = []
            if structure == "one":
                for j in range(k):
                    col = samps[:, j]
                    diff = col - mu0
                    # Remove zeros (Wilcoxon requirement)
                    diff = diff[diff != 0]
                    if len(diff) < 10:
                        wil_pvals.append(float("nan"))
                        continue
                    try:
                        _, wp = stats.wilcoxon(diff, alternative=scipy_alt)
                        wil_pvals.append(float(wp))
                    except Exception:
                        wil_pvals.append(float("nan"))

            elif structure == "two":
                for j in range(k):
                    a_col, b_col = s1[:, j], s2[:, j]
                    try:
                        _, wp = stats.mannwhitneyu(a_col, b_col, alternative=scipy_alt)
                        wil_pvals.append(float(wp))
                    except Exception:
                        wil_pvals.append(float("nan"))

            else:  # paired
                for j in range(k):
                    d_col = diffs[:, j]
                    d_nz = d_col[d_col != 0]
                    if len(d_nz) < 10:
                        wil_pvals.append(float("nan"))
                        continue
                    try:
                        _, wp = stats.wilcoxon(d_nz, alternative=scipy_alt)
                        wil_pvals.append(float(wp))
                    except Exception:
                        wil_pvals.append(float("nan"))

            # Filter out NaNs
            valid = [p for p in wil_pvals if not np.isnan(p)]
            wil_new_rej = sum(1 for p in valid if p < alpha)
            pv_wilcoxon_rejected.set(pv_wilcoxon_rejected() + wil_new_rej)

            wpv = deque(pv_wilcoxon_pvalues(), maxlen=MAX_DATA)
            wpv.extend(valid)
            pv_wilcoxon_pvalues.set(wpv)

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
    def pv_wilcoxon_rate():
        td = pv_total()
        wn = len(pv_wilcoxon_pvalues())
        if td == 0 or wn == 0:
            return "\u2014"
        return f"{100 * pv_wilcoxon_rejected() / wn:.1f}%"

    @render.text
    def pv_fpr_value():
        fp = pv_false_positives()
        tp = pv_true_positives()
        total_rej = fp + tp
        if total_rej == 0:
            return "\u2014"
        return f"{100 * fp / total_rej:.1f}%"

    @render.ui
    def pv_fpr_stat_card():
        if _get_mode() != "pipeline":
            return ui.div()
        return ui.div(
            ui.div(
                "FPR\u00a0",
                tip(
                    "False Positive Risk = P(H\u2080 | reject) = "
                    "false positives / all rejections. "
                    "The share of your 'significant' findings that are actually null. "
                    "Driven by \u03c0 (base rate), \u03b1, and power \u2014 often 20\u201350\u202f% "
                    "in real A/B testing even with \u03b1 = 0.05."
                ),
                class_="stat-label",
            ),
            ui.div(ui.output_text("pv_fpr_value", inline=True),
                   class_="stat-value missed"),
            class_="stat-card",
        )

    @render.ui
    def pv_reject_stat_card():
        try:
            wil_on = bool(input.pv_wilcoxon_on())
        except Exception:
            wil_on = False

        structure = "one"
        try:
            structure = input.pv_test_structure()
        except Exception:
            pass

        param_card = ui.div(
            ui.div(
                "REJECT RATE\u00a0",
                tip(
                    "Fraction of tests where H\u2080 was rejected. "
                    "Equals empirical power when true value \u2260 H\u2080, "
                    "or Type\u00a0I error rate when true value = H\u2080."
                ),
                class_="stat-label",
            ),
            ui.div(ui.output_text("pv_reject_rate", inline=True),
                   class_="stat-value included"),
            class_="stat-card",
        )

        if not wil_on:
            return param_card

        wil_label = "MWU RATE" if structure == "two" else "WILCOXON RATE"
        wil_card = ui.div(
            ui.div(
                wil_label + "\u00a0",
                tip("Empirical reject rate for the nonparametric test."),
                class_="stat-label",
            ),
            ui.div(ui.output_text("pv_wilcoxon_rate", inline=True),
                   class_="stat-value wilcoxon"),
            class_="stat-card",
        )
        return ui.TagList(param_card, wil_card)

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
        return fig_to_ui(fig)

    @render.ui
    def pv_hist_plot():
        try:
            wil_on = bool(input.pv_wilcoxon_on())
        except Exception:
            wil_on = False

        structure = "one"
        try:
            structure = input.pv_test_structure()
        except Exception:
            pass

        wil_pvals = list(pv_wilcoxon_pvalues()) if wil_on else None
        wil_label = "Mann-Whitney U" if structure == "two" else "Wilcoxon"
        method = "t-test"
        try:
            method = "z-test" if input.pv_test_method() == "z" else "t-test"
        except Exception:
            pass

        fig = draw_pvalue_hist(
            list(pv_all_pvalues()),
            alpha=input.pv_alpha(),
            dark=is_dark(),
            pvalues_wilcoxon=wil_pvals,
            wilcoxon_label=wil_label,
            param_label=method,
        )
        return fig_to_ui(fig)

    @render.ui
    def pv_effect_scatter_plot():
        mode        = _get_mode()
        true_effect = _get_mu_true() - _get_mu0()
        null_vec    = list(pv_is_null_true()) if mode == "pipeline" else None
        fig = draw_effect_scatter(
            effects=list(pv_all_effects()),
            pvalues=list(pv_all_pvalues()),
            alpha=input.pv_alpha(),
            true_effect=true_effect,
            dark=is_dark(),
            is_null=null_vec,
            null_effect=0.0,                # effects are centered on H₀ (x̄ − μ₀)
        )
        return fig_to_ui(fig)

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
        return fig_to_ui(fig)
