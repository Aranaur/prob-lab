# =============================================================================
# Variance Reduction Explorer — server logic
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from shiny import reactive, render, ui

from utils import tip
from vr_plots import (
    draw_cuped_scatter,
    draw_cuped_variance_bars,
    draw_cuped_power_curve,
    draw_vwe_population,
    draw_vwe_variance_decomp,
    draw_vwe_ci_bars,
    draw_vwe_ci_bars,
    draw_vr_pvalue_hist,
)
from theme import fig_to_ui




def vr_server(input, output, session, is_dark):

    MAX_DATA = 10_000

    # ── Reactive state ──────────────────────────────────────────────────────
    _total         = reactive.value(0)
    _pv_naive      = reactive.value(deque(maxlen=MAX_DATA))
    _pv_method     = reactive.value(deque(maxlen=MAX_DATA))
    _last_data     = reactive.value(None)
    _sum_ci_naive  = reactive.value(0.0)
    _sum_ci_method = reactive.value(0.0)
    is_playing     = reactive.value(False)
    speed_ms       = reactive.value(0.5)
    _active_preset = reactive.value(None)

    @render.ui
    def vr_misconception_banner():
        if input.vr_mode() == "cuped":
            return ui.div(
                ui.tags.i(class_="info-icon"),
                ui.tags.strong(" Common Misconception: "),
                "\u201cI can center the covariate X by group mean in CUPED.\u201d",
                ui.tags.br(),
                ui.tags.strong("Reality: "),
                "In Y", ui.tags.sub("cuped"), " = Y \u2212 \u03b8\u0302(X \u2212 X\u0304), the mean X\u0304 must be the ",
                ui.tags.em("global"), " mean (control\u2009+\u2009treatment). Per-group centering biases the treatment-effect estimate.",
                class_="info-banner-text",
            )
        else:
            return ui.div(
                ui.tags.i(class_="info-icon"),
                ui.tags.strong(" Common Misconception: "),
                "\u201cI can classify 'power users' based on their behavior ",
                ui.tags.em("during"), " the A/B test.\u201d",
                ui.tags.br(),
                ui.tags.strong("Reality: "),
                "This causes ", ui.tags.strong("post-treatment bias"), ". Strata assignment must be based purely on ",
                ui.tags.em("pre-experiment"), " data to remain independent of the treatment effect.",
                class_="info-banner-text",
            )

    # ── Dynamic mode-specific parameters ────────────────────────────────────
    @render.ui
    def vr_mode_params():
        mode = input.vr_mode()
        if mode == "cuped":
            return ui.div(
                ui.input_slider(
                    "vr_rho",
                    ui.TagList(
                        "\u03c1 (covariate correlation)",
                        tip("Correlation between the pre-experiment covariate X "
                            "and outcome Y.  Higher \u03c1 \u2192 greater variance "
                            "reduction: Var(Y\u1d9c\u1d58\u1d56\u1d49\u1d48) = "
                            "Var(Y)\u00b7(1\u2212\u03c1\u00b2)."),
                    ),
                    min=0.0, max=0.95, value=0.50, step=0.05, width="100%",
                ),
                ui.input_numeric(
                    "vr_sigma",
                    ui.TagList("\u03c3", tip("Population standard deviation.")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
            )
        else:  # vwe
            return ui.div(
                ui.input_slider(
                    "vr_pct_regular",
                    ui.TagList(
                        "% regular users",
                        tip("Proportion of low-variance users.  The rest are "
                            "\u2018power users\u2019 (whales) whose high variance "
                            "dominates the naive pooled estimate."),
                    ),
                    min=50, max=99, value=90, step=1, width="100%",
                ),
                ui.div(
                    ui.div(
                        ui.input_numeric(
                            "vr_sigma_reg",
                            ui.TagList(
                                "\u03c3 (regular)",
                                tip("Std dev for regular users."),
                            ),
                            value=2.0, min=0.1, step=1.0, width="100%",
                        ),
                    ),
                    ui.div(
                        ui.input_numeric(
                            "vr_sigma_pow",
                            ui.TagList(
                                "\u03c3 (power)",
                                tip("Std dev for power users — typically much "
                                    "larger than \u03c3(regular)."),
                            ),
                            value=50.0, min=0.1, step=5.0, width="100%",
                        ),
                    ),
                    class_="group-params-cols",
                ),
            )

    # ── VWE presets ─────────────────────────────────────────────────────────
    _PRESET_DESC = {
        "marketplace": {
            "title": "Marketplace  (order sum)",
            "body":  "90\u2009% buyers: 1\u20133 orders/mo, \u03c3\u2009\u2248\u20092k. "
                     "10\u2009% power buyers: 10\u201350 orders, \u03c3\u2009\u2248\u200950k. "
                     "These 10\u2009% generate \u224860\u2009% of total variance.",
        },
        "streaming": {
            "title": "Streaming  (time in app)",
            "body":  "85\u2009% users: 15\u201330 min/day, \u03c3\u2009\u2248\u200920 min. "
                     "15\u2009% power users: 3\u20136 h/day, \u03c3\u2009\u2248\u2009120 min. "
                     "Naive t-test is driven by the heavy tail.",
        },
    }

    @render.ui
    def vr_presets():
        if input.vr_mode() != "vwe":
            return ui.div()
        return ui.div(
            ui.tags.label("Presets", class_="presets-label"),
            ui.div(
                ui.input_action_button("vr_preset_marketplace", "Marketplace",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("vr_preset_streaming", "Streaming",
                                       class_="btn-ctrl btn-sample btn-flex"),
                class_="sidebar-btn-row",
            ),
            ui.output_ui("vr_preset_desc"),
        )

    @render.ui
    def vr_preset_desc():
        key = _active_preset()
        if key is None or key not in _PRESET_DESC:
            return ui.div(
                "Select a preset to see a real-world scenario.",
                class_="np-preset-hint",
            )
        d = _PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(d["title"]),
            ui.tags.br(), d["body"],
            class_="np-preset-hint np-preset-hint--active",
        )

    @reactive.effect
    @reactive.event(input.vr_preset_marketplace)
    def _preset_mp():
        _active_preset.set("marketplace")
        ui.update_slider("vr_pct_regular", value=90)
        ui.update_numeric("vr_sigma_reg", value=2.0)
        ui.update_numeric("vr_sigma_pow", value=50.0)
        ui.update_numeric("vr_delta", value=0.5)
        ui.update_numeric("vr_n", value=500)

    @reactive.effect
    @reactive.event(input.vr_preset_streaming)
    def _preset_st():
        _active_preset.set("streaming")
        ui.update_slider("vr_pct_regular", value=85)
        ui.update_numeric("vr_sigma_reg", value=20.0)
        ui.update_numeric("vr_sigma_pow", value=120.0)
        ui.update_numeric("vr_delta", value=5.0)
        ui.update_numeric("vr_n", value=500)

    # ── Reset ───────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.vr_btn_reset, input.vr_mode)
    def _reset():
        _total.set(0)
        _pv_naive.set(deque(maxlen=MAX_DATA))
        _pv_method.set(deque(maxlen=MAX_DATA))
        _last_data.set(None)
        _sum_ci_naive.set(0.0)
        _sum_ci_method.set(0.0)
        is_playing.set(False)
        _active_preset.set(None)
        ui.update_action_button("vr_btn_play", label="Play")

    # ── Speed ───────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.vr_speed_minus)
    def _spd_dn():
        speed_ms.set(min(speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.vr_speed_plus)
    def _spd_up():
        speed_ms.set(max(speed_ms() - 0.05, 0.05))

    # ── Play / Pause ────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.vr_btn_play)
    def _toggle():
        is_playing.set(not is_playing())
        ui.update_action_button(
            "vr_btn_play", label="Pause" if is_playing() else "Play")

    @reactive.effect
    def _auto():
        if is_playing():
            reactive.invalidate_later(speed_ms())
            with reactive.isolate():
                _draw_samples(1)

    # ── Manual buttons ──────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.vr_btn_sample_1)
    def _s1():
        _draw_samples(1)

    @reactive.effect
    @reactive.event(input.vr_btn_sample_50)
    def _s50():
        _draw_samples(50)

    @reactive.effect
    @reactive.event(input.vr_btn_sample_100)
    def _s100():
        _draw_samples(100)

    # ═══════════════════════════════════════════════════════════════════════
    # Core sampling
    # ═══════════════════════════════════════════════════════════════════════

    def _draw_samples(k: int):
        if input.vr_mode() == "cuped":
            _draw_cuped(k)
        else:
            _draw_vwe(k)

    # ── CUPED ───────────────────────────────────────────────────────────────
    def _draw_cuped(k: int):
        n     = max(10, int(input.vr_n() or 200))
        delta = float(input.vr_delta() or 0.0)
        try:
            sigma = max(0.01, float(input.vr_sigma()))
        except Exception:
            sigma = 1.0
        try:
            rho = float(input.vr_rho())
        except Exception:
            rho = 0.5
        alpha = float(input.vr_alpha() or 0.05)

        # ── Generate correlated (X, Y) ──
        # X ~ N(0, σ²);  Y = ρX + √(1−ρ²)·σ·Z  ⟹  Var(Y) = σ²,  Cor(X,Y) = ρ
        X_c = np.random.normal(0, sigma, (n, k))
        X_t = np.random.normal(0, sigma, (n, k))
        noise_std = sigma * np.sqrt(max(1 - rho ** 2, 0))
        Y_c = rho * X_c + np.random.normal(0, noise_std, (n, k))
        Y_t = delta + rho * X_t + np.random.normal(0, noise_std, (n, k))

        # ── Naive Welch t-test ──
        mc, mt = Y_c.mean(0), Y_t.mean(0)
        vc, vt = Y_c.var(0, ddof=1), Y_t.var(0, ddof=1)
        se_n   = np.sqrt(vc / n + vt / n)
        t_n    = (mt - mc) / se_n
        df_n   = (vc / n + vt / n) ** 2 / (
            (vc / n) ** 2 / (n - 1) + (vt / n) ** 2 / (n - 1))
        pv_n   = 2 * (1 - stats.t.cdf(np.abs(t_n), df_n))

        # ── CUPED adjustment  (global X̄!) ──
        X_all = np.vstack([X_c, X_t])                       # (2n, k)
        Y_all = np.vstack([Y_c, Y_t])                       # (2n, k)
        X_bar = X_all.mean(axis=0, keepdims=True)            # (1, k)
        # Batched θ̂ = ΣXc·Yc / ΣXc²
        Xc_ = X_all - X_bar
        Yc_ = Y_all - Y_all.mean(axis=0, keepdims=True)
        theta = np.sum(Xc_ * Yc_, axis=0) / (np.sum(Xc_ ** 2, axis=0) + 1e-12)

        Y_cc = Y_c - theta[np.newaxis, :] * (X_c - X_bar)
        Y_ct = Y_t - theta[np.newaxis, :] * (X_t - X_bar)

        mcc, mct = Y_cc.mean(0), Y_ct.mean(0)
        vcc, vct = Y_cc.var(0, ddof=1), Y_ct.var(0, ddof=1)
        se_cup   = np.sqrt(vcc / n + vct / n)
        t_cup    = (mct - mcc) / se_cup
        df_cup   = (vcc / n + vct / n) ** 2 / (
            (vcc / n) ** 2 / (n - 1) + (vct / n) ** 2 / (n - 1))
        pv_cup   = 2 * (1 - stats.t.cdf(np.abs(t_cup), df_cup))

        # ── Update state ──
        _total.set(_total() + k)

        pn = deque(_pv_naive(), maxlen=MAX_DATA)
        pn.extend(pv_n.tolist())
        _pv_naive.set(pn)

        pm = deque(_pv_method(), maxlen=MAX_DATA)
        pm.extend(pv_cup.tolist())
        _pv_method.set(pm)

        z_a = stats.norm.ppf(1 - alpha / 2)
        _sum_ci_naive.set(_sum_ci_naive() + float(np.sum(2 * z_a * se_n)))
        _sum_ci_method.set(_sum_ci_method() + float(np.sum(2 * z_a * se_cup)))

        # Last experiment for scatter visualisation
        _last_data.set({
            "x_ctrl": X_c[:, -1], "y_ctrl": Y_c[:, -1],
            "x_treat": X_t[:, -1], "y_treat": Y_t[:, -1],
            "theta": float(theta[-1]),
            "delta_hat": float(mct[-1] - mcc[-1]),
        })

    # ── VWE ─────────────────────────────────────────────────────────────────
    def _draw_vwe(k: int):
        n     = max(10, int(input.vr_n() or 200))
        delta = float(input.vr_delta() or 0.0)
        alpha = float(input.vr_alpha() or 0.05)
        try:
            pct = float(input.vr_pct_regular())
        except Exception:
            pct = 90.0
        try:
            sr = max(0.01, float(input.vr_sigma_reg()))
        except Exception:
            sr = 2.0
        try:
            sp = max(0.01, float(input.vr_sigma_pow()))
        except Exception:
            sp = 50.0

        nr  = max(2, int(n * pct / 100))
        npw = max(2, n - nr)

        # ── Heteroscedastic data ──
        yc_r = np.random.normal(0, sr, (nr, k))
        yc_p = np.random.normal(0, sp, (npw, k))
        yt_r = np.random.normal(delta, sr, (nr, k))
        yt_p = np.random.normal(delta, sp, (npw, k))

        yc = np.vstack([yc_r, yc_p])   # (n, k)
        yt = np.vstack([yt_r, yt_p])

        # ── Naive Welch t-test ──
        mc, mt = yc.mean(0), yt.mean(0)
        vc, vt = yc.var(0, ddof=1), yt.var(0, ddof=1)
        se_n   = np.sqrt(vc / n + vt / n)
        t_n    = (mt - mc) / se_n
        df_n   = (vc / n + vt / n) ** 2 / (
            (vc / n) ** 2 / (n - 1) + (vt / n) ** 2 / (n - 1))
        pv_n   = 2 * (1 - stats.t.cdf(np.abs(t_n), df_n))

        # ── VWE: weighted by inverse group variance ──
        # Pool group variances across arms for stability
        s2r = (yc_r.var(0, ddof=1) + yt_r.var(0, ddof=1)) / 2  # (k,)
        s2p = (yc_p.var(0, ddof=1) + yt_p.var(0, ddof=1)) / 2

        wr = nr  / (s2r + 1e-12)     # w_g = n_g / s_g²
        wp = npw / (s2p + 1e-12)
        W  = wr + wp                  # total weight per experiment

        # Weighted means for each arm
        mc_w = (wr * yc_r.mean(0) + wp * yc_p.mean(0)) / W
        mt_w = (wr * yt_r.mean(0) + wp * yt_p.mean(0)) / W

        # SE of difference:  Var(ȳ_w) = 1/W  for one arm ⟹ SE = √(2/W)
        se_vwe = np.sqrt(2.0 / W)
        z_vwe  = (mt_w - mc_w) / se_vwe
        pv_vwe = 2 * (1 - stats.norm.cdf(np.abs(z_vwe)))

        # ── Update state ──
        _total.set(_total() + k)

        pn = deque(_pv_naive(), maxlen=MAX_DATA)
        pn.extend(pv_n.tolist())
        _pv_naive.set(pn)

        pm = deque(_pv_method(), maxlen=MAX_DATA)
        pm.extend(pv_vwe.tolist())
        _pv_method.set(pm)

        z_a = stats.norm.ppf(1 - alpha / 2)
        _sum_ci_naive.set(_sum_ci_naive() + float(np.sum(2 * z_a * se_n)))
        _sum_ci_method.set(_sum_ci_method() + float(np.sum(2 * z_a * se_vwe)))

        # Last experiment for population chart
        _last_data.set({
            "y_reg": np.concatenate([yc_r[:, -1], yt_r[:, -1]]),
            "y_pow": np.concatenate([yc_p[:, -1], yt_p[:, -1]]),
        })

    # ═══════════════════════════════════════════════════════════════════════
    # Stats row  (fully dynamic — depends on mode)
    # ═══════════════════════════════════════════════════════════════════════

    @render.ui
    def vr_stats_row():
        mode  = input.vr_mode()
        tot   = _total()
        alpha = float(input.vr_alpha() or 0.05)

        if mode == "cuped":
            try:
                rho = float(input.vr_rho())
            except Exception:
                rho = 0.5
            n = max(10, int(input.vr_n() or 200))
            try:
                sigma = max(0.01, float(input.vr_sigma()))
            except Exception:
                sigma = 1.0

            var_red  = (1 - rho ** 2) * 100
            z_a      = stats.norm.ppf(1 - alpha / 2)
            z_b      = stats.norm.ppf(0.80)
            se_naive = sigma * np.sqrt(2 / n)
            mde_n    = (z_a + z_b) * se_naive
            mde_c    = mde_n * np.sqrt(max(1 - rho ** 2, 0))
            n_eq     = n / (1 - rho ** 2) if rho < 1 else float("inf")

            return ui.div(
                _stat_card("VAR REDUCTION\u00a0",
                           tip("Variance reduction: (1\u2212\u03c1\u00b2)\u00d7100%."),
                           f"{var_red:.1f}%", "coverage"),
                _stat_card("MDE (NAIVE)\u00a0",
                           tip("Minimum detectable effect at 80% power, naive test."),
                           f"{mde_n:.3f}", "missed"),
                _stat_card("MDE (CUPED)\u00a0",
                           tip(f"MDE\u00d7\u221a(1\u2212\u03c1\u00b2).  "
                               f"Equivalent to {n_eq:.0f} obs/group."),
                           f"{mde_c:.3f}", "included"),
                _stat_card("TOTAL EXPERIMENTS\u00a0",
                           tip("Total simulated experiments."),
                           f"{tot:,}", "total"),
                class_="stats-row",
            )
        else:  # vwe
            try:
                pct = float(input.vr_pct_regular())
            except Exception:
                pct = 90.0
            try:
                sr = max(0.01, float(input.vr_sigma_reg()))
            except Exception:
                sr = 2.0
            try:
                sp = max(0.01, float(input.vr_sigma_pow()))
            except Exception:
                sp = 50.0

            p = pct / 100
            total_var   = p * sr ** 2 + (1 - p) * sp ** 2
            whale_share = (1 - p) * sp ** 2 / total_var * 100

            pv_n = list(_pv_naive())
            pv_m = list(_pv_method())
            rej_n = (f"{np.mean(np.array(pv_n) < alpha) * 100:.1f}%"
                     if len(pv_n) > 0 else "\u2014")
            rej_m = (f"{np.mean(np.array(pv_m) < alpha) * 100:.1f}%"
                     if len(pv_m) > 0 else "\u2014")

            return ui.div(
                _stat_card("WHALE VAR SHARE\u00a0",
                           tip("% of total pooled variance from power users."),
                           f"{whale_share:.1f}%", "missed"),
                _stat_card("NAIVE REJECT\u00a0",
                           tip("Empirical reject rate (naive t-test)."),
                           rej_n, "missed"),
                _stat_card("VWE REJECT\u00a0",
                           tip("Empirical reject rate (VWE)."),
                           rej_m, "coverage"),
                _stat_card("TOTAL EXPERIMENTS\u00a0",
                           tip("Total simulated experiments."),
                           f"{tot:,}", "total"),
                class_="stats-row",
            )

    # ── Chart titles ────────────────────────────────────────────────────────
    @render.text
    def vr_title1():
        return ("CUPED: PRE vs POST SCATTER"
                if input.vr_mode() == "cuped"
                else "USER POPULATION")

    @render.text
    def vr_title2():
        return ("VARIANCE COMPARISON"
                if input.vr_mode() == "cuped"
                else "VARIANCE DECOMPOSITION")

    @render.text
    def vr_title3():
        return ("POWER vs \u03c1"
                if input.vr_mode() == "cuped"
                else "CI WIDTH COMPARISON")

    # ═══════════════════════════════════════════════════════════════════════
    # Chart renderers
    # ═══════════════════════════════════════════════════════════════════════

    @render.ui
    def vr_chart1():
        dark = is_dark()
        data = _last_data()

        if input.vr_mode() == "cuped":
            if data is None or "x_ctrl" not in data:
                return ui.div("Draw samples to see the scatter plot.",
                              class_="chart-placeholder")
            fig = draw_cuped_scatter(
                data["x_ctrl"], data["y_ctrl"],
                data["x_treat"], data["y_treat"],
                data["theta"], data["delta_hat"], dark,
            )
        else:
            if data is None or "y_reg" not in data:
                return ui.div("Draw samples to see the distribution.",
                              class_="chart-placeholder")
            fig = draw_vwe_population(data["y_reg"], data["y_pow"], dark)
        return fig_to_ui(fig)

    @render.ui
    def vr_chart2():
        dark = is_dark()

        if input.vr_mode() == "cuped":
            try:
                rho = float(input.vr_rho())
            except Exception:
                rho = 0.5
            try:
                sigma = max(0.01, float(input.vr_sigma()))
            except Exception:
                sigma = 1.0
            n = max(10, int(input.vr_n() or 200))
            var_naive = 2 * sigma ** 2 / n
            var_cuped = var_naive * (1 - rho ** 2)
            fig = draw_cuped_variance_bars(var_naive, var_cuped, rho, n, dark)
        else:
            try:
                pct = float(input.vr_pct_regular())
            except Exception:
                pct = 90.0
            try:
                sr = max(0.01, float(input.vr_sigma_reg()))
            except Exception:
                sr = 2.0
            try:
                sp = max(0.01, float(input.vr_sigma_pow()))
            except Exception:
                sp = 50.0
            fig = draw_vwe_variance_decomp(pct, sr, sp, dark)
        return fig_to_ui(fig)

    @render.ui
    def vr_chart3():
        dark = is_dark()

        if input.vr_mode() == "cuped":
            n     = max(10, int(input.vr_n() or 200))
            delta = float(input.vr_delta() or 0.0)
            try:
                sigma = max(0.01, float(input.vr_sigma()))
            except Exception:
                sigma = 1.0
            alpha   = float(input.vr_alpha() or 0.05)
            try:
                rho_now = float(input.vr_rho())
            except Exception:
                rho_now = 0.5
            z_a       = stats.norm.ppf(1 - alpha / 2)
            rho_range = np.linspace(0, 0.95, 50)

            def _pw(r):
                se = sigma * np.sqrt(2 / n) * np.sqrt(max(1 - r ** 2, 0))
                if se <= 0 or delta == 0:
                    return alpha if delta == 0 else 1.0
                return float(stats.norm.cdf(abs(delta) / se - z_a))

            pow_naive = [_pw(0)] * len(rho_range)
            pow_cuped = [_pw(r) for r in rho_range]
            fig = draw_cuped_power_curve(rho_range, pow_naive, pow_cuped,
                                         rho_now, dark)
        else:
            tot = _total()
            ci_n = _sum_ci_naive() / tot if tot > 0 else 0.0
            ci_v = _sum_ci_method() / tot if tot > 0 else 0.0
            fig = draw_vwe_ci_bars(ci_n, ci_v, dark)
        return fig_to_ui(fig)

    @render.ui
    def vr_pvalue_plot():
        dark  = is_dark()
        alpha = float(input.vr_alpha() or 0.05)
        label = "CUPED" if input.vr_mode() == "cuped" else "VWE"
        fig = draw_vr_pvalue_hist(
            list(_pv_naive()), list(_pv_method()),
            alpha, method_label=label, dark=dark,
        )
        return fig_to_ui(fig)


# ── helper ──────────────────────────────────────────────────────────────────
def _stat_card(label_text, label_tip, value_text, value_class):
    return ui.div(
        ui.div(label_text, label_tip, class_="stat-label"),
        ui.div(value_text, class_=f"stat-value {value_class}"),
        class_="stat-card",
    )
