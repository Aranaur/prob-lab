# =============================================================================
# Bayesian Explorer — server logic
#
# Three sub-tabs:
#   Tab 1 — Beta-Binomial Updating (prefix bys1_)
#   Tab 2 — Frequentist vs Bayesian CI (prefix bys2_)
#   Tab 3 — Bayesian A/B Testing (prefix bys3_)
# =============================================================================

from __future__ import annotations

import numpy as np
from scipy import stats
from shiny import reactive, render, ui

from utils import tip
from ci_methods import compute_ci_proportion
from bayes_plots import (
    draw_bys1_prior_posterior,
    draw_bys1_coin_sequence,
    draw_bys1_evolution_forest,
    draw_bys2_forest,
    draw_bys2_width_hist,
    draw_bys2_running_coverage,
    draw_bys3_posteriors,
    draw_bys3_joint,
    draw_bys3_prob_evolution,
)

_PLOTLY_CFG = {"displayModeBar": False, "responsive": True}


def _fig_html(fig):
    return ui.div(
        ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False,
                            config=_PLOTLY_CFG)),
        class_="plotly-container",
    )


def _stat_card(label_text, label_tip, value_html, value_class, sub_html=None):
    body = [ui.div(value_html, class_=f"stat-value {value_class}")]
    if sub_html is not None:
        body.append(ui.div(sub_html, class_="stat-sub"))
    return ui.div(
        ui.div(label_text, label_tip, class_="stat-label"),
        *body,
        class_="stat-card",
    )


# ── Module-level math helpers ────────────────────────────────────────────────

def _update_beta(alpha: float, beta: float, k: int, n: int) -> tuple[float, float]:
    """Beta-Binomial posterior: Beta(α + k, β + n − k)."""
    return alpha + k, beta + n - k


def _cred_interval(alpha: float, beta: float,
                   level: float = 0.95) -> tuple[float, float]:
    """Equal-tail credible interval for Beta(α, β)."""
    a_lo = (1.0 - level) / 2.0
    a_hi = 1.0 - a_lo
    return float(stats.beta.ppf(a_lo, alpha, beta)), \
           float(stats.beta.ppf(a_hi, alpha, beta))


def _post_mode(alpha: float, beta: float) -> float | None:
    """Posterior mode; undefined when α ≤ 1 or β ≤ 1 (boundary case)."""
    if alpha > 1.0 and beta > 1.0:
        return (alpha - 1.0) / (alpha + beta - 2.0)
    return None


def _sample_posteriors(
    aA: float, bA: float, aB: float, bB: float,
    draws: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded Beta draws for the two variants."""
    rng = np.random.default_rng(seed)
    pA = rng.beta(aA, bA, draws)
    pB = rng.beta(aB, bB, draws)
    return pA, pB


def _ab_metrics(
    aA: float, bA: float, aB: float, bB: float,
    draws: int, seed: int,
) -> dict[str, float]:
    """Decision-theory summary from posterior draws."""
    pA, pB = _sample_posteriors(aA, bA, aB, bB, draws, seed)
    diff = pB - pA
    meanA = float(np.mean(pA))
    e_lift_abs = float(np.mean(diff))
    return {
        "prob_b_beats_a": float(np.mean(diff > 0.0)),
        "e_lift_abs":     e_lift_abs,
        "e_lift_rel":     e_lift_abs / meanA if meanA > 0.0 else 0.0,
        "e_loss_A":       float(np.mean(np.maximum(diff, 0.0))),
        "e_loss_B":       float(np.mean(np.maximum(-diff, 0.0))),
        "meanA":          meanA,
        "meanB":          float(np.mean(pB)),
    }


# ═════════════════════════════════════════════════════════════════════════════
def bayes_server(input, output, session, is_dark):

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 1 — Beta-Binomial Updating
    # ═══════════════════════════════════════════════════════════════════════

    _bys1_obs = reactive.value([])                  # list of 0/1 draws
    _bys1_preset = reactive.value(None)             # active prior preset key

    # Prior presets: (label, alpha, beta)
    _BYS1_PRIOR_PRESETS = {
        "uniform":   ("Uniform Beta(1, 1)",          1.0,  1.0),
        "jeffreys":  ("Jeffreys Beta(½, ½)",         0.5,  0.5),
        "weak":      ("Weak Beta(2, 2)",             2.0,  2.0),
        "skeptic":   ("Strong-skeptic Beta(20, 20)", 20.0, 20.0),
        "confident": ("Strong-confident Beta(20, 5)", 20.0, 5.0),
    }

    _BYS1_PRESET_DESC = {
        "uniform": (
            "Uniform Beta(1, 1)",
            "Flat prior \u2014 no preference over p. Posterior mode equals the MLE "
            "k\u2215n. Use as a baseline to see what the data alone say.",
        ),
        "jeffreys": (
            "Jeffreys Beta(\u00bd, \u00bd)",
            "Non-informative, invariant under reparameterisation. "
            "Pulls slightly toward extremes; matches the Wilson CI on Tab 2.",
        ),
        "weak": (
            "Weak Beta(2, 2)",
            "Gentle pull toward p\u2009=\u20090.5 (\u03b1+\u03b2 = 4 pseudo-obs). "
            "Barely visible once n exceeds a few dozen.",
        ),
        "skeptic": (
            "Strong-skeptic Beta(20, 20)",
            "Concentrated at 0.5 with 40 pseudo-observations. "
            "With small n the posterior stays near 0.5; resample to n\u2009=\u2009500 "
            "to watch the data override the prior.",
        ),
        "confident": (
            "Strong-confident Beta(20, 5)",
            "Believes p \u2248 0.8 a priori. If truth is 0.30, the prior will fight "
            "the data until n is large \u2014 a cautionary tale for overconfident priors.",
        ),
    }

    for key, (_label, a, b) in _BYS1_PRIOR_PRESETS.items():
        def _make_handler(a=a, b=b, key=key):
            @reactive.effect
            @reactive.event(input[f"bys1_prior_{key}"])
            def _apply_prior():
                ui.update_numeric("bys1_alpha", value=a)
                ui.update_numeric("bys1_beta",  value=b)
                _bys1_preset.set(key)
        _make_handler()

    @render.ui
    def bys1_preset_desc():
        key = _bys1_preset()
        if key is None or key not in _BYS1_PRESET_DESC:
            return ui.div(
                "\u2190 Pick a prior preset to see what it demonstrates.",
                class_="np-preset-hint",
            )
        title, body = _BYS1_PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(title + ": "),
            body,
            class_="np-preset-hint np-preset-hint--active",
        )

    # Sample buttons — append to the observation list
    def _sample_bys1(count: int):
        if bool(input.bys1_lock()):
            return
        try:
            p = float(input.bys1_true_p() or 0.3)
        except Exception:
            p = 0.3
        p = max(0.001, min(0.999, p))
        flips = np.random.binomial(1, p, count).astype(int).tolist()
        _bys1_obs.set(list(_bys1_obs()) + flips)

    @reactive.effect
    @reactive.event(input.bys1_sample_1)
    def _s1(): _sample_bys1(1)

    @reactive.effect
    @reactive.event(input.bys1_sample_50)
    def _s50(): _sample_bys1(50)

    @reactive.effect
    @reactive.event(input.bys1_sample_100)
    def _s100(): _sample_bys1(100)

    @reactive.effect
    @reactive.event(input.bys1_reset)
    def _bys1_reset():
        _bys1_obs.set([])

    # Reset observations when true p changes (old flips are no longer from this DGP)
    @reactive.effect
    @reactive.event(input.bys1_true_p)
    def _bys1_reset_on_truep():
        if not bool(input.bys1_lock()):
            _bys1_obs.set([])

    # Speed / Play
    _bys1_playing  = reactive.value(False)
    _bys1_speed_ms = reactive.value(0.3)

    @reactive.effect
    @reactive.event(input.bys1_speed_minus)
    def _bys1_sp_dn():
        _bys1_speed_ms.set(min(_bys1_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.bys1_speed_plus)
    def _bys1_sp_up():
        _bys1_speed_ms.set(max(_bys1_speed_ms() - 0.05, 0.05))

    @reactive.effect
    @reactive.event(input.bys1_btn_play)
    def _bys1_toggle():
        _bys1_playing.set(not _bys1_playing())
        ui.update_action_button(
            "bys1_btn_play",
            label="Pause" if _bys1_playing() else "Play",
        )

    # Stop playing when lock is engaged
    @reactive.effect
    @reactive.event(input.bys1_lock)
    def _bys1_lock_stops_play():
        if bool(input.bys1_lock()) and _bys1_playing():
            _bys1_playing.set(False)
            ui.update_action_button("bys1_btn_play", label="Play")

    @reactive.effect
    def _bys1_auto():
        if _bys1_playing() and not bool(input.bys1_lock()):
            reactive.invalidate_later(_bys1_speed_ms())
            with reactive.isolate():
                _sample_bys1(1)

    # ── Derived scalars ────────────────────────────────────────────────────
    def _bys1_pp():
        try:
            a = float(input.bys1_alpha() or 1.0)
            b = float(input.bys1_beta()  or 1.0)
        except Exception:
            a, b = 1.0, 1.0
        return max(0.01, a), max(0.01, b)

    def _bys1_kn() -> tuple[int, int]:
        obs = _bys1_obs()
        return int(sum(obs)), len(obs)

    # ── Lock-data badge ────────────────────────────────────────────────────
    @render.ui
    def bys1_lock_badge():
        if not bool(input.bys1_lock()):
            return ui.div()   # empty
        _, n = _bys1_kn()
        return ui.div(
            ui.span("🔒 ", class_="bys-lock-icon"),
            ui.tags.strong(f"Data locked  (n = {n})"),
            class_="bys-lock-badge",
        )

    # Sample + Speed/Play rows (disabled placeholder when locked)
    @render.ui
    def bys1_sample_controls():
        if bool(input.bys1_lock()):
            return ui.div(
                ui.em("Unlock to sample new data"),
                class_="bys-sample-disabled",
            )
        return ui.TagList(
            ui.div(
                ui.tags.span("Sample:", class_="btn-row-label"),
                ui.input_action_button("bys1_sample_1",   "\u00d71",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("bys1_sample_50",  "\u00d750",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("bys1_sample_100", "\u00d7100",
                                       class_="btn-ctrl btn-sample btn-flex"),
                class_="sidebar-btn-row",
            ),
            ui.div(
                ui.div(
                    ui.tags.label("Speed"),
                    ui.input_action_button("bys1_speed_minus", "\u2212",
                                           class_="btn-ctrl btn-pm"),
                    ui.input_action_button("bys1_btn_play", "Play",
                                           class_="btn-ctrl btn-play btn-flex"),
                    ui.input_action_button("bys1_speed_plus", "+",
                                           class_="btn-ctrl btn-pm"),
                    class_="ctrl-group ctrl-group-full",
                ),
                class_="sidebar-btn-row",
            ),
        )

    # ── Stats row ─────────────────────────────────────────────────────────
    @render.ui
    def bys1_stats_row():
        a_prior, b_prior = _bys1_pp()
        k, n = _bys1_kn()
        a_post, b_post = _update_beta(a_prior, b_prior, k, n)

        mean = a_post / (a_post + b_post)
        mode = _post_mode(a_post, b_post)
        mode_txt = f"{mode:.3f}" if mode is not None else "undefined (boundary case)"
        lo, hi = _cred_interval(a_post, b_post, level=0.95)
        prior_w = a_prior + b_prior

        return ui.div(
            _stat_card("N OBSERVED",
                       tip("Number of Bernoulli trials observed so far."),
                       f"{n}", "coverage"),
            _stat_card("POSTERIOR MEAN",
                       tip("E[p | data] = (α + k) / (α + β + n)."),
                       f"{mean:.3f}", "included"),
            _stat_card("POSTERIOR MODE",
                       tip("argmax of posterior Beta; undefined when α ≤ 1 or β ≤ 1."),
                       mode_txt, "pw-d"),
            _stat_card("95% CrI",
                       tip("Credible interval — 95% posterior mass given prior and data."),
                       f"[{lo:.3f}, {hi:.3f}]", "total"),
            _stat_card("PRIOR vs DATA",
                       tip("Prior weight (α+β) acts like pseudo-observations; "
                           "when n >> (α+β) the posterior is dominated by data."),
                       f"{prior_w:.1f} pseudo",
                       "pw-alpha",
                       sub_html=f"{n} observed"),
            class_="stats-row bys1-stats-row",
        )

    # ── Chart 1 title (dynamic) ───────────────────────────────────────────
    @render.ui
    def bys1_chart1_title():
        if bool(input.bys1_lock()):
            _, n = _bys1_kn()
            return ui.div(
                f"POSTERIOR RECOMPUTED ON THE SAME {n} OBSERVATIONS",
                class_="card-title",
            )
        return ui.div(
            "PRIOR · LIKELIHOOD · POSTERIOR",
            class_="section-title",
        )

    # ── Charts ─────────────────────────────────────────────────────────────
    @render.ui
    def bys1_prior_posterior():
        a, b = _bys1_pp()
        k, n = _bys1_kn()
        fig = draw_bys1_prior_posterior(a, b, k, n, dark=is_dark())
        return _fig_html(fig)

    @render.ui
    def bys1_coin_sequence():
        fig = draw_bys1_coin_sequence(list(_bys1_obs()), dark=is_dark())
        return _fig_html(fig)

    @render.ui
    def bys1_evolution_forest():
        a, b = _bys1_pp()
        fig = draw_bys1_evolution_forest(a, b, list(_bys1_obs()), dark=is_dark())
        return _fig_html(fig)


    # ═══════════════════════════════════════════════════════════════════════
    # Tab 2 — Frequentist vs Bayesian CI
    # ═══════════════════════════════════════════════════════════════════════

    # Raw k-values are the source of truth; history = derived CI tuples.
    _bys2_raw_k   = reactive.value([])
    _bys2_history = reactive.value([])    # list of (flo, fhi, blo, bhi, covf, covb)
    _bys2_preset  = reactive.value(None)

    _BYS2_PRESETS = {
        # key: (label, true_p, n, alpha, beta, freq_method)
        "dominates":  ("Prior dominates",  0.30, 10,  20.0, 5.0,  "wilson"),
        "agreement":  ("Agreement",        0.50, 500, 1.0,  1.0,  "wilson"),
        "jeffreys":   ("Jeffreys ≈ Wilson", 0.50, 100, 0.5,  0.5,  "wilson"),
    }

    _BYS2_PRESET_DESC = {
        "dominates": (
            "Prior dominates",
            "Beta(20, 5) prior with true p\u2009=\u20090.30, n\u2009=\u200910. "
            "The CrI sits near 0.8 (prior mean) while the Wilson CI tracks the data. "
            "Bayesian coverage plummets \u2014 a calibration cost of a wrong prior.",
        ),
        "agreement": (
            "Agreement (large n)",
            "Uniform Beta(1, 1) prior with n\u2009=\u2009500, true p\u2009=\u20090.50. "
            "Prior weight is negligible; CI and CrI widths and coverage converge.",
        ),
        "jeffreys": (
            "Jeffreys \u2248 Wilson",
            "Jeffreys prior Beta(\u00bd, \u00bd) + Wilson CI, n\u2009=\u2009100. "
            "The two intervals nearly coincide \u2014 not a coincidence, they are "
            "mathematically related through the arcsine transformation.",
        ),
    }

    for key, (_l, tp, nn, aa, bb, fm) in _BYS2_PRESETS.items():
        def _make_h(tp=tp, nn=nn, aa=aa, bb=bb, fm=fm, key=key):
            @reactive.effect
            @reactive.event(input[f"bys2_pre_{key}"])
            def _apply():
                ui.update_numeric("bys2_true_p",    value=tp)
                ui.update_numeric("bys2_n",         value=nn)
                ui.update_numeric("bys2_alpha",     value=aa)
                ui.update_numeric("bys2_beta",      value=bb)
                ui.update_select ("bys2_freq_method", selected=fm)
                _bys2_raw_k.set([])
                _bys2_history.set([])
                _bys2_preset.set(key)
        _make_h()

    @render.ui
    def bys2_preset_desc():
        key = _bys2_preset()
        if key is None or key not in _BYS2_PRESET_DESC:
            return ui.div(
                "\u2190 Pick a scenario preset to see what it demonstrates.",
                class_="np-preset-hint",
            )
        title, body = _BYS2_PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(title + ": "),
            body,
            class_="np-preset-hint np-preset-hint--active",
        )

    # ── Sampling ───────────────────────────────────────────────────────────
    def _bys2_params() -> tuple[float, int, float, float, str, float]:
        try:    tp = float(input.bys2_true_p() or 0.5)
        except Exception: tp = 0.5
        try:    n  = int  (input.bys2_n()      or 30)
        except Exception: n = 30
        try:    a  = float(input.bys2_alpha()  or 1.0)
        except Exception: a = 1.0
        try:    b  = float(input.bys2_beta()   or 1.0)
        except Exception: b = 1.0
        try:    conf = float(input.bys2_conf() or 0.95)
        except Exception: conf = 0.95
        tp = max(0.001, min(0.999, tp))
        return tp, max(2, n), max(0.01, a), max(0.01, b), \
               input.bys2_freq_method() or "wilson", conf

    def _sample_bys2(reps: int):
        tp, n, a, b, fm, conf = _bys2_params()
        new_ks = np.random.binomial(n, tp, reps).astype(int)

        # Freq CI (batched via ci_methods)
        f_lo, f_hi = compute_ci_proportion(new_ks, n, method=fm, level=conf)
        f_lo = np.atleast_1d(f_lo); f_hi = np.atleast_1d(f_hi)

        new_hist = []
        for i, k in enumerate(new_ks):
            k_int = int(k)
            a_post, b_post = _update_beta(a, b, k_int, n)
            b_lo, b_hi = _cred_interval(a_post, b_post, level=conf)
            new_hist.append((
                float(f_lo[i]), float(f_hi[i]),
                b_lo, b_hi,
                bool(f_lo[i] <= tp <= f_hi[i]),
                bool(b_lo    <= tp <= b_hi),
            ))
        _bys2_raw_k.set(list(_bys2_raw_k()) + [int(k) for k in new_ks])
        _bys2_history.set(list(_bys2_history()) + new_hist)

    @reactive.effect
    @reactive.event(input.bys2_sample_1)
    def _s2_1(): _sample_bys2(1)

    @reactive.effect
    @reactive.event(input.bys2_sample_50)
    def _s2_50(): _sample_bys2(50)

    @reactive.effect
    @reactive.event(input.bys2_sample_100)
    def _s2_100(): _sample_bys2(100)

    @reactive.effect
    @reactive.event(input.bys2_reset)
    def _s2_reset():
        _bys2_raw_k.set([])
        _bys2_history.set([])

    # Speed / Play
    _bys2_playing  = reactive.value(False)
    _bys2_speed_ms = reactive.value(0.3)

    @reactive.effect
    @reactive.event(input.bys2_speed_minus)
    def _bys2_sp_dn():
        _bys2_speed_ms.set(min(_bys2_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.bys2_speed_plus)
    def _bys2_sp_up():
        _bys2_speed_ms.set(max(_bys2_speed_ms() - 0.05, 0.05))

    @reactive.effect
    @reactive.event(input.bys2_btn_play)
    def _bys2_toggle():
        _bys2_playing.set(not _bys2_playing())
        ui.update_action_button(
            "bys2_btn_play",
            label="Pause" if _bys2_playing() else "Play",
        )

    @reactive.effect
    def _bys2_auto():
        if _bys2_playing():
            reactive.invalidate_later(_bys2_speed_ms())
            with reactive.isolate():
                _sample_bys2(1)

    # Structural changes reset state (confusing to keep old k's under new n/true_p).
    @reactive.effect
    @reactive.event(input.bys2_true_p, input.bys2_n, input.bys2_freq_method,
                    input.bys2_conf)
    def _s2_reset_on_change():
        _bys2_raw_k.set([])
        _bys2_history.set([])

    # "Recompute with same data" — full replacement, not partial append.
    @reactive.effect
    @reactive.event(input.bys2_recompute)
    def _bys2_recompute():
        tp, n, a, b, _fm, conf = _bys2_params()
        raws = list(_bys2_raw_k())
        old  = list(_bys2_history())
        if not raws:
            return
        new_hist = []
        for i, k in enumerate(raws):
            f_lo, f_hi, _, _, covf, _ = old[i]   # freq invariant to prior
            a_post, b_post = _update_beta(a, b, k, n)
            b_lo, b_hi = _cred_interval(a_post, b_post, level=conf)
            new_hist.append((
                f_lo, f_hi, b_lo, b_hi, covf,
                bool(b_lo <= tp <= b_hi),
            ))
        _bys2_history.set(new_hist)     # full replacement

    # ── Stats row ─────────────────────────────────────────────────────────
    @render.ui
    def bys2_stats_row():
        hist = list(_bys2_history())
        _, _, _, _, _, conf = _bys2_params()
        nominal = conf * 100.0

        if not hist:
            empty = "—"
            return ui.div(
                _stat_card("MEAN WIDTH (F)", tip("Average frequentist CI width."),
                           empty, "coverage"),
                _stat_card("MEAN WIDTH (B)", tip("Average Bayesian CrI width."),
                           empty, "total"),
                _stat_card(f"COVERAGE F",
                           tip("% of frequentist CIs containing true p; should ≈ nominal."),
                           empty, "included"),
                _stat_card(f"COVERAGE B",
                           tip("% of Bayesian CrIs containing true p — "
                               "depends on prior; NOT a calibration error."),
                           empty, "pw-d"),
                class_="stats-row bys2-stats-row",
            )

        fw = np.array([h[1] - h[0] for h in hist])
        bw = np.array([h[3] - h[2] for h in hist])
        covf = np.array([h[4] for h in hist], dtype=float)
        covb = np.array([h[5] for h in hist], dtype=float)
        n = covf.size
        cov_f_pct = covf.mean() * 100.0
        cov_b_pct = covb.mean() * 100.0
        se_pct = np.sqrt(conf * (1 - conf) / n) * 100.0

        # Frequentist marker: ⚠ if |cov−nominal| > 2·SE → real calibration error.
        f_off = abs(cov_f_pct - nominal) > 2 * se_pct
        b_off = abs(cov_b_pct - nominal) > 2 * se_pct

        f_value = ui.span(f"{cov_f_pct:.1f}%", (" ⚠" if f_off else ""),
                          title=("Coverage differs from nominal by more than ±2·SE "
                                 "— possible calibration issue." if f_off else ""))
        b_value = ui.span(f"{cov_b_pct:.1f}%", (" 🟡" if b_off else ""),
                          title=("Depends on prior — Bayesian intervals are not calibrated "
                                 "for frequentist coverage. This is expected behaviour "
                                 "when the prior is informative." if b_off else ""))

        return ui.div(
            _stat_card("MEAN WIDTH (F)", tip("Average frequentist CI width."),
                       f"{fw.mean():.3f}", "coverage"),
            _stat_card("MEAN WIDTH (B)", tip("Average Bayesian CrI width."),
                       f"{bw.mean():.3f}", "total"),
            _stat_card(f"COVERAGE F  (nominal {nominal:.0f}%)",
                       tip("% of frequentist CIs containing true p; should ≈ nominal."),
                       f_value, "included",
                       sub_html=f"n = {n} experiments"),
            _stat_card(f"COVERAGE B",
                       tip("% of Bayesian CrIs containing true p — depends on prior."),
                       b_value, "pw-d",
                       sub_html="🟡 = depends on prior (not an error)"
                                if b_off else "ok"),
            class_="stats-row bys2-stats-row",
        )

    # ── Charts ────────────────────────────────────────────────────────────
    @render.ui
    def bys2_forest():
        tp, *_ = _bys2_params()
        fig = draw_bys2_forest(list(_bys2_history()), tp, dark=is_dark())
        return _fig_html(fig)

    @render.ui
    def bys2_width_hist():
        fig = draw_bys2_width_hist(list(_bys2_history()), dark=is_dark())
        return _fig_html(fig)

    @render.ui
    def bys2_running_coverage():
        _, _, _, _, _, conf = _bys2_params()
        fig = draw_bys2_running_coverage(list(_bys2_history()),
                                         nominal=conf * 100.0, dark=is_dark())
        return _fig_html(fig)


    # ═══════════════════════════════════════════════════════════════════════
    # Tab 3 — Bayesian A/B Testing
    # ═══════════════════════════════════════════════════════════════════════

    _bys3_kA = reactive.value(0); _bys3_nA_obs = reactive.value(0)
    _bys3_kB = reactive.value(0); _bys3_nB_obs = reactive.value(0)
    _bys3_seed = reactive.value(0)          # updated on each sample
    _bys3_trajectory = reactive.value([])   # list of (n_per_variant, P(B>A))
    _bys3_preset = reactive.value(None)

    # Tab 3 presets: (label, pA, pB, n_per_variant)
    _BYS3_PRESETS = {
        "no_effect":   ("No effect",          0.10,  0.10,  2000),
        "small":       ("Small lift",         0.10,  0.11,  2000),
        "large":       ("Large lift",         0.10,  0.15,  2000),
        "early":       ("Early stop",         0.10,  0.15,   200),
        "tiny_lift":   ("High prob, tiny lift", 0.100, 0.101, 50000),
    }

    _BYS3_PRESET_DESC = {
        "no_effect": (
            "No effect",
            "p_A = p_B = 0.10, n\u2009=\u20092000 per batch. "
            "Ground truth is a tie \u2014 watch P(B\u2009>\u2009A) oscillate around 0.5. "
            "Stopping on a lucky 0.95 crossing would be a false win.",
        ),
        "small": (
            "Small lift (+10% rel.)",
            "p_A\u2009=\u20090.10 vs p_B\u2009=\u20090.11, n\u2009=\u20092000. "
            "Realistic e-commerce scenario. Takes many batches before P(B\u2009>\u2009A) "
            "and E[lift] stabilise.",
        ),
        "large": (
            "Large lift (+50% rel.)",
            "p_A\u2009=\u20090.10 vs p_B\u2009=\u20090.15, n\u2009=\u20092000. "
            "Strong signal \u2014 P(B\u2009>\u2009A) crosses the 0.95 threshold quickly and stays.",
        ),
        "early": (
            "Early stop",
            "Large lift with n\u2009=\u2009200 per batch \u2014 few trials per decision point. "
            "Shows how P(B\u2009>\u2009A) can briefly hit 0.95 on noise alone before more "
            "data settles the answer.",
        ),
        "tiny_lift": (
            "High prob, tiny lift",
            "p_A\u2009=\u20090.100 vs p_B\u2009=\u20090.101 with n\u2009=\u200950 000. "
            "P(B\u2009>\u2009A) climbs near 1 yet E[lift] \u2248 0.001 \u2014 "
            "statistical confidence without practical significance.",
        ),
    }

    for key, (_l, pa, pb, nn) in _BYS3_PRESETS.items():
        def _mk(pa=pa, pb=pb, nn=nn, key=key):
            @reactive.effect
            @reactive.event(input[f"bys3_pre_{key}"])
            def _apply():
                ui.update_numeric("bys3_true_pA", value=pa)
                ui.update_numeric("bys3_true_pB", value=pb)
                ui.update_numeric("bys3_nA",      value=nn)
                ui.update_numeric("bys3_nB",      value=nn)
                _bys3_reset_state()
                _bys3_preset.set(key)
        _mk()

    @render.ui
    def bys3_preset_desc():
        key = _bys3_preset()
        if key is None or key not in _BYS3_PRESET_DESC:
            return ui.div(
                "\u2190 Pick a preset to see what it demonstrates.",
                class_="np-preset-hint",
            )
        title, body = _BYS3_PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(title + ": "),
            body,
            class_="np-preset-hint np-preset-hint--active",
        )

    def _bys3_reset_state():
        _bys3_kA.set(0); _bys3_nA_obs.set(0)
        _bys3_kB.set(0); _bys3_nB_obs.set(0)
        _bys3_trajectory.set([])
        _bys3_seed.set(0)

    @reactive.effect
    @reactive.event(input.bys3_reset)
    def _bys3_rst(): _bys3_reset_state()

    # Structural resets
    @reactive.effect
    @reactive.event(input.bys3_true_pA, input.bys3_true_pB,
                    input.bys3_nA, input.bys3_nB,
                    input.bys3_alpha, input.bys3_beta)
    def _bys3_structural_reset():
        _bys3_reset_state()

    # ── Sampling a batch ──────────────────────────────────────────────────
    def _bys3_params():
        try:    pa = float(input.bys3_true_pA() or 0.10)
        except Exception: pa = 0.10
        try:    pb = float(input.bys3_true_pB() or 0.11)
        except Exception: pb = 0.11
        try:    nA = int  (input.bys3_nA() or 1000)
        except Exception: nA = 1000
        try:    nB = int  (input.bys3_nB() or 1000)
        except Exception: nB = 1000
        try:    a  = float(input.bys3_alpha() or 1.0)
        except Exception: a  = 1.0
        try:    b  = float(input.bys3_beta()  or 1.0)
        except Exception: b  = 1.0
        try:    draws = int(input.bys3_draws() or 20000)
        except Exception: draws = 20000
        try:    thr = float(input.bys3_threshold() or 0.95)
        except Exception: thr = 0.95
        return (max(0.001, min(0.999, pa)),
                max(0.001, min(0.999, pb)),
                max(10, nA), max(10, nB),
                max(0.01, a), max(0.01, b),
                max(100, draws), max(0.5, min(0.999, thr)))

    def _bys3_run_batches(n_batches: int):
        pa, pb, nA, nB, a, b, draws, _thr = _bys3_params()
        traj = list(_bys3_trajectory())
        kA   = int(_bys3_kA());    nA_obs = int(_bys3_nA_obs())
        kB   = int(_bys3_kB());    nB_obs = int(_bys3_nB_obs())
        seed = int(_bys3_seed())
        for _ in range(max(1, n_batches)):
            sA = int(np.random.binomial(nA, pa))
            sB = int(np.random.binomial(nB, pb))
            kA += sA;  nA_obs += nA
            kB += sB;  nB_obs += nB
            # Fresh seed per batch — stable UI until next batch
            seed = int(np.random.randint(0, 2**31 - 1))
            aA, bA = _update_beta(a, b, kA, nA_obs)
            aB, bB = _update_beta(a, b, kB, nB_obs)
            m = _ab_metrics(aA, bA, aB, bB, draws, seed)
            traj.append(((nA_obs + nB_obs) // 2, m["prob_b_beats_a"]))
        _bys3_kA.set(kA);  _bys3_nA_obs.set(nA_obs)
        _bys3_kB.set(kB);  _bys3_nB_obs.set(nB_obs)
        _bys3_seed.set(seed)
        _bys3_trajectory.set(traj)

    @reactive.effect
    @reactive.event(input.bys3_sample_1)
    def _bys3_s1(): _bys3_run_batches(1)

    @reactive.effect
    @reactive.event(input.bys3_sample_50)
    def _bys3_s50(): _bys3_run_batches(50)

    @reactive.effect
    @reactive.event(input.bys3_sample_100)
    def _bys3_s100(): _bys3_run_batches(100)

    # Speed / Play
    _bys3_playing  = reactive.value(False)
    _bys3_speed_ms = reactive.value(0.3)

    @reactive.effect
    @reactive.event(input.bys3_speed_minus)
    def _bys3_sp_dn():
        _bys3_speed_ms.set(min(_bys3_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.bys3_speed_plus)
    def _bys3_sp_up():
        _bys3_speed_ms.set(max(_bys3_speed_ms() - 0.05, 0.05))

    @reactive.effect
    @reactive.event(input.bys3_btn_play)
    def _bys3_toggle():
        _bys3_playing.set(not _bys3_playing())
        ui.update_action_button(
            "bys3_btn_play",
            label="Pause" if _bys3_playing() else "Play",
        )

    @reactive.effect
    def _bys3_auto():
        if _bys3_playing():
            reactive.invalidate_later(_bys3_speed_ms())
            with reactive.isolate():
                _bys3_run_batches(1)

    # ── Derived posterior params ──────────────────────────────────────────
    def _bys3_post():
        pa, pb, nA, nB, a, b, draws, thr = _bys3_params()
        aA, bA = _update_beta(a, b, int(_bys3_kA()), int(_bys3_nA_obs()))
        aB, bB = _update_beta(a, b, int(_bys3_kB()), int(_bys3_nB_obs()))
        return aA, bA, aB, bB, draws, thr

    # ── Stats row ─────────────────────────────────────────────────────────
    @render.ui
    def bys3_stats_row():
        kA = int(_bys3_kA()); nA = int(_bys3_nA_obs())
        kB = int(_bys3_kB()); nB = int(_bys3_nB_obs())
        aA, bA, aB, bB, draws, thr = _bys3_post()
        seed = int(_bys3_seed())

        if nA == 0 or nB == 0:
            m = dict(prob_b_beats_a=0.5, e_lift_abs=0.0, e_lift_rel=0.0,
                     e_loss_A=0.0, e_loss_B=0.0, meanA=0.0, meanB=0.0)
        else:
            m = _ab_metrics(aA, bA, aB, bB, draws, seed)

        # Decision rule (Phase 1: probability threshold)
        prob = m["prob_b_beats_a"]
        if prob >= thr:
            decision_txt, decision_cls = "B wins", "included"
        elif prob <= 1.0 - thr:
            decision_txt, decision_cls = "A wins", "missed"
        else:
            decision_txt, decision_cls = "inconclusive", "pw-alpha"

        rel_pct = m["e_lift_rel"] * 100.0
        lift_html = f"{m['e_lift_abs']:+.4f}  ({rel_pct:+.1f}%)"

        return ui.div(
            _stat_card("OBSERVED A", tip("k_A / n_A — successes over trials for variant A."),
                       f"{kA} / {nA}", "coverage"),
            _stat_card("OBSERVED B", tip("k_B / n_B for variant B."),
                       f"{kB} / {nB}", "total"),
            _stat_card("P(B > A)",
                       tip("Posterior probability that variant B outperforms A."),
                       f"{prob:.3f}", "pw-power"),
            _stat_card("E[lift]",
                       tip("Posterior expected lift = E[p_B − p_A]; "
                           "relative to mean p_A."),
                       lift_html, "pw-d"),
            _stat_card("E[loss | A]",
                       tip("Expected regret from choosing A if B is actually better."),
                       f"{m['e_loss_A']:.4f}", "pw-alpha"),
            _stat_card("E[loss | B]",
                       tip("Expected regret from choosing B if A is actually better."),
                       f"{m['e_loss_B']:.4f}", "pw-alpha"),
            _stat_card(f"DECISION (thr {thr:.2f})",
                       tip("Phase-1 rule: probability threshold. Loss-based rule is Phase 2; "
                           "see E[loss] columns above."),
                       decision_txt, decision_cls),
            class_="stats-row bys3-stats-row",
        )

    # ── Classical p-value comparison (side-by-side) ───────────────────────
    @render.ui
    def bys3_classical():
        kA = int(_bys3_kA()); nA = int(_bys3_nA_obs())
        kB = int(_bys3_kB()); nB = int(_bys3_nB_obs())
        if nA == 0 or nB == 0:
            return ui.div(
                ui.em("Sample to compare against the classical test."),
                class_="bys3-classical",
            )
        pA = kA / nA; pB = kB / nB
        p_pool = (kA + kB) / (nA + nB)
        var_z = p_pool * (1.0 - p_pool) * (1.0 / nA + 1.0 / nB)
        if var_z <= 0:
            pval = 1.0; z = 0.0
        else:
            z = (pB - pA) / np.sqrt(var_z)
            pval = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        aA, bA, aB, bB, draws, _thr = _bys3_post()
        seed = int(_bys3_seed())
        prob = _ab_metrics(aA, bA, aB, bB, draws, seed)["prob_b_beats_a"]

        return ui.div(
            ui.div(
                ui.tags.strong("Classical 2-sample z-test p-value: "),
                ui.span(f"{pval:.3f}",
                        class_="bys3-clsk" + (" bys3-sig" if pval < 0.05 else "")),
                class_="bys3-classical-row",
            ),
            ui.div(
                ui.tags.strong("Bayesian P(B > A): "),
                ui.span(f"{prob:.3f}", class_="bys3-clsk"),
                class_="bys3-classical-row",
            ),
            ui.div(
                ui.em("These answer different questions — not the same test."),
                class_="bys3-classical-caption",
            ),
            class_="bys3-classical",
        )

    # ── Charts ────────────────────────────────────────────────────────────
    @render.ui
    def bys3_posteriors():
        aA, bA, aB, bB, *_ = _bys3_post()
        fig = draw_bys3_posteriors(aA, bA, aB, bB, dark=is_dark())
        return _fig_html(fig)

    @render.ui
    def bys3_joint():
        aA, bA, aB, bB, draws, _thr = _bys3_post()
        seed = int(_bys3_seed())
        kA = int(_bys3_kA()); kB = int(_bys3_kB())
        if kA == 0 and kB == 0 and int(_bys3_nA_obs()) == 0:
            pA = np.array([]); pB = np.array([])
        else:
            pA, pB = _sample_posteriors(aA, bA, aB, bB, draws, seed)
        fig = draw_bys3_joint(pA, pB, dark=is_dark())
        return _fig_html(fig)

    @render.ui
    def bys3_prob_evolution():
        _, _, _, _, _, thr = _bys3_post()
        fig = draw_bys3_prob_evolution(list(_bys3_trajectory()),
                                       threshold=thr, dark=is_dark())
        return _fig_html(fig)
