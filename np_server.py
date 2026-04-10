# =============================================================================
# Nonparametric Explorer — server logic
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from shiny import reactive, render, ui

from utils import tip
from np_plots import (
    draw_np_sample_kde,
    draw_np_pvalue_hist,
    draw_np_reject_bars,
    draw_np_rank_plot,
)

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}

_DIST_CHOICES = {
    "normal":       "Normal",
    "lognormal":    "Log-normal (skewed)",
    "exponential":  "Exponential",
    "uniform":      "Uniform",
    "contaminated": "Contaminated Normal",
    "cauchy":       "Cauchy (heavy tails)",
}

_DIST_CHOICES_PAIRED = {
    "normal":       "Normal",
    "lognormal":    "Log-normal (skewed)",
    "contaminated": "Contaminated Normal",
    "cauchy":       "Cauchy (heavy tails)",
}


def _fig_to_ui(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)
    return ui.div(ui.HTML(html), class_="plotly-container")


def np_server(input, output, session, is_dark):

    MAX_DATA = 10_000

    # ── Reactive state ────────────────────────────────────────────────────
    np_total        = reactive.value(0)
    np_param_rej    = reactive.value(0)
    np_nonparam_rej = reactive.value(0)
    np_pvals_param:    reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    np_pvals_nonparam: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    np_last_a: reactive.Value[np.ndarray | None] = reactive.value(None)
    np_last_b: reactive.Value[np.ndarray | None] = reactive.value(None)
    np_prob_a_gt_b: reactive.Value[float | None]  = reactive.value(None)
    np_is_playing = reactive.value(False)
    np_speed_ms   = reactive.value(0.5)

    # Preset initial values consumed by the dynamic UI renderer
    _init_a = reactive.value({})
    _init_b = reactive.value({})
    _init_p = reactive.value({})

    # ── Safe input helpers ────────────────────────────────────────────────
    def _safe(input_id, default):
        try:
            v = getattr(input, input_id)()
            return type(default)(v) if v is not None else default
        except Exception:
            return default

    def _get_delta(): return _safe("np_delta", 0.0)
    def _get_n():     return max(int(_safe("np_n", 30)), 5)
    def _get_alpha(): return _safe("np_alpha", 0.05)

    def _get_dist(prefix):
        return _safe(f"{prefix}_dist", "normal")

    def _get_params(prefix):
        d = _get_dist(prefix)
        if d == "normal":
            return {"mu": _safe(f"{prefix}_mu", 0.0),
                    "sigma": max(_safe(f"{prefix}_sigma", 1.0), 0.1)}
        if d == "lognormal":
            return {"mu_log": _safe(f"{prefix}_mu_log", 0.0),
                    "sigma_log": max(_safe(f"{prefix}_sigma_log", 0.5), 0.1)}
        if d == "exponential":
            return {"rate": max(_safe(f"{prefix}_rate", 1.0), 0.1)}
        if d == "uniform":
            a = _safe(f"{prefix}_lo", 0.0)
            b = _safe(f"{prefix}_hi", 1.0)
            if a > b: a, b = b, a
            if abs(b - a) < 1e-9: b = a + 1.0
            return {"a": a, "b": b}
        if d == "contaminated":
            return {"mu": _safe(f"{prefix}_mu", 0.0),
                    "sigma": max(_safe(f"{prefix}_sigma", 1.0), 0.1),
                    "eps": max(0.01, min(0.5, _safe(f"{prefix}_eps", 0.1))),
                    "sigma_mult": max(1.5, _safe(f"{prefix}_sigma_mult", 5.0))}
        if d == "cauchy":
            return {"x0": _safe(f"{prefix}_x0", 0.0),
                    "gamma": max(_safe(f"{prefix}_gamma", 1.0), 0.1)}
        return {"mu": 0.0, "sigma": 1.0}

    def _read_params(prefix, dist_id):
        """Read current param input values (call inside reactive.isolate)."""
        if dist_id == "normal":
            return {"mu": _safe(f"{prefix}_mu", 0.0),
                    "sigma": _safe(f"{prefix}_sigma", 1.0)}
        if dist_id == "lognormal":
            return {"mu_log": _safe(f"{prefix}_mu_log", 0.0),
                    "sigma_log": _safe(f"{prefix}_sigma_log", 0.5)}
        if dist_id == "exponential":
            return {"rate": _safe(f"{prefix}_rate", 1.0)}
        if dist_id == "uniform":
            return {"a": _safe(f"{prefix}_lo", 0.0),
                    "b": _safe(f"{prefix}_hi", 1.0)}
        if dist_id == "contaminated":
            return {"mu": _safe(f"{prefix}_mu", 0.0),
                    "sigma": _safe(f"{prefix}_sigma", 1.0),
                    "eps": _safe(f"{prefix}_eps", 0.1),
                    "sigma_mult": _safe(f"{prefix}_sigma_mult", 5.0)}
        if dist_id == "cauchy":
            return {"x0": _safe(f"{prefix}_x0", 0.0),
                    "gamma": _safe(f"{prefix}_gamma", 1.0)}
        return {}

    # ── Sampling engine ───────────────────────────────────────────────────
    def _sample_dist(dist_id, params, n):
        if dist_id == "normal":
            return np.random.normal(params["mu"], params["sigma"], n)
        if dist_id == "lognormal":
            return np.random.lognormal(params["mu_log"], params["sigma_log"], n)
        if dist_id == "exponential":
            return np.random.exponential(1.0 / max(params["rate"], 1e-6), n)
        if dist_id == "uniform":
            return np.random.uniform(params["a"], params["b"], n)
        if dist_id == "contaminated":
            sigma_out = params["sigma"] * params["sigma_mult"]
            mask = np.random.random(n) < params["eps"]
            base = np.random.normal(params["mu"], params["sigma"], n)
            base[mask] = np.random.normal(params["mu"], sigma_out, int(mask.sum()))
            return base
        if dist_id == "cauchy":
            return np.random.standard_cauchy(n) * params["gamma"] + params["x0"]
        return np.random.normal(0, 1, n)

    # ── Dynamic distribution UI ───────────────────────────────────────────
    def _make_params(prefix, dist_id, init):
        iv = init or {}
        if dist_id == "normal":
            return ui.div(
                ui.div(ui.input_numeric(f"{prefix}_mu",
                    ui.TagList("\u03bc ", tip("Mean")),
                    value=iv.get("mu", 0.0), step=0.5, width="100%")),
                ui.div(ui.input_numeric(f"{prefix}_sigma",
                    ui.TagList("\u03c3 ", tip("Standard deviation")),
                    value=iv.get("sigma", 1.0), min=0.1, step=0.5, width="100%")),
                class_="group-params-cols",
            )
        if dist_id == "lognormal":
            return ui.div(
                ui.div(ui.input_numeric(f"{prefix}_mu_log",
                    ui.TagList("\u03bc\u2097\u2099 ", tip("Log-scale mean")),
                    value=iv.get("mu_log", 0.0), step=0.25, width="100%")),
                ui.div(ui.input_numeric(f"{prefix}_sigma_log",
                    ui.TagList("\u03c3\u2097\u2099 ", tip("Log-scale std dev")),
                    value=iv.get("sigma_log", 0.5), min=0.1, step=0.25, width="100%")),
                class_="group-params-cols",
            )
        if dist_id == "exponential":
            return ui.input_numeric(f"{prefix}_rate",
                ui.TagList("Rate (\u03bb) ", tip("Rate parameter; mean = 1/\u03bb")),
                value=iv.get("rate", 1.0), min=0.1, step=0.5, width="100%")
        if dist_id == "uniform":
            return ui.div(
                ui.div(ui.input_numeric(f"{prefix}_lo",
                    ui.TagList("a (min) ", tip("Lower bound")),
                    value=iv.get("a", 0.0), step=0.5, width="100%")),
                ui.div(ui.input_numeric(f"{prefix}_hi",
                    ui.TagList("b (max) ", tip("Upper bound")),
                    value=iv.get("b", 1.0), step=0.5, width="100%")),
                class_="group-params-cols",
            )
        if dist_id == "contaminated":
            return ui.div(
                ui.div(
                    ui.div(ui.input_numeric(f"{prefix}_mu",
                        ui.TagList("\u03bc ", tip("Mean of base normal")),
                        value=iv.get("mu", 0.0), step=0.5, width="100%")),
                    ui.div(ui.input_numeric(f"{prefix}_sigma",
                        ui.TagList("\u03c3 ", tip("Std dev of base")),
                        value=iv.get("sigma", 1.0), min=0.1, step=0.5, width="100%")),
                    class_="group-params-cols",
                ),
                ui.div(
                    ui.div(ui.input_numeric(f"{prefix}_eps",
                        ui.TagList("\u03b5 ", tip("Contamination fraction (0\u20130.5)")),
                        value=iv.get("eps", 0.1), min=0.01, max=0.5, step=0.05, width="100%")),
                    ui.div(ui.input_numeric(f"{prefix}_sigma_mult",
                        ui.TagList("\u03c3\u00d7 ", tip("Outlier \u03c3 = base \u03c3 \u00d7 this")),
                        value=iv.get("sigma_mult", 5.0), min=1.5, step=0.5, width="100%")),
                    class_="group-params-cols",
                ),
            )
        if dist_id == "cauchy":
            return ui.div(
                ui.div(ui.input_numeric(f"{prefix}_x0",
                    ui.TagList("x\u2080 ", tip("Location")),
                    value=iv.get("x0", 0.0), step=0.5, width="100%")),
                ui.div(ui.input_numeric(f"{prefix}_gamma",
                    ui.TagList("\u03b3 ", tip("Scale")),
                    value=iv.get("gamma", 1.0), min=0.1, step=0.5, width="100%")),
                class_="group-params-cols",
            )
        return ui.div()

    @render.ui
    def np_dist_section():
        mode = input.np_mode()

        # Read dist selectors (reactive dependencies — trigger re-render)
        try:    cur_a = input.np_a_dist()
        except: cur_a = "normal"
        try:    cur_b = input.np_b_dist()
        except: cur_b = "normal"
        try:    cur_p = input.np_p_dist()
        except: cur_p = "normal"

        # Consume preset values + preserve existing params (isolated)
        with reactive.isolate():
            ia = dict(_init_a()); _init_a.set({})
            ib = dict(_init_b()); _init_b.set({})
            ip = dict(_init_p()); _init_p.set({})

        dist_a = ia.pop("dist", cur_a)
        dist_b = ib.pop("dist", cur_b)
        dist_p = ip.pop("dist", cur_p)

        with reactive.isolate():
            if not ia: ia = _read_params("np_a", dist_a)
            if not ib: ib = _read_params("np_b", dist_b)
            if not ip: ip = _read_params("np_p", dist_p)

        if mode == "independent":
            return ui.div(
                ui.div(
                    ui.div(
                        ui.tags.span("Group A", class_="group-col-label"),
                        ui.input_select("np_a_dist", "Distribution",
                                        choices=_DIST_CHOICES, selected=dist_a, width="100%"),
                    ),
                    ui.div(
                        ui.tags.span("Group B", class_="group-col-label"),
                        ui.input_select("np_b_dist", "Distribution",
                                        choices=_DIST_CHOICES, selected=dist_b, width="100%"),
                    ),
                    class_="group-params-cols",
                ),
                ui.tags.span("A parameters:",
                             style="font-size:0.68rem; color:var(--c-muted); display:block; margin:4px 0 2px;"),
                _make_params("np_a", dist_a, ia),
                ui.tags.span("B parameters:",
                             style="font-size:0.68rem; color:var(--c-muted); display:block; margin:4px 0 2px;"),
                _make_params("np_b", dist_b, ib),
                class_="group-params-block",
            )

        # Paired
        return ui.div(
            ui.input_select("np_p_dist",
                ui.TagList("Differences distribution",
                           tip("Base distribution for paired differences before adding \u03b4. "
                               "Wilcoxon signed-rank assumes symmetry under H\u2080.")),
                choices=_DIST_CHOICES_PAIRED, selected=dist_p, width="100%"),
            _make_params("np_p", dist_p, ip),
            class_="group-params-block",
        )

    # ── Presets ───────────────────────────────────────────────────────────
    _active_preset = reactive.value(None)

    _PRESET_DESC = {
        "normal_h0": (
            "Normal H\u2080",
            "Both groups N(0,1), \u03b4\u200a=\u200a0. "
            "Simulates Type\u00a0I error under ideal conditions. "
            "Expect both tests to reject \u2248\u00a0\u03b1 of the time.",
        ),
        "normal_h1": (
            "Normal H\u2081",
            "Both groups N(0,1), \u03b4\u200a=\u200a0.5. "
            "Power comparison under normality. "
            "t-test is optimal here; MW-U is slightly less powerful.",
        ),
        "outlier": (
            "Outliers",
            "A\u200a~\u200aN(0,1), B\u200a~\u200aContaminated N (10\u202f% outliers at 5\u03c3), \u03b4\u200a=\u200a0.5. "
            "Outliers inflate t-test variance and reduce its power. "
            "MW-U is more robust and often detects the effect better.",
        ),
        "skewed": (
            "Skewed",
            "Both groups Exp(1), \u03b4\u200a=\u200a0.5. "
            "Skewed, right-tailed distributions. "
            "MW-U gains efficiency over t-test under non-normality.",
        ),
        "myth": (
            "Shape myth",
            "A\u200a~\u200aExp(1), B\u200a~\u200aUniform(0,\u00a01.386), \u03b4\u200a=\u200a0. "
            "Both have median \u2248\u00a0ln(2)\u00a0\u2248\u00a00.693, yet P(A\u200a>\u200aB)\u200a\u2248\u200a0.54. "
            "MW-U rejects H\u2080 because shapes differ \u2014 it tests "
            "\u2018stochastic dominance\u2019, not medians.",
        ),
        "cauchy": (
            "Cauchy",
            "Both groups Cauchy(0,1), \u03b4\u200a=\u200a0. "
            "Undefined mean and variance \u2014 CLT fails. "
            "t-test loses Type\u00a0I control; MW-U maintains it.",
        ),
    }

    def _apply_preset(mode, ia, ib, ip, delta, n):
        _init_a.set(ia)
        _init_b.set(ib)
        _init_p.set(ip)
        ui.update_select("np_mode", selected=mode)
        ui.update_numeric("np_delta", value=delta)
        ui.update_numeric("np_n", value=n)

    @reactive.effect
    @reactive.event(input.np_preset_normal_h0)
    def _pr_h0():
        _active_preset.set("normal_h0")
        _apply_preset("independent",
                       {"dist": "normal", "mu": 0.0, "sigma": 1.0},
                       {"dist": "normal", "mu": 0.0, "sigma": 1.0},
                       {}, delta=0.0, n=30)

    @reactive.effect
    @reactive.event(input.np_preset_normal_h1)
    def _pr_h1():
        _active_preset.set("normal_h1")
        _apply_preset("independent",
                       {"dist": "normal", "mu": 0.0, "sigma": 1.0},
                       {"dist": "normal", "mu": 0.0, "sigma": 1.0},
                       {}, delta=0.5, n=30)

    @reactive.effect
    @reactive.event(input.np_preset_outlier)
    def _pr_out():
        _active_preset.set("outlier")
        _apply_preset("independent",
                       {"dist": "normal", "mu": 0.0, "sigma": 1.0},
                       {"dist": "contaminated", "mu": 0.0, "sigma": 1.0,
                        "eps": 0.1, "sigma_mult": 5.0},
                       {}, delta=0.5, n=30)

    @reactive.effect
    @reactive.event(input.np_preset_skewed)
    def _pr_skew():
        _active_preset.set("skewed")
        _apply_preset("independent",
                       {"dist": "exponential", "rate": 1.0},
                       {"dist": "exponential", "rate": 1.0},
                       {}, delta=0.5, n=30)

    @reactive.effect
    @reactive.event(input.np_preset_myth)
    def _pr_myth():
        _active_preset.set("myth")
        _apply_preset("independent",
                       {"dist": "exponential", "rate": 1.0},
                       {"dist": "uniform", "a": 0.0, "b": 1.386},
                       {}, delta=0.0, n=50)

    @reactive.effect
    @reactive.event(input.np_preset_cauchy)
    def _pr_cauchy():
        _active_preset.set("cauchy")
        _apply_preset("independent",
                       {"dist": "cauchy", "x0": 0.0, "gamma": 1.0},
                       {"dist": "cauchy", "x0": 0.0, "gamma": 1.0},
                       {}, delta=0.0, n=30)

    @render.ui
    def np_preset_desc():
        key = _active_preset()
        if key is None:
            return ui.div(
                "\u2190 Select a preset to see what it demonstrates.",
                class_="np-preset-hint",
            )
        title, body = _PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(title + ": "),
            body,
            class_="np-preset-hint np-preset-hint--active",
        )

    # ── Speed ± ──────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.np_speed_minus)
    def _np_spd_dn():
        np_speed_ms.set(min(np_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.np_speed_plus)
    def _np_spd_up():
        np_speed_ms.set(max(np_speed_ms() - 0.05, 0.05))

    # ── Play / Pause ─────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.np_btn_play)
    def _np_toggle():
        np_is_playing.set(not np_is_playing())
        ui.update_action_button("np_btn_play",
                                label="Pause" if np_is_playing() else "Play")

    @reactive.effect
    def _np_auto():
        if np_is_playing():
            reactive.invalidate_later(np_speed_ms())
            with reactive.isolate():
                _draw_samples(1)

    # ── Manual buttons ───────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.np_btn_sample_1)
    def _s1():  _draw_samples(1)

    @reactive.effect
    @reactive.event(input.np_btn_sample_50)
    def _s50(): _draw_samples(50)

    @reactive.effect
    @reactive.event(input.np_btn_sample_100)
    def _s100(): _draw_samples(100)

    # ── Reset ────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.np_btn_reset, input.np_mode,
                    input.np_preset_normal_h0, input.np_preset_normal_h1,
                    input.np_preset_outlier, input.np_preset_skewed,
                    input.np_preset_myth, input.np_preset_cauchy)
    def _np_reset():
        np_total.set(0)
        np_param_rej.set(0)
        np_nonparam_rej.set(0)
        np_pvals_param.set(deque(maxlen=MAX_DATA))
        np_pvals_nonparam.set(deque(maxlen=MAX_DATA))
        np_last_a.set(None)
        np_last_b.set(None)
        np_prob_a_gt_b.set(None)
        np_is_playing.set(False)
        ui.update_action_button("np_btn_play", label="Play")

    # ── Core sampling ────────────────────────────────────────────────────
    def _draw_samples(k: int):
        mode  = input.np_mode()
        alpha = _get_alpha()
        n     = _get_n()

        pvals_t  = []
        pvals_np = []
        last_a: np.ndarray | None = None
        last_b: np.ndarray | None = None

        if mode == "independent":
            da, pa = _get_dist("np_a"), _get_params("np_a")
            db, pb = _get_dist("np_b"), _get_params("np_b")
            delta  = _get_delta()

            for _ in range(k):
                a = _sample_dist(da, pa, n)
                b = _sample_dist(db, pb, n) + delta

                _, tp = stats.ttest_ind(a, b, equal_var=False)
                try:
                    _, mp = stats.mannwhitneyu(a, b, alternative="two-sided")
                except ValueError:
                    mp = 1.0

                pvals_t.append(float(tp))
                pvals_np.append(float(mp))
                last_a, last_b = a, b

            # P(A > B) from last sample
            if last_a is not None and last_b is not None:
                pairs = np.subtract.outer(last_a, last_b)
                prob = float(np.mean(pairs > 0) + 0.5 * np.mean(pairs == 0))
                np_prob_a_gt_b.set(prob)

        else:  # paired
            dp, pp = _get_dist("np_p"), _get_params("np_p")
            delta  = _get_delta()

            for _ in range(k):
                diffs = _sample_dist(dp, pp, n) + delta

                _, tp = stats.ttest_1samp(diffs, 0)
                try:
                    _, wp = stats.wilcoxon(diffs, alternative="two-sided")
                except ValueError:
                    wp = 1.0

                pvals_t.append(float(tp))
                pvals_np.append(float(wp))
                last_a = diffs

            last_b = None
            np_prob_a_gt_b.set(None)

        # Update state
        new_t  = sum(p < alpha for p in pvals_t)
        new_np = sum(p < alpha for p in pvals_np)

        np_total.set(np_total() + k)
        np_param_rej.set(np_param_rej() + new_t)
        np_nonparam_rej.set(np_nonparam_rej() + new_np)

        pv_t = deque(np_pvals_param(), maxlen=MAX_DATA)
        pv_t.extend(pvals_t)
        np_pvals_param.set(pv_t)

        pv_np = deque(np_pvals_nonparam(), maxlen=MAX_DATA)
        pv_np.extend(pvals_np)
        np_pvals_nonparam.set(pv_np)

        np_last_a.set(last_a)
        np_last_b.set(last_b)

    # ── Stat-card labels ─────────────────────────────────────────────────
    @render.text
    def np_param_stat_label():
        return ("T-TEST REJECT RATE" if input.np_mode() == "independent"
                else "PAIRED T REJECT RATE")

    @render.text
    def np_nonparam_stat_label():
        return ("MANN-WHITNEY U REJECT" if input.np_mode() == "independent"
                else "WILCOXON REJECT RATE")

    # ── Stat-card values ─────────────────────────────────────────────────
    @render.text
    def np_param_rate():
        t = np_total()
        return "\u2014" if t == 0 else f"{100 * np_param_rej() / t:.1f}%"

    @render.text
    def np_nonparam_rate():
        t = np_total()
        return "\u2014" if t == 0 else f"{100 * np_nonparam_rej() / t:.1f}%"

    @render.text
    def np_total_tests():
        return f"{np_total():,}"

    @render.text
    def np_advantage():
        t = np_total()
        if t == 0:
            return "\u2014"
        diff = (np_nonparam_rej() - np_param_rej()) / t * 100
        sign = "+" if diff > 0 else ""
        return f"{sign}{diff:.1f}pp"

    # ── Chart renderers ──────────────────────────────────────────────────
    @render.ui
    def np_sample_plot():
        mode = input.np_mode()
        a = np_last_a()
        b = np_last_b() if mode == "independent" else None
        prob = np_prob_a_gt_b() if mode == "independent" else None
        fig = draw_np_sample_kde(a, b, prob_a_gt_b=prob, dark=is_dark())
        return _fig_to_ui(fig)

    @render.ui
    def np_pvalue_plot():
        mode = input.np_mode()
        p_lab  = "Welch t" if mode == "independent" else "Paired t"
        np_lab = "Mann-Whitney U" if mode == "independent" else "Wilcoxon"
        fig = draw_np_pvalue_hist(
            list(np_pvals_param()), list(np_pvals_nonparam()),
            alpha=_get_alpha(), param_label=p_lab, nonparam_label=np_lab,
            dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def np_reject_plot():
        mode = input.np_mode()
        p_lab  = "Welch t" if mode == "independent" else "Paired t"
        np_lab = "MW-U" if mode == "independent" else "Wilcoxon"
        fig = draw_np_reject_bars(
            np_total(), np_param_rej(), np_nonparam_rej(),
            alpha=_get_alpha(), param_label=p_lab, nonparam_label=np_lab,
            dark=is_dark(),
        )
        return _fig_to_ui(fig)

    @render.ui
    def np_rank_plot():
        fig = draw_np_rank_plot(np_last_a(), np_last_b(), dark=is_dark())
        return _fig_to_ui(fig)
