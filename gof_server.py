# =============================================================================
# GoF Explorer — server logic
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from shiny import reactive, render, ui

from utils import tip
from gof_plots import (
    draw_ks1_ecdf, draw_ks2_ecdf, draw_chi2_bars,
    draw_qq_plot, draw_gof_null_dist, draw_gof_pvalue_hist,
)

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
MAX_DATA = 10_000


def _fig_to_ui(fig):
    html = fig.to_html(full_html=False, include_plotlyjs=False, config=_PLOTLY_CONFIG)
    return ui.div(ui.HTML(html), class_="plotly-container")


# ── Distribution helpers ─────────────────────────────────────────────────────

_DIST_CHOICES = {
    "normal":      "Normal",
    "uniform":     "Uniform",
    "exponential": "Exponential",
    "lognormal":   "Log-normal",
    "t":           "Student\u2019s t",
    "chi2":        "\u03c7\u00b2",
    "beta":        "Beta",
}


def _dist_param_ui(prefix: str, dist: str):
    """Return parameter inputs for a distribution with the given prefix."""
    if dist == "normal":
        return ui.div(
            ui.div(ui.input_numeric(f"{prefix}_mu", ui.TagList("\u03bc", tip("Mean")),
                                    value=0.0, step=0.5, width="100%")),
            ui.div(ui.input_numeric(f"{prefix}_sigma", ui.TagList("\u03c3", tip("Standard deviation")),
                                    value=1.0, min=0.1, step=0.5, width="100%")),
            class_="group-params-cols",
        )
    elif dist == "uniform":
        return ui.div(
            ui.div(ui.input_numeric(f"{prefix}_a", ui.TagList("a", tip("Lower bound")),
                                    value=0.0, step=0.5, width="100%")),
            ui.div(ui.input_numeric(f"{prefix}_b", ui.TagList("b", tip("Upper bound")),
                                    value=1.0, step=0.5, width="100%")),
            class_="group-params-cols",
        )
    elif dist == "exponential":
        return ui.div(
            ui.input_numeric(f"{prefix}_lam", ui.TagList("\u03bb", tip("Rate parameter")),
                             value=1.0, min=0.1, step=0.5, width="100%"),
        )
    elif dist == "lognormal":
        return ui.div(
            ui.div(ui.input_numeric(f"{prefix}_mu", ui.TagList("\u03bc\u2097\u2099", tip("Log-mean")),
                                    value=0.0, step=0.25, width="100%")),
            ui.div(ui.input_numeric(f"{prefix}_sigma", ui.TagList("\u03c3\u2097\u2099", tip("Log-std")),
                                    value=0.5, min=0.1, step=0.25, width="100%")),
            class_="group-params-cols",
        )
    elif dist == "t":
        return ui.div(
            ui.input_numeric(f"{prefix}_df", ui.TagList("df", tip("Degrees of freedom")),
                             value=5, min=1, max=200, step=1, width="100%"),
        )
    elif dist == "chi2":
        return ui.div(
            ui.input_numeric(f"{prefix}_df", ui.TagList("df", tip("Degrees of freedom")),
                             value=5, min=1, max=200, step=1, width="100%"),
        )
    elif dist == "beta":
        return ui.div(
            ui.div(ui.input_numeric(f"{prefix}_a", ui.TagList("\u03b1", tip("Shape parameter \u03b1")),
                                    value=2.0, min=0.1, step=0.5, width="100%")),
            ui.div(ui.input_numeric(f"{prefix}_b", ui.TagList("\u03b2", tip("Shape parameter \u03b2")),
                                    value=5.0, min=0.1, step=0.5, width="100%")),
            class_="group-params-cols",
        )
    return ui.div()


def _read_frozen(prefix: str, dist: str, input):
    """Build a scipy frozen distribution from current input values."""
    def _v(name, default):
        try:
            v = getattr(input, f"{prefix}_{name}")()
            return float(v) if v is not None else default
        except Exception:
            return default

    if dist == "normal":
        return stats.norm(loc=_v("mu", 0), scale=max(_v("sigma", 1), 0.01))
    elif dist == "uniform":
        a, b = _v("a", 0), _v("b", 1)
        if a > b:
            a, b = b, a
        return stats.uniform(loc=a, scale=max(b - a, 0.01))
    elif dist == "exponential":
        return stats.expon(scale=1.0 / max(_v("lam", 1), 0.01))
    elif dist == "lognormal":
        mu, sg = _v("mu", 0), max(_v("sigma", 0.5), 0.01)
        return stats.lognorm(s=sg, scale=np.exp(mu))
    elif dist == "t":
        return stats.t(df=max(_v("df", 5), 1))
    elif dist == "chi2":
        return stats.chi2(df=max(_v("df", 5), 1))
    elif dist == "beta":
        return stats.beta(a=max(_v("a", 2), 0.01), b=max(_v("b", 5), 0.01))
    return stats.norm()


def _dist_label(dist: str) -> str:
    return _DIST_CHOICES.get(dist, dist)


def _sample_from(prefix: str, dist: str, n: int, input) -> np.ndarray:
    """Draw n iid observations from the distribution."""
    frozen = _read_frozen(prefix, dist, input)
    return frozen.rvs(size=n)


# ── Server function ──────────────────────────────────────────────────────────

def gof_server(input, output, session, is_dark):

    # ── Reactive state ───────────────────────────────────────────────────────
    gof_is_playing   = reactive.value(False)
    gof_speed_ms     = reactive.value(0.5)
    gof_pvalues      = reactive.value(deque(maxlen=MAX_DATA))
    gof_last_stat    = reactive.value(None)
    gof_last_pval    = reactive.value(None)
    gof_total        = reactive.value(0)
    gof_rejected     = reactive.value(0)
    # Per-sample artefacts for the main chart
    gof_last_sample  = reactive.value(None)   # sample 1 or only sample
    gof_last_sample2 = reactive.value(None)   # sample 2 (KS2)
    gof_last_d_loc   = reactive.value(None)   # x where D is max (KS)
    # Chi-squared specific
    gof_last_obs     = reactive.value(None)
    gof_last_exp     = reactive.value(None)
    gof_last_edges   = reactive.value(None)
    gof_last_chi2_df = reactive.value(1)
    # Shapiro–Wilk bootstrap null
    gof_sw_null      = reactive.value(np.array([]))

    # ── Safe input readers ───────────────────────────────────────────────────
    def _n():
        try:
            v = input.gof_n()
            return max(int(v), 3) if v is not None else 30
        except Exception:
            return 30

    def _n2():
        try:
            v = input.gof_n2()
            return max(int(v), 3) if v is not None else 30
        except Exception:
            return 30

    def _k_bins():
        try:
            v = input.gof_k()
            return max(int(v), 2) if v is not None else None
        except Exception:
            return None

    def _data_dist():
        try:
            return input.gof_data_dist()
        except Exception:
            return "normal"

    def _h0_dist():
        try:
            return input.gof_h0_dist()
        except Exception:
            return "normal"

    def _dist2():
        try:
            return input.gof_dist2()
        except Exception:
            return "normal"

    def _alternative():
        try:
            return input.gof_alternative()
        except Exception:
            return "two-sided"

    def _binning():
        try:
            return input.gof_binning()
        except Exception:
            return "equal_width"

    # ── Dynamic parameter block ──────────────────────────────────────────────
    @render.ui
    def gof_params_block():
        test = input.gof_test()

        if test == "ks1":
            return ui.div(
                ui.tags.strong("Data distribution"),
                ui.input_select("gof_data_dist", None, choices=_DIST_CHOICES,
                                selected="normal", width="100%"),
                ui.output_ui("gof_data_params"),
                ui.hr(style="border-color: var(--c-border); margin: 6px 0;"),
                ui.tags.strong("H\u2080 distribution"),
                ui.input_select("gof_h0_dist", None, choices=_DIST_CHOICES,
                                selected="normal", width="100%"),
                ui.output_ui("gof_h0_params"),
                ui.hr(style="border-color: var(--c-border); margin: 6px 0;"),
                ui.div(
                    ui.div(ui.input_numeric("gof_n", ui.TagList("n", tip("Sample size")),
                                            value=30, min=3, max=5000, step=1, width="100%")),
                    ui.div(ui.input_select(
                        "gof_alternative",
                        ui.TagList("Alternative", tip("Direction of the KS test.")),
                        choices={"two-sided": "Two-sided", "greater": "Greater", "less": "Less"},
                        selected="two-sided", width="100%",
                    )),
                    class_="group-params-cols",
                ),
            )

        elif test == "ks2":
            return ui.div(
                ui.tags.strong("Sample 1"),
                ui.input_select("gof_data_dist", None, choices=_DIST_CHOICES,
                                selected="normal", width="100%"),
                ui.output_ui("gof_data_params"),
                ui.input_numeric("gof_n", ui.TagList("n\u2081", tip("Sample 1 size")),
                                 value=30, min=3, max=5000, step=1, width="100%"),
                ui.hr(style="border-color: var(--c-border); margin: 6px 0;"),
                ui.tags.strong("Sample 2"),
                ui.input_select("gof_dist2", None, choices=_DIST_CHOICES,
                                selected="normal", width="100%"),
                ui.output_ui("gof_dist2_params"),
                ui.input_numeric("gof_n2", ui.TagList("n\u2082", tip("Sample 2 size")),
                                 value=30, min=3, max=5000, step=1, width="100%"),
            )

        elif test == "chi2":
            return ui.div(
                ui.tags.strong("Data distribution"),
                ui.input_select("gof_data_dist", None, choices=_DIST_CHOICES,
                                selected="normal", width="100%"),
                ui.output_ui("gof_data_params"),
                ui.hr(style="border-color: var(--c-border); margin: 6px 0;"),
                ui.tags.strong("H\u2080 distribution"),
                ui.input_select("gof_h0_dist", None, choices=_DIST_CHOICES,
                                selected="normal", width="100%"),
                ui.output_ui("gof_h0_params"),
                ui.hr(style="border-color: var(--c-border); margin: 6px 0;"),
                ui.div(
                    ui.div(ui.input_numeric("gof_n", ui.TagList("n", tip("Sample size")),
                                            value=50, min=5, max=5000, step=1, width="100%")),
                    ui.div(ui.input_numeric("gof_k", ui.TagList("k (bins)", tip("Number of bins. Default = \u2308\u221an\u2309.")),
                                            value=None, min=2, max=50, step=1, width="100%")),
                    class_="group-params-cols",
                ),
                ui.input_select(
                    "gof_binning",
                    ui.TagList("Binning", tip("Equal-width: equal-sized intervals. "
                                              "Equal-probability: each bin has equal expected probability.")),
                    choices={"equal_width": "Equal-width", "equal_prob": "Equal-probability"},
                    selected="equal_width", width="100%",
                ),
            )

        elif test == "sw":
            return ui.div(
                ui.tags.strong("Data distribution"),
                ui.input_select("gof_data_dist", None, choices=_DIST_CHOICES,
                                selected="normal", width="100%"),
                ui.output_ui("gof_data_params"),
                ui.hr(style="border-color: var(--c-border); margin: 6px 0;"),
                ui.input_numeric("gof_n", ui.TagList("n", tip("Sample size (3\u20135\u202f000)")),
                                 value=30, min=3, max=5000, step=1, width="100%"),
                ui.div(
                    ui.tags.small("H\u2080: data are normally distributed"),
                    style="color: var(--c-text2); margin-top: 4px;",
                ),
            )

        return ui.div()

    # ── Sub-parameter blocks ─────────────────────────────────────────────────
    @render.ui
    def gof_data_params():
        return _dist_param_ui("gof_d", _data_dist())

    @render.ui
    def gof_h0_params():
        return _dist_param_ui("gof_h", _h0_dist())

    @render.ui
    def gof_dist2_params():
        return _dist_param_ui("gof_d2", _dist2())

    # ── Reset ────────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.gof_btn_reset, input.gof_test)
    def _reset():
        _do_reset()

    def _do_reset():
        gof_pvalues.set(deque(maxlen=MAX_DATA))
        gof_last_stat.set(None)
        gof_last_pval.set(None)
        gof_total.set(0)
        gof_rejected.set(0)
        gof_last_sample.set(None)
        gof_last_sample2.set(None)
        gof_last_d_loc.set(None)
        gof_last_obs.set(None)
        gof_last_exp.set(None)
        gof_last_edges.set(None)
        gof_last_chi2_df.set(1)
        gof_sw_null.set(np.array([]))
        gof_is_playing.set(False)
        ui.update_action_button("gof_btn_play", label="Play")

    # ── Play / Pause / Speed ─────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.gof_btn_play)
    def _toggle():
        gof_is_playing.set(not gof_is_playing())
        ui.update_action_button(
            "gof_btn_play",
            label="Pause" if gof_is_playing() else "Play",
        )

    @reactive.effect
    @reactive.event(input.gof_speed_minus)
    def _speed_down():
        gof_speed_ms.set(min(gof_speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.gof_speed_plus)
    def _speed_up():
        gof_speed_ms.set(max(gof_speed_ms() - 0.05, 0.05))

    @reactive.effect
    def _auto_draw():
        if gof_is_playing():
            reactive.invalidate_later(gof_speed_ms())
            with reactive.isolate():
                _run_tests(1)

    # ── Manual buttons ───────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.gof_btn_sample_1)
    def _s1():
        _run_tests(1)

    @reactive.effect
    @reactive.event(input.gof_btn_sample_50)
    def _s50():
        _run_tests(50)

    @reactive.effect
    @reactive.event(input.gof_btn_sample_100)
    def _s100():
        _run_tests(100)

    # ── Core test logic ──────────────────────────────────────────────────────
    def _run_tests(k: int):
        test = input.gof_test()
        alpha = input.gof_alpha()
        n = _n()

        pvs = deque(gof_pvalues(), maxlen=MAX_DATA)
        total = gof_total()
        rejected = gof_rejected()

        last_s = last_p = last_dl = None
        last_s2 = None

        for _ in range(k):
            if test == "ks1":
                s = _sample_from("gof_d", _data_dist(), n, input)
                h0_frozen = _read_frozen("gof_h", _h0_dist(), input)
                alt = _alternative()
                result = stats.kstest(s, h0_frozen.cdf, alternative=alt)
                stat_val, pv = result.statistic, result.pvalue

                # Find D location
                xs_sorted = np.sort(s)
                ecdf_vals = np.arange(1, n + 1) / n
                ecdf_minus = np.concatenate([[0.0], ecdf_vals[:-1]])
                cdf_vals = h0_frozen.cdf(xs_sorted)
                if alt == "greater":
                    diffs = ecdf_vals - cdf_vals
                elif alt == "less":
                    diffs = cdf_vals - ecdf_minus
                else:
                    diffs = np.maximum(ecdf_vals - cdf_vals, cdf_vals - ecdf_minus)
                d_idx = int(np.argmax(diffs))
                d_loc = float(xs_sorted[d_idx])

                last_s, last_dl = s, d_loc

            elif test == "ks2":
                s1 = _sample_from("gof_d", _data_dist(), n, input)
                s2 = _sample_from("gof_d2", _dist2(), _n2(), input)
                result = stats.ks_2samp(s1, s2)
                stat_val, pv = result.statistic, result.pvalue

                # Find D location
                combined = np.sort(np.concatenate([s1, s2]))
                n1, n2_ = len(s1), len(s2)
                e1 = np.searchsorted(np.sort(s1), combined, side="right") / n1
                e2 = np.searchsorted(np.sort(s2), combined, side="right") / n2_
                diffs = np.abs(e1 - e2)
                d_idx = int(np.argmax(diffs))
                d_loc = float(combined[d_idx])

                last_s, last_s2, last_dl = s1, s2, d_loc

            elif test == "chi2":
                s = _sample_from("gof_d", _data_dist(), n, input)
                h0_frozen = _read_frozen("gof_h", _h0_dist(), input)
                k_bins = _k_bins()
                if k_bins is None:
                    k_bins = int(np.ceil(np.sqrt(n)))
                k_bins = max(2, min(k_bins, n // 2))

                binning = _binning()
                if binning == "equal_prob":
                    probs = np.linspace(0, 1, k_bins + 1)
                    edges = h0_frozen.ppf(probs)
                    edges[0] = -np.inf
                    edges[-1] = np.inf
                else:
                    lo = min(float(s.min()), float(h0_frozen.ppf(0.001)))
                    hi = max(float(s.max()), float(h0_frozen.ppf(0.999)))
                    edges = np.linspace(lo, hi, k_bins + 1)
                    edges[0] = -np.inf
                    edges[-1] = np.inf

                obs_counts = np.histogram(s, bins=edges)[0].astype(float)
                exp_probs = np.diff(h0_frozen.cdf(edges))
                exp_counts = exp_probs * n
                # Avoid zero expected counts
                exp_counts = np.maximum(exp_counts, 1e-8)

                df = k_bins - 1
                chi2_val = float(np.sum((obs_counts - exp_counts) ** 2 / exp_counts))
                pv = float(1 - stats.chi2.cdf(chi2_val, df))
                stat_val = chi2_val

                last_s = s
                gof_last_obs.set(obs_counts)
                gof_last_exp.set(exp_counts)
                gof_last_edges.set(edges)
                gof_last_chi2_df.set(df)

            elif test == "sw":
                s = _sample_from("gof_d", _data_dist(), min(n, 5000), input)
                result = stats.shapiro(s)
                stat_val, pv = result.statistic, result.pvalue
                last_s = s

                # Bootstrap null: generate SW stats under normality
                if len(gof_sw_null()) < 2000:
                    boot_w = []
                    for _ in range(200):
                        boot_s = np.random.normal(size=min(n, 5000))
                        boot_w.append(stats.shapiro(boot_s).statistic)
                    prev = gof_sw_null()
                    gof_sw_null.set(np.concatenate([prev, boot_w]) if len(prev) else np.array(boot_w))

            else:
                continue

            pvs.append(pv)
            total += 1
            if pv < alpha:
                rejected += 1

            last_p = pv

        gof_pvalues.set(pvs)
        gof_total.set(total)
        gof_rejected.set(rejected)
        gof_last_stat.set(stat_val)
        gof_last_pval.set(last_p)
        if last_s is not None:
            gof_last_sample.set(last_s)
        if last_s2 is not None:
            gof_last_sample2.set(last_s2)
        if last_dl is not None:
            gof_last_d_loc.set(last_dl)

    # ── Stat card text outputs ───────────────────────────────────────────────
    @render.text
    def gof_stat_label():
        test = input.gof_test()
        return {"ks1": "D-STATISTIC", "ks2": "D-STATISTIC",
                "chi2": "\u03c7\u00b2 STATISTIC", "sw": "W-STATISTIC"}.get(test, "STATISTIC")

    @render.text
    def gof_stat_value():
        v = gof_last_stat()
        return f"{v:.4f}" if v is not None else "\u2014"

    @render.text
    def gof_pvalue():
        v = gof_last_pval()
        return f"{v:.4f}" if v is not None else "\u2014"

    @render.text
    def gof_reject_rate():
        t = gof_total()
        if t == 0:
            return "\u2014"
        return f"{100 * gof_rejected() / t:.1f}%"

    @render.text
    def gof_total_tests():
        return f"{gof_total():,}"

    @render.text
    def gof_main_chart_title():
        test = input.gof_test()
        return {
            "ks1":  "EMPIRICAL CDF vs THEORETICAL CDF",
            "ks2":  "EMPIRICAL CDFs COMPARISON",
            "chi2": "OBSERVED vs EXPECTED FREQUENCIES",
            "sw":   "Q-Q PLOT (NORMALITY)",
        }.get(test, "MAIN CHART")

    # ── Chart renderers ──────────────────────────────────────────────────────
    @render.ui
    def gof_main_plot():
        test = input.gof_test()
        dark = is_dark()
        alpha = input.gof_alpha()

        if test == "ks1":
            s = gof_last_sample()
            if s is None:
                return _empty_msg("Draw a sample to see the ECDF", dark)
            h0_frozen = _read_frozen("gof_h", _h0_dist(), input)
            fig = draw_ks1_ecdf(
                sample=s, dist_frozen=h0_frozen,
                d_stat=gof_last_stat() or 0,
                d_loc=gof_last_d_loc(),
                alpha=alpha,
                alternative=_alternative(),
                dist_label=_dist_label(_h0_dist()),
                dark=dark,
            )

        elif test == "ks2":
            s1, s2 = gof_last_sample(), gof_last_sample2()
            if s1 is None or s2 is None:
                return _empty_msg("Draw a sample to see both ECDFs", dark)
            fig = draw_ks2_ecdf(
                sample1=s1, sample2=s2,
                d_stat=gof_last_stat() or 0,
                d_loc=gof_last_d_loc(),
                alpha=alpha, dark=dark,
            )

        elif test == "chi2":
            obs = gof_last_obs()
            if obs is None:
                return _empty_msg("Draw a sample to see frequencies", dark)
            fig = draw_chi2_bars(
                bin_edges=gof_last_edges(),
                observed=obs,
                expected=gof_last_exp(),
                chi2_stat=gof_last_stat() or 0,
                df=gof_last_chi2_df(),
                pvalue=gof_last_pval() or 1,
                alpha=alpha, dark=dark,
            )

        elif test == "sw":
            s = gof_last_sample()
            if s is None:
                return _empty_msg("Draw a sample to see the Q-Q plot", dark)
            fig = draw_qq_plot(
                sample=s,
                w_stat=gof_last_stat(),
                pvalue=gof_last_pval(),
                dark=dark,
            )
        else:
            return ui.div()

        return _fig_to_ui(fig)

    @render.ui
    def gof_null_plot():
        test = input.gof_test()
        dark = is_dark()
        alpha = input.gof_alpha()

        null_params = {}
        if test in ("ks1", "ks2"):
            null_params["n"] = _n()
            if test == "ks2":
                null_params["n2"] = _n2()
        elif test == "chi2":
            null_params["df"] = gof_last_chi2_df()
        elif test == "sw":
            null_params["w_null"] = gof_sw_null()

        alt = _alternative() if test == "ks1" else "two-sided"

        fig = draw_gof_null_dist(
            stat_value=gof_last_stat(),
            test_type=test,
            null_params=null_params,
            alpha=alpha,
            alternative=alt,
            dark=dark,
        )
        return _fig_to_ui(fig)

    @render.ui
    def gof_pval_hist():
        fig = draw_gof_pvalue_hist(
            pvalues=list(gof_pvalues()),
            alpha=input.gof_alpha(),
            dark=is_dark(),
        )
        return _fig_to_ui(fig)


def _empty_msg(text: str, dark: bool):
    from plots import _base_fig, _theme
    t = _theme(dark)
    fig = _base_fig(dark=dark)
    fig.add_annotation(
        xref="paper", yref="paper", x=0.5, y=0.5,
        text=text, showarrow=False,
        font=dict(size=12, color=t["muted"]),
    )
    html = fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})
    return ui.div(ui.HTML(html), class_="plotly-container")
