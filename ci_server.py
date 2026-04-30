from collections import deque
import numpy as np
from scipy import stats
from shiny import reactive, render, ui

from utils import tip
from plots import draw_ci_plot, draw_prop_plot, draw_width_plot, draw_means_plot, draw_population_plot
from ci_methods import compute_ci_mean, compute_ci_proportion, compute_ci_bootstrap
from theme import fig_to_ui

def ci_server(input, output, session, is_dark):
    MAX_DISPLAY = 50      # CI intervals shown on chart
    MAX_DATA    = 10_000  # rolling window for histogram / proportion data

    # ── Reactive state ──
    total_drawn = reactive.value(0)
    total_covered = reactive.value(0)
    history = reactive.value([])                          # last MAX_DISPLAY CI entries
    all_widths: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    all_estimates: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    prop_x:     reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    prop_y:     reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    last_sample: reactive.Value[list] = reactive.value([])
    is_playing = reactive.value(False)
    speed_ms = reactive.value(0.5)     # seconds
    _ci_active_preset  = reactive.value(None)
    _ci_preset_params  = reactive.value({})   # dist-specific param overrides for ci_dynamic_params
    _ci_preset_stat    = reactive.value(None) # statistic override for ci_statistic_ui
    _ci_preset_method  = reactive.value(None) # method override for _sync_ci_method

    # ── CI Scenario presets ───────────────────────────────────────────────
    _CI_PRESET_DESC = {
        "ideal": (
            "Ideal conditions",
            "Normal(0,\u00a01), n\u200a=\u200a30, t-interval. "
            "Textbook case: CLT is satisfied, coverage \u2248 nominal. "
            "Baseline for comparing all other scenarios.",
        ),
        "skewed": (
            "Skewed + small n",
            "Log-normal(\u03bc\u2097\u2099\u200a=\u200a0, \u03c3\u2097\u2099\u200a=\u200a1), n\u200a=\u200a15, t-interval. "
            "Strong right skew and few observations \u2014 CLT has not yet kicked in. "
            "t-interval systematically under-covers.",
        ),
        "rare": (
            "Rare events",
            "Binomial(m\u200a=\u200a40, p\u200a=\u200a0.05), proportion, Wald CI. "
            "np\u200a=\u200a2\u200a<\u200a5: normal approximation breaks down. "
            "Wald CI severely under-covers \u2014 switch to Wilson or Clopper-Pearson.",
        ),
        "boot": (
            "Bootstrap rescue",
            "Exponential(\u03bb\u200a=\u200a1), n\u200a=\u200a10, Bootstrap. "
            "Skewed distribution with very small n \u2014 t-interval fails here. "
            "Bootstrap outperforms t-interval by resampling without distributional assumptions.",
        ),
        "poisson": (
            "Poisson counts",
            "Poisson(\u03bb\u200a=\u200a2), n\u200a=\u200a20, t-interval. "
            "Discrete, right-skewed counts with small \u03bb. "
            "t-interval under-covers because the sampling distribution is not yet symmetric.",
        ),
        "heavy_skew": (
            "Heavy skew / outlier shock",
            "Log-normal(\u03bc\u2097\u2099\u200a=\u200a0, \u03c3\u2097\u2099\u200a=\u200a2.5), n\u200a=\u200a10, t-interval. "
            "Extreme right skew: a single rare observation dominates the sample mean. "
            "CI width explodes sample-to-sample and coverage drops sharply \u2014 "
            "most of your uncertainty comes from the tail, not the body.",
        ),
        "false_conf": (
            "False confidence trap",
            "Normal(0,\u00a01), n\u200a=\u200a30, t-interval, confidence\u200a=\u200a80%, null\u200a=\u200a0. "
            "H\u2080 is true by construction, and the CI is well-calibrated. "
            "Because the nominal level is 80%, \u2248 20% of intervals exclude 0 by design \u2014 "
            "pure Type I error. Watch the decision verdict flash red on ~1 in 5 samples: "
            "\u201csignificant\u201d doesn't mean \u201creal effect.\u201d",
        ),
    }

    def _ci_set_preset(dist, method, n, statistic="mean", conf=None, **params):
        # Store values in reactive vars so re-rendered dynamic UIs pick them up
        _ci_preset_stat.set(statistic)
        _ci_preset_method.set(method)
        _ci_preset_params.set({"ci_sample_size": n, **params})
        # Reset accumulated state so each preset starts clean \u2014 handles cases
        # where two presets share dist/method/statistic but differ in n or conf
        # (e.g. "Ideal" and "False confidence"), which wouldn't trigger _reset.
        total_drawn.set(0)
        total_covered.set(0)
        history.set([])
        all_widths.set(deque(maxlen=MAX_DATA))
        all_estimates.set(deque(maxlen=MAX_DATA))
        prop_x.set(deque(maxlen=MAX_DATA))
        prop_y.set(deque(maxlen=MAX_DATA))
        last_sample.set([])
        is_playing.set(False)
        ui.update_action_button("ci_btn_play", label="Play")
        # Trigger distribution change (re-renders ci_dynamic_params + ci_statistic_ui)
        ui.update_select("ci_pop_dist", selected=dist)
        # Belt-and-suspenders: also send direct updates for same-dist case
        # (when dist doesn't change, the renderers won't re-fire, so these are needed)
        ui.update_select("ci_statistic",    selected=statistic)
        ui.update_select("ci_method",       selected=method)
        ui.update_numeric("ci_sample_size", value=n)
        if conf is not None:
            ui.update_slider("ci_conf_level", value=conf)

    @reactive.effect
    @reactive.event(input.ci_pre_ideal)
    def _ci_pr_ideal():
        _ci_active_preset.set("ideal")
        _ci_set_preset("normal", "t", 30, conf=95,
                       ci_pop_mean=0.0, ci_pop_sd=1.0)

    @reactive.effect
    @reactive.event(input.ci_pre_skewed)
    def _ci_pr_skewed():
        _ci_active_preset.set("skewed")
        _ci_set_preset("lognormal", "t", 15,
                       ci_lnorm_mu=0.0, ci_lnorm_sigma=1.0)

    @reactive.effect
    @reactive.event(input.ci_pre_rare)
    def _ci_pr_rare():
        _ci_active_preset.set("rare")
        _ci_set_preset("binomial", "wald", 30,
                       statistic="proportion",
                       ci_binom_n=40, ci_binom_p=0.05)

    @reactive.effect
    @reactive.event(input.ci_pre_boot)
    def _ci_pr_boot():
        _ci_active_preset.set("boot")
        _ci_set_preset("exponential", "bootstrap", 10,
                       ci_pop_lambda=1.0)

    @reactive.effect
    @reactive.event(input.ci_pre_poisson)
    def _ci_pr_poisson():
        _ci_active_preset.set("poisson")
        _ci_set_preset("poisson", "t", 20,
                       ci_pois_lam=2.0)

    @reactive.effect
    @reactive.event(input.ci_pre_heavy)
    def _ci_pr_heavy():
        _ci_active_preset.set("heavy_skew")
        _ci_set_preset("lognormal", "t", 10,
                       ci_lnorm_mu=0.0, ci_lnorm_sigma=2.5)

    @reactive.effect
    @reactive.event(input.ci_pre_false_conf)
    def _ci_pr_false_conf():
        _ci_active_preset.set("false_conf")
        _ci_set_preset("normal", "t", 30, conf=80,
                       ci_pop_mean=0.0, ci_pop_sd=1.0)

    @render.ui
    def ci_preset_desc():
        key = _ci_active_preset()
        if key is None:
            return ui.div(
                "\u2190 Select a preset to see what it demonstrates.",
                class_="np-preset-hint",
            )
        title, body = _CI_PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(title + ": "),
            body,
            class_="np-preset-hint np-preset-hint--active",
        )

    # ── Dynamic statistic dropdown (adds Proportion for Binomial) ────────────
    @render.ui
    def ci_statistic_ui():
        dist = input.ci_pop_dist()
        with reactive.isolate():
            preset_stat = _ci_preset_stat()
        choices = {
            "mean":       "Mean",
            "median":     "Median",
            "variance":   "Variance",
            "percentile": "Percentile",
        }
        if dist == "binomial":
            choices["proportion"] = "Proportion  (p\u0302 = successes / n\u00b7m)"
        try:
            current = input.ci_statistic()
            selected = current if current in choices else "mean"
        except Exception:
            selected = "mean"
        # Preset overrides the preserved selection when re-rendering on dist change
        if preset_stat is not None and preset_stat in choices:
            selected = preset_stat
        return ui.input_select(
            "ci_statistic",
            ui.TagList("Statistic", tip(
                "The population parameter to estimate. "
                "Median, Variance, and Percentile use Bootstrap only. "
                "Proportion (Binomial only) uses Wald, Wilson, or Clopper-Pearson."
            )),
            choices=choices,
            selected=selected,
            width="100%",
        )

    # ── Sync ci_method choices based on chosen statistic ─────────────────────
    @reactive.effect
    @reactive.event(input.ci_statistic)
    def _sync_ci_method():
        stat = input.ci_statistic()
        with reactive.isolate():
            pm = _ci_preset_method()
        if stat in ("median", "variance", "percentile"):
            ui.update_select("ci_method",
                choices={"bootstrap": "Bootstrap   (percentile, B\u200a=\u200a500)"},
                selected="bootstrap")
        elif stat == "proportion":
            valid = {"wald", "wilson", "clopper_pearson"}
            ui.update_select("ci_method",
                choices={
                    "wald":            "Wald \u26a0 (poor for small n / extreme p)",
                    "wilson":          "Wilson  (score, recommended)",
                    "clopper_pearson": "Clopper-Pearson  (exact)",
                },
                selected=pm if pm in valid else "wald")
        else:
            valid = {"t", "z", "bootstrap"}
            ui.update_select("ci_method",
                choices={
                    "t":         "t-interval  (unknown \u03c3)",
                    "z":         "z-interval  (known \u03c3) \u26a0 rarely realistic",
                    "bootstrap": "Bootstrap   (percentile, B\u200a=\u200a500)",
                },
                selected=pm if pm in valid else "t")

    # ── Reset ci_statistic to "mean" when switching away from Binomial ────────
    @reactive.effect
    @reactive.event(input.ci_pop_dist)
    def _sync_statistic_on_dist_change():
        if input.ci_pop_dist() != "binomial":
            try:
                if input.ci_statistic() == "proportion":
                    ui.update_select("ci_statistic", selected="mean")
            except Exception:
                pass

    # ── Percentile level slider (visible only when Percentile selected) ───────
    @render.ui
    def ci_percentile_param():
        if input.ci_statistic() != "percentile":
            return ui.div()
        return ui.input_slider(
            "ci_percentile_level",
            ui.TagList("Percentile level (p)",
                       tip("The p-th percentile of the population distribution to estimate via Bootstrap CI.")),
            min=1, max=99, value=25, step=1, width="100%",
        )

    # ── Dynamic Parameters UI ──────────────────────────────────────────────
    @render.ui
    def ci_dynamic_params():
        dist = input.ci_pop_dist()
        with reactive.isolate():
            _p = _ci_preset_params()

        def pval(key, default):
            return _p.get(key, default)

        n_col = ui.div(
            ui.input_numeric("ci_sample_size",
                ui.TagList("Sample size (n)\u00a0", tip("Number of observations in each sample.")),
                value=pval("ci_sample_size", 5), min=2, max=500, step=1, width="100%")
        )

        if dist == "normal":
            r1 = ui.div(
                ui.div(ui.input_numeric("ci_pop_mean", ui.TagList("Population \u03bc\u00a0", tip("The expected value (center) of the normal distribution.")), value=pval("ci_pop_mean", 0.0), step=0.5, width="100%")),
                ui.div(ui.input_numeric("ci_pop_sd", ui.TagList("Population \u03c3\u00a0", tip("Measures the spread of the distribution around the mean.")), value=pval("ci_pop_sd", 1.0), min=0.1, step=0.5, width="100%")),
                class_="group-params-cols"
            )
            return ui.div(r1, ui.div(n_col, ui.div(), class_="group-params-cols"), class_="group-params-block")

        elif dist == "uniform":
            r1 = ui.div(
                ui.div(ui.input_numeric("ci_pop_min", ui.TagList("Minimum (a)\u00a0", tip("The lower bound of the uniform distribution.")), value=pval("ci_pop_min", 0.0), step=0.5, width="100%")),
                ui.div(ui.input_numeric("ci_pop_max", ui.TagList("Maximum (b)\u00a0", tip("The upper bound of the uniform distribution.")), value=pval("ci_pop_max", 1.0), step=0.5, width="100%")),
                class_="group-params-cols"
            )
            return ui.div(r1, ui.div(n_col, ui.div(), class_="group-params-cols"), class_="group-params-block")

        elif dist == "exponential":
            r1 = ui.div(
                ui.div(ui.input_numeric("ci_pop_lambda", ui.TagList("Rate (\u03bb)\u00a0", tip("The rate parameter. Higher \u03bb means more frequent events and a smaller mean (1/\u03bb).")), value=pval("ci_pop_lambda", 1.0), min=0.1, step=0.5, width="100%")),
                n_col,
                class_="group-params-cols"
            )
            return ui.div(r1, class_="group-params-block")

        elif dist == "lognormal":
            r1 = ui.div(
                ui.div(ui.input_numeric("ci_lnorm_mu", ui.TagList("Log-mean (\u03bc\u2097\u2099)\u00a0", tip("Mean of the underlying normal distribution on the log scale.")), value=pval("ci_lnorm_mu", 0.0), step=0.25, width="100%")),
                ui.div(ui.input_numeric("ci_lnorm_sigma", ui.TagList("Log-std (\u03c3\u2097\u2099)\u00a0", tip("Std dev on the log scale. Larger values give stronger right skew.")), value=pval("ci_lnorm_sigma", 0.5), min=0.1, max=3.0, step=0.25, width="100%")),
                class_="group-params-cols"
            )
            return ui.div(r1, ui.div(n_col, ui.div(), class_="group-params-cols"), class_="group-params-block")

        elif dist == "poisson":
            r1 = ui.div(
                ui.div(ui.input_numeric("ci_pois_lam", ui.TagList("Rate (\u03bb)\u00a0", tip("Expected number of events. Both mean and variance equal \u03bb.")), value=pval("ci_pois_lam", 3.0), min=0.1, step=0.5, width="100%")),
                n_col,
                class_="group-params-cols"
            )
            return ui.div(r1, class_="group-params-block")

        elif dist == "binomial":
            r1 = ui.div(
                ui.div(ui.input_numeric("ci_binom_n", ui.TagList("Trials (m)\u00a0", tip("Number of independent Bernoulli trials per observation.")), value=pval("ci_binom_n", 10), min=1, max=500, step=1, width="100%")),
                ui.div(ui.input_numeric("ci_binom_p", ui.TagList("Probability (p)\u00a0", tip("Probability of success on each trial (0 < p < 1).")), value=pval("ci_binom_p", 0.5), min=0.01, max=0.99, step=0.05, width="100%")),
                class_="group-params-cols"
            )
            return ui.div(r1, ui.div(n_col, ui.div(), class_="group-params-cols"), class_="group-params-block")

    # ── True Parameters ────────────────────────────────────────────────────
    @reactive.calc
    def true_params():
        """Return (mu, sigma, true_median, true_variance) for the current distribution."""
        dist = input.ci_pop_dist()
        if dist == "normal":
            try:
                mu = float(input.ci_pop_mean() or 0.0)
                sigma = float(input.ci_pop_sd() or 1.0)
            except Exception:
                mu, sigma = 0.0, 1.0
            median = mu
            variance = sigma ** 2
        elif dist == "uniform":
            try:
                a = float(input.ci_pop_min() or 0.0)
                b = float(input.ci_pop_max() or 1.0)
                if a > b: a, b = b, a
            except Exception:
                a, b = 0.0, 1.0
            mu = (a + b) / 2.0
            sigma = (b - a) / np.sqrt(12)
            median = mu
            variance = sigma ** 2
        elif dist == "exponential":
            try:
                lam = float(input.ci_pop_lambda() or 1.0)
            except Exception:
                lam = 1.0
            if lam <= 0: lam = 1e-6
            mu = 1.0 / lam
            sigma = 1.0 / lam
            median = float(np.log(2) / lam)
            variance = sigma ** 2
        elif dist == "lognormal":
            try:
                lnmu = float(input.ci_lnorm_mu() or 0.0)
                lnsg = float(input.ci_lnorm_sigma() or 0.5)
                if lnsg <= 0: lnsg = 0.1
            except Exception:
                lnmu, lnsg = 0.0, 0.5
            mu    = float(np.exp(lnmu + lnsg**2 / 2))
            sigma = float(np.sqrt((np.exp(lnsg**2) - 1) * np.exp(2*lnmu + lnsg**2)))
            median = float(np.exp(lnmu))
            variance = sigma ** 2
        elif dist == "poisson":
            try:
                lam = float(input.ci_pois_lam() or 3.0)
                if lam <= 0: lam = 0.1
            except Exception:
                lam = 3.0
            mu    = lam
            sigma = float(np.sqrt(lam))
            median = float(stats.poisson.median(lam))
            variance = lam
        elif dist == "binomial":
            try:
                m = int(input.ci_binom_n() or 10)
                p = float(input.ci_binom_p() or 0.5)
                m = max(1, m)
                p = max(0.001, min(0.999, p))
            except Exception:
                m, p = 10, 0.5
            mu    = float(m * p)
            sigma = float(np.sqrt(m * p * (1 - p)))
            median = float(stats.binom.median(m, p))
            variance = sigma ** 2
        else:
            mu, sigma = 0.0, 1.0
            median = mu
            variance = sigma ** 2

        if sigma <= 0:
            sigma = 1e-6
        return mu, sigma, median, variance

    @reactive.calc
    def true_value():
        """The true population value of the chosen statistic."""
        mu, sigma, median, variance = true_params()
        stat = input.ci_statistic()
        if stat == "proportion":
            try:
                p = float(input.ci_binom_p() or 0.5)
                return max(0.001, min(0.999, p))
            except Exception:
                return 0.5
        if stat == "median":
            return median
        if stat == "variance":
            return variance
        if stat == "percentile":
            try:
                p = (input.ci_percentile_level() or 25) / 100.0
            except Exception:
                p = 0.25
            dist = input.ci_pop_dist()
            # Use scipy ppf for the exact population percentile
            if dist == "normal":
                try: mu_ = float(input.ci_pop_mean() or 0.0); sg_ = float(input.ci_pop_sd() or 1.0)
                except Exception: mu_, sg_ = 0.0, 1.0
                return float(stats.norm.ppf(p, mu_, sg_))
            elif dist == "uniform":
                try:
                    a_ = float(input.ci_pop_min() or 0.0); b_ = float(input.ci_pop_max() or 1.0)
                    if a_ > b_: a_, b_ = b_, a_
                except Exception: a_, b_ = 0.0, 1.0
                return float(stats.uniform.ppf(p, a_, b_ - a_))
            elif dist == "exponential":
                try: lam_ = float(input.ci_pop_lambda() or 1.0)
                except Exception: lam_ = 1.0
                if lam_ <= 0: lam_ = 1e-6
                return float(stats.expon.ppf(p, scale=1.0 / lam_))
            elif dist == "lognormal":
                try:
                    lnmu_ = float(input.ci_lnorm_mu() or 0.0)
                    lnsg_ = float(input.ci_lnorm_sigma() or 0.5)
                    if lnsg_ <= 0: lnsg_ = 0.1
                except Exception: lnmu_, lnsg_ = 0.0, 0.5
                return float(stats.lognorm.ppf(p, s=lnsg_, scale=float(np.exp(lnmu_))))
            elif dist == "poisson":
                try: lam_ = float(input.ci_pois_lam() or 3.0)
                except Exception: lam_ = 3.0
                if lam_ <= 0: lam_ = 0.1
                return float(stats.poisson.ppf(p, lam_))
            elif dist == "binomial":
                try:
                    m_ = int(input.ci_binom_n() or 10); p_ = float(input.ci_binom_p() or 0.5)
                    m_ = max(1, m_); p_ = max(0.001, min(0.999, p_))
                except Exception: m_, p_ = 10, 0.5
                return float(stats.binom.ppf(p, m_, p_))
            return float(np.percentile([mu], 50))  # fallback
        return mu

    # ── Sample size +/- ────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.ci_n_minus)
    def _n_minus():
        cur = input.ci_sample_size()
        if cur is not None and cur > 2:
            ui.update_numeric("ci_sample_size", value=cur - 1)

    @reactive.effect
    @reactive.event(input.ci_n_plus)
    def _n_plus():
        cur = input.ci_sample_size()
        if cur is not None and cur < 500:
            ui.update_numeric("ci_sample_size", value=cur + 1)

    # ── Speed +/- ────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.ci_speed_minus)
    def _speed_down():
        speed_ms.set(min(speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.ci_speed_plus)
    def _speed_up():
        speed_ms.set(max(speed_ms() - 0.05, 0.05))

    # ── Play / Pause ──────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.ci_btn_play)
    def _toggle_play():
        is_playing.set(not is_playing())
        playing = is_playing()
        if playing:
            ui.update_action_button("ci_btn_play", label="Pause")
        else:
            ui.update_action_button("ci_btn_play", label="Play")

    # ── Continuous animation ──────────────────────────────────────────────
    @reactive.effect
    def _auto_draw():
        if is_playing():
            reactive.invalidate_later(speed_ms())
            with reactive.isolate():
                draw_samples(1)

    # ── Manual buttons ────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.ci_btn_sample_1)
    def _sample_1():
        draw_samples(1)

    @reactive.effect
    @reactive.event(input.ci_btn_sample_50)
    def _sample_50():
        draw_samples(50)

    @reactive.effect
    @reactive.event(input.ci_btn_sample_100)
    def _sample_100():
        draw_samples(100)

    # ── Reset ─────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.ci_btn_reset, input.ci_pop_dist, input.ci_method, input.ci_statistic)
    def _reset():
        total_drawn.set(0)
        total_covered.set(0)
        history.set([])
        all_widths.set(deque(maxlen=MAX_DATA))
        all_estimates.set(deque(maxlen=MAX_DATA))
        prop_x.set(deque(maxlen=MAX_DATA))
        prop_y.set(deque(maxlen=MAX_DATA))
        last_sample.set([])
        is_playing.set(False)
        ui.update_action_button("ci_btn_play", label="Play")

    # Reset when percentile level changes (conditional input — needs try/except)
    @reactive.effect
    def _reset_on_percentile_level():
        try:
            input.ci_percentile_level()
        except Exception:
            return
        if input.ci_statistic() == "percentile":
            total_drawn.set(0)
            total_covered.set(0)
            history.set([])
            all_widths.set(deque(maxlen=MAX_DATA))
            all_estimates.set(deque(maxlen=MAX_DATA))
            prop_x.set(deque(maxlen=MAX_DATA))
            prop_y.set(deque(maxlen=MAX_DATA))
            last_sample.set([])
            is_playing.set(False)
            ui.update_action_button("ci_btn_play", label="Play")

    # ── Core sampling logic ───────────────────────────────────────────────
    def draw_samples(k: int):
        n = input.ci_sample_size()
        if n is None or n < 2:
            n = 5
        n = int(n)
        conf = input.ci_conf_level() / 100.0

        current_drawn = total_drawn()
        current_covered = total_covered()

        mu, sigma, *_ = true_params()

        # Vectorized data generation: shape (n, k)
        dist_choice = input.ci_pop_dist()
        if dist_choice == "normal":
            samps = np.random.normal(mu, sigma, size=(n, k))
        elif dist_choice == "uniform":
            try:
                a = float(input.ci_pop_min() or 0.0)
                b = float(input.ci_pop_max() or 1.0)
                if a > b: a, b = b, a
            except Exception:
                a, b = 0.0, 1.0
            samps = np.random.uniform(a, b, size=(n, k))
        elif dist_choice == "exponential":
            try:
                lam = float(input.ci_pop_lambda() or 1.0)
            except Exception:
                lam = 1.0
            if lam <= 0: lam = 1e-6
            samps = np.random.exponential(1.0 / lam, size=(n, k))
        elif dist_choice == "lognormal":
            try:
                lnmu = float(input.ci_lnorm_mu() or 0.0)
                lnsg = float(input.ci_lnorm_sigma() or 0.5)
                if lnsg <= 0: lnsg = 0.1
            except Exception:
                lnmu, lnsg = 0.0, 0.5
            samps = np.random.lognormal(lnmu, lnsg, size=(n, k))
        elif dist_choice == "poisson":
            try:
                lam = float(input.ci_pois_lam() or 3.0)
                if lam <= 0: lam = 0.1
            except Exception:
                lam = 3.0
            samps = np.random.poisson(lam, size=(n, k)).astype(float)
        elif dist_choice == "binomial":
            try:
                m = int(input.ci_binom_n() or 10)
                p = float(input.ci_binom_p() or 0.5)
                m = max(1, m)
                p = max(0.001, min(0.999, p))
            except Exception:
                m, p = 10, 0.5
            samps = np.random.binomial(m, p, size=(n, k)).astype(float)
        else:
            samps = np.random.normal(mu, sigma, size=(n, k))

        # Store the most recent single sample for the population plot
        last_sample.set(samps[:, -1].tolist())

        stat_type = input.ci_statistic()
        try:
            p_level = int(input.ci_percentile_level() or 25)
        except Exception:
            p_level = 25

        # Binomial m parameter (needed for proportion)
        try:
            binom_m = max(1, int(input.ci_binom_n() or 10))
        except Exception:
            binom_m = 10

        # Point estimates for the chosen statistic
        if stat_type == "mean":
            estimates = np.mean(samps, axis=0)
        elif stat_type == "median":
            estimates = np.median(samps, axis=0)
        elif stat_type == "variance":
            estimates = np.var(samps, axis=0, ddof=1)
        elif stat_type == "percentile":
            estimates = np.percentile(samps, p_level, axis=0)
        elif stat_type == "proportion":
            # p̂ = total successes / total Bernoulli trials = mean(X_i) / m
            estimates = np.mean(samps, axis=0) / binom_m
        else:
            estimates = np.mean(samps, axis=0)

        method = input.ci_method()

        if method == "t" and stat_type == "mean":
            los, his = compute_ci_mean(samps, method="t", level=conf)

        elif method == "z" and stat_type == "mean":
            los, his = compute_ci_mean(samps, method="z", level=conf, sigma=sigma)

        elif stat_type == "proportion":
            n_eff = n * binom_m   # total Bernoulli trials per sample
            k_succ = np.round(estimates * n_eff).astype(int)
            los, his = compute_ci_proportion(k_succ, n_eff, method=method, level=conf)

        else:  # bootstrap — percentile method, B=500 resamples
            los, his = compute_ci_bootstrap(
                samps, level=conf, statistic=stat_type, p_level=p_level, B=500,
            )

        ws  = his - los
        tv  = true_value()
        cvs = (tv >= los) & (tv <= his)

        new_entries = []
        for i in range(k):
            new_entries.append(dict(
                id=current_drawn + i + 1,
                estimate=float(estimates[i]),
                lower=float(los[i]),
                upper=float(his[i]),
                covered=bool(cvs[i]),
                width=float(ws[i])
            ))

        num_covered_new = int(np.sum(cvs))
        current_drawn += k
        current_covered += num_covered_new

        total_drawn.set(current_drawn)
        total_covered.set(current_covered)

        hist = history() + new_entries
        if len(hist) > MAX_DISPLAY:
            hist = hist[-MAX_DISPLAY:]
        history.set(hist)

        w = deque(all_widths(), maxlen=MAX_DATA)
        w.extend(e["width"] for e in new_entries)
        all_widths.set(w)

        m = deque(all_estimates(), maxlen=MAX_DATA)
        m.extend(e["estimate"] for e in new_entries)
        all_estimates.set(m)

        px = deque(prop_x(), maxlen=MAX_DATA)
        px.append(current_drawn)
        prop_x.set(px)

        py = deque(prop_y(), maxlen=MAX_DATA)
        py.append(current_covered / current_drawn)
        prop_y.set(py)

    # ── Text outputs ──────────────────────────────────────────────────────
    @render.text
    def ci_conf_pct():
        return f"{input.ci_conf_level()}%"

    @render.text
    def ci_conf_pct2():
        return f"{input.ci_conf_level()}%"

    @render.text
    def ci_conf_pct3():
        return f"{input.ci_conf_level()}%"

    @render.text
    def ci_cov_rate():
        td = total_drawn()
        if td == 0:
            return "\u2014"
        return f"{100 * total_covered() / td:.1f}%"

    @render.text
    def ci_num_covered():
        return f"{total_covered():,}"

    @render.text
    def ci_num_missed():
        return f"{total_drawn() - total_covered():,}"

    @render.text
    def ci_num_total():
        return f"{total_drawn():,}"

    # ── Undercoverage verdict (coverage vs nominal with SE guard) ────────
    @render.ui
    def ci_cov_verdict():
        td = total_drawn()
        if td < 30:
            return ui.div(
                "collecting samples\u2026",
                style=("font-size:0.68rem; color:var(--c-text3); "
                       "font-style:italic; margin-top:2px;"),
            )
        cov = total_covered() / td
        nominal = input.ci_conf_level() / 100.0
        se = (cov * (1 - cov) / td) ** 0.5
        gap = cov - nominal
        if gap < -2 * se:
            return ui.div(
                "\u26a0 Undercoverage (below ", f"{nominal*100:.0f}%", ")",
                style=("font-size:0.72rem; color:#f87171; "
                       "font-weight:600; margin-top:2px;"),
            )
        if gap > 2 * se:
            return ui.div(
                "Overcoverage (above ", f"{nominal*100:.0f}%", ")",
                style=("font-size:0.72rem; color:#fbbf24; "
                       "font-weight:600; margin-top:2px;"),
            )
        return ui.div(
            "\u2713 on target (\u2248 ", f"{nominal*100:.0f}%", ")",
            style=("font-size:0.72rem; color:#34d399; "
                   "font-weight:600; margin-top:2px;"),
        )

    # ── Decision framing: does the latest CI contain the null value? ─────
    # Since this module knows the true \u03b8, we also label Type I / Type II
    # outcomes so users don't mistake a Type-I interval for a real finding.
    @render.ui
    def ci_decision():
        hist = history()
        if not hist:
            return ui.div()
        last = hist[-1]
        lo, hi = last["lower"], last["upper"]
        stat = input.ci_statistic()
        # Pick a reasonable null per statistic
        if stat == "proportion":
            null_val, null_lbl = 0.5, "0.5 (fair)"
        elif stat == "variance":
            null_val, null_lbl = 1.0, "1"
        else:
            null_val, null_lbl = 0.0, "0"

        try:
            true_val = float(true_value())
        except Exception:
            true_val = null_val
        # Tolerance relative to the null magnitude (handles both 0 and non-zero)
        tol = 1e-6 * max(1.0, abs(null_val))
        h0_true = abs(true_val - null_val) < tol

        contains = lo <= null_val <= hi

        if contains and h0_true:
            verdict = "Yes \u2192 not significant (H\u2080 retained \u2014 correct)"
            color = "#34d399"
        elif contains and not h0_true:
            verdict = ("Yes \u2192 not significant "
                       "\u26a0 Type II (H\u2080 is false, but retained)")
            color = "#fbbf24"
        elif (not contains) and h0_true:
            verdict = ("No \u2192 significant "
                       "\u26a0 Type I / false positive under H\u2080")
            color = "#f87171"
        else:
            verdict = "No \u2192 significant (effect detected \u2014 correct)"
            color = "#34d399"

        return ui.div(
            ui.div(
                ui.tags.span(
                    f"Last CI: [{lo:.3g}, {hi:.3g}] \u00b7 ",
                    style="color:var(--c-text3);",
                ),
                ui.tags.span(
                    f"Includes null ({null_lbl}): ",
                    style="color:var(--c-text3);",
                ),
                ui.tags.span(verdict, style=f"color:{color}; font-weight:600;"),
            ),
            ui.div(
                "In real experiments the truth is unknown \u2014 "
                "Type\u202fI / Type\u202fII labels are only visible here because this is a simulation.",
                style=("font-size:0.68rem; color:var(--c-text3); "
                       "font-style:italic; margin-top:4px;"),
            ),
            style=("font-size:0.75rem; text-align:center; "
                   "margin:6px 12px 2px; padding-top:6px; "
                   "border-top:1px solid var(--c-border);"),
        )

    # ── Dynamic labels ───────────────────────────────────────────────────
    @render.text
    def ci_stat_label_inc():
        s = input.ci_statistic()
        if s == "percentile":
            try: p = int(input.ci_percentile_level() or 25)
            except Exception: p = 25
            return f"P{p} INCLUDED"
        return {"mean": "\u03bc INCLUDED", "median": "MEDIAN INCLUDED",
                "variance": "\u03c3\u00b2 INCLUDED",
                "proportion": "p INCLUDED"}.get(s, "\u03bc INCLUDED")

    @render.text
    def ci_stat_label_miss():
        s = input.ci_statistic()
        if s == "percentile":
            try: p = int(input.ci_percentile_level() or 25)
            except Exception: p = 25
            return f"P{p} MISSED"
        return {"mean": "\u03bc MISSED", "median": "MEDIAN MISSED",
                "variance": "\u03c3\u00b2 MISSED",
                "proportion": "p MISSED"}.get(s, "\u03bc MISSED")

    @render.text
    def ci_stat_plot_title():
        s = input.ci_statistic()
        if s == "percentile":
            try: p = int(input.ci_percentile_level() or 25)
            except Exception: p = 25
            return f"SAMPLE P{p} DISTRIBUTION"
        return {"mean": "SAMPLE MEANS DISTRIBUTION (CLT)",
                "median": "SAMPLE MEDIANS DISTRIBUTION",
                "variance": "SAMPLE VARIANCES DISTRIBUTION",
                "proportion": "SAMPLE PROPORTIONS DISTRIBUTION (CLT)"}.get(s, "SAMPLE STATISTICS DISTRIBUTION")

    @render.text
    def ci_prop_plot_title():
        s = input.ci_statistic()
        if s == "proportion":
            return "PROPORTION OF CIs INCLUDING p"
        return "PROPORTION OF CIs INCLUDING \u03bc"

    # ── Chart renderers (Plotly → HTML) ──────────────────────────────────
    @render.ui
    def ci_plot():
        n = input.ci_sample_size()
        if n is None or n < 2:
            n = 5
        mu, sigma, *_ = true_params()
        tv = true_value()
        stat = input.ci_statistic()
        try: p_level = int(input.ci_percentile_level() or 25)
        except Exception: p_level = 25
        fig = draw_ci_plot(history(), tv, sigma, int(n), input.ci_method(),
                           statistic=stat, p_level=p_level, dark=is_dark())
        return fig_to_ui(fig)

    @render.ui
    def ci_prop_plot():
        fig = draw_prop_plot(list(prop_x()), list(prop_y()),
                             input.ci_conf_level() / 100.0, dark=is_dark())
        return fig_to_ui(fig)

    @render.ui
    def ci_width_plot():
        fig = draw_width_plot(list(all_widths()), dark=is_dark())
        return fig_to_ui(fig)

    @render.ui
    def ci_means_plot():
        n = input.ci_sample_size()
        if n is None or n < 2:
            n = 5
        n = int(n)
        mu, sigma, *_ = true_params()
        tv = true_value()
        stat = input.ci_statistic()
        try: p_level = int(input.ci_percentile_level() or 25)
        except Exception: p_level = 25
        # For proportion: sigma=sqrt(p(1-p)), n=n_eff → SE = sqrt(p(1-p)/n_eff)
        if stat == "proportion":
            try: binom_m = max(1, int(input.ci_binom_n() or 10))
            except Exception: binom_m = 10
            sigma_plot = float(np.sqrt(tv * (1 - tv)))
            n_plot = n * binom_m
        else:
            sigma_plot, n_plot = sigma, n
        fig = draw_means_plot(list(all_estimates()), tv, sigma_plot, n_plot,
                              statistic=stat, p_level=p_level, dark=is_dark())
        return fig_to_ui(fig)

    @render.ui
    def ci_population_plot():
        dist = input.ci_pop_dist()
        stat = input.ci_statistic()
        try: p_level = int(input.ci_percentile_level() or 25)
        except Exception: p_level = 25
        tv = true_value()

        # Build distribution params dict for the plot function
        try:
            if dist == "normal":
                params = {"mu": float(input.ci_pop_mean() or 0.0),
                          "sigma": float(input.ci_pop_sd() or 1.0)}
            elif dist == "uniform":
                a_ = float(input.ci_pop_min() or 0.0)
                b_ = float(input.ci_pop_max() or 1.0)
                if a_ > b_: a_, b_ = b_, a_
                params = {"a": a_, "b": b_}
            elif dist == "exponential":
                params = {"lam": float(input.ci_pop_lambda() or 1.0)}
            elif dist == "lognormal":
                params = {"lnmu": float(input.ci_lnorm_mu() or 0.0),
                          "lnsg": float(input.ci_lnorm_sigma() or 0.5)}
            elif dist == "poisson":
                params = {"lam": float(input.ci_pois_lam() or 3.0)}
            elif dist == "binomial":
                params = {"m": int(input.ci_binom_n() or 10),
                          "p": float(input.ci_binom_p() or 0.5)}
            else:
                params = {}
        except Exception:
            params = {}

        params["p_level"] = p_level
        fig = draw_population_plot(dist, params, last_sample(), tv,
                                   statistic=stat, dark=is_dark())
        return fig_to_ui(fig)
