# =============================================================================
# Server: reactive logic, event handlers, render functions
# =============================================================================

from collections import deque

import numpy as np
from scipy import stats
from shiny import reactive, render, ui

from utils import tip
from plots import draw_ci_plot, draw_prop_plot, draw_width_plot, draw_means_plot


def server(input, output, session):

    MAX_DISPLAY = 50      # CI intervals shown on chart
    MAX_DATA    = 10_000  # rolling window for histogram / proportion data

    # ── Reactive state ──
    total_drawn = reactive.value(0)
    total_covered = reactive.value(0)
    history = reactive.value([])                          # last MAX_DISPLAY CI entries
    all_widths: reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    all_means:  reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    prop_x:     reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    prop_y:     reactive.Value[deque] = reactive.value(deque(maxlen=MAX_DATA))
    is_playing = reactive.value(False)
    speed_ms = reactive.value(0.5)     # seconds

    # ── Dynamic Parameters UI ──────────────────────────────────────────────
    @render.ui
    def dynamic_params():
        dist = input.pop_dist()
        if dist == "normal":
            return ui.div(
                ui.input_numeric("pop_mean",
                    ui.TagList("Population Mean (\u03bc)", tip("The expected value (center) of the normal distribution.")),
                    value=0.0, step=0.5, width="100%"),
                ui.input_numeric("pop_sd",
                    ui.TagList("Population Std Dev (\u03c3)", tip("Measures the spread of the distribution around the mean.")),
                    value=1.0, min=0.1, step=0.5, width="100%"),
                class_="slider-row",
            )
        elif dist == "uniform":
            return ui.div(
                ui.input_numeric("pop_min",
                    ui.TagList("Minimum (a)", tip("The lower bound of the uniform distribution.")),
                    value=0.0, step=0.5, width="100%"),
                ui.input_numeric("pop_max",
                    ui.TagList("Maximum (b)", tip("The upper bound of the uniform distribution.")),
                    value=1.0, step=0.5, width="100%"),
                class_="slider-row",
            )
        elif dist == "exponential":
            return ui.div(
                ui.input_numeric("pop_lambda",
                    ui.TagList("Rate (\u03bb)", tip("The rate parameter. Higher \u03bb means more frequent events and a smaller mean (1/\u03bb).")),
                    value=1.0, min=0.1, step=0.5, width="100%"),
                ui.div(style="width: 100%;"),
                class_="slider-row",
            )
        elif dist == "lognormal":
            return ui.div(
                ui.input_numeric("lnorm_mu",
                    ui.TagList("Log-mean (\u03bc\u2097\u2099)", tip("Mean of the underlying normal distribution on the log scale.")),
                    value=0.0, step=0.25, width="100%"),
                ui.input_numeric("lnorm_sigma",
                    ui.TagList("Log-std (\u03c3\u2097\u2099)", tip("Std dev on the log scale. Larger values give stronger right skew.")),
                    value=0.5, min=0.1, max=3.0, step=0.25, width="100%"),
                class_="slider-row",
            )
        elif dist == "poisson":
            return ui.div(
                ui.input_numeric("pois_lam",
                    ui.TagList("Rate (\u03bb)", tip("Expected number of events. Both mean and variance equal \u03bb.")),
                    value=3.0, min=0.1, step=0.5, width="100%"),
                ui.div(style="width: 100%;"),
                class_="slider-row",
            )
        elif dist == "binomial":
            return ui.div(
                ui.input_numeric("binom_n",
                    ui.TagList("Trials (m)", tip("Number of independent Bernoulli trials per observation.")),
                    value=10, min=1, max=500, step=1, width="100%"),
                ui.input_numeric("binom_p",
                    ui.TagList("Probability (p)", tip("Probability of success on each trial (0 < p < 1).")),
                    value=0.5, min=0.01, max=0.99, step=0.05, width="100%"),
                class_="slider-row",
            )

    # ── Interactive Formulas ───────────────────────────────────────────────
    @render.ui
    def formulas_ui():
        dist = input.pop_dist()

        n = input.sample_size()
        if n is None or n < 2:
            n = 5
        n = int(n)
        conf = input.conf_level() / 100.0
        method = input.ci_method()

        if dist == "normal":
            dist_name = "Normal Distribution"
            param_math = r"\[ \mu = \text{Mean}, \quad \sigma = \text{SD} \]"
        elif dist == "uniform":
            dist_name = "Uniform Distribution"
            param_math = r"\[ \mu = \frac{a+b}{2}, \quad \sigma = \frac{b-a}{\sqrt{12}} \]"
        elif dist == "exponential":
            dist_name = "Exponential Distribution"
            param_math = r"\[ \mu = \frac{1}{\lambda}, \quad \sigma = \frac{1}{\lambda} \]"
        elif dist == "lognormal":
            dist_name = "Log-normal Distribution"
            param_math = (r"\[ \mu = e^{\mu_{\ln}+\sigma_{\ln}^2/2},"
                          r"\quad \sigma = \sqrt{(e^{\sigma_{\ln}^2}-1)\,e^{2\mu_{\ln}+\sigma_{\ln}^2}} \]")
        elif dist == "poisson":
            dist_name = "Poisson Distribution"
            param_math = r"\[ \mu = \lambda, \quad \sigma = \sqrt{\lambda} \]"
        elif dist == "binomial":
            dist_name = "Binomial Distribution"
            param_math = r"\[ \mu = m \cdot p, \quad \sigma = \sqrt{m \cdot p \cdot (1-p)} \]"
        else:
            dist_name = "Distribution"
            param_math = ""

        if method == "t":
            tc = float(stats.t.ppf(1 - (1 - conf) / 2, df=n - 1))
            ci_math = (
                rf"\[ \text{{CI}} = \bar{{x}} \pm t_{{{n-1},\,{conf*100:g}\%}}"
                rf" \times \frac{{s}}{{\sqrt{{n}}}}"
                rf" \implies \bar{{x}} \pm {tc:.3f}"
                rf" \times \frac{{s}}{{\sqrt{{{n}}}}} \]"
            )
        elif method == "z":
            zc = float(stats.norm.ppf(1 - (1 - conf) / 2))
            ci_math = (
                rf"\[ \text{{CI}} = \bar{{x}} \pm z_{{{conf*100:g}\%}}"
                rf" \times \frac{{\sigma}}{{\sqrt{{n}}}}"
                rf" \implies \bar{{x}} \pm {zc:.3f}"
                rf" \times \frac{{\sigma}}{{\sqrt{{{n}}}}} \]"
            )
        else:  # bootstrap
            alpha = (1 - conf) * 100
            ci_math = (
                rf"\[ \text{{CI}} = \Bigl["
                rf"Q_{{{alpha/2:g}\%}}(\bar{{x}}^*),\;"
                rf"Q_{{{100-alpha/2:g}\%}}(\bar{{x}}^*)"
                rf"\Bigr], \quad B = 500 \]"
            )

        js_trigger = ui.tags.script("if (window.MathJax) { MathJax.typesetPromise(); }")

        return ui.TagList(
            ui.div(
                ui.div(dist_name + " Parameters",
                       style="font-weight: 500; color: #94a3b8; font-size: 0.85rem; text-align: center;"),
                ui.HTML(param_math),
                style="margin-bottom: 2px;"
            ),
            ui.div(
                ui.div("Confidence Interval",
                       style="font-weight: 500; color: #94a3b8; font-size: 0.85rem; text-align: center;"),
                ui.HTML(ci_math),
                style="margin-bottom: 2px;"
            ),
            js_trigger
        )

    # ── True Parameters ────────────────────────────────────────────────────
    @reactive.calc
    def true_params():
        dist = input.pop_dist()
        if dist == "normal":
            try:
                mu = float(input.pop_mean() or 0.0)
                sigma = float(input.pop_sd() or 1.0)
            except Exception:
                mu, sigma = 0.0, 1.0
        elif dist == "uniform":
            try:
                a = float(input.pop_min() or 0.0)
                b = float(input.pop_max() or 1.0)
                if a > b: a, b = b, a
            except Exception:
                a, b = 0.0, 1.0
            mu = (a + b) / 2.0
            sigma = (b - a) / np.sqrt(12)
        elif dist == "exponential":
            try:
                lam = float(input.pop_lambda() or 1.0)
            except Exception:
                lam = 1.0
            if lam <= 0: lam = 1e-6
            mu = 1.0 / lam
            sigma = 1.0 / lam
        elif dist == "lognormal":
            try:
                lnmu = float(input.lnorm_mu() or 0.0)
                lnsg = float(input.lnorm_sigma() or 0.5)
                if lnsg <= 0: lnsg = 0.1
            except Exception:
                lnmu, lnsg = 0.0, 0.5
            mu    = float(np.exp(lnmu + lnsg**2 / 2))
            sigma = float(np.sqrt((np.exp(lnsg**2) - 1) * np.exp(2*lnmu + lnsg**2)))
        elif dist == "poisson":
            try:
                lam = float(input.pois_lam() or 3.0)
                if lam <= 0: lam = 0.1
            except Exception:
                lam = 3.0
            mu    = lam
            sigma = float(np.sqrt(lam))
        elif dist == "binomial":
            try:
                m = int(input.binom_n() or 10)
                p = float(input.binom_p() or 0.5)
                m = max(1, m)
                p = max(0.001, min(0.999, p))
            except Exception:
                m, p = 10, 0.5
            mu    = float(m * p)
            sigma = float(np.sqrt(m * p * (1 - p)))
        else:
            mu, sigma = 0.0, 1.0

        if sigma <= 0:
            sigma = 1e-6
        return mu, sigma

    # ── Sample size +/- ────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.n_minus)
    def _n_minus():
        cur = input.sample_size()
        if cur is not None and cur > 2:
            ui.update_numeric("sample_size", value=cur - 1)

    @reactive.effect
    @reactive.event(input.n_plus)
    def _n_plus():
        cur = input.sample_size()
        if cur is not None and cur < 500:
            ui.update_numeric("sample_size", value=cur + 1)

    # ── Speed +/- ──────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.speed_minus)
    def _speed_down():
        speed_ms.set(min(speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.speed_plus)
    def _speed_up():
        speed_ms.set(max(speed_ms() - 0.05, 0.05))

    # ── Play / Pause ──────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.btn_play)
    def _toggle_play():
        is_playing.set(not is_playing())
        playing = is_playing()
        if playing:
            ui.update_action_button("btn_play", label="Pause")
        else:
            ui.update_action_button("btn_play", label="Play")

    # ── Continuous animation ──────────────────────────────────────────────
    @reactive.effect
    def _auto_draw():
        if is_playing():
            reactive.invalidate_later(speed_ms())
            with reactive.isolate():
                draw_samples(1)

    # ── Manual buttons ────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.btn_sample_1)
    def _sample_1():
        draw_samples(1)

    @reactive.effect
    @reactive.event(input.btn_sample_50)
    def _sample_50():
        draw_samples(50)

    @reactive.effect
    @reactive.event(input.btn_sample_100)
    def _sample_100():
        draw_samples(100)

    # ── Reset ─────────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.btn_reset, input.pop_dist, input.ci_method)
    def _reset():
        total_drawn.set(0)
        total_covered.set(0)
        history.set([])
        all_widths.set(deque(maxlen=MAX_DATA))
        all_means.set(deque(maxlen=MAX_DATA))
        prop_x.set(deque(maxlen=MAX_DATA))
        prop_y.set(deque(maxlen=MAX_DATA))
        is_playing.set(False)
        ui.update_action_button("btn_play", label="Play")

    # ── Core sampling logic ───────────────────────────────────────────────
    def draw_samples(k: int):
        n = input.sample_size()
        if n is None or n < 2:
            n = 5
        n = int(n)
        conf = input.conf_level() / 100.0

        current_drawn = total_drawn()
        current_covered = total_covered()

        mu, sigma = true_params()

        # Vectorized data generation: shape (n, k)
        dist_choice = input.pop_dist()
        if dist_choice == "normal":
            samps = np.random.normal(mu, sigma, size=(n, k))
        elif dist_choice == "uniform":
            try:
                a = float(input.pop_min() or 0.0)
                b = float(input.pop_max() or 1.0)
                if a > b: a, b = b, a
            except Exception:
                a, b = 0.0, 1.0
            samps = np.random.uniform(a, b, size=(n, k))
        elif dist_choice == "exponential":
            try:
                lam = float(input.pop_lambda() or 1.0)
            except Exception:
                lam = 1.0
            if lam <= 0: lam = 1e-6
            samps = np.random.exponential(1.0 / lam, size=(n, k))
        elif dist_choice == "lognormal":
            try:
                lnmu = float(input.lnorm_mu() or 0.0)
                lnsg = float(input.lnorm_sigma() or 0.5)
                if lnsg <= 0: lnsg = 0.1
            except Exception:
                lnmu, lnsg = 0.0, 0.5
            samps = np.random.lognormal(lnmu, lnsg, size=(n, k))
        elif dist_choice == "poisson":
            try:
                lam = float(input.pois_lam() or 3.0)
                if lam <= 0: lam = 0.1
            except Exception:
                lam = 3.0
            samps = np.random.poisson(lam, size=(n, k)).astype(float)
        elif dist_choice == "binomial":
            try:
                m = int(input.binom_n() or 10)
                p = float(input.binom_p() or 0.5)
                m = max(1, m)
                p = max(0.001, min(0.999, p))
            except Exception:
                m, p = 10, 0.5
            samps = np.random.binomial(m, p, size=(n, k)).astype(float)
        else:
            samps = np.random.normal(mu, sigma, size=(n, k))

        means = np.mean(samps, axis=0)
        method = input.ci_method()

        if method == "t":
            stds = np.std(samps, axis=0, ddof=1)
            ses  = stds / np.sqrt(n)
            tc   = float(stats.t.ppf(1 - (1 - conf) / 2, df=n - 1))
            los  = means - tc * ses
            his  = means + tc * ses

        elif method == "z":
            # z-interval uses the TRUE population σ (known-variance case)
            zc  = float(stats.norm.ppf(1 - (1 - conf) / 2))
            los = means - zc * (sigma / np.sqrt(n))
            his = means + zc * (sigma / np.sqrt(n))

        else:  # bootstrap — percentile method, B=500 resamples
            B   = 500
            # Vectorised: idx (B, n, k) indexes into samps (n, k) per column j
            idx        = np.random.randint(0, n, size=(B, n, k))
            j_idx      = np.arange(k)[np.newaxis, np.newaxis, :]
            boot_means = samps[idx, j_idx].mean(axis=1)          # (B, k)
            alpha_pct  = (1 - conf) * 100
            los = np.percentile(boot_means, alpha_pct / 2,       axis=0)
            his = np.percentile(boot_means, 100 - alpha_pct / 2, axis=0)

        ws  = his - los
        cvs = (mu >= los) & (mu <= his)

        new_entries = []
        for i in range(k):
            new_entries.append(dict(
                id=current_drawn + i + 1,
                mean=float(means[i]),
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

        m = deque(all_means(), maxlen=MAX_DATA)
        m.extend(e["mean"] for e in new_entries)
        all_means.set(m)

        px = deque(prop_x(), maxlen=MAX_DATA)
        px.append(current_drawn)
        prop_x.set(px)

        py = deque(prop_y(), maxlen=MAX_DATA)
        py.append(current_covered / current_drawn)
        prop_y.set(py)

    # ── Text outputs ──────────────────────────────────────────────────────
    @render.text
    def conf_pct():
        return f"{input.conf_level()}%"

    @render.text
    def conf_pct2():
        return f"{input.conf_level()}%"

    @render.text
    def conf_pct3():
        return f"{input.conf_level()}%"

    @render.text
    def cov_rate():
        td = total_drawn()
        if td == 0:
            return "\u2014"
        return f"{100 * total_covered() / td:.1f}%"

    @render.text
    def num_covered():
        return f"{total_covered():,}"

    @render.text
    def num_missed():
        return f"{total_drawn() - total_covered():,}"

    @render.text
    def num_total():
        return f"{total_drawn():,}"

    # ── Plot renderers (delegate to plots.py) ─────────────────────────────
    @render.plot(alt="Confidence Intervals")
    def ci_plot():
        n = input.sample_size()
        if n is None or n < 2:
            n = 5
        mu, sigma = true_params()
        return draw_ci_plot(history(), mu, sigma, int(n), input.ci_method())

    @render.plot(alt="Proportion of CIs including mu")
    def prop_plot():
        return draw_prop_plot(list(prop_x()), list(prop_y()), input.conf_level() / 100.0)

    @render.plot(alt="CI Width Distribution")
    def width_plot():
        return draw_width_plot(list(all_widths()))

    @render.plot(alt="Sample Means Distribution")
    def means_plot():
        n = input.sample_size()
        if n is None or n < 2:
            n = 5
        mu, sigma = true_params()
        return draw_means_plot(list(all_means()), mu, sigma, int(n))
