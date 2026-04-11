# =============================================================================
# Bootstrap Explorer — server logic
# =============================================================================

import numpy as np
from scipy import stats
from scipy.stats import trim_mean as _scipy_trim_mean
from shiny import reactive, render, ui

from utils import tip
from boot_plots import (
    draw_boot_sample,
    draw_boot_distribution,
    draw_boot_ci_forest,
    draw_boot_coverage,
)

_PLOTLY_CFG = {"displayModeBar": False, "responsive": True}


def _fig_html(fig):
    return ui.div(
        ui.HTML(fig.to_html(full_html=False, include_plotlyjs=False,
                            config=_PLOTLY_CFG)),
        class_="plotly-container",
    )


# ── Data generation ─────────────────────────────────────────────────────────
def _generate(dist: str, n: int) -> np.ndarray:
    if dist == "normal":
        return np.random.normal(0, 1, n)
    if dist == "lognormal":
        return np.random.lognormal(0, 0.5, n)
    if dist == "heavy":
        return np.random.standard_t(3, n)
    if dist == "uniform":
        return np.random.uniform(0, 1, n)
    if dist == "bimodal":
        mask = np.random.random(n) < 0.5
        out = np.empty(n)
        out[mask] = np.random.normal(-2, 1, int(mask.sum()))
        out[~mask] = np.random.normal(2, 1, int((~mask).sum()))
        return out
    return np.random.normal(0, 1, n)


# ── True parameter values ──────────────────────────────────────────────────
_TRUE_CACHE: dict = {}


def _true_param(dist: str, stat: str) -> float:
    """Return the true population value of *stat* under *dist*."""
    key = (dist, stat)
    if key in _TRUE_CACHE:
        return _TRUE_CACHE[key]

    known = {
        ("normal", "mean"): 0.0, ("normal", "median"): 0.0,
        ("normal", "trimmed"): 0.0, ("normal", "std"): 1.0,
        ("lognormal", "mean"): float(np.exp(0.125)),
        ("lognormal", "median"): 1.0,
        ("heavy", "mean"): 0.0, ("heavy", "median"): 0.0,
        ("heavy", "trimmed"): 0.0,
        ("uniform", "mean"): 0.5, ("uniform", "median"): 0.5,
        ("uniform", "trimmed"): 0.5,
        ("uniform", "std"): float(np.sqrt(1.0 / 12)),
        ("bimodal", "mean"): 0.0, ("bimodal", "median"): 0.0,
        ("bimodal", "trimmed"): 0.0,
    }
    if key in known:
        _TRUE_CACHE[key] = known[key]
        return known[key]

    # Monte Carlo fallback
    big = _generate(dist, 500_000)
    val = float(_stat_single(stat, big))
    _TRUE_CACHE[key] = val
    return val


# ── Statistic helpers ───────────────────────────────────────────────────────
def _stat_single(stat: str, x: np.ndarray) -> float:
    if stat == "mean":
        return float(np.mean(x))
    if stat == "median":
        return float(np.median(x))
    if stat == "trimmed":
        return float(_scipy_trim_mean(x, 0.1))
    if stat == "std":
        return float(np.std(x, ddof=1))
    if stat == "percentile90":
        return float(np.percentile(x, 90))
    return float(np.mean(x))


def _stat_batch(stat: str, x: np.ndarray) -> np.ndarray:
    """Compute statistic along axis-1 for shape (B, n) → (B,)."""
    if stat == "mean":
        return x.mean(axis=1)
    if stat == "median":
        return np.median(x, axis=1)
    if stat == "trimmed":
        n = x.shape[1]
        k = max(1, int(n * 0.1))
        s = np.sort(x, axis=1)
        return s[:, k:n - k].mean(axis=1)
    if stat == "std":
        return x.std(axis=1, ddof=1)
    if stat == "percentile90":
        return np.percentile(x, 90, axis=1)
    return x.mean(axis=1)


# ── CI computation ──────────────────────────────────────────────────────────
def _compute_cis(boot_stats, boot_ses, theta_hat, sample, stat,
                 alpha, methods):
    """Return (cis_dict, z0, a_acc)."""
    B = len(boot_stats)
    se_boot = float(np.std(boot_stats, ddof=1))
    z_a = stats.norm.ppf(1 - alpha / 2)
    cis: dict = {}
    z0, a_acc = 0.0, 0.0

    if "Percentile" in methods:
        cis["Percentile"] = (
            float(np.percentile(boot_stats, alpha / 2 * 100)),
            float(np.percentile(boot_stats, (1 - alpha / 2) * 100)),
        )

    if "Normal" in methods:
        cis["Normal"] = (
            theta_hat - z_a * se_boot,
            theta_hat + z_a * se_boot,
        )

    if "Basic" in methods:
        cis["Basic"] = (
            2 * theta_hat - float(np.percentile(boot_stats, (1 - alpha / 2) * 100)),
            2 * theta_hat - float(np.percentile(boot_stats, alpha / 2 * 100)),
        )

    if "Studentized" in methods and boot_ses is not None and stat == "mean":
        t_star = (boot_stats - theta_hat) / (boot_ses + 1e-12)
        t_lo = float(np.percentile(t_star, (1 - alpha / 2) * 100))
        t_hi = float(np.percentile(t_star, alpha / 2 * 100))
        se_orig = float(np.std(sample, ddof=1) / np.sqrt(len(sample)))
        cis["Studentized"] = (
            theta_hat - t_lo * se_orig,
            theta_hat - t_hi * se_orig,
        )

    if "BCa" in methods and B >= 20:
        # Bias correction
        prop = np.clip(np.mean(boot_stats < theta_hat), 1e-10, 1 - 1e-10)
        z0 = float(stats.norm.ppf(prop))

        # Acceleration (jackknife)
        n = len(sample)
        jack = np.array([_stat_single(stat, np.delete(sample, i))
                         for i in range(n)])
        jm = jack.mean()
        diff = jm - jack
        num = float(np.sum(diff ** 3))
        den = 6.0 * float(np.sum(diff ** 2)) ** 1.5
        a_acc = num / (den + 1e-12)

        z_lo = stats.norm.ppf(alpha / 2)
        z_hi = stats.norm.ppf(1 - alpha / 2)
        a1 = float(stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a_acc * (z0 + z_lo))))
        a2 = float(stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a_acc * (z0 + z_hi))))
        a1 = np.clip(a1, 0.5 / B, 1 - 0.5 / B)
        a2 = np.clip(a2, 0.5 / B, 1 - 0.5 / B)
        cis["BCa"] = (
            float(np.percentile(boot_stats, a1 * 100)),
            float(np.percentile(boot_stats, a2 * 100)),
        )

    return cis, z0, a_acc


# ── Stat card helper ────────────────────────────────────────────────────────
def _stat_card(label_text, label_tip, value_text, value_class):
    return ui.div(
        ui.div(label_text, label_tip, class_="stat-label"),
        ui.div(value_text, class_=f"stat-value {value_class}"),
        class_="stat-card",
    )


# ═══════════════════════════════════════════════════════════════════════════════
def boot_server(input, output, session, is_dark):

    # ── Reactive state ──────────────────────────────────────────────────────
    _total         = reactive.value(0)
    _last_sample   = reactive.value(None)
    _last_theta    = reactive.value(0.0)
    _last_boot     = reactive.value(None)      # ndarray (B,) or partial list
    _last_cis      = reactive.value({})
    _last_counts   = reactive.value(None)
    _last_z0       = reactive.value(0.0)
    _last_a        = reactive.value(0.0)
    _cov_counts    = reactive.value({})

    # Step-by-step
    _step_active   = reactive.value(False)
    _step_sample   = reactive.value(None)
    _step_theta    = reactive.value(0.0)
    _step_boot     = reactive.value([])
    _step_boot_ses = reactive.value([])
    _step_counts   = reactive.value(None)

    is_playing     = reactive.value(False)
    speed_ms       = reactive.value(0.3)
    _active_preset = reactive.value(None)

    # ── Cached true theta ───────────────────────────────────────────────────
    @reactive.calc
    def _true_theta():
        return _true_param(input.boot_dist(), input.boot_statistic())

    # ── Dynamic CI method checkboxes ────────────────────────────────────────
    @render.ui
    def boot_ci_methods_ui():
        stat = input.boot_statistic()
        choices = {
            "Percentile":  ui.span("Percentile\u00a0", tip("Uses empirical quantiles of the bootstrap distribution")),
            "Normal":      ui.span("Normal\u00a0", tip("Assumes normality: θ̂ ± z·SE")),
            "Basic":       ui.span("Basic\u00a0", tip("Pivotal method, reflects quantiles around θ̂")),
            "BCa":         ui.span("BCa\u00a0", tip("Bias-Corrected and Accelerated. Best for skewed distributions")),
        }
        if stat == "mean":
            choices["Studentized"] = ui.span("Studentized\u00a0", tip("Bootstrap-t. Uses original sample standard error"))
        with reactive.isolate():
            try:
                cur = list(input.boot_ci_methods())
                sel = [c for c in cur if c in choices]
            except Exception:
                sel = []
        if not sel:
            sel = ["Percentile", "Normal", "BCa"]
        return ui.input_checkbox_group(
            "boot_ci_methods",
            ui.TagList("CI Methods",
                       tip("Select which confidence-interval methods to "
                           "compute and compare.")),
            choices=choices, selected=sel, width="100%",
        )

    # ── Presets ─────────────────────────────────────────────────────────────
    _PRESET_DESC = {
        "skewed_median": {
            "title": "Skewed + Median",
            "body":  "Log-normal data with median. BCa corrects for "
                     "asymmetric bias \u2014 Percentile CI is shifted.",
        },
        "small_heavy": {
            "title": "Small n + Heavy tails",
            "body":  "t(df\u2009=\u20093) with n\u2009=\u200915.  "
                     "Normal CI is too narrow; BCa handles skewness.",
        },
        "bimodal": {
            "title": "Bimodal",
            "body":  "Mixture of two normals. Bootstrap works; "
                     "Normal CI assumes unimodality and under-covers.",
        },
        "tiny": {
            "title": "n\u2009=\u20095 extreme",
            "body":  "Only 5 observations. All methods are unreliable "
                     "\u2014 shows the limits of resampling.",
        },
    }

    @render.ui
    def boot_presets_ui():
        return ui.div(
            ui.tags.label("Presets", class_="presets-label"),
            ui.div(
                ui.input_action_button("boot_pre_skewed", "Skewed+Med",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("boot_pre_heavy", "Heavy",
                                       class_="btn-ctrl btn-sample btn-flex"),
                class_="sidebar-btn-row",
                style="margin-bottom: 0.5rem;",
            ),
            ui.div(
                ui.input_action_button("boot_pre_bimodal", "Bimodal",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("boot_pre_tiny", "n=5",
                                       class_="btn-ctrl btn-sample btn-flex"),
                class_="sidebar-btn-row",
            ),
            ui.output_ui("boot_preset_desc"),
        )

    @render.ui
    def boot_preset_desc():
        key = _active_preset()
        if key is None or key not in _PRESET_DESC:
            return ui.div("Select a preset for a pathological case.",
                          class_="np-preset-hint")
        d = _PRESET_DESC[key]
        return ui.div(
            ui.tags.strong(d["title"]), ui.tags.br(), d["body"],
            class_="np-preset-hint np-preset-hint--active",
        )

    def _apply_preset(dist, n, stat):
        ui.update_select("boot_dist", selected=dist)
        ui.update_numeric("boot_n", value=n)
        ui.update_select("boot_statistic", selected=stat)

    @reactive.effect
    @reactive.event(input.boot_pre_skewed)
    def _pr1():
        _active_preset.set("skewed_median")
        _apply_preset("lognormal", 30, "median")

    @reactive.effect
    @reactive.event(input.boot_pre_heavy)
    def _pr2():
        _active_preset.set("small_heavy")
        _apply_preset("heavy", 15, "mean")

    @reactive.effect
    @reactive.event(input.boot_pre_bimodal)
    def _pr3():
        _active_preset.set("bimodal")
        _apply_preset("bimodal", 50, "mean")

    @reactive.effect
    @reactive.event(input.boot_pre_tiny)
    def _pr4():
        _active_preset.set("tiny")
        _apply_preset("normal", 5, "mean")

    # ── Reset ───────────────────────────────────────────────────────────────
    def _do_reset():
        _total.set(0)
        _last_sample.set(None)
        _last_theta.set(0.0)
        _last_boot.set(None)
        _last_cis.set({})
        _last_counts.set(None)
        _last_z0.set(0.0)
        _last_a.set(0.0)
        _cov_counts.set({})
        _step_active.set(False)
        _step_sample.set(None)
        _step_boot.set([])
        _step_boot_ses.set([])
        _step_counts.set(None)
        is_playing.set(False)
        ui.update_action_button("boot_btn_play", label="Play")

    @reactive.effect
    @reactive.event(input.boot_btn_reset, input.boot_dist,
                    input.boot_statistic, input.boot_n, input.boot_conf)
    def _reset():
        key = _active_preset()
        if key is not None:
            match = False
            try:
                d = input.boot_dist()
                s = input.boot_statistic()
                n = int(input.boot_n() or 0)
                if key == "skewed_median" and d == "lognormal" and s == "median" and n == 30: match = True
                elif key == "small_heavy" and d == "heavy" and s == "mean" and n == 15: match = True
                elif key == "bimodal" and d == "bimodal" and s == "mean" and n == 50: match = True
                elif key == "tiny" and d == "normal" and s == "mean" and n == 5: match = True
            except:
                pass
            if not match:
                _active_preset.set(None)
        _do_reset()

    # Abandon step experiment on B change or step toggle
    @reactive.effect
    @reactive.event(input.boot_B, input.boot_step_mode)
    def _abandon_step():
        if _step_active():
            _step_active.set(False)
            _step_boot.set([])
            _step_boot_ses.set([])

    # ── Speed / Play ────────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.boot_speed_minus)
    def _sp_dn():
        speed_ms.set(min(speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.boot_speed_plus)
    def _sp_up():
        speed_ms.set(max(speed_ms() - 0.05, 0.05))

    @reactive.effect
    @reactive.event(input.boot_btn_play)
    def _toggle():
        is_playing.set(not is_playing())
        ui.update_action_button("boot_btn_play",
                                label="Pause" if is_playing() else "Play")

    @reactive.effect
    def _auto():
        if is_playing():
            reactive.invalidate_later(speed_ms())
            with reactive.isolate():
                _draw_samples(1)

    # ── Manual buttons ──────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.boot_btn_sample_1)
    def _s1():
        _draw_samples(1)

    @reactive.effect
    @reactive.event(input.boot_btn_sample_50)
    def _s50():
        _draw_samples(50)

    @reactive.effect
    @reactive.event(input.boot_btn_sample_100)
    def _s100():
        _draw_samples(100)

    # ═══════════════════════════════════════════════════════════════════════
    # Core sampling
    # ═══════════════════════════════════════════════════════════════════════

    def _get_params():
        dist  = input.boot_dist()
        n     = max(5, int(input.boot_n() or 30))
        B     = max(100, int(input.boot_B() or 2000))
        stat  = input.boot_statistic()
        alpha = 1 - (input.boot_conf() or 95) / 100
        try:
            methods = list(input.boot_ci_methods())
        except Exception:
            methods = ["Percentile", "BCa"]
        return dist, n, B, stat, alpha, methods

    def _draw_samples(k: int):
        if input.boot_step_mode():
            _advance_step(k)
        else:
            _run_fast(k)

    # ── Fast mode ───────────────────────────────────────────────────────────
    def _run_fast(k: int):
        dist, n, B, stat, alpha, methods = _get_params()
        true_t = _true_theta()
        cov = dict(_cov_counts())

        sample = theta_hat = None
        boot_stats = last_counts = None
        cis = {}
        z0 = a_acc = 0.0

        for _ in range(k):
            sample = _generate(dist, n)
            theta_hat = _stat_single(stat, sample)

            idx = np.random.randint(0, n, (B, n))
            boot_samps = sample[idx]
            boot_stats = _stat_batch(stat, boot_samps)

            boot_ses = None
            if stat == "mean":
                boot_ses = boot_samps.std(axis=1, ddof=1) / np.sqrt(n)

            last_counts = np.bincount(idx[-1], minlength=n)

            cis, z0, a_acc = _compute_cis(
                boot_stats, boot_ses, theta_hat, sample, stat, alpha, methods)

            # Coverage
            for method, (lo, hi) in cis.items():
                if method not in cov:
                    cov[method] = 0
                if true_t is not None and lo <= true_t <= hi:
                    cov[method] += 1

        _total.set(_total() + k)
        _cov_counts.set(cov)

        # Store last experiment for rendering
        if sample is not None:
            _last_sample.set(sample)
            _last_theta.set(theta_hat)
            _last_boot.set(boot_stats)
            _last_cis.set(cis)
            _last_counts.set(last_counts)
            _last_z0.set(z0)
            _last_a.set(a_acc)

    # ── Step-by-step mode ───────────────────────────────────────────────────
    def _advance_step(k: int):
        dist, n, B, stat, alpha, methods = _get_params()

        if not _step_active():
            # Start new experiment
            sample = _generate(dist, n)
            theta_hat = _stat_single(stat, sample)
            _step_sample.set(sample)
            _step_theta.set(theta_hat)
            _step_boot.set([])
            _step_boot_ses.set([])
            _step_counts.set(None)
            _step_active.set(True)
            _last_sample.set(sample)
            _last_theta.set(theta_hat)
            _last_boot.set(np.array([]))
            _last_cis.set({})
            _last_counts.set(None)
            return  # show original sample; resamples start next tick

        sample = _step_sample()
        boot_list = list(_step_boot())
        ses_list  = list(_step_boot_ses())
        remaining = B - len(boot_list)

        if remaining <= 0:
            # Experiment already done; start a new one next tick
            _step_active.set(False)
            return

        # ×100 = complete experiment; otherwise advance k resamples
        to_do = remaining if k >= 100 else min(k, remaining)

        if to_do > 1:
            # Batched resamples (fast)
            idx_all = np.random.randint(0, n, (to_do, n))
            boot_samps = sample[idx_all]
            new_stats = _stat_batch(stat, boot_samps).tolist()
            boot_list.extend(new_stats)
            if stat == "mean":
                new_ses = (boot_samps.std(axis=1, ddof=1) / np.sqrt(n)).tolist()
                ses_list.extend(new_ses)
            last_counts = np.bincount(idx_all[-1], minlength=n)
        else:
            # Single resample (animation tick)
            idx = np.random.randint(0, n, n)
            resamp = sample[idx]
            boot_list.append(_stat_single(stat, resamp))
            if stat == "mean":
                ses_list.append(float(np.std(resamp, ddof=1) / np.sqrt(n)))
            last_counts = np.bincount(idx, minlength=n)

        _step_boot.set(boot_list)
        _step_boot_ses.set(ses_list)
        _step_counts.set(last_counts)

        # Update rendering state (growing histogram)
        _last_sample.set(sample)
        _last_theta.set(_step_theta())
        _last_boot.set(np.array(boot_list))
        _last_counts.set(last_counts)

        if len(boot_list) >= B:
            # Experiment complete — compute CIs, update coverage
            boot_arr = np.array(boot_list)
            ses_arr  = np.array(ses_list) if ses_list else None
            cis, z0, a_acc = _compute_cis(
                boot_arr, ses_arr, _step_theta(), sample, stat, alpha, methods)
            _last_cis.set(cis)
            _last_z0.set(z0)
            _last_a.set(a_acc)

            true_t = _true_theta()
            cov = dict(_cov_counts())
            for method, (lo, hi) in cis.items():
                if method not in cov:
                    cov[method] = 0
                if true_t is not None and lo <= true_t <= hi:
                    cov[method] += 1
            _cov_counts.set(cov)
            _total.set(_total() + 1)
            _step_active.set(False)
        else:
            # In progress — no CIs yet
            _last_cis.set({})

    # ═══════════════════════════════════════════════════════════════════════
    # Stats row
    # ═══════════════════════════════════════════════════════════════════════

    @render.ui
    def boot_stats_row():
        boot = _last_boot()
        theta = _last_theta()
        tot   = _total()

        if boot is not None and len(boot) > 0:
            bias = float(np.mean(boot) - theta)
            se   = float(np.std(boot, ddof=1))
        else:
            bias, se = None, None

        bias_txt = f"{bias:.4f}" if bias is not None else "\u2014"
        se_txt   = f"{se:.4f}" if se is not None else "\u2014"
        b_done   = len(boot) if boot is not None else 0
        try:
            B_total = int(input.boot_B())
        except Exception:
            B_total = 2000

        bias_class = "coverage"
        if bias is not None and se is not None and se > 0:
            if abs(bias) > 0.02 * se:
                bias_class = "missed"

        return ui.div(
            _stat_card("BIAS\u00a0",
                       tip("Mean(\u03b8\u0302*) \u2212 \u03b8\u0302. "
                           "Non-zero bias motivates BCa correction."),
                       bias_txt, bias_class),
            _stat_card("SE\u1d47\u2092\u2092\u209c\u00a0",
                       tip("Standard deviation of the bootstrap distribution."),
                       se_txt, "coverage"),
            _stat_card("RESAMPLES\u00a0",
                       tip("Bootstrap resamples completed in the current "
                           "experiment / total requested."),
                       f"{b_done:,} / {B_total:,}", "included"),
            _stat_card("TOTAL EXPERIMENTS\u00a0",
                       tip("Completed bootstrap experiments (each with B "
                           "resamples)."),
                       f"{tot:,}", "total"),
            class_="stats-row",
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Chart renderers
    # ═══════════════════════════════════════════════════════════════════════

    @render.ui
    def boot_sample_plot():
        dark   = is_dark()
        sample = _last_sample()
        if sample is None:
            return ui.div("Draw samples to begin.", class_="chart-placeholder")
        fig = draw_boot_sample(
            sample, _last_counts(), _last_theta(), _true_theta(), dark)
        return _fig_html(fig)

    @render.ui
    def boot_dist_plot():
        dark = is_dark()
        boot = _last_boot()
        if boot is None or len(boot) == 0:
            return ui.div("No bootstrap resamples yet.",
                          class_="chart-placeholder")
        theta = _last_theta()
        fig = draw_boot_distribution(
            boot, theta, _true_theta(), _last_cis(),
            float(np.mean(boot) - theta),
            float(np.std(boot, ddof=1)),
            _last_z0(), _last_a(), dark,
        )
        return _fig_html(fig)

    @render.ui
    def boot_ci_plot():
        dark = is_dark()
        cis = _last_cis()
        if not cis:
            return ui.div("Complete an experiment to see CIs.",
                          class_="chart-placeholder")
        fig = draw_boot_ci_forest(cis, _last_theta(), _true_theta(), dark)
        return _fig_html(fig)

    @render.ui
    def boot_coverage_plot():
        dark = is_dark()
        conf = float(input.boot_conf() or 95)
        fig = draw_boot_coverage(_cov_counts(), _total(), conf, dark)
        return _fig_html(fig)
