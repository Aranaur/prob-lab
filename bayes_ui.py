# =============================================================================
# Bayesian Explorer — UI panel
#
# One outer sidebar + main panel with navset_underline across three sub-tabs
# (Bootstrap Explorer layout). Sidebar holds three control groups; only the
# group matching the active sub-tab is visible — a small JS listener toggles
# a `data-active` attribute. All inputs live in the DOM at all times so their
# values survive tab switches.
# =============================================================================

from shiny import ui
from utils import tip


# ── Sidebar control groups ──────────────────────────────────────────────────

def _sidebar_tab1() -> ui.Tag:
    """Beta-Binomial Updating controls."""
    return ui.div(
        # Misconception banner
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Common Misconception: "),
            "\u201cA prior always biases the result.\u201d",
            ui.tags.br(),
            ui.tags.strong("Reality: "),
            "An informative prior only shortens the posterior when n is small. "
            "Large n dominates any reasonable prior \u2014 try Beta(20, 5) "
            "with n = 5, then resample to n = 500.",
            class_="info-banner-text",
        ),

        # True p
        ui.input_slider(
            "bys1_true_p",
            ui.TagList("True p (data-generating)",
                       tip("The unknown success probability used to generate "
                           "synthetic coin flips.")),
            min=0.01, max=0.99, value=0.30, step=0.01, width="100%",
        ),

        # Prior parameters
        ui.div(
            ui.div(
                ui.input_numeric(
                    "bys1_alpha",
                    ui.TagList("Prior \u03b1",
                               tip("Prior successes (pseudo-observations).")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
            ),
            ui.div(
                ui.input_numeric(
                    "bys1_beta",
                    ui.TagList("Prior \u03b2",
                               tip("Prior failures (pseudo-observations).")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
            ),
            class_="group-params-cols",
        ),

        # Prior presets
        ui.tags.label(
            "Prior presets",
            style="font-weight:500; color:var(--c-text3); "
                  "font-size:0.82rem; margin-bottom:2px;",
        ),
        ui.div(
            ui.input_action_button("bys1_prior_uniform",   "Uniform",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys1_prior_jeffreys",  "Jeffreys",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys1_prior_weak",      "Weak",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys1_prior_skeptic",   "Skeptic",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys1_prior_confident", "Confident",
                                   class_="btn-ctrl btn-preset"),
            class_="np-preset-grid",
        ),
        ui.output_ui("bys1_preset_desc"),

        # Lock-data toggle
        ui.input_switch(
            "bys1_lock",
            ui.TagList(
                "Lock data, change prior",
                tip("When on, sampling is disabled and changing the prior "
                    "recomputes the posterior on the same observations. "
                    "Demonstrates that posterior = prior \u00d7 likelihood, "
                    "not magic."),
            ),
            value=False,
        ),
        ui.output_ui("bys1_lock_badge"),

        # Sampling controls (sample + speed/play rows; dynamic for lock state)
        ui.div(
            ui.output_ui("bys1_sample_controls"),
            ui.div(
                ui.input_action_button("bys1_reset", "Reset",
                                       class_="btn-ctrl btn-reset btn-full"),
                class_="sidebar-btn-row",
            ),
            class_="sidebar-controls",
        ),

        class_="bys-sidebar-group", **{"data-bys-tab": "1"},
    )


def _sidebar_tab2() -> ui.Tag:
    """Freq vs Bayes CI controls."""
    return ui.div(
        # Misconception banner
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Misconception: "),
            "\u201cCredible interval = Confidence interval.\u201d",
            ui.tags.br(),
            ui.tags.strong("Correct (Freq CI): "),
            "95% of these intervals contain p across repeated sampling.",
            ui.tags.br(),
            ui.tags.strong("Correct (Bayes CrI): "),
            "With probability 95%, p lies in this interval ",
            ui.tags.em("given this prior and these data"),
            ".",
            ui.tags.br(),
            ui.tags.strong("They answer different questions "),
            "\u2014 and may give different coverage.",
            class_="info-banner-text",
        ),

        ui.input_slider(
            "bys2_true_p",
            ui.TagList("True p",
                       tip("Unknown parameter the CIs aim to cover.")),
            min=0.01, max=0.99, value=0.30, step=0.01, width="100%",
        ),
        ui.div(
            ui.div(
                ui.input_numeric(
                    "bys2_n",
                    ui.TagList("n per experiment",
                               tip("Number of Bernoulli trials per CI.")),
                    value=30, min=2, step=1, width="100%",
                ),
            ),
            ui.div(
                ui.input_slider(
                    "bys2_conf",
                    ui.TagList("Confidence level",
                               tip("Nominal level for both CI and CrI.")),
                    min=0.80, max=0.99, value=0.95, step=0.01, width="100%",
                ),
            ),
            class_="group-params-cols",
        ),
        ui.div(
            ui.div(
                ui.input_numeric(
                    "bys2_alpha",
                    ui.TagList("Prior \u03b1", tip("Prior successes.")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
            ),
            ui.div(
                ui.input_numeric(
                    "bys2_beta",
                    ui.TagList("Prior \u03b2", tip("Prior failures.")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
            ),
            class_="group-params-cols",
        ),

        ui.input_select(
            "bys2_freq_method",
            ui.TagList("Frequentist method",
                       tip("CI method for proportions.")),
            choices={
                "wilson":          "Wilson  (score, recommended)",
                "wald":            "Wald \u26a0 (poor at extremes)",
                "clopper_pearson": "Clopper-Pearson  (exact)",
            },
            selected="wilson", width="100%",
        ),

        # Presets
        ui.tags.label(
            "Scenario presets",
            style="font-weight:500; color:var(--c-text3); "
                  "font-size:0.82rem; margin-bottom:2px;",
        ),
        ui.div(
            ui.input_action_button("bys2_pre_dominates", "Prior dominates",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys2_pre_agreement", "Agreement",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys2_pre_jeffreys",  "Jeffreys \u2248 Wilson",
                                   class_="btn-ctrl btn-preset"),
            class_="np-preset-grid",
        ),
        ui.output_ui("bys2_preset_desc"),

        # Sampling controls
        ui.div(
            ui.div(
                ui.tags.span("Sample:", class_="btn-row-label"),
                ui.input_action_button("bys2_sample_1",   "\u00d71",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("bys2_sample_50",  "\u00d750",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("bys2_sample_100", "\u00d7100",
                                       class_="btn-ctrl btn-sample btn-flex"),
                class_="sidebar-btn-row",
            ),
            ui.div(
                ui.div(
                    ui.tags.label("Speed"),
                    ui.input_action_button("bys2_speed_minus", "\u2212",
                                           class_="btn-ctrl btn-pm"),
                    ui.input_action_button("bys2_btn_play", "Play",
                                           class_="btn-ctrl btn-play btn-flex"),
                    ui.input_action_button("bys2_speed_plus", "+",
                                           class_="btn-ctrl btn-pm"),
                    class_="ctrl-group ctrl-group-full",
                ),
                class_="sidebar-btn-row",
            ),
            ui.div(
                ui.input_action_button(
                    "bys2_recompute",
                    "\u21bb Recompute with same data",
                    class_="btn-ctrl btn-sample btn-full",
                ),
                class_="sidebar-btn-row",
            ),
            ui.div(
                ui.input_action_button("bys2_reset", "Reset",
                                       class_="btn-ctrl btn-reset btn-full"),
                class_="sidebar-btn-row",
            ),
            class_="sidebar-controls",
        ),

        class_="bys-sidebar-group", **{"data-bys-tab": "2"},
    )


def _sidebar_tab3() -> ui.Tag:
    """Bayesian A/B Testing controls."""
    return ui.div(
        # Misconception banner
        ui.div(
            ui.tags.i(class_="info-icon"),
            ui.tags.strong(" Misconception: "),
            "\u201cBayesian A/B has better statistical power.\u201d",
            ui.tags.br(),
            ui.tags.strong("Reality: "),
            "It reframes the question \u2014 ",
            ui.tags.em("P(B > A) instead of P(data | H\u2080)"),
            ". High probability \u2260 large effect: always check ",
            ui.tags.em("E[lift]"),
            " and ",
            ui.tags.em("E[loss]"),
            ".",
            class_="info-banner-text",
        ),

        # Truth
        ui.div(
            ui.div(
                ui.input_numeric(
                    "bys3_true_pA",
                    ui.TagList("True p_A",
                               tip("Underlying conversion rate for A.")),
                    value=0.10, min=0.001, max=0.999, step=0.001,
                    width="100%",
                ),
            ),
            ui.div(
                ui.input_numeric(
                    "bys3_true_pB",
                    ui.TagList("True p_B",
                               tip("Underlying conversion rate for B.")),
                    value=0.11, min=0.001, max=0.999, step=0.001,
                    width="100%",
                ),
            ),
            class_="group-params-cols",
        ),

        # n per batch
        ui.div(
            ui.div(
                ui.input_numeric(
                    "bys3_nA",
                    ui.TagList("n_A per batch",
                               tip("Trials added to A each Sample click.")),
                    value=2000, min=10, step=100, width="100%",
                ),
            ),
            ui.div(
                ui.input_numeric(
                    "bys3_nB",
                    ui.TagList("n_B per batch",
                               tip("Trials added to B each Sample click.")),
                    value=2000, min=10, step=100, width="100%",
                ),
            ),
            class_="group-params-cols",
        ),

        # Shared prior
        ui.div(
            ui.div(
                ui.input_numeric(
                    "bys3_alpha",
                    ui.TagList("Prior \u03b1",
                               tip("Shared prior successes for both variants.")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
            ),
            ui.div(
                ui.input_numeric(
                    "bys3_beta",
                    ui.TagList("Prior \u03b2",
                               tip("Shared prior failures for both variants.")),
                    value=1.0, min=0.1, step=0.5, width="100%",
                ),
            ),
            class_="group-params-cols",
        ),

        ui.input_slider(
            "bys3_threshold",
            ui.TagList("Decision threshold",
                       tip("Call the winner when P(B > A) \u2265 threshold "
                           "or \u2264 1 \u2212 threshold.")),
            min=0.80, max=0.99, value=0.95, step=0.01, width="100%",
        ),

        ui.input_numeric(
            "bys3_draws",
            ui.TagList("MC draws",
                       tip("Posterior Monte-Carlo draws per metric update. "
                           "Seed is frozen between samples for stable UI.")),
            value=20000, min=1000, step=1000, width="100%",
        ),

        # Presets
        ui.tags.label(
            "Presets",
            style="font-weight:500; color:var(--c-text3); "
                  "font-size:0.82rem; margin-bottom:2px;",
        ),
        ui.div(
            ui.input_action_button("bys3_pre_no_effect", "No effect",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys3_pre_small",     "Small lift",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys3_pre_large",     "Large lift",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys3_pre_early",     "Early stop",
                                   class_="btn-ctrl btn-preset"),
            ui.input_action_button("bys3_pre_tiny_lift", "High prob, tiny lift",
                                   class_="btn-ctrl btn-preset"),
            class_="np-preset-grid",
        ),
        ui.output_ui("bys3_preset_desc"),

        # Sample controls
        ui.div(
            ui.div(
                ui.tags.span("Sample:", class_="btn-row-label"),
                ui.input_action_button("bys3_sample_1",   "\u00d71",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("bys3_sample_50",  "\u00d750",
                                       class_="btn-ctrl btn-sample btn-flex"),
                ui.input_action_button("bys3_sample_100", "\u00d7100",
                                       class_="btn-ctrl btn-sample btn-flex"),
                class_="sidebar-btn-row",
            ),
            ui.div(
                ui.div(
                    ui.tags.label("Speed"),
                    ui.input_action_button("bys3_speed_minus", "\u2212",
                                           class_="btn-ctrl btn-pm"),
                    ui.input_action_button("bys3_btn_play", "Play",
                                           class_="btn-ctrl btn-play btn-flex"),
                    ui.input_action_button("bys3_speed_plus", "+",
                                           class_="btn-ctrl btn-pm"),
                    class_="ctrl-group ctrl-group-full",
                ),
                class_="sidebar-btn-row",
            ),
            ui.div(
                ui.input_action_button("bys3_reset", "Reset",
                                       class_="btn-ctrl btn-reset btn-full"),
                class_="sidebar-btn-row",
            ),
            class_="sidebar-controls",
        ),

        class_="bys-sidebar-group", **{"data-bys-tab": "3"},
    )


# ── Sub-tab content (main-panel side) ───────────────────────────────────────

def _main_tab1() -> ui.Tag:
    return ui.nav_panel(
        "Beta-Binomial Updating",

        ui.output_ui("bys1_stats_row"),

        ui.div(
            ui.div(
                ui.div(
                    ui.output_ui("bys1_chart1_title"),
                    ui.output_ui("bys1_prior_posterior"),
                    class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div("COIN FLIPS \u00b7 RUNNING p\u0302",
                           class_="card-title"),
                    ui.output_ui("bys1_coin_sequence"),
                    class_="glass-card chart-card",
                ),
                class_="charts-col-left",
            ),
            ui.div(
                ui.div(
                    ui.div("POSTERIOR EVOLUTION (95% CrI)",
                           class_="card-title"),
                    ui.output_ui("bys1_evolution_forest"),
                    class_="glass-card chart-card",
                ),
                class_="charts-col-right",
            ),
            class_="charts-area",
        ),
    )


def _main_tab2() -> ui.Tag:
    return ui.nav_panel(
        "Freq vs Bayes CI",

        ui.output_ui("bys2_stats_row"),

        ui.div(
            ui.tags.strong("Coverage \u2260 Credibility. "),
            "Bayesian coverage depends on the prior and true parameter \u2014 ",
            ui.tags.em("a Bayesian interval is not designed to have fixed "
                       "frequentist coverage"),
            ". It answers a different question: ",
            ui.tags.em("\u201cgiven this prior and data, where is the "
                       "parameter?\u201d"),
            class_="bys-banner",
        ),

        ui.div(
            ui.div(
                ui.div(
                    ui.div("FOREST: CI vs CrI  (last 50)",
                           class_="card-title"),
                    ui.output_ui("bys2_forest"),
                    class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div("WIDTH DISTRIBUTION", class_="card-title"),
                    ui.output_ui("bys2_width_hist"),
                    class_="glass-card chart-card",
                ),
                class_="charts-col-left",
            ),
            ui.div(
                ui.div(
                    ui.div("RUNNING COVERAGE", class_="card-title"),
                    ui.output_ui("bys2_running_coverage"),
                    class_="glass-card chart-card",
                ),
                class_="charts-col-right",
            ),
            class_="charts-area",
        ),
    )


def _main_tab3() -> ui.Tag:
    return ui.nav_panel(
        "A/B Testing",

        ui.output_ui("bys3_stats_row"),

        ui.div(
            ui.tags.strong("High probability \u2260 large effect. "),
            "P(B > A) tells you ", ui.tags.em("who"), " is better, not ",
            ui.tags.em("by how much"),
            ". Read E[lift] and E[loss] alongside to see the ",
            ui.tags.em("size"), " of the difference.",
            class_="bys-banner",
        ),

        ui.div(
            ui.div(
                ui.div(
                    ui.div("POSTERIOR DENSITIES", class_="card-title"),
                    ui.output_ui("bys3_posteriors"),
                    class_="glass-card chart-card",
                ),
                ui.div(
                    ui.div("P(B > A) EVOLUTION", class_="card-title"),
                    ui.output_ui("bys3_prob_evolution"),
                    class_="glass-card chart-card",
                ),
                class_="charts-col-left",
            ),
            ui.div(
                ui.div(
                    ui.div("JOINT POSTERIOR  (p_A vs p_B)",
                           class_="card-title"),
                    ui.output_ui("bys3_joint"),
                    class_="glass-card chart-card",
                ),
                class_="charts-col-right",
            ),
            class_="charts-area",
        ),

        ui.div(
            ui.div("CLASSICAL vs BAYESIAN", class_="card-title"),
            ui.output_ui("bys3_classical"),
            class_="glass-card",
            style="margin-top:0.5rem; padding:0.75rem 1rem;",
        ),
    )


# ── JS: toggle sidebar group visibility based on active sub-tab ─────────────

_BYS_TAB_SCRIPT = ui.tags.script("""
    (function() {
        function byTab(label) {
            if (label.indexOf('Beta-Binomial') !== -1) return '1';
            if (label.indexOf('Freq') !== -1)          return '2';
            if (label.indexOf('A/B') !== -1)           return '3';
            return '1';
        }
        function sync(active) {
            document.querySelectorAll('.bys-sidebar-group').forEach(function(el) {
                el.style.display = (el.getAttribute('data-bys-tab') === active)
                    ? '' : 'none';
            });
        }
        // Initial pick after DOM ready
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                var active = document.querySelector(
                    '#bys_subtabs .nav-link.active, a.nav-link.active[data-bs-toggle]'
                );
                sync(byTab(active ? active.textContent : 'Beta-Binomial'));
            }, 50);
        });
        // React to Bootstrap tab changes
        document.addEventListener('shown.bs.tab', function(e) {
            var t = e.target;
            if (!t) return;
            var container = t.closest('#bys_subtabs');
            if (!container) return;
            sync(byTab(t.textContent || ''));
        });
    })();
""")


# ── Nav panel ───────────────────────────────────────────────────────────────

def bayes_panel() -> ui.Tag:
    return ui.nav_panel(
        "Bayesian Explorer",

        _BYS_TAB_SCRIPT,

        ui.div(
            # ── Single outer sidebar ────────────────────────────────────────
            ui.div(
                _sidebar_tab1(),
                _sidebar_tab2(),
                _sidebar_tab3(),

                # Footer
                ui.div(
                    ui.tags.a("LinkedIn",
                              href="https://www.linkedin.com/in/ihormiroshnychenko/",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Telegram", href="https://t.me/araprof",
                              target="_blank"),
                    " \u2022 ",
                    ui.tags.a("Website", href="https://aranaur.rbind.io/",
                              target="_blank"),
                    class_="footer-links",
                ),

                class_="sidebar bys-sidebar",
            ),

            # ── Main panel with nested navset ───────────────────────────────
            ui.div(
                ui.navset_underline(
                    _main_tab1(),
                    _main_tab2(),
                    _main_tab3(),
                    id="bys_subtabs",
                ),
                class_="main-panel",
            ),

            class_="app-body",
        ),
    )
