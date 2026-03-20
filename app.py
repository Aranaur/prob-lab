# =============================================================================
# Confidence Intervals Explorer - Interactive Shiny for Python App
# Inspired by https://rpsychologist.com/d3/ci/
# =============================================================================

import numpy as np
from scipy import stats
from shiny import App, reactive, render, ui
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import asyncio
from pathlib import Path

# ─── Matplotlib dark style ───────────────────────────────────────────────────
DARK_BG = "#1e293b00"  # transparent
GRID_COLOR = "#334155"
AXIS_COLOR = "#94a3b8"
LABEL_COLOR = "#cbd5e1"

def dark_style(ax: plt.Axes):
    """Apply consistent dark theme to a matplotlib Axes."""
    ax.set_facecolor(DARK_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=AXIS_COLOR, labelsize=8)
    ax.xaxis.label.set_color(LABEL_COLOR)
    ax.yaxis.label.set_color(LABEL_COLOR)
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.3, alpha=0.6)


# =============================================================================
# UI
# =============================================================================
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(rel="stylesheet", href="style.css")
    ),

    # JS for toggling play / pause button class
    ui.tags.head(ui.tags.script("""
        Shiny.addCustomMessageHandler('togglePlayClass', function(msg) {
            var btn = document.getElementById('btn_play');
            if (msg.playing) {
                btn.classList.remove('btn-play');
                btn.classList.add('btn-pause');
            } else {
                btn.classList.remove('btn-pause');
                btn.classList.add('btn-play');
            }
        });
    """)),

    # Title
    ui.tags.h1("Interpreting Confidence Intervals", class_="main-title"),

    # Top controls row 1
    ui.div(
        ui.input_slider("conf_level", "Confidence Level (%)",
                        min=50, max=99, value=95, step=1, width="100%"),
        ui.input_select("pop_dist", "Population Distribution", 
                        choices={"normal": "Normal", "uniform": "Uniform", "exponential": "Exponential (Right-skewed)"},
                        selected="normal", width="100%"),
        class_="slider-row",
    ),

    # Top controls row 2
    ui.div(
        ui.input_numeric("pop_mean", "Population Mean (\u03bc)", value=0.0, step=0.5, width="100%"),
        ui.input_numeric("pop_sd", "Population Std Dev (\u03c3)", value=1.0, min=0.1, step=0.5, width="100%"),
        class_="slider-row",
    ),

    # Stats cards
    ui.div(
        ui.div(
            ui.div("CI COVERAGE", class_="stat-label"),
            ui.div(ui.output_text("cov_rate", inline=True), class_="stat-value coverage"),
            class_="stat-card",
        ),
        ui.div(
            ui.div("\u03bc INCLUDED", class_="stat-label"),
            ui.div(ui.output_text("num_covered", inline=True), class_="stat-value included"),
            class_="stat-card",
        ),
        ui.div(
            ui.div("\u03bc MISSED", class_="stat-label"),
            ui.div(ui.output_text("num_missed", inline=True), class_="stat-value missed"),
            class_="stat-card",
        ),
        ui.div(
            ui.div("SAMPLES DRAWN", class_="stat-label"),
            ui.div(ui.output_text("num_total", inline=True), class_="stat-value total"),
            class_="stat-card",
        ),
        class_="stats-row",
    ),

    # Two-panel plots
    ui.div(
        # Left panel – proportion + width dist
        ui.div(
            ui.div(
                ui.div("PROPORTION OF CIs INCLUDING \u03bc", class_="card-title"),
                ui.output_plot("prop_plot", height="190px"),
                class_="glass-card",
            ),
            ui.div(
                ui.div("CI WIDTH DISTRIBUTION", class_="card-title"),
                ui.output_plot("width_plot", height="190px"),
                class_="glass-card",
            ),
            class_="panel-left",
        ),
        # Right panel – main CI chart
        ui.div(
            ui.div(
                ui.div("CONFIDENCE INTERVALS", class_="card-title"),
                ui.output_plot("ci_plot", height="420px"),
                class_="glass-card",
            ),
            class_="panel-right",
        ),
        class_="plot-wrapper",
    ),

    # Controls row
    ui.div(
        ui.div(
            ui.tags.label("Sample size"),
            ui.input_action_button("n_minus", "\u2212", class_="btn-ctrl btn-pm"),
            ui.input_numeric("sample_size", label="", value=5, min=2, max=500, step=1, width="44px"),
            ui.input_action_button("n_plus", "+", class_="btn-ctrl btn-pm"),
            class_="ctrl-group",
            style="margin-left: auto;"
        ),
        ui.input_action_button("btn_sample_1", "Sample \u00d71", class_="btn-ctrl btn-sample"),
        ui.input_action_button("btn_sample_50", "Sample \u00d750", class_="btn-ctrl btn-sample"),
        ui.div(
            ui.tags.label("Speed"),
            ui.input_action_button("speed_minus", "\u2212", class_="btn-ctrl btn-pm"),
            ui.input_action_button("btn_play", "Play", class_="btn-ctrl btn-play"),
            ui.input_action_button("speed_plus", "+", class_="btn-ctrl btn-pm"),
            class_="ctrl-group",
            style="margin-left: auto;"
        ),
        ui.input_action_button("btn_reset", "Reset", class_="btn-ctrl btn-reset"),
        class_="controls-row",
    ),

    # Footer links
    ui.div(
        ui.tags.a("LinkedIn", href="https://www.linkedin.com/in/ihormiroshnychenko/", target="_blank"),
        " \u2022 ",
        ui.tags.a("Telegram", href="https://t.me/araprof", target="_blank"),
        " \u2022 ",
        ui.tags.a("Website", href="https://aranaur.rbind.io/", target="_blank"),
        class_="footer-links",
    ),
)


# =============================================================================
# SERVER
# =============================================================================
def server(input, output, session):

    MAX_DISPLAY = 50

    # ── Reactive state ──
    total_drawn = reactive.value(0)
    total_covered = reactive.value(0)
    # Each entry: dict(id, mean, lower, upper, covered, width)
    history = reactive.value([])
    all_widths = reactive.value([])
    prop_x = reactive.value([])
    prop_y = reactive.value([])
    is_playing = reactive.value(False)
    speed_ms = reactive.value(0.5)  # seconds

    # ── Sample size +/- ──
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

    # ── Speed +/- ──
    @reactive.effect
    @reactive.event(input.speed_minus)
    def _speed_down():
        speed_ms.set(min(speed_ms() + 0.05, 1.0))

    @reactive.effect
    @reactive.event(input.speed_plus)
    def _speed_up():
        speed_ms.set(max(speed_ms() - 0.05, 0.05))

    # ── Play / Pause ──
    @reactive.effect
    @reactive.event(input.btn_play)
    def _toggle_play():
        is_playing.set(not is_playing())
        playing = is_playing()
        if playing:
            ui.update_action_button("btn_play", label="Pause")
        else:
            ui.update_action_button("btn_play", label="Play")

    # ── Continuous animation ──
    @reactive.effect
    def _auto_draw():
        if is_playing():
            # Invalidate this effect so it runs again after speed_ms seconds
            reactive.invalidate_later(speed_ms())
            # Safely draw one sample
            with reactive.isolate():
                draw_samples(1)

    # ── Manual buttons ──
    @reactive.effect
    @reactive.event(input.btn_sample_1)
    def _sample_1():
        draw_samples(1)

    @reactive.effect
    @reactive.event(input.btn_sample_50)
    def _sample_50():
        draw_samples(50)

    # ── Reset ──
    @reactive.effect
    @reactive.event(input.btn_reset, input.pop_dist)
    def _reset():
        total_drawn.set(0)
        total_covered.set(0)
        history.set([])
        all_widths.set([])
        prop_x.set([])
        prop_y.set([])
        is_playing.set(False)
        ui.update_action_button("btn_play", label="Play")

    # ── Core sampling ──
    def draw_samples(k: int):
        n = input.sample_size()
        if n is None or n < 2:
            n = 5
        n = int(n)
        conf = input.conf_level() / 100.0

        current_drawn = total_drawn()
        current_covered = total_covered()

        mu = float(input.pop_mean() or 0.0)
        sigma = float(input.pop_sd() or 1.0)
        if sigma <= 0:
            sigma = 1e-6

        # Vectorized data generation
        # shape: (n, k) -> we take mean along axis 0
        dist_choice = input.pop_dist()
        if dist_choice == "normal":
            samps = np.random.normal(mu, sigma, size=(n, k))
        elif dist_choice == "uniform":
            limit = np.sqrt(3) * sigma
            samps = np.random.uniform(mu - limit, mu + limit, size=(n, k))
        elif dist_choice == "exponential":
            samps = (np.random.exponential(1.0, size=(n, k)) - 1.0) * sigma + mu
        else:
            samps = np.random.normal(mu, sigma, size=(n, k))
        means = np.mean(samps, axis=0)
        stds = np.std(samps, axis=0, ddof=1)
        ses = stds / np.sqrt(n)
        
        tc = float(stats.t.ppf(1 - (1 - conf) / 2, df=n - 1))
        
        los = means - tc * ses
        his = means + tc * ses
        ws = his - los
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

        # Update visible history buffer
        hist = history() + new_entries
        if len(hist) > MAX_DISPLAY:
            hist = hist[-MAX_DISPLAY:]
        history.set(hist)

        # All widths
        all_widths.set(all_widths() + [e["width"] for e in new_entries])

        # Running proportion
        prop_x.set(prop_x() + [current_drawn])
        prop_y.set(prop_y() + [current_covered / current_drawn])

    # ── Text outputs ──
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

    # ── Main CI plot ──
    @render.plot(alt="Confidence Intervals")
    def ci_plot():
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=DARK_BG)
        dark_style(ax)

        n = input.sample_size()
        if n is None or n < 2:
            n = 5
        
        mu = float(input.pop_mean() or 0.0)
        sigma = float(input.pop_sd() or 1.0)
        if sigma <= 0:
            sigma = 1e-6

        se_theory = sigma / np.sqrt(n)
        x_lo = mu - 5 * se_theory
        x_hi = mu + 5 * se_theory

        # Population mean line
        ax.axvline(mu, color="#f59e0b", linewidth=1.1, linestyle="--", zorder=5)

        dat = history()
        if len(dat) == 0:
            ax.text(mu, 0.5, "Press Sample or Play to begin",
                    ha="center", va="center", color="#64748b", fontsize=12)
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Sample mean", fontsize=10)
            ax.set_yticks([])
            ax.grid(axis="y", visible=False)
            fig.tight_layout()
            plt.close(fig)
            return fig

        for idx, entry in enumerate(dat):
            y = idx + 1
            color = "#94a3b8" if entry["covered"] else "#f87171"
            ax.plot([entry["lower"], entry["upper"]], [y, y],
                    color=color, linewidth=1.0, solid_capstyle="round", zorder=2)
            ax.plot(entry["mean"], y, "o",
                    color="#38bdf8", markersize=4, alpha=0.9, zorder=3)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(0.5, len(dat) + 0.5)
        ax.set_xlabel("Sample mean", fontsize=10)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        plt.close(fig)
        return fig

    # ── Proportion line chart ──
    @render.plot(alt="Proportion of CIs including mu")
    def prop_plot():
        fig, ax = plt.subplots(figsize=(4, 1.9), facecolor=DARK_BG)
        dark_style(ax)

        conf_target = input.conf_level() / 100.0
        ax.axhline(conf_target, color="#38bdf8", linewidth=0.8,
                   linestyle="--", zorder=3)

        px = prop_x()
        py = prop_y()

        if len(px) == 0:
            ax.set_xlim(0, 100)
            ax.set_ylim(max(0, conf_target - 0.2), 1.0)
        else:
            ax.plot(px, py, color="#e2e8f0", linewidth=0.9, zorder=4)
            y_min = max(0, min(min(py), conf_target) - 0.05)
            y_max = min(1, max(max(py), conf_target) + 0.05)
            ax.set_ylim(y_min, y_max)

        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Samples drawn", fontsize=9)
        fig.tight_layout()
        plt.close(fig)
        return fig

    # ── CI width distribution ──
    @render.plot(alt="CI Width Distribution")
    def width_plot():
        fig, ax = plt.subplots(figsize=(4, 1.9), facecolor=DARK_BG)
        dark_style(ax)

        widths = all_widths()
        if len(widths) < 3:
            ax.text(0.5, 0.5, "Collecting data\u2026",
                    ha="center", va="center", color="#64748b",
                    fontsize=11, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.hist(widths, bins="auto", density=True,
                    color="#818cf8", alpha=0.5, edgecolor="#a5b4fc", linewidth=0.6)
            # KDE overlay
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(widths)
            xs = np.linspace(min(widths), max(widths), 200)
            ax.plot(xs, kde(xs), color="#a5b4fc", linewidth=1.0)
            ax.set_xlabel("CI Width", fontsize=9)
            ax.set_yticks([])
        fig.tight_layout()
        plt.close(fig)
        return fig


# =============================================================================
# Run
# =============================================================================
css_dir = Path(__file__).parent / "css"

app = App(app_ui, server, static_assets=css_dir)
