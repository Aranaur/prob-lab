import plotly.graph_objects as go
from plots import draw_ci_plot, draw_population_plot

def test_draw_ci_plot_smoke():
    history = [
        {"id": 1, "estimate": 0.1, "lower": -0.5, "upper": 0.7, "covered": True, "width": 1.2},
        {"id": 2, "estimate": 2.1, "lower": 1.5, "upper": 2.7, "covered": False, "width": 1.2},
    ]
    
    fig = draw_ci_plot(history, true_val=0.0, sigma=1.0, n=30, method="t", statistic="mean", p_level=25, dark=True)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_draw_population_plot_smoke():
    params = {"mu": 0.0, "sigma": 1.0}
    sample = [0.1, -0.2, 0.5, 1.2, -0.8]
    
    fig = draw_population_plot("normal", params, sample, true_val=0.0, statistic="mean", dark=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
