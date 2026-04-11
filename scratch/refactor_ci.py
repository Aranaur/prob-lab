import os

def replace_in_file(filepath, replacements):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# 1. ui_layout.py replacements
ui_replacements = [
    ('"conf_pct"', '"ci_conf_pct"'),
    ('"conf_pct2"', '"ci_conf_pct2"'),
    ('"conf_pct3"', '"ci_conf_pct3"'),
    ('"conf_level"', '"ci_conf_level"'),
    ('"pop_dist"', '"ci_pop_dist"'),
    ('"dynamic_params"', '"ci_dynamic_params"'),
    ('"btn_sample_1"', '"ci_btn_sample_1"'),
    ('"btn_sample_50"', '"ci_btn_sample_50"'),
    ('"btn_sample_100"', '"ci_btn_sample_100"'),
    ('"speed_minus"', '"ci_speed_minus"'),
    ('"btn_play"', '"ci_btn_play"'),
    ('"speed_plus"', '"ci_speed_plus"'),
    ('"btn_reset"', '"ci_btn_reset"'),
    ('"cov_rate"', '"ci_cov_rate"'),
    ('"stat_label_inc"', '"ci_stat_label_inc"'),
    ('"num_covered"', '"ci_num_covered"'),
    ('"stat_label_miss"', '"ci_stat_label_miss"'),
    ('"num_missed"', '"ci_num_missed"'),
    ('"num_total"', '"ci_num_total"'),
    ('"stat_plot_title"', '"ci_stat_plot_title"'),
    ('"means_plot"', '"ci_means_plot"'),
    ('"prop_plot_title"', '"ci_prop_plot_title"'),
    ('"prop_plot"', '"ci_prop_plot"'),
    ('"width_plot"', '"ci_width_plot"'),
    ('"population_plot"', '"ci_population_plot"'),
    ('"Variance Reduction"', '"Variance Reduction Explorer"'),
    ("getElementById('btn_play')", "getElementById('ci_btn_play')"),
    ('"CI COVERAGE "', '"CI COVERAGE\\u00a0"'),
    ('"SAMPLES DRAWN "', '"TOTAL EXPERIMENTS\\u00a0"'),
    ('"sample_size"', '"ci_sample_size"'),
    ('"pop_mean"', '"ci_pop_mean"'),
    ('"pop_sd"', '"ci_pop_sd"'),
    ('"pop_min"', '"ci_pop_min"'),
    ('"pop_max"', '"ci_pop_max"'),
    ('"pop_lambda"', '"ci_pop_lambda"'),
    ('"lnorm_mu"', '"ci_lnorm_mu"'),
    ('"lnorm_sigma"', '"ci_lnorm_sigma"'),
    ('"pois_lam"', '"ci_pois_lam"'),
    ('"binom_n"', '"ci_binom_n"'),
    ('"binom_p"', '"ci_binom_p"'),
]
replace_in_file('ui_layout.py', ui_replacements)

# 2. server.py replacements
server_replacements = [
    # Function names (renderers)
    ('def conf_pct(', 'def ci_conf_pct('),
    ('def conf_pct2(', 'def ci_conf_pct2('),
    ('def conf_pct3(', 'def ci_conf_pct3('),
    ('def cov_rate(', 'def ci_cov_rate('),
    ('def num_covered(', 'def ci_num_covered('),
    ('def num_missed(', 'def ci_num_missed('),
    ('def num_total(', 'def ci_num_total('),
    ('def stat_label_inc(', 'def ci_stat_label_inc('),
    ('def stat_label_miss(', 'def ci_stat_label_miss('),
    ('def stat_plot_title(', 'def ci_stat_plot_title('),
    ('def prop_plot_title(', 'def ci_prop_plot_title('),
    ('def means_plot(', 'def ci_means_plot('),
    ('def prop_plot(', 'def ci_prop_plot('),
    ('def width_plot(', 'def ci_width_plot('),
    ('def population_plot(', 'def ci_population_plot('),
    ('def dynamic_params(', 'def ci_dynamic_params('),
    
    # Input usages
    ('input.conf_level', 'input.ci_conf_level'),
    ('input.pop_dist', 'input.ci_pop_dist'),
    ('input.sample_size', 'input.ci_sample_size'),
    ('input.pop_mean', 'input.ci_pop_mean'),
    ('input.pop_sd', 'input.ci_pop_sd'),
    ('input.pop_min', 'input.ci_pop_min'),
    ('input.pop_max', 'input.ci_pop_max'),
    ('input.pop_lambda', 'input.ci_pop_lambda'),
    ('input.lnorm_mu', 'input.ci_lnorm_mu'),
    ('input.lnorm_sigma', 'input.ci_lnorm_sigma'),
    ('input.pois_lam', 'input.ci_pois_lam'),
    ('input.binom_n', 'input.ci_binom_n'),
    ('input.binom_p', 'input.ci_binom_p'),
    ('input.n_minus', 'input.ci_n_minus'),
    ('input.n_plus', 'input.ci_n_plus'),
    ('input.speed_minus', 'input.ci_speed_minus'),
    ('input.speed_plus', 'input.ci_speed_plus'),
    ('input.btn_play', 'input.ci_btn_play'),
    ('input.btn_sample_1', 'input.ci_btn_sample_1'),
    ('input.btn_sample_50', 'input.ci_btn_sample_50'),
    ('input.btn_sample_100', 'input.ci_btn_sample_100'),
    ('input.btn_reset', 'input.ci_btn_reset'),
    
    # Text updates in server.py
    ('"btn_play"', '"ci_btn_play"'),
]
replace_in_file('server.py', server_replacements)

# 3. Label standardizations across UI files
label_repls = [
    ('TOTAL TESTS ', 'TOTAL EXPERIMENTS\\u00a0'),
    ('TOTAL TESTS\\u00a0', 'TOTAL EXPERIMENTS\\u00a0'),
    ('EXPERIMENTS\\u00a0', 'TOTAL EXPERIMENTS\\u00a0'),
    ('REJECT RATE ', 'REJECT RATE\\u00a0'),
    ('EFFECT SIZE (d) ', 'EFFECT SIZE (d)\\u00a0'),
    ('SAMPLE SIZE (n) ', 'SAMPLE SIZE (n)\\u00a0'),
    ('\\u03b1 (TYPE I) ', '\\u03b1 (TYPE I)\\u00a0'),
    ('POWER (1\\u2212\\u03b2) ', 'POWER (1\\u2212\\u03b2)\\u00a0'),
]

for f in ['gof_ui.py', 'mt_ui.py', 'np_ui.py', 'power_ui.py', 'pvalue_ui.py', 'seq_ui.py', 'vr_server.py']:
    if os.path.exists(f):
        replace_in_file(f, label_repls)
        
# For Card Titles
card_title_repls = [
    ('"p-VALUE HISTOGRAM"', '"p-VALUE DISTRIBUTION"'),
    ('\"NULL DISTRIBUTION & TEST STATISTIC\"', '\"NULL DISTRIBUTION & TEST STATISTIC\"'),
]
for f in ['gof_ui.py', 'mt_ui.py', 'np_ui.py', 'power_ui.py', 'pvalue_ui.py', 'seq_ui.py', 'vr_ui.py', 'ui_layout.py', 'pvalue_server.py']:
    if os.path.exists(f):
        replace_in_file(f, card_title_repls)

print("ID and Label Refactoring Complete!")
