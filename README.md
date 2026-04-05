---
title: Prob Lab
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Prob Lab

An interactive educational platform for exploring probability theory and statistical concepts through simulation. Built with **Shiny for Python**, deployed on Hugging Face Spaces via Docker.

## Modules

### 🎯 CI Explorer
Visualize confidence interval coverage in real time. Sample from multiple distributions, compare CI methods (t, z, bootstrap), and see how the Central Limit Theorem drives coverage.

**Controls:** confidence level, population distribution, CI method, sample size, animation speed.  
**Charts:** CI intervals, proportion covering μ, CI width distribution, sample means (CLT).

### 🔬 p-value Explorer
Simulate hypothesis tests repeatedly and watch p-values accumulate. Understand Type I/II errors, the null distribution, and statistical power through direct observation.

**Controls:** test structure (one-sample, two-sample, paired), test method (t / z), alternative hypothesis, μ₀, true μ, σ, n, α.  
**Charts:** null distribution with rejection region, p-value histogram, power diagram.

### ⚡ Power Explorer
Design studies and understand the relationship between effect size, sample size, significance level, and statistical power. Based on exact power formulas with a "Solve for" mode.

**Controls:** solve for (Power / n / d / α), test type (Z, one-sample t, two-sample t, paired t), alternative hypothesis, Cohen's d, n, α, power — plus preset scenarios.  
**Charts:** H₀/H₁ sampling distributions with α / β / power regions, power curve (power vs n).

**Presets:** Clinical trial · A/B test · Psychology study · Small effect detection.

## Features

- Dark / light theme toggle
- Responsive layout (desktop & mobile)
- All plots rendered with Plotly (interactive hover, zoom)
- MathJax for LaTeX formulas

## Stack

- [Shiny for Python](https://shiny.posit.co/py/)
- [Plotly](https://plotly.com/python/)
- [SciPy](https://scipy.org/)
- Docker

## Run locally

```bash
uv sync
shiny run app.py --host 0.0.0.0 --port 7860
```

Or with Docker:

```bash
docker build -t prob-lab .
docker run -p 7860:7860 prob-lab
```
