"""Reusable dashboard components: Plotly charts with a consistent dark theme."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


ACCENT = "#00D9FF"
SUCCESS = "#00D98F"
DANGER = "#FF4E6B"
WARNING = "#FFB547"
MUTED = "#7A869A"
BG = "#0E1117"
PAPER = "#1C2130"


def _style_layout(fig: go.Figure, title: Optional[str] = None, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=ACCENT, size=16)) if title else None,
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(family="Inter, sans-serif", size=13, color="#F0F2F6"),
        xaxis=dict(gridcolor="#2B3142", zerolinecolor="#2B3142"),
        yaxis=dict(gridcolor="#2B3142", zerolinecolor="#2B3142"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.1),
        height=height,
    )
    return fig


def equity_chart(
    equities: Dict[str, pd.Series],
    title: str = "Equity Curve",
    benchmark_name: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    palette = [ACCENT, SUCCESS, WARNING, DANGER, "#B794F4", "#FFD66B", "#4FD1C5", "#F687B3"]
    for i, (name, eq) in enumerate(equities.items()):
        is_bench = name == benchmark_name
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            name=name,
            mode="lines",
            line=dict(
                color="#FFFFFF" if is_bench else palette[i % len(palette)],
                width=2 if is_bench else 2.5,
                dash="dash" if is_bench else "solid",
            ),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
    return _style_layout(fig, title, height=450)


def drawdown_chart(equity: pd.Series, title: str = "Drawdown") -> go.Figure:
    dd = equity / equity.cummax() - 1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        fill="tozeroy",
        line=dict(color=DANGER, width=1),
        fillcolor="rgba(255, 78, 107, 0.25)",
        name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_yaxes(ticksuffix="%")
    return _style_layout(fig, title, height=280)


def returns_histogram(returns: pd.Series, title: str = "Return Distribution") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=60,
        marker=dict(color=ACCENT, line=dict(width=0)),
        name="Returns",
    ))
    mean_v = float(returns.mean() * 100)
    fig.add_vline(x=mean_v, line=dict(color=SUCCESS, dash="dash"),
                  annotation_text=f"μ={mean_v:.3f}%", annotation_position="top")
    fig.update_xaxes(ticksuffix="%")
    return _style_layout(fig, title, height=300)


def rolling_metric_chart(series: pd.Series, title: str, y_suffix: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines",
        line=dict(color=ACCENT, width=2),
        name=title,
    ))
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    return _style_layout(fig, title, height=280)


def heatmap(df: pd.DataFrame, title: str = "Correlation") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale=[[0, DANGER], [0.5, PAPER], [1, SUCCESS]],
        zmid=0,
        hovertemplate="<b>%{x}</b> × <b>%{y}</b>: %{z:.2f}<extra></extra>",
    ))
    return _style_layout(fig, title, height=400)


def weights_bar(weights: pd.DataFrame, title: str = "Portfolio Weights") -> go.Figure:
    palette = [ACCENT, SUCCESS, WARNING, DANGER, "#B794F4", "#4FD1C5"]
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_trace(go.Bar(
            x=weights.index, y=weights[col],
            name=col,
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(barmode="group")
    return _style_layout(fig, title, height=320)


def efficient_frontier_chart(ef: pd.DataFrame, assets: Optional[pd.DataFrame] = None,
                              tangent: Optional[tuple] = None,
                              title: str = "Efficient Frontier") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ef["risk"], y=ef["return"],
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=ACCENT, size=6),
        name="Efficient Frontier",
        hovertemplate="Risk: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>",
    ))
    if assets is not None:
        fig.add_trace(go.Scatter(
            x=assets["vol"], y=assets["ret"],
            mode="markers+text",
            marker=dict(color=WARNING, size=12, symbol="diamond"),
            text=assets.index,
            textposition="top right",
            textfont=dict(color=WARNING),
            name="Individual assets",
        ))
    if tangent is not None:
        risk, ret = tangent
        fig.add_trace(go.Scatter(
            x=[risk], y=[ret],
            mode="markers+text",
            marker=dict(color=SUCCESS, size=16, symbol="star"),
            text=["Max Sharpe"], textposition="middle right",
            name="Max Sharpe",
        ))
    return _style_layout(fig, title, height=460)


def greeks_gauge(greeks: Dict[str, float]) -> go.Figure:
    """Gauge-style cards for delta/gamma/vega/theta/rho."""
    fig = make_subplots(rows=1, cols=5, specs=[[{"type": "indicator"}] * 5],
                         subplot_titles=["Delta", "Gamma", "Vega", "Theta", "Rho"])
    configs = [
        ("delta", -1, 1, ACCENT),
        ("gamma", 0, 0.1, SUCCESS),
        ("vega", -100, 100, WARNING),
        ("theta", -50, 50, DANGER),
        ("rho", -100, 100, "#B794F4"),
    ]
    for i, (k, lo, hi, col) in enumerate(configs, 1):
        v = greeks.get(k, 0)
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=v,
            gauge={"axis": {"range": [lo, hi]}, "bar": {"color": col}},
            number={"font": {"color": col, "size": 28}},
        ), row=1, col=i)
    return _style_layout(fig, height=220)


def iv_smile_chart(strikes, injected, recovered) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=strikes, y=[v * 100 for v in injected],
                              mode="lines+markers",
                              line=dict(color=MUTED, dash="dot"),
                              name="Injected IV"))
    fig.add_trace(go.Scatter(x=strikes, y=[v * 100 for v in recovered],
                              mode="lines+markers",
                              line=dict(color=ACCENT, width=3),
                              name="Recovered IV"))
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(title="Strike")
    return _style_layout(fig, "Implied Volatility Smile", height=380)
