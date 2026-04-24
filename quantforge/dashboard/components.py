"""Premium UI components — Bloomberg Terminal × Linear/Stripe aesthetic."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Design tokens
# =============================================================================
ACCENT = "#00D9FF"
ACCENT_GLOW = "rgba(0, 217, 255, 0.35)"
SUCCESS = "#00D98F"
DANGER = "#FF4E6B"
WARNING = "#FFB547"
PURPLE = "#B794F4"
MUTED = "#7A869A"
BG = "#0B0F19"
BG_LIGHT = "#0E1117"
PAPER = "#1C2130"
BORDER = "#2B3142"
TEXT = "#F0F2F6"
TEXT_DIM = "#9AA5B8"


_COLOR_CYCLE = [ACCENT, SUCCESS, PURPLE, WARNING, DANGER, "#FFD66B", "#4FD1C5", "#F687B3"]


def _style_layout(fig: go.Figure, title: str | None = None, height: int = 400,
                   show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:14px;color:{TEXT_DIM};letter-spacing:0.08em;'>{title.upper()}</span>"
            if title else None,
            x=0.0, xanchor="left", y=0.97, yanchor="top",
            pad=dict(l=4),
        ) if title else None,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color=TEXT),
        xaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
            tickfont=dict(color=TEXT_DIM, size=11),
        ),
        yaxis=dict(
            gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
            tickfont=dict(color=TEXT_DIM, size=11),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=PAPER, bordercolor=ACCENT, font_size=12),
        margin=dict(l=12, r=12, t=40 if title else 12, b=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", orientation="h",
            y=-0.18, x=0.5, xanchor="center",
            font=dict(color=TEXT_DIM, size=11),
        ) if show_legend else dict(),
        height=height,
        showlegend=show_legend,
    )
    return fig


# =============================================================================
# CSS injection — call once at top of app
# =============================================================================
GLOBAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

  html, body, [data-testid="stAppViewContainer"] {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
  }}
  [data-testid="stAppViewContainer"] > div {{
    background: radial-gradient(ellipse at top, rgba(0, 217, 255, 0.04) 0%, transparent 60%), {BG} !important;
  }}

  .block-container {{ padding-top: 1rem; padding-bottom: 4rem; max-width: 1480px; }}

  /* Headers */
  h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }}
  h1 {{ font-size: 2.2rem !important; font-weight: 700 !important; }}
  h2 {{ font-size: 1.6rem !important; color: {TEXT} !important; margin-top: 1.8rem !important; }}
  h3 {{ font-size: 1.1rem !important; color: {TEXT_DIM} !important; text-transform: uppercase; letter-spacing: 0.1em; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: #07090F !important;
    border-right: 1px solid {BORDER} !important;
    width: 268px !important;
  }}
  [data-testid="stSidebar"] > div {{ padding-top: 1.5rem; }}

  .sidebar-brand {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem; font-weight: 700;
    background: linear-gradient(90deg, {ACCENT}, {SUCCESS});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex; align-items: center; gap: 8px;
    padding: 0 1.5rem 0.1rem;
  }}
  .sidebar-tagline {{
    color: {TEXT_DIM}; font-size: 0.78rem; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 0 1.5rem 1.2rem;
  }}
  .sidebar-divider {{
    border: 0; border-top: 1px solid {BORDER};
    margin: 0 1.5rem 1rem !important;
  }}

  /* Radio buttons in sidebar (nav) */
  [data-testid="stSidebar"] [role="radiogroup"] {{ gap: 2px !important; padding: 0 0.75rem; }}
  [data-testid="stSidebar"] [role="radiogroup"] > label {{
    padding: 0.55rem 0.85rem !important;
    border-radius: 8px !important;
    transition: all 0.15s !important;
    cursor: pointer;
    border: 1px solid transparent;
  }}
  [data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
    background: rgba(0, 217, 255, 0.07) !important;
  }}
  [data-testid="stSidebar"] [role="radiogroup"] > label[data-checked="true"],
  [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
    background: rgba(0, 217, 255, 0.12) !important;
    border-left: 2px solid {ACCENT} !important;
    border-radius: 0 8px 8px 0 !important;
  }}

  /* KPI metric cards */
  [data-testid="stMetric"] {{
    background: linear-gradient(180deg, {PAPER} 0%, #161A26 100%);
    padding: 20px 22px;
    border-radius: 14px;
    border: 1px solid {BORDER};
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
  }}
  [data-testid="stMetric"]:hover {{
    border-color: rgba(0, 217, 255, 0.3);
    box-shadow: 0 0 0 1px rgba(0, 217, 255, 0.1), 0 8px 24px rgba(0, 217, 255, 0.05);
    transform: translateY(-1px);
  }}
  [data-testid="stMetricLabel"] {{
    color: {TEXT_DIM} !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
  }}
  [data-testid="stMetricValue"] {{
    color: {TEXT} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
  }}
  [data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
  }}

  /* Buttons */
  .stButton > button {{
    background: linear-gradient(135deg, {ACCENT} 0%, #0099DD 100%) !important;
    color: {BG} !important;
    font-weight: 600 !important;
    border: 0 !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 0 0 {ACCENT_GLOW};
  }}
  .stButton > button:hover {{
    box-shadow: 0 0 20px 2px {ACCENT_GLOW} !important;
    transform: translateY(-1px);
  }}
  .stDownloadButton > button {{
    background: {PAPER} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
  }}
  .stDownloadButton > button:hover {{
    border-color: {ACCENT} !important; color: {ACCENT} !important;
  }}

  /* Tabs */
  div[data-testid="stTabs"] button {{
    background: transparent !important; color: {TEXT_DIM} !important;
    border: 0 !important; font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
  }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {ACCENT} !important; border-bottom: 2px solid {ACCENT} !important;
  }}

  /* Inputs */
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div {{
    background: {PAPER} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT} !important;
  }}

  /* DataFrames */
  [data-testid="stDataFrame"] {{
    background: {PAPER} !important;
    border-radius: 12px;
    border: 1px solid {BORDER};
  }}

  /* Expanders */
  div[data-testid="stExpander"] {{
    background: {PAPER} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
  }}

  /* Custom hero title */
  .hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 700; line-height: 1.05;
    background: linear-gradient(135deg, {TEXT} 30%, {ACCENT} 70%, {SUCCESS} 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem; letter-spacing: -0.03em;
  }}
  .hero-subtitle {{
    color: {TEXT_DIM}; font-size: 1.05rem; font-weight: 400;
    max-width: 720px; margin-bottom: 1.8rem;
  }}

  /* Status pills */
  .pill-row {{ display: flex; gap: 8px; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; font-size: 0.78rem; font-weight: 500;
    background: {PAPER}; border: 1px solid {BORDER};
    border-radius: 100px; font-family: 'JetBrains Mono', monospace;
  }}
  .pill-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
    animation: pulse 2.2s ease-in-out infinite;
  }}
  .pill-green  {{ color: {SUCCESS}; border-color: rgba(0, 217, 143, 0.25); }}
  .pill-green  .pill-dot {{ background: {SUCCESS}; box-shadow: 0 0 8px {SUCCESS}; }}
  .pill-cyan   {{ color: {ACCENT};  border-color: rgba(0, 217, 255, 0.25); }}
  .pill-cyan   .pill-dot {{ background: {ACCENT};  box-shadow: 0 0 8px {ACCENT}; }}
  .pill-purple {{ color: {PURPLE};  border-color: rgba(183, 148, 244, 0.25); }}
  .pill-purple .pill-dot {{ background: {PURPLE};  box-shadow: 0 0 8px {PURPLE}; }}
  .pill-amber  {{ color: {WARNING}; border-color: rgba(255, 181, 71, 0.25); }}
  .pill-amber  .pill-dot {{ background: {WARNING}; box-shadow: 0 0 8px {WARNING}; }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%      {{ opacity: 0.45; }}
  }}

  /* Section card */
  .qf-card {{
    background: {PAPER}; border: 1px solid {BORDER}; border-radius: 16px;
    padding: 22px; margin: 12px 0;
  }}
  .qf-card-title {{
    font-size: 0.78rem; color: {TEXT_DIM}; letter-spacing: 0.12em;
    text-transform: uppercase; font-weight: 600; margin-bottom: 0.8rem;
  }}

  /* Strategy badges */
  .strat-badge {{
    display: inline-block; padding: 2px 8px; font-size: 0.65rem;
    font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    border-radius: 4px; font-family: 'JetBrains Mono', monospace;
    margin-right: 6px;
  }}
  .badge-mom    {{ background: rgba(0, 217, 255, 0.15);  color: {ACCENT}; }}
  .badge-mr     {{ background: rgba(183, 148, 244, 0.15); color: {PURPLE}; }}
  .badge-factor {{ background: rgba(255, 181, 71, 0.15);  color: {WARNING}; }}
  .badge-ml     {{ background: rgba(0, 217, 143, 0.15);   color: {SUCCESS}; }}
  .badge-bench  {{ background: rgba(122, 134, 154, 0.2);  color: {MUTED}; }}

  /* Number color coding */
  .num-pos {{ color: {SUCCESS} !important; font-family: 'JetBrains Mono', monospace; }}
  .num-neg {{ color: {DANGER} !important;  font-family: 'JetBrains Mono', monospace; }}
  .num-neu {{ color: {TEXT_DIM} !important; font-family: 'JetBrains Mono', monospace; }}

  /* Hide streamlit chrome */
  #MainMenu, footer, [data-testid="stHeader"] {{ visibility: hidden !important; height: 0 !important; }}
</style>
"""


def render_pills(items: list[tuple[str, str]]) -> str:
    """Render a row of status pills. Items: (tone, text). Tone in green/cyan/purple/amber."""
    html = '<div class="pill-row">'
    for tone, text in items:
        html += f'<span class="pill pill-{tone}"><span class="pill-dot"></span>{text}</span>'
    html += "</div>"
    return html


def render_hero(title: str, subtitle: str) -> str:
    return f'<div class="hero-title">{title}</div><div class="hero-subtitle">{subtitle}</div>'


# =============================================================================
# Chart primitives
# =============================================================================
def equity_chart(
    equities: dict[str, pd.Series],
    title: str = "Equity Curve",
    benchmark_name: str | None = None,
    height: int = 450,
) -> go.Figure:
    fig = go.Figure()
    i = 0
    for name, eq in equities.items():
        is_bench = name == benchmark_name
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values, name=name,
            mode="lines",
            line=dict(
                color="#FFFFFF" if is_bench else _COLOR_CYCLE[i % len(_COLOR_CYCLE)],
                width=1.6 if is_bench else 2.2,
                dash="dash" if is_bench else "solid",
            ),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %d, %Y}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
        if not is_bench:
            i += 1
    return _style_layout(fig, title, height=height)


def drawdown_chart(equity: pd.Series, title: str = "Drawdown", height: int = 260) -> go.Figure:
    dd = (equity / equity.cummax() - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, fill="tozeroy",
        line=dict(color=DANGER, width=1.4),
        fillcolor="rgba(255, 78, 107, 0.22)",
        name="Drawdown",
        hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}%<extra></extra>",
    ))
    fig.update_yaxes(ticksuffix="%")
    return _style_layout(fig, title, height=height, show_legend=False)


def sparkline(series: pd.Series, positive: bool = True, height: int = 60) -> go.Figure:
    color = SUCCESS if positive else DANGER
    fill = "rgba(0, 217, 143, 0.2)" if positive else "rgba(255, 78, 107, 0.2)"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(series))), y=series.values,
        mode="lines", fill="tozeroy",
        line=dict(color=color, width=1.5),
        fillcolor=fill,
        hoverinfo="skip",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=False,
    )
    return fig


def returns_histogram(returns: pd.Series, title: str = "Return Distribution", height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns.values * 100, nbinsx=60,
        marker=dict(color=ACCENT, line=dict(width=0)),
        opacity=0.85,
    ))
    mean_v = float(returns.mean() * 100)
    fig.add_vline(x=mean_v, line=dict(color=SUCCESS, dash="dash", width=1.5),
                   annotation_text=f"μ = {mean_v:+.3f}%", annotation_position="top",
                   annotation_font=dict(color=SUCCESS))
    fig.update_xaxes(ticksuffix="%")
    return _style_layout(fig, title, height=height, show_legend=False)


def rolling_metric_chart(series: pd.Series, title: str,
                          y_suffix: str = "", height: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        line=dict(color=ACCENT, width=2.2),
        fill="tozeroy", fillcolor="rgba(0, 217, 255, 0.08)",
        hovertemplate="%{x|%b %d}<br>%{y:.3f}<extra></extra>",
    ))
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    return _style_layout(fig, title, height=height, show_legend=False)


def heatmap(df: pd.DataFrame, title: str = "Correlation", height: int = 380) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=df.values, x=df.columns, y=df.index,
        colorscale=[[0, DANGER], [0.5, "#1C2130"], [1, SUCCESS]],
        zmid=0,
        hovertemplate="<b>%{x}</b> × <b>%{y}</b>: %{z:.2f}<extra></extra>",
        colorbar=dict(bgcolor=PAPER, bordercolor=BORDER,
                      tickfont=dict(color=TEXT_DIM)),
    ))
    return _style_layout(fig, title, height=height, show_legend=False)


def weights_bar(weights: pd.DataFrame, title: str = "Portfolio Weights", height: int = 340) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(weights.columns):
        fig.add_trace(go.Bar(
            x=weights.index, y=weights[col], name=col,
            marker_color=_COLOR_CYCLE[i % len(_COLOR_CYCLE)],
            hovertemplate="<b>%{x}</b><br>%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    return _style_layout(fig, title, height=height)


def efficient_frontier_chart(
    ef: pd.DataFrame,
    assets: pd.DataFrame | None = None,
    tangent: tuple | None = None,
    title: str = "Efficient Frontier",
    height: int = 460,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ef["risk"], y=ef["return"], mode="lines",
        line=dict(color=ACCENT, width=2.5),
        fill="tonexty", fillcolor="rgba(0, 217, 255, 0.05)",
        name="Efficient Frontier",
        hovertemplate="Risk: %{x:.3f}<br>Return: %{y:.3f}<extra></extra>",
    ))
    if assets is not None:
        fig.add_trace(go.Scatter(
            x=assets["vol"], y=assets["ret"],
            mode="markers+text",
            marker=dict(color=WARNING, size=11, symbol="diamond",
                         line=dict(color=BG, width=1)),
            text=assets.index, textposition="top right",
            textfont=dict(color=WARNING, family="JetBrains Mono", size=10),
            name="Assets",
        ))
    if tangent is not None:
        risk, ret = tangent
        fig.add_trace(go.Scatter(
            x=[risk], y=[ret], mode="markers+text",
            marker=dict(color=SUCCESS, size=18, symbol="star",
                         line=dict(color=BG, width=1.5)),
            text=["Max Sharpe"], textposition="middle right",
            textfont=dict(color=SUCCESS, family="JetBrains Mono", size=11),
            name="Max Sharpe",
        ))
    fig.update_xaxes(ticksuffix=" σ")
    return _style_layout(fig, title, height=height)


def greeks_gauge(greeks: dict[str, float]) -> go.Figure:
    fig = make_subplots(rows=1, cols=5, specs=[[{"type": "indicator"}] * 5],
                         subplot_titles=["Delta", "Gamma", "Vega", "Theta", "Rho"])
    configs = [
        ("delta", -1, 1, ACCENT),
        ("gamma", 0, 0.1, SUCCESS),
        ("vega", -100, 100, WARNING),
        ("theta", -50, 50, DANGER),
        ("rho", -100, 100, PURPLE),
    ]
    for i, (k, lo, hi, col) in enumerate(configs, 1):
        v = greeks.get(k, 0)
        fig.add_trace(go.Indicator(
            mode="number+gauge",
            value=v,
            gauge={"axis": {"range": [lo, hi], "tickcolor": TEXT_DIM, "tickfont": {"size": 9}},
                   "bar": {"color": col}, "bgcolor": PAPER,
                   "borderwidth": 1, "bordercolor": BORDER},
            number={"font": {"color": col, "size": 26, "family": "Space Grotesk"}},
        ), row=1, col=i)
    fig.update_layout(
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(color=TEXT_DIM, family="Inter"),
        margin=dict(l=12, r=12, t=50, b=12), height=220,
    )
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(color=TEXT_DIM, size=11, family="Inter")
    return fig


def iv_smile_chart(strikes, injected, recovered, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes, y=[v * 100 for v in injected],
        mode="lines+markers",
        line=dict(color=MUTED, dash="dot", width=1.5),
        marker=dict(color=MUTED, size=8), name="Injected IV",
    ))
    fig.add_trace(go.Scatter(
        x=strikes, y=[v * 100 for v in recovered],
        mode="lines+markers",
        line=dict(color=ACCENT, width=3),
        marker=dict(color=ACCENT, size=10, line=dict(color=BG, width=1)),
        name="Recovered IV",
    ))
    fig.update_yaxes(ticksuffix="%")
    fig.update_xaxes(title=dict(text="Strike", font=dict(color=TEXT_DIM)))
    return _style_layout(fig, "Implied Volatility Smile", height=height)


def horizontal_bar(df: pd.DataFrame, x_col: str, y_col: str,
                    title: str = "", color: str | None = None, height: int = 420) -> go.Figure:
    color = color or ACCENT
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x_col], y=df[y_col], orientation="h",
        marker=dict(color=color, line=dict(width=0)),
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _style_layout(fig, title, height=height, show_legend=False)


def dual_axis(primary: pd.Series, secondary: pd.Series,
               primary_label: str = "Primary", secondary_label: str = "Secondary",
               title: str = "", height: int = 360) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=primary.index, y=primary.values, name=primary_label,
                              line=dict(color=ACCENT, width=2.2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=secondary.index, y=secondary.values, name=secondary_label,
                              line=dict(color=PURPLE, width=2, dash="dot")), secondary_y=True)
    return _style_layout(fig, title, height=height)
