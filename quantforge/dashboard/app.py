"""QuantForge Streamlit dashboard — dark theme, Plotly charts, benchmark overlays."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from quantforge.analytics.benchmark import benchmark_report
from quantforge.analytics.performance import summary_stats
from quantforge.analytics.tearsheet import tearsheet_markdown
from quantforge.backtest import BacktestEngine
from quantforge.dashboard.components import (
    ACCENT, SUCCESS, DANGER, PAPER,
    drawdown_chart, efficient_frontier_chart, equity_chart,
    greeks_gauge, heatmap, iv_smile_chart, returns_histogram,
    rolling_metric_chart, weights_bar,
)
from quantforge.data.loader import DataLoader
from quantforge.data.synthetic import (
    generate_correlated_returns, generate_ohlcv, generate_panel,
)
from quantforge.indicators.technical import atr, bollinger_bands, ema, macd, rsi
from quantforge.indicators.statistical import realized_vol, rolling_zscore
from quantforge.options.binomial import crr_american, crr_price
from quantforge.options.black_scholes import bs_call, bs_implied_vol, bs_put
from quantforge.options.greeks import all_greeks
from quantforge.options.monte_carlo import mc_asian, mc_barrier, mc_european
from quantforge.portfolio.hrp import hierarchical_risk_parity
from quantforge.portfolio.markowitz import efficient_frontier, max_sharpe, min_variance
from quantforge.portfolio.risk_parity import equal_risk_contribution
from quantforge.risk.drawdown import drawdown_series, drawdown_table, max_drawdown
from quantforge.risk.var import (
    cornish_fisher_var, historical_cvar, historical_var, parametric_var,
)
from quantforge.strategies import (
    BollingerMeanReversion, BuyAndHoldStrategy, CrossSectionalMomentum,
    DonchianBreakout, DualMomentum, FactorStrategy, MACrossoverStrategy,
    MomentumStrategy, RegimeSwitch, RSIReversalStrategy, VolTarget,
)


st.set_page_config(
    page_title="QuantForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- custom CSS polish ---------------------------------------------------------
st.markdown(
    """
<style>
  .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
  h1, h2, h3 { color: #00D9FF; }
  [data-testid="stMetric"] {
      background: #1C2130;
      padding: 18px;
      border-radius: 12px;
      border: 1px solid #2B3142;
  }
  [data-testid="stMetricLabel"] { color: #7A869A; font-size: 0.85rem !important; }
  [data-testid="stMetricValue"] { color: #F0F2F6; }
  [data-testid="stSidebar"] { background: #0B0F19; border-right: 1px solid #1C2130; }
  .main-title {
      font-size: 2.4rem; font-weight: 700;
      background: linear-gradient(90deg, #00D9FF, #00D98F);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.2rem;
  }
  .subtitle { color: #7A869A; font-size: 1rem; margin-bottom: 2rem; }
  .stButton>button {
      background: linear-gradient(90deg, #00D9FF, #00D98F);
      color: #0E1117; font-weight: 600; border: 0; border-radius: 8px;
      padding: 0.6rem 1.4rem;
  }
  .stDownloadButton>button {
      background: #1C2130; color: #F0F2F6;
      border: 1px solid #2B3142; border-radius: 8px;
  }
  div.stTabs button { font-size: 1rem; font-weight: 500; }
  div.stTabs button[aria-selected="true"] { color: #00D9FF; }
  .strat-badge {
      display: inline-block; padding: 3px 10px; font-size: 0.75rem;
      border-radius: 12px; background: #1C2130; border: 1px solid #2B3142;
      color: #00D9FF; margin-right: 4px;
  }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_market(tickers: tuple, start: str, end: str) -> dict:
    dl = DataLoader(cache_dir="data/cache")
    return dl.yfinance_many(list(tickers), start=start, end=end)


def sidebar() -> str:
    st.sidebar.markdown("### ⚡ **QuantForge**")
    st.sidebar.caption("Quant research, end-to-end")
    st.sidebar.divider()
    return st.sidebar.radio(
        "Navigate",
        [
            "📊 Overview",
            "📈 Data & Indicators",
            "🧪 Strategy Backtest",
            "🏆 Tournament (Real Data)",
            "💹 Options Lab",
            "🧮 Portfolio Optimizer",
            "⚠️ Risk Analysis",
            "🤖 ML Training",
        ],
        label_visibility="collapsed",
    )


# ============================== OVERVIEW ======================================
def page_overview() -> None:
    st.markdown('<div class="main-title">QuantForge</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Full-stack quant research platform — data, backtesting, options, portfolio, ML, risk.</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategies", "12", "+4 new")
    c2.metric("Indicators", "25+")
    c3.metric("Optimizers", "5")
    c4.metric("Tests", "304", "all green")

    st.divider()
    st.markdown("### What's inside")
    modules = [
        ("📊 Data", "Synthetic GBM, yfinance loader, 2516 bars of SPY/QQQ/IWM/TLT/GLD cached"),
        ("📈 Indicators", "15 technical + 10 statistical, vectorized"),
        ("💹 Options", "Black-Scholes, CRR, MC exotics, full Greeks, IV solver"),
        ("🧮 Portfolio", "Markowitz, ERC, Black-Litterman, HRP, CVaR-LP"),
        ("⚠️ Risk", "VaR (4 methods), drawdown, stress tests, Kelly, MC simulation"),
        ("🤖 ML", "Features, regime detection, trainer with walk-forward + feature importance"),
        ("🧪 Backtest", "Event-driven, broker sim, slippage/commission, TCA, monthly rebalance"),
        ("🏆 Strategies", "Momentum, MA cross, MR, Donchian, RSI, pairs, factor, cross-sectional, dual momentum, vol-target, regime switch, ML"),
    ]
    cols = st.columns(2)
    for i, (name, desc) in enumerate(modules):
        with cols[i % 2]:
            st.markdown(f"**{name}**")
            st.caption(desc)


# ============================== DATA ==========================================
def page_data() -> None:
    st.header("📈 Data & Indicators")
    tab_synth, tab_real = st.tabs(["Synthetic", "Real Market (yfinance)"])

    with tab_synth:
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input("Bars", 100, 5000, 500, 50)
        s0 = c2.number_input("Start price", 1.0, 1000.0, 100.0)
        mu = c3.slider("μ (ann.)", -0.2, 0.3, 0.08, 0.01)
        sigma = c4.slider("σ (ann.)", 0.05, 0.8, 0.20, 0.01)
        seed = st.number_input("Seed", 0, 9999, 42, 1)
        df = generate_ohlcv(n, s0=s0, mu=mu, sigma=sigma, seed=int(seed))
        _render_indicators(df)

    with tab_real:
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", "SPY")
        start = c2.text_input("Start", "2020-01-01")
        end = c3.text_input("End", "2025-01-01")
        if st.button("📥 Load from yfinance", key="load_real"):
            try:
                with st.spinner(f"Fetching {ticker}..."):
                    panel = load_market((ticker,), start, end)
                    df = panel[ticker]
                _render_indicators(df)
            except Exception as e:
                st.error(f"Load failed: {e}")


def _render_indicators(df: pd.DataFrame) -> None:
    close = df["close"]
    bb = bollinger_bands(close)
    c_fig = {
        "close": close,
        "SMA(20)": close.rolling(20).mean(),
        "EMA(50)": ema(close, 50),
        "BB upper": bb["upper"],
        "BB lower": bb["lower"],
    }
    st.plotly_chart(equity_chart(c_fig, "Price + Moving Averages + Bollinger Bands"),
                     use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(rolling_metric_chart(rsi(close), "RSI(14)"), use_container_width=True)
        st.plotly_chart(rolling_metric_chart(realized_vol(close.pct_change(), 21), "Realized Vol (21d)",
                                               y_suffix=""),
                         use_container_width=True)
    with c2:
        mac = macd(close)
        st.plotly_chart(equity_chart({"MACD": mac["macd"], "signal": mac["signal"]}, "MACD"),
                         use_container_width=True)
        st.plotly_chart(rolling_metric_chart(atr(df), "ATR(14)"), use_container_width=True)

    with st.expander("🔍 Raw OHLCV"):
        st.dataframe(df.tail(20), use_container_width=True)
        st.download_button("⬇️ Download CSV", df.to_csv().encode("utf-8"),
                            "data.csv", "text/csv")


# ============================== BACKTEST ======================================
_STRATS = {
    "Momentum(60)": lambda: MomentumStrategy(lookback=60, allow_short=False),
    "Momentum(120)": lambda: MomentumStrategy(lookback=120, allow_short=False),
    "MA 20/100": lambda: MACrossoverStrategy(20, 100),
    "MA 50/200": lambda: MACrossoverStrategy(50, 200),
    "RSI Reversal": lambda: RSIReversalStrategy(oversold=30, overbought=70),
    "Bollinger MR": lambda: BollingerMeanReversion(20, 2.0),
    "Donchian": lambda: DonchianBreakout(20, 10, allow_short=False),
    "Factor Composite": lambda: FactorStrategy(120, 60),
    "Cross-Sec Momentum": lambda: CrossSectionalMomentum(),
    "Dual Momentum": lambda: DualMomentum(),
    "Regime Switch": lambda: RegimeSwitch(),
    "VolTarget(Momentum)": lambda: VolTarget(base=MomentumStrategy(lookback=120), target_vol=0.10),
    "Buy & Hold": lambda: BuyAndHoldStrategy(),
}


def page_backtest() -> None:
    st.header("🧪 Strategy Backtest")

    c1, c2, c3 = st.columns(3)
    source = c1.radio("Data source", ["Synthetic", "Real (SPY)"], horizontal=True)
    rebalance = c2.selectbox("Rebalance", ["bar", "weekly", "monthly"])
    capital = c3.number_input("Capital", 10_000, 10_000_000, 100_000, 10_000)

    strat_name = st.selectbox("Strategy", list(_STRATS.keys()))
    sizing = st.slider("Sizing fraction / target weight", 0.05, 1.0, 0.5, 0.05)

    if source == "Synthetic":
        n = st.slider("Bars", 200, 2000, 600, 50)
        seed = st.slider("Seed", 0, 999, 11)
        n_assets = st.slider("Universe size", 1, 6, 3)
        data = generate_panel([f"S{i}" for i in range(n_assets)], n=n, seed=seed)
    else:
        try:
            with st.spinner("Loading real SPY/QQQ/IWM/TLT/GLD..."):
                data = load_market(("SPY", "QQQ", "IWM", "TLT", "GLD"), "2015-01-01", "2025-01-01")
        except Exception as e:
            st.error(f"Load failed: {e}")
            return

    if st.button("▶️ Run backtest", type="primary"):
        strat = _STRATS[strat_name]()
        with st.spinner("Running..."):
            engine = BacktestEngine(strategy=strat, data=data, initial_capital=capital,
                                     sizing_fraction=sizing, rebalance=rebalance)
            res = engine.run()

        stats = summary_stats(res.equity_curve, trades=res.trades)

        # Benchmark
        bench_eng = BacktestEngine(strategy=BuyAndHoldStrategy(),
                                     data={list(data.keys())[0]: data[list(data.keys())[0]]},
                                     initial_capital=capital, sizing_fraction=1.0)
        bench_res = bench_eng.run()
        bench_stats = summary_stats(bench_res.equity_curve)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{stats['total_return']*100:+.2f}%",
                   f"{(stats['total_return']-bench_stats['total_return'])*100:+.2f}% vs B&H")
        m2.metric("Sharpe", f"{stats['sharpe']:+.2f}",
                   f"{stats['sharpe']-bench_stats['sharpe']:+.2f}")
        m3.metric("Max DD", f"{stats['max_drawdown']*100:+.2f}%",
                   f"{(stats['max_drawdown']-bench_stats['max_drawdown'])*100:+.2f}%",
                   delta_color="inverse")
        m4.metric("Trades", f"{len(res.trades):,}",
                   f"Cost: {stats.get('cost_bps', 0):.1f} bps")

        st.plotly_chart(
            equity_chart({strat_name: res.equity_curve, "Buy & Hold": bench_res.equity_curve},
                          "Equity Curve vs Buy & Hold",
                          benchmark_name="Buy & Hold"),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.plotly_chart(drawdown_chart(res.equity_curve, f"{strat_name} Drawdown"),
                         use_container_width=True)
        c2.plotly_chart(returns_histogram(res.equity_curve.pct_change().dropna(), "Daily Returns"),
                         use_container_width=True)

        if not res.trades.empty:
            strat_r = res.equity_curve.pct_change().dropna()
            bench_r = bench_res.equity_curve.pct_change().dropna()
            rep = benchmark_report(strat_r, bench_r)
            with st.expander("📊 Full metrics vs benchmark"):
                cb1, cb2, cb3 = st.columns(3)
                cb1.metric("Alpha (ann.)", f"{rep.alpha*100:+.2f}%")
                cb2.metric("Beta", f"{rep.beta:+.3f}")
                cb3.metric("Info Ratio", f"{rep.info_ratio:+.2f}")
                st.markdown(tearsheet_markdown(res.equity_curve, strat_name))
                st.download_button("⬇️ Download trades CSV",
                                    res.trades.to_csv(index=False).encode("utf-8"),
                                    "trades.csv", "text/csv")


# ============================== TOURNAMENT ====================================
def page_tournament() -> None:
    st.header("🏆 Real-Data Strategy Tournament")
    st.caption("12 strategies × 10 years × SPY / QQQ / IWM / TLT / GLD — cost-aware (5bps slip, 1bp commission), monthly rebalance")

    try:
        data = load_market(("SPY", "QQQ", "IWM", "TLT", "GLD"), "2015-01-01", "2025-01-01")
    except Exception as e:
        st.error(f"Load failed: {e}")
        return
    st.success(f"✅ Loaded {min(len(v) for v in data.values())} bars × {len(data)} tickers")

    chosen = st.multiselect("Strategies to race", list(_STRATS.keys()),
                             default=["Buy & Hold", "Momentum(120)", "MA 20/100", "Donchian",
                                      "Dual Momentum", "Regime Switch", "Cross-Sec Momentum",
                                      "VolTarget(Momentum)"])
    if st.button("▶️ Run tournament", type="primary") and chosen:
        rows, equities = [], {}
        prog = st.progress(0.0)
        for i, name in enumerate(chosen, 1):
            strat = _STRATS[name]()
            subset = data if name != "Buy & Hold" else {"SPY": data["SPY"]}
            size = 1.0 if name == "Buy & Hold" else 0.2
            rebalance = "bar" if name == "Buy & Hold" else "monthly"
            eng = BacktestEngine(strategy=strat, data=subset,
                                  initial_capital=100_000, sizing_fraction=size,
                                  rebalance=rebalance)
            res = eng.run()
            s = summary_stats(res.equity_curve, trades=res.trades)
            rows.append({"strategy": name, **{k: s[k] for k in ("total_return", "annual_return", "annual_vol",
                                                                  "sharpe", "sortino", "max_drawdown")},
                         "turnover": s.get("turnover", 0), "trades": int(s.get("n_trades", 0) or 0)})
            equities[name] = res.equity_curve
            prog.progress(i / len(chosen))

        df = pd.DataFrame(rows).set_index("strategy")
        df_display = df.copy()
        for col in ("total_return", "annual_return", "annual_vol", "max_drawdown"):
            df_display[col] = df_display[col].map(lambda v: f"{v*100:+.2f}%")
        df_display["sharpe"] = df_display["sharpe"].map(lambda v: f"{v:+.2f}")
        df_display["sortino"] = df_display["sortino"].map(lambda v: f"{v:+.2f}")
        df_display["turnover"] = df_display["turnover"].map(lambda v: f"{v:.1f}×")
        st.dataframe(df_display, use_container_width=True)

        st.plotly_chart(equity_chart(equities, "Equity Curves (real data, 10y)",
                                       benchmark_name="Buy & Hold"),
                         use_container_width=True)

        best = df["sharpe"].idxmax()
        st.success(f"🏆 Best Sharpe: **{best}** ({df.loc[best, 'sharpe']:+.2f})")

        st.download_button("⬇️ Download tournament CSV",
                            df.reset_index().to_csv(index=False).encode("utf-8"),
                            "tournament.csv", "text/csv")


# ============================== OPTIONS =======================================
def page_options() -> None:
    st.header("💹 Options Lab")
    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Spot", 1.0, 10_000.0, 100.0)
    K = c2.number_input("Strike", 1.0, 10_000.0, 100.0)
    T = c3.number_input("T (years)", 0.01, 10.0, 1.0, 0.05)
    r = c4.number_input("r", -0.05, 0.2, 0.04, 0.005, format="%.4f")
    sigma = c5.slider("σ", 0.01, 1.5, 0.20, 0.01)
    opt = st.radio("Option type", ["call", "put"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    bs = bs_call(S, K, T, r, sigma) if opt == "call" else bs_put(S, K, T, r, sigma)
    crr = crr_price(S, K, T, r, sigma, opt, 500)
    mc, mc_se = mc_european(S, K, T, r, sigma, opt, n_paths=20_000, seed=42)
    col1.metric("Black-Scholes", f"${bs:.4f}")
    col2.metric("CRR (N=500)", f"${crr:.4f}", f"{(crr-bs):+.4f}")
    col3.metric("Monte Carlo", f"${mc:.4f}", f"±{mc_se:.4f}")

    greeks = all_greeks(S, K, T, r, sigma, opt)
    st.plotly_chart(greeks_gauge(greeks), use_container_width=True)

    st.subheader("Vol smile recovery")
    strikes = np.linspace(S * 0.7, S * 1.3, 21)
    injected = [sigma + 0.08 * ((k / S - 1) ** 2 - 0.01) for k in strikes]
    mkt = [bs_call(S, k, T, r, iv) for k, iv in zip(strikes, injected)]
    recovered = [bs_implied_vol(m, S, k, T, r, "call") for m, k in zip(mkt, strikes)]
    st.plotly_chart(iv_smile_chart(strikes, injected, recovered), use_container_width=True)

    st.subheader("Exotic options (Monte Carlo)")
    c1, c2, c3 = st.columns(3)
    asi, ase = mc_asian(S, K, T, r, sigma, opt, seed=42)
    bar, bse = mc_barrier(S, K, T, r, sigma, barrier=S * 0.8,
                           barrier_type="down-and-out", option=opt, seed=42)
    amer = crr_american(S, K, T, r, sigma, opt, 500)
    c1.metric("Asian (arith.)", f"${asi:.4f}", f"±{ase:.4f}")
    c2.metric(f"Down-&-out ({S*0.8:.0f})", f"${bar:.4f}", f"±{bse:.4f}")
    c3.metric("American", f"${amer:.4f}", f"{amer-bs:+.4f} prem")


# ============================== PORTFOLIO =====================================
def page_portfolio() -> None:
    st.header("🧮 Portfolio Optimizer")
    tab1, tab2 = st.tabs(["Synthetic", "Real Data"])

    with tab1:
        n_a = st.slider("# assets", 3, 10, 5, key="syn_n")
        n_d = st.slider("Days", 252, 2000, 504, key="syn_d")
        symbols = [f"A{i}" for i in range(n_a)]
        rng = np.random.default_rng(42)
        factor = rng.normal(0, 0.01, (n_a, n_a))
        cov_daily = 0.5 * (factor @ factor.T) + np.diag(rng.uniform(0.01, 0.1, n_a)) / 252
        ret_df = generate_correlated_returns(n_d, symbols, cov=cov_daily, seed=42)
        _render_portfolio(ret_df, symbols)

    with tab2:
        try:
            data = load_market(("SPY", "QQQ", "IWM", "TLT", "GLD"), "2015-01-01", "2025-01-01")
            ret_df = pd.DataFrame({t: v["close"].pct_change() for t, v in data.items()}).dropna()
            _render_portfolio(ret_df, list(ret_df.columns))
        except Exception as e:
            st.error(f"Load failed: {e}")


def _render_portfolio(ret_df: pd.DataFrame, symbols: list) -> None:
    mu = ret_df.mean().values * 252
    cov = ret_df.cov().values * 252
    vols = np.sqrt(np.diag(cov))

    stats_df = pd.DataFrame({"μ": mu, "σ": vols, "Sharpe": mu / vols}, index=symbols)
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Asset statistics (annualized)**")
        st.dataframe(stats_df.round(3), use_container_width=True)
    with c2:
        corr = ret_df.corr()
        st.plotly_chart(heatmap(corr, "Correlation"), use_container_width=True)

    w_ms = max_sharpe(mu, cov, risk_free=0.02)
    w_mv = min_variance(cov)
    w_erc = equal_risk_contribution(cov)
    w_hrp = hierarchical_risk_parity(cov)
    wdf = pd.DataFrame({"Max Sharpe": w_ms, "Min Var": w_mv, "ERC": w_erc, "HRP": w_hrp},
                        index=symbols)
    st.plotly_chart(weights_bar(wdf, "Portfolio Weights"), use_container_width=True)

    ef = efficient_frontier(mu, cov, n_points=25)
    tangent = None
    if len(ef):
        tangent_vol = float(np.sqrt(w_ms @ cov @ w_ms))
        tangent_ret = float(w_ms @ mu)
        tangent = (tangent_vol, tangent_ret)
    assets_pts = pd.DataFrame({"vol": vols, "ret": mu}, index=symbols)
    st.plotly_chart(efficient_frontier_chart(ef, assets_pts, tangent),
                     use_container_width=True)

    rows = []
    for name, w in wdf.items():
        pr = float(w.values @ mu)
        pv = float(np.sqrt(w.values @ cov @ w.values))
        rows.append({"Portfolio": name, "Return": f"{pr*100:+.2f}%",
                     "Vol": f"{pv*100:.2f}%",
                     "Sharpe": f"{(pr - 0.02) / pv if pv > 0 else 0:+.2f}"})
    st.dataframe(pd.DataFrame(rows).set_index("Portfolio"), use_container_width=True)


# ============================== RISK ==========================================
def page_risk() -> None:
    st.header("⚠️ Risk Analysis")
    c1, c2, c3, c4 = st.columns(4)
    n = c1.number_input("Bars", 252, 5000, 1000, 50)
    mu = c2.slider("μ", -0.2, 0.3, 0.08, 0.01)
    sigma = c3.slider("σ", 0.05, 0.8, 0.25, 0.01)
    conf = c4.slider("VaR confidence", 0.90, 0.995, 0.95, 0.005)

    from quantforge.data.synthetic import generate_gbm
    prices = generate_gbm(int(n), mu=mu, sigma=sigma, seed=7)
    rets = prices.pct_change().dropna()

    c1, c2 = st.columns(2)
    c1.plotly_chart(equity_chart({"Price": prices}, "Price path"), use_container_width=True)
    c2.plotly_chart(drawdown_chart(prices, "Drawdown"), use_container_width=True)

    st.subheader(f"Value-at-Risk ({conf*100:.1f}%)")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Historical", f"{historical_var(rets, conf)*100:.2f}%")
    v2.metric("Parametric", f"{parametric_var(rets, conf)*100:.2f}%")
    v3.metric("Cornish-Fisher", f"{cornish_fisher_var(rets, conf)*100:.2f}%")
    v4.metric("Historical CVaR", f"{historical_cvar(rets, conf)*100:.2f}%")

    mdd, peak, trough = max_drawdown(prices)
    m1, m2, m3 = st.columns(3)
    m1.metric("Max DD", f"{mdd*100:.2f}%")
    m2.metric("Peak date", f"{peak.date()}")
    m3.metric("Trough date", f"{trough.date()}")

    st.plotly_chart(returns_histogram(rets, "Return distribution"), use_container_width=True)

    ddt = drawdown_table(prices, top_n=10)
    if not ddt.empty:
        st.subheader("Top 10 drawdowns")
        st.dataframe(ddt, use_container_width=True)


# ============================== ML TRAINING ===================================
def page_ml() -> None:
    st.header("🤖 ML Training on Real Data")
    st.caption("Real SPY data → 20+ features → time-ordered train/val/test → gradient boosting → honest metrics.")

    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", "SPY")
    start = c2.text_input("Start", "2010-01-01")
    end = c3.text_input("End", "2025-01-01")

    if st.button("🏋️ Train real model", type="primary"):
        try:
            with st.spinner("Downloading data..."):
                panel = load_market((ticker,), start, end)
                df = panel[ticker]
        except Exception as e:
            st.error(f"Load failed: {e}")
            return

        from sklearn.ensemble import GradientBoostingClassifier
        from quantforge.ml.features import build_feature_matrix, target_labels
        from quantforge.ml.trainer import train_classifier

        with st.spinner("Building features..."):
            feats = build_feature_matrix(df)
            target = target_labels(df["close"], horizon=1, kind="binary")
            data = feats.join(target.rename("y")).dropna()
            X = data.drop(columns="y")
            y = data["y"]

        st.success(f"✅ Features: **{len(X.columns)}**, samples: **{len(X)}**")
        st.caption(f"Target distribution: up **{(y == 1).mean():.2%}**, down **{(y == 0).mean():.2%}**")

        with st.spinner("Training (hyperparameter search over 9 configs)..."):
            hp_grid = [{"n_estimators": n, "max_depth": d, "learning_rate": 0.05, "random_state": 42}
                        for n in [50, 100, 200] for d in [2, 3, 5]]
            model, report = train_classifier(X, y, model_cls=GradientBoostingClassifier,
                                               hp_grid=hp_grid, verbose=False)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Train acc", f"{report.train_acc:.4f}")
        m2.metric("Val acc", f"{report.val_acc:.4f}")
        m3.metric("Test acc", f"{report.test_acc:.4f}",
                   f"{(report.test_acc - report.baseline_acc):+.4f} vs baseline")
        m4.metric("Test AUC", f"{report.test_auc:.4f}")

        fi_df = pd.DataFrame(
            sorted(report.feature_importances.items(), key=lambda kv: kv[1], reverse=True),
            columns=["feature", "importance"],
        ).head(15)
        st.subheader("Top-15 feature importances")
        import plotly.express as px
        fi_fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                         color_discrete_sequence=[ACCENT])
        fi_fig.update_layout(paper_bgcolor=PAPER, plot_bgcolor="#0E1117",
                              font=dict(color="#F0F2F6"), height=480,
                              yaxis=dict(autorange="reversed"))
        st.plotly_chart(fi_fig, use_container_width=True)

        st.subheader("Test-set confusion matrix")
        cm_df = pd.DataFrame(report.confusion_test,
                              index=["Actual Down", "Actual Up"],
                              columns=["Pred Down", "Pred Up"])
        st.dataframe(cm_df, use_container_width=True)

        if report.hp_search is not None:
            with st.expander("🔬 Hyperparameter search results"):
                st.dataframe(report.hp_search.round(4), use_container_width=True)

        st.info(
            f"🎯 **Baseline** (always predict majority): {report.baseline_acc:.4f}   |   "
            f"**Test AUC**: {report.test_auc:.4f} (0.50 = random)"
        )


PAGES = {
    "📊 Overview": page_overview,
    "📈 Data & Indicators": page_data,
    "🧪 Strategy Backtest": page_backtest,
    "🏆 Tournament (Real Data)": page_tournament,
    "💹 Options Lab": page_options,
    "🧮 Portfolio Optimizer": page_portfolio,
    "⚠️ Risk Analysis": page_risk,
    "🤖 ML Training": page_ml,
}


def main() -> None:
    choice = sidebar()
    PAGES[choice]()


if __name__ == "__main__":
    main()
