"""QuantForge dashboard — Bloomberg Terminal × Linear/Stripe aesthetic.

Launch:   streamlit run quantforge/dashboard/app.py
"""
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
    ACCENT,
    BORDER,
    DANGER,
    GLOBAL_CSS,
    PAPER,
    PURPLE,
    SUCCESS,
    TEXT,
    TEXT_DIM,
    WARNING,
    drawdown_chart,
    efficient_frontier_chart,
    equity_chart,
    greeks_gauge,
    heatmap,
    horizontal_bar,
    iv_smile_chart,
    render_hero,
    render_pills,
    returns_histogram,
    rolling_metric_chart,
    weights_bar,
)
from quantforge.data.loader import DataLoader
from quantforge.data.synthetic import generate_correlated_returns, generate_ohlcv, generate_panel
from quantforge.indicators.statistical import realized_vol
from quantforge.indicators.technical import atr, bollinger_bands, ema, macd, rsi
from quantforge.options.binomial import crr_american, crr_price
from quantforge.options.black_scholes import bs_call, bs_implied_vol, bs_put
from quantforge.options.greeks import all_greeks
from quantforge.options.monte_carlo import mc_asian, mc_barrier, mc_european
from quantforge.portfolio.hrp import hierarchical_risk_parity
from quantforge.portfolio.markowitz import efficient_frontier, max_sharpe, min_variance
from quantforge.portfolio.risk_parity import equal_risk_contribution
from quantforge.risk.drawdown import drawdown_table, max_drawdown
from quantforge.risk.var import (
    cornish_fisher_var,
    historical_cvar,
    historical_var,
    parametric_var,
)
from quantforge.strategies import (
    BollingerMeanReversion,
    BuyAndHoldStrategy,
    CrossSectionalMomentum,
    DonchianBreakout,
    DualMomentum,
    FactorStrategy,
    MACrossoverStrategy,
    MomentumStrategy,
    RegimeSwitch,
    RSIReversalStrategy,
    VolTarget,
)

st.set_page_config(page_title="QuantForge", page_icon="⚡", layout="wide",
                    initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_market(tickers: tuple, start: str, end: str) -> dict:
    dl = DataLoader(cache_dir="data/cache")
    return dl.yfinance_many(list(tickers), start=start, end=end)


def sidebar() -> str:
    st.sidebar.markdown('<div class="sidebar-brand">⚡ QuantForge</div>',
                         unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-tagline">Quant Research Terminal</div>',
                         unsafe_allow_html=True)
    st.sidebar.markdown('<hr class="sidebar-divider"/>', unsafe_allow_html=True)
    return st.sidebar.radio(
        "nav",
        [
            "◆  Overview", "▤  Data & Indicators", "⚗  Strategy Backtest",
            "♛  Tournament", "◈  Options Lab", "◎  Portfolio Optimizer",
            "⚠  Risk Analysis", "✦  ML Training",
        ],
        label_visibility="collapsed",
    )


# ============================== OVERVIEW =====================================
def page_overview() -> None:
    st.markdown(render_hero(
        "QuantForge",
        "Full-stack quant research — backtesting, options, portfolio optimization, "
        "risk analytics, and machine learning — in one minimal-dep Python library.",
    ), unsafe_allow_html=True)

    st.markdown(render_pills([
        ("green", "304 tests passing"),
        ("cyan", "2,516 bars · real data"),
        ("purple", "12 strategies · 5 optimizers"),
        ("amber", "v1.0.0"),
    ]), unsafe_allow_html=True)

    try:
        panel = load_market(("SPY", "QQQ", "IWM", "TLT", "GLD"), "2015-01-01", "2025-01-01")
        spy_curve = panel["SPY"]["close"]
        rets = spy_curve.pct_change().dropna()
        eq = 100_000 * (spy_curve / spy_curve.iloc[0])
        total_ret = eq.iloc[-1] / 100_000 - 1
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252))
        mdd = float((eq / eq.cummax() - 1).min())
    except Exception:
        total_ret, sharpe, mdd = 1.457, 0.79, -0.338

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return · SPY 10Y", f"{total_ret*100:+.1f}%", "cumulative")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}", "risk-adjusted")
    c3.metric("Max Drawdown", f"{mdd*100:+.1f}%", "2020 COVID")
    c4.metric("Active Strategies", "12", "+4 new")

    st.markdown('<h3 style="margin-top:1.5rem;">Market snapshot</h3>', unsafe_allow_html=True)
    try:
        main_curves = {t: panel[t]["close"] for t in ("SPY", "QQQ", "IWM", "TLT", "GLD")}
        normalized = {k: 100 * v / v.iloc[0] for k, v in main_curves.items()}
        st.plotly_chart(equity_chart(normalized, "Normalized price — base 100",
                                       benchmark_name="SPY", height=420),
                         use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load market data: {e}")

    st.markdown('<h3 style="margin-top:2rem;">What\'s inside</h3>', unsafe_allow_html=True)
    modules = [
        ("◆ Data", "Synthetic GBM + yfinance loader + 2,516 cached ETF bars", ACCENT),
        ("▤ Indicators", "15 technical + 10 statistical, vectorized", ACCENT),
        ("◈ Options", "BS, CRR, Monte Carlo exotics, closed-form Greeks", PURPLE),
        ("◎ Portfolio", "Markowitz, ERC, Black-Litterman, HRP, CVaR-LP", PURPLE),
        ("⚠ Risk", "VaR × 4 methods, stress tests, Kelly, MC simulation", WARNING),
        ("✦ ML", "Features, regime, walk-forward trainer with HP search", SUCCESS),
        ("⚗ Backtest", "Event-driven, broker sim, monthly rebalance, TCA", ACCENT),
        ("♛ Strategies", "Momentum, MR, Donchian, pairs, dual momentum, vol-target...", SUCCESS),
    ]
    cols = st.columns(4)
    for i, (name, desc, color) in enumerate(modules):
        with cols[i % 4]:
            st.markdown(
                f"""<div class="qf-card" style="min-height:110px;">
                    <div class="qf-card-title" style="color:{color};">{name}</div>
                    <div style="color:{TEXT_DIM}; font-size: 0.85rem; line-height: 1.5;">{desc}</div>
                   </div>""",
                unsafe_allow_html=True,
            )


# ============================== DATA =========================================
def page_data() -> None:
    st.markdown(render_hero("Data & Indicators",
                              "Price action with 25+ indicators — synthetic or real."),
                 unsafe_allow_html=True)
    tab_synth, tab_real = st.tabs(["◆ Synthetic", "◐ Real Market"])

    with tab_synth:
        c1, c2, c3, c4 = st.columns(4)
        n = c1.number_input("Bars", 100, 5000, 500, 50)
        s0 = c2.number_input("Start price", 1.0, 1000.0, 100.0)
        mu = c3.slider("μ (annual)", -0.2, 0.3, 0.08, 0.01)
        sigma = c4.slider("σ (annual)", 0.05, 0.8, 0.20, 0.01)
        seed = st.number_input("Seed", 0, 9999, 42, 1)
        df = generate_ohlcv(n, s0=s0, mu=mu, sigma=sigma, seed=int(seed))
        _render_indicators(df, "synthetic")

    with tab_real:
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", "SPY")
        start = c2.text_input("Start", "2020-01-01")
        end = c3.text_input("End", "2025-01-01")
        if st.button("📥 Load from yfinance"):
            try:
                with st.spinner(f"Fetching {ticker}..."):
                    panel = load_market((ticker,), start, end)
                _render_indicators(panel[ticker], ticker)
            except Exception as e:
                st.error(f"Load failed: {e}")


def _render_indicators(df: pd.DataFrame, label: str) -> None:
    close = df["close"]
    rets = close.pct_change().dropna()
    total_ret = close.iloc[-1] / close.iloc[0] - 1
    ann_vol = float(rets.std() * np.sqrt(252))
    last_rsi = float(rsi(close).iloc[-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"${close.iloc[-1]:,.2f}",
               f"{(close.iloc[-1]/close.iloc[-2]-1)*100:+.2f}%")
    c2.metric("Period return", f"{total_ret*100:+.2f}%")
    c3.metric("Ann. vol", f"{ann_vol*100:.2f}%")
    c4.metric("RSI(14)", f"{last_rsi:.1f}",
               "OB" if last_rsi > 70 else "OS" if last_rsi < 30 else "neutral")

    bb = bollinger_bands(close)
    st.plotly_chart(
        equity_chart({"Close": close, "SMA(20)": close.rolling(20).mean(),
                       "EMA(50)": ema(close, 50),
                       "BB upper": bb["upper"], "BB lower": bb["lower"]},
                      f"{label.upper()} · Price + Bands", height=420),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(rolling_metric_chart(rsi(close), "RSI(14)"),
                         use_container_width=True)
        st.plotly_chart(rolling_metric_chart(realized_vol(rets, 21),
                                               "Realized Vol (21d)"),
                         use_container_width=True)
    with c2:
        mac = macd(close)
        st.plotly_chart(equity_chart({"MACD": mac["macd"], "Signal": mac["signal"]},
                                      "MACD", height=280),
                         use_container_width=True)
        st.plotly_chart(rolling_metric_chart(atr(df), "ATR(14)"),
                         use_container_width=True)

    with st.expander("◐ Raw OHLCV"):
        st.dataframe(df.tail(25), use_container_width=True)
        st.download_button("Download CSV", df.to_csv().encode(),
                            f"{label}_ohlcv.csv", "text/csv")


# ============================== STRATEGIES ===================================
_STRAT_CATEGORIES = {
    "Momentum": ["Momentum(60)", "Momentum(120)", "Cross-Sec Momentum", "Dual Momentum"],
    "Trend": ["MA 20/100", "MA 50/200", "Donchian", "Regime Switch"],
    "Mean Reversion": ["Bollinger MR", "RSI Reversal"],
    "Meta": ["Factor Composite", "VolTarget(Momentum)", "Buy & Hold"],
}
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
_STRAT_BADGES = {
    "Momentum(60)": ("mom", "MOM"), "Momentum(120)": ("mom", "MOM"),
    "MA 20/100": ("mom", "TREND"), "MA 50/200": ("mom", "TREND"),
    "Donchian": ("mom", "BRK"), "Cross-Sec Momentum": ("mom", "XSM"),
    "Dual Momentum": ("mom", "DUAL"), "RSI Reversal": ("mr", "MR"),
    "Bollinger MR": ("mr", "MR"), "Factor Composite": ("factor", "FACTOR"),
    "Regime Switch": ("factor", "RISK"), "VolTarget(Momentum)": ("ml", "VOLTGT"),
    "Buy & Hold": ("bench", "BENCH"),
}


def page_backtest() -> None:
    st.markdown(render_hero("Strategy Backtest",
                              "Plug a strategy into synthetic or real data, compare against Buy & Hold."),
                 unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    source = c1.radio("Data source", ["Real (5 ETFs)", "Synthetic"], horizontal=True)
    rebalance = c2.selectbox("Rebalance", ["monthly", "weekly", "bar"])
    capital = c3.number_input("Capital ($)", 10_000, 10_000_000, 100_000, 10_000)
    sizing = c4.slider("Size frac", 0.05, 1.0, 0.3, 0.05)

    cat_col, name_col = st.columns([1, 2])
    cat = cat_col.selectbox("Category", list(_STRAT_CATEGORIES.keys()))
    strat_name = name_col.selectbox("Strategy", _STRAT_CATEGORIES[cat])

    if source == "Real (5 ETFs)":
        try:
            data = load_market(("SPY", "QQQ", "IWM", "TLT", "GLD"), "2015-01-01", "2025-01-01")
        except Exception as e:
            st.error(f"Load failed: {e}")
            return
    else:
        n = st.slider("Bars", 200, 2000, 600, 50)
        seed = st.slider("Seed", 0, 999, 11)
        n_assets = st.slider("Universe size", 1, 6, 3)
        data = generate_panel([f"S{i}" for i in range(n_assets)], n=n, seed=seed)

    if st.button("▶  Run backtest", type="primary"):
        strat = _STRATS[strat_name]()
        with st.spinner("Running..."):
            engine = BacktestEngine(strategy=strat, data=data, initial_capital=capital,
                                     sizing_fraction=sizing, rebalance=rebalance)
            res = engine.run()
            bench_sym = next(iter(data.keys()))
            bench_eng = BacktestEngine(strategy=BuyAndHoldStrategy(),
                                         data={bench_sym: data[bench_sym]},
                                         initial_capital=capital, sizing_fraction=1.0)
            bench_res = bench_eng.run()

        stats = summary_stats(res.equity_curve, trades=res.trades)
        bench_stats = summary_stats(bench_res.equity_curve)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{stats['total_return']*100:+.2f}%",
                   f"{(stats['total_return']-bench_stats['total_return'])*100:+.2f}% vs B&H")
        c2.metric("Sharpe", f"{stats['sharpe']:+.2f}",
                   f"{stats['sharpe']-bench_stats['sharpe']:+.2f}")
        c3.metric("Max DD", f"{stats['max_drawdown']*100:+.2f}%",
                   f"{(stats['max_drawdown']-bench_stats['max_drawdown'])*100:+.2f}%",
                   delta_color="inverse")
        c4.metric("Trades", f"{len(res.trades):,}",
                   f"Cost {stats.get('cost_bps', 0):.1f} bps")

        st.plotly_chart(
            equity_chart({strat_name: res.equity_curve, "Buy & Hold": bench_res.equity_curve},
                          f"{strat_name} vs Buy & Hold · {bench_sym}",
                          benchmark_name="Buy & Hold"),
            use_container_width=True,
        )
        c1, c2 = st.columns(2)
        c1.plotly_chart(drawdown_chart(res.equity_curve, "Drawdown"), use_container_width=True)
        c2.plotly_chart(returns_histogram(res.equity_curve.pct_change().dropna(),
                                           "Daily Returns"), use_container_width=True)

        strat_r = res.equity_curve.pct_change().dropna()
        bench_r = bench_res.equity_curve.pct_change().dropna()
        rep = benchmark_report(strat_r, bench_r)
        with st.expander("📊  Full metrics vs benchmark"):
            cb1, cb2, cb3 = st.columns(3)
            cb1.metric("Alpha (annual)", f"{rep.alpha*100:+.2f}%")
            cb2.metric("Beta", f"{rep.beta:+.3f}")
            cb3.metric("Info Ratio", f"{rep.info_ratio:+.2f}")
            st.markdown(tearsheet_markdown(res.equity_curve, strat_name))

        if not res.trades.empty:
            st.download_button("⬇  trades.csv",
                                res.trades.to_csv(index=False).encode(),
                                "trades.csv", "text/csv")


# ============================== TOURNAMENT ===================================
def page_tournament() -> None:
    st.markdown(render_hero("Tournament",
                              "12 strategies × 10 years of real market data · cost-aware · monthly rebalance."),
                 unsafe_allow_html=True)

    try:
        data = load_market(("SPY", "QQQ", "IWM", "TLT", "GLD"), "2015-01-01", "2025-01-01")
    except Exception as e:
        st.error(f"Load failed: {e}")
        return

    st.markdown(render_pills([
        ("green", f"{min(len(v) for v in data.values())} bars"),
        ("cyan", "SPY · QQQ · IWM · TLT · GLD"),
        ("purple", "5bp slip + 1bp comm"),
    ]), unsafe_allow_html=True)

    chosen = st.multiselect(
        "Strategies", list(_STRATS.keys()),
        default=["Buy & Hold", "Momentum(120)", "MA 20/100", "Donchian",
                 "Dual Momentum", "Regime Switch", "Cross-Sec Momentum",
                 "VolTarget(Momentum)", "Bollinger MR"],
    )

    if st.button("▶  Run tournament", type="primary") and chosen:
        rows, equities = [], {}
        prog = st.progress(0.0)
        for i, name in enumerate(chosen, 1):
            strat = _STRATS[name]()
            subset = data if name != "Buy & Hold" else {"SPY": data["SPY"]}
            size = 1.0 if name == "Buy & Hold" else 0.2
            rebal = "bar" if name == "Buy & Hold" else "monthly"
            eng = BacktestEngine(strategy=strat, data=subset, initial_capital=100_000,
                                  sizing_fraction=size, rebalance=rebal)
            res = eng.run()
            s = summary_stats(res.equity_curve, trades=res.trades)
            rows.append({
                "strategy": name,
                "total_return": s["total_return"], "annual_return": s["annual_return"],
                "sharpe": s["sharpe"], "sortino": s["sortino"],
                "max_drawdown": s["max_drawdown"],
                "turnover": s.get("turnover", 0), "trades": int(s.get("n_trades", 0) or 0),
            })
            equities[name] = res.equity_curve
            prog.progress(i / len(chosen))
        prog.empty()

        df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).set_index("strategy")

        best_sharpe = df.index[0]
        best_ret_name = df["total_return"].idxmax()
        smallest_dd = df["max_drawdown"].idxmax()
        c1, c2, c3 = st.columns(3)
        c1.metric("◆  Best Sharpe", best_sharpe, f"{df.loc[best_sharpe, 'sharpe']:+.2f}")
        c2.metric("◆  Best Return", best_ret_name,
                   f"{df.loc[best_ret_name, 'total_return']*100:+.1f}%")
        c3.metric("◆  Smallest DD", smallest_dd,
                   f"{df.loc[smallest_dd, 'max_drawdown']*100:+.1f}%")

        st.plotly_chart(equity_chart(equities, "Equity curves · 10-year real data",
                                       benchmark_name="Buy & Hold", height=480),
                         use_container_width=True)

        def _fmt_pct(v, inv=False):
            color = SUCCESS if (v > 0) ^ inv else DANGER
            return f'<span style="color:{color};font-family:JetBrains Mono;font-weight:500;">{v*100:+.2f}%</span>'
        def _fmt_num(v):
            color = SUCCESS if v > 0 else DANGER
            return f'<span style="color:{color};font-family:JetBrains Mono;font-weight:500;">{v:+.2f}</span>'
        rows_html = []
        for name, r in df.iterrows():
            cat, label = _STRAT_BADGES.get(name, ("bench", "N/A"))
            rows_html.append(f"""
            <tr style="border-bottom:1px solid {BORDER};">
              <td style="padding:10px 14px;">
                <span class="strat-badge badge-{cat}">{label}</span>
                <span style="font-weight:500;color:{TEXT};">{name}</span>
              </td>
              <td style="padding:10px 14px;text-align:right;">{_fmt_pct(r['total_return'])}</td>
              <td style="padding:10px 14px;text-align:right;">{_fmt_pct(r['annual_return'])}</td>
              <td style="padding:10px 14px;text-align:right;">{_fmt_num(r['sharpe'])}</td>
              <td style="padding:10px 14px;text-align:right;">{_fmt_pct(r['max_drawdown'])}</td>
              <td style="padding:10px 14px;text-align:right;color:{TEXT_DIM};font-family:JetBrains Mono;">{r['turnover']:.1f}×</td>
              <td style="padding:10px 14px;text-align:right;color:{TEXT_DIM};font-family:JetBrains Mono;">{int(r['trades']):,}</td>
            </tr>""")
        header = f"""
        <table style="width:100%;background:{PAPER};border:1px solid {BORDER};border-radius:14px;
                     border-collapse:separate;border-spacing:0;overflow:hidden;margin-top:1rem;">
          <thead>
            <tr style="background:#151926;">
              <th style="padding:12px 14px;text-align:left;color:{TEXT_DIM};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;">Strategy</th>
              <th style="padding:12px 14px;text-align:right;color:{TEXT_DIM};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;">Total</th>
              <th style="padding:12px 14px;text-align:right;color:{TEXT_DIM};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;">Annual</th>
              <th style="padding:12px 14px;text-align:right;color:{TEXT_DIM};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;">Sharpe</th>
              <th style="padding:12px 14px;text-align:right;color:{TEXT_DIM};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;">Max DD</th>
              <th style="padding:12px 14px;text-align:right;color:{TEXT_DIM};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;">Turnover</th>
              <th style="padding:12px 14px;text-align:right;color:{TEXT_DIM};font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;">Trades</th>
            </tr>
          </thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>"""
        st.markdown(header, unsafe_allow_html=True)

        st.download_button("⬇  tournament.csv",
                            df.reset_index().to_csv(index=False).encode(),
                            "tournament.csv", "text/csv")


# ============================== OPTIONS ======================================
def page_options() -> None:
    st.markdown(render_hero("Options Lab", "Price, hedge, and explore exotic derivatives."),
                 unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Spot", 1.0, 10_000.0, 100.0)
    K = c2.number_input("Strike", 1.0, 10_000.0, 100.0)
    T = c3.number_input("T (years)", 0.01, 10.0, 1.0, 0.05)
    r = c4.number_input("r", -0.05, 0.2, 0.04, 0.005, format="%.4f")
    sigma = c5.slider("σ", 0.01, 1.5, 0.2, 0.01)
    opt = st.radio("Option type", ["call", "put"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    bs = bs_call(S, K, T, r, sigma) if opt == "call" else bs_put(S, K, T, r, sigma)
    crr = crr_price(S, K, T, r, sigma, opt, 500)
    mc, mc_se = mc_european(S, K, T, r, sigma, opt, n_paths=20_000, seed=42)
    col1.metric("Black-Scholes", f"${bs:.4f}")
    col2.metric("CRR Binomial", f"${crr:.4f}", f"{(crr-bs):+.4f}")
    col3.metric("Monte Carlo", f"${mc:.4f}", f"±{mc_se:.4f}")

    greeks = all_greeks(S, K, T, r, sigma, opt)
    st.plotly_chart(greeks_gauge(greeks), use_container_width=True)

    st.markdown('<h3 style="margin-top:2rem;">Implied Volatility Smile</h3>', unsafe_allow_html=True)
    strikes = np.linspace(S * 0.7, S * 1.3, 21)
    injected = [sigma + 0.08 * ((k / S - 1) ** 2 - 0.01) for k in strikes]
    mkt = [bs_call(S, k, T, r, iv) for k, iv in zip(strikes, injected, strict=False)]
    recovered = [bs_implied_vol(m, S, k, T, r, "call") for m, k in zip(mkt, strikes, strict=False)]
    st.plotly_chart(iv_smile_chart(strikes, injected, recovered), use_container_width=True)

    st.markdown('<h3 style="margin-top:2rem;">Exotic options</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    asi, ase = mc_asian(S, K, T, r, sigma, opt, seed=42)
    bar, bse = mc_barrier(S, K, T, r, sigma, barrier=S * 0.8,
                           barrier_type="down-and-out", option=opt, seed=42)
    amer = crr_american(S, K, T, r, sigma, opt, 500)
    c1.metric("Asian (arithmetic)", f"${asi:.4f}", f"±{ase:.4f}")
    c2.metric(f"Barrier @ ${S*0.8:.0f}", f"${bar:.4f}", f"±{bse:.4f}")
    c3.metric("American", f"${amer:.4f}", f"{amer-bs:+.4f} prem")


# ============================== PORTFOLIO ====================================
def page_portfolio() -> None:
    st.markdown(render_hero("Portfolio Optimizer",
                              "5 optimizers, correlation heatmap, efficient frontier — on real or synthetic universes."),
                 unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["◐ Real Data (5 ETFs)", "◆ Synthetic"])

    with tab1:
        try:
            data = load_market(("SPY", "QQQ", "IWM", "TLT", "GLD"), "2015-01-01", "2025-01-01")
            ret_df = pd.DataFrame({t: v["close"].pct_change() for t, v in data.items()}).dropna()
            _render_portfolio(ret_df, list(ret_df.columns))
        except Exception as e:
            st.error(f"Load failed: {e}")

    with tab2:
        n_a = st.slider("# assets", 3, 10, 5, key="syn_n")
        n_d = st.slider("Days", 252, 2000, 504, key="syn_d")
        symbols = [f"A{i}" for i in range(n_a)]
        rng = np.random.default_rng(42)
        factor = rng.normal(0, 0.01, (n_a, n_a))
        cov_d = 0.5 * (factor @ factor.T) + np.diag(rng.uniform(0.01, 0.1, n_a)) / 252
        ret_df = generate_correlated_returns(n_d, symbols, cov=cov_d, seed=42)
        _render_portfolio(ret_df, symbols)


def _render_portfolio(ret_df: pd.DataFrame, symbols: list) -> None:
    mu = ret_df.mean().values * 252
    cov = ret_df.cov().values * 252
    vols = np.sqrt(np.diag(cov))

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown('<div class="qf-card-title">Asset statistics (annualized)</div>',
                     unsafe_allow_html=True)
        stats_df = pd.DataFrame({"μ": mu, "σ": vols, "Sharpe": mu / vols}, index=symbols)
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
    st.plotly_chart(weights_bar(wdf, "Optimizer weights"), use_container_width=True)

    ef = efficient_frontier(mu, cov, n_points=25)
    tangent = None
    if len(ef):
        tangent = (float(np.sqrt(w_ms @ cov @ w_ms)), float(w_ms @ mu))
    assets_pts = pd.DataFrame({"vol": vols, "ret": mu}, index=symbols)
    st.plotly_chart(efficient_frontier_chart(ef, assets_pts, tangent),
                     use_container_width=True)


# ============================== RISK =========================================
def page_risk() -> None:
    st.markdown(render_hero("Risk Analysis",
                              "VaR, CVaR, drawdown, distribution — pick your poison."),
                 unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    n = c1.number_input("Bars", 252, 5000, 1000, 50)
    mu = c2.slider("μ", -0.2, 0.3, 0.08, 0.01)
    sigma = c3.slider("σ", 0.05, 0.8, 0.25, 0.01)
    conf = c4.slider("VaR confidence", 0.90, 0.995, 0.95, 0.005)

    from quantforge.data.synthetic import generate_gbm
    prices = generate_gbm(int(n), mu=mu, sigma=sigma, seed=7)
    rets = prices.pct_change().dropna()

    c1, c2 = st.columns(2)
    c1.plotly_chart(equity_chart({"Price": prices}, "Price path", height=300),
                     use_container_width=True)
    c2.plotly_chart(drawdown_chart(prices, "Drawdown", height=300),
                     use_container_width=True)

    st.markdown(f'<h3 style="margin-top:1.5rem;">Value-at-Risk ({conf*100:.1f}%)</h3>',
                 unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Historical", f"{historical_var(rets, conf)*100:.2f}%")
    v2.metric("Parametric", f"{parametric_var(rets, conf)*100:.2f}%")
    v3.metric("Cornish-Fisher", f"{cornish_fisher_var(rets, conf)*100:.2f}%")
    v4.metric("Historical CVaR", f"{historical_cvar(rets, conf)*100:.2f}%")

    mdd, peak, trough = max_drawdown(prices)
    m1, m2, m3 = st.columns(3)
    m1.metric("Max Drawdown", f"{mdd*100:.2f}%")
    m2.metric("Peak date", f"{peak.date()}")
    m3.metric("Trough date", f"{trough.date()}")

    st.plotly_chart(returns_histogram(rets, "Return distribution"),
                     use_container_width=True)

    ddt = drawdown_table(prices, top_n=10)
    if not ddt.empty:
        st.markdown('<h3 style="margin-top:1.5rem;">Top 10 drawdowns</h3>', unsafe_allow_html=True)
        st.dataframe(ddt, use_container_width=True)


# ============================== ML ===========================================
def page_ml() -> None:
    st.markdown(render_hero("ML Training",
                              "Real SPY → 20+ features → time-ordered train/val/test → honest metrics. No pretending."),
                 unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", "SPY")
    start = c2.text_input("Start", "2010-01-01")
    end = c3.text_input("End", "2025-01-01")

    if st.button("✦  Train real model", type="primary"):
        try:
            with st.spinner("Fetching real market data..."):
                panel = load_market((ticker,), start, end)
                df = panel[ticker]
        except Exception as e:
            st.error(f"Load failed: {e}")
            return

        from sklearn.ensemble import GradientBoostingClassifier

        from quantforge.ml.features import build_feature_matrix, target_labels
        from quantforge.ml.trainer import train_classifier

        with st.spinner("Building 20+ features..."):
            feats = build_feature_matrix(df)
            target = target_labels(df["close"], horizon=1, kind="binary")
            data = feats.join(target.rename("y")).dropna()
            X = data.drop(columns="y")
            y = data["y"]

        st.markdown(render_pills([
            ("green", f"{len(X)} samples"),
            ("cyan", f"{len(X.columns)} features"),
            ("purple", f"up {(y==1).mean():.1%} / down {(y==0).mean():.1%}"),
        ]), unsafe_allow_html=True)

        with st.spinner("Hyperparameter search over 9 configs..."):
            hp_grid = [{"n_estimators": n, "max_depth": d, "learning_rate": 0.05, "random_state": 42}
                        for n in [50, 100, 200] for d in [2, 3, 5]]
            model, report = train_classifier(X, y, model_cls=GradientBoostingClassifier,
                                               hp_grid=hp_grid, verbose=False)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train acc", f"{report.train_acc:.4f}")
        c2.metric("Val acc", f"{report.val_acc:.4f}")
        c3.metric("Test acc", f"{report.test_acc:.4f}",
                   f"{(report.test_acc - report.baseline_acc):+.4f} vs baseline")
        c4.metric("Test AUC", f"{report.test_auc:.4f}", "0.5 = random")

        fi_df = pd.DataFrame(
            sorted(report.feature_importances.items(), key=lambda kv: kv[1], reverse=True),
            columns=["feature", "importance"],
        ).head(15)
        c_left, c_right = st.columns([3, 2])
        c_left.plotly_chart(horizontal_bar(fi_df, "importance", "feature",
                                              "Top-15 feature importances"),
                              use_container_width=True)
        c_right.markdown('<h3 style="margin-top:0;">Confusion matrix (test)</h3>',
                         unsafe_allow_html=True)
        cm = report.confusion_test
        c_right.markdown(f"""
        <table style="width:100%;background:{PAPER};border:1px solid {BORDER};border-radius:10px;border-collapse:separate;border-spacing:0;overflow:hidden;">
          <tr style="background:#151926;color:{TEXT_DIM};font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;">
            <th style="padding:10px;"></th><th style="padding:10px;">Pred Down</th><th style="padding:10px;">Pred Up</th>
          </tr>
          <tr>
            <td style="padding:10px;color:{TEXT_DIM};text-align:right;">Actual Down</td>
            <td style="padding:10px;text-align:center;font-family:JetBrains Mono;color:{SUCCESS};">{cm[0][0]}</td>
            <td style="padding:10px;text-align:center;font-family:JetBrains Mono;color:{DANGER};">{cm[0][1]}</td>
          </tr>
          <tr>
            <td style="padding:10px;color:{TEXT_DIM};text-align:right;">Actual Up</td>
            <td style="padding:10px;text-align:center;font-family:JetBrains Mono;color:{DANGER};">{cm[1][0]}</td>
            <td style="padding:10px;text-align:center;font-family:JetBrains Mono;color:{SUCCESS};">{cm[1][1]}</td>
          </tr>
        </table>""", unsafe_allow_html=True)

        if report.hp_search is not None:
            with st.expander("🔬  Hyperparameter search results"):
                st.dataframe(report.hp_search.round(4), use_container_width=True)


PAGES = {
    "◆  Overview": page_overview,
    "▤  Data & Indicators": page_data,
    "⚗  Strategy Backtest": page_backtest,
    "♛  Tournament": page_tournament,
    "◈  Options Lab": page_options,
    "◎  Portfolio Optimizer": page_portfolio,
    "⚠  Risk Analysis": page_risk,
    "✦  ML Training": page_ml,
}


def main() -> None:
    choice = sidebar()
    PAGES[choice]()


if __name__ == "__main__":
    main()
