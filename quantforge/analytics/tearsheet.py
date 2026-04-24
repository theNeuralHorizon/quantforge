"""Tearsheet generation (text + markdown + optional matplotlib)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantforge.analytics.performance import summary_stats
from quantforge.risk.drawdown import drawdown_table


@dataclass
class Tearsheet:
    equity: pd.Series
    name: str = "Strategy"
    benchmark: pd.Series | None = None

    def stats(self) -> dict:
        return summary_stats(self.equity)

    def drawdowns(self, top_n: int = 5) -> pd.DataFrame:
        return drawdown_table(self.equity, top_n=top_n)

    def to_markdown(self) -> str:
        return tearsheet_markdown(self.equity, self.name)

    def to_text(self) -> str:
        return tearsheet_text(self.equity, self.name)


def _fmt_pct(x: float) -> str:
    try:
        return f"{x * 100:+.2f}%"
    except Exception:
        return str(x)


def _fmt_num(x: float, n: int = 2) -> str:
    try:
        return f"{x:,.{n}f}"
    except Exception:
        return str(x)


def tearsheet_text(equity: pd.Series, name: str = "Strategy") -> str:
    s = summary_stats(equity)
    lines = [
        f"=== {name} Tearsheet ===",
        f"Total return:       {_fmt_pct(s['total_return'])}",
        f"Annual return:      {_fmt_pct(s['annual_return'])}",
        f"Annual vol:         {_fmt_pct(s['annual_vol'])}",
        f"Sharpe:             {_fmt_num(s['sharpe'])}",
        f"Sortino:            {_fmt_num(s['sortino'])}",
        f"Calmar:             {_fmt_num(s['calmar'])}",
        f"Omega:              {_fmt_num(s['omega'])}",
        f"Max Drawdown:       {_fmt_pct(s['max_drawdown'])}",
        f"VaR(95):            {_fmt_pct(s['var_95'])}",
        f"CVaR(95):           {_fmt_pct(s['cvar_95'])}",
        f"Best day:           {_fmt_pct(s['best_day'])}",
        f"Worst day:          {_fmt_pct(s['worst_day'])}",
        f"Skew:               {_fmt_num(s['skew'])}",
        f"Kurtosis:           {_fmt_num(s['kurt'])}",
    ]
    return "\n".join(lines)


def tearsheet_markdown(equity: pd.Series, name: str = "Strategy") -> str:
    s = summary_stats(equity)
    rows = [
        ("Total Return", _fmt_pct(s["total_return"])),
        ("Annual Return", _fmt_pct(s["annual_return"])),
        ("Annual Vol", _fmt_pct(s["annual_vol"])),
        ("Sharpe", _fmt_num(s["sharpe"])),
        ("Sortino", _fmt_num(s["sortino"])),
        ("Calmar", _fmt_num(s["calmar"])),
        ("Omega", _fmt_num(s["omega"])),
        ("Max Drawdown", _fmt_pct(s["max_drawdown"])),
        ("VaR (95%)", _fmt_pct(s["var_95"])),
        ("CVaR (95%)", _fmt_pct(s["cvar_95"])),
        ("Best Day", _fmt_pct(s["best_day"])),
        ("Worst Day", _fmt_pct(s["worst_day"])),
        ("Skew", _fmt_num(s["skew"])),
        ("Kurtosis", _fmt_num(s["kurt"])),
    ]
    md = [f"# {name} Tearsheet", "", "| Metric | Value |", "|---|---|"]
    md.extend(f"| {k} | {v} |" for k, v in rows)
    return "\n".join(md)
