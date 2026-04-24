"""Drawdown analytics."""
from __future__ import annotations

import pandas as pd


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Return (max_dd, peak_date, trough_date)."""
    dd = drawdown_series(equity)
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def underwater_duration(equity: pd.Series) -> pd.Series:
    """Duration (in index steps) since last all-time high."""
    peak = equity.cummax()
    is_uw = equity < peak
    groups = (is_uw != is_uw.shift()).cumsum()
    counts = is_uw.groupby(groups).cumsum()
    return counts


def drawdown_table(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Top-N drawdowns with peak/trough/recovery dates."""
    dd = drawdown_series(equity)
    peak = equity.cummax()
    in_dd = dd < 0
    change = in_dd.ne(in_dd.shift())
    groups = change.cumsum()

    rows = []
    for _grp_id, grp in dd[in_dd].groupby(groups[in_dd]):
        if grp.empty:
            continue
        trough = grp.idxmin()
        peak_date = equity.loc[:trough].idxmax()
        recovery = None
        post = equity.loc[trough:]
        back = post[post >= peak.loc[trough]]
        if not back.empty:
            recovery = back.index[0]
        rows.append({
            "peak": peak_date,
            "trough": trough,
            "recovery": recovery,
            "depth": float(grp.min()),
            "length_days": (trough - peak_date).days if hasattr(trough - peak_date, "days") else None,
            "recovery_days": (recovery - trough).days if recovery is not None and hasattr(recovery - trough, "days") else None,
        })
    df = pd.DataFrame(rows).sort_values("depth").head(top_n).reset_index(drop=True)
    return df
