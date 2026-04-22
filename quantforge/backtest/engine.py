"""Event-driven backtest engine orchestrating strategy + broker + portfolio.

Performance notes
-----------------
The strategy API takes a pandas DataFrame of history at each `on_bar` call. The
naive implementation (re-building the DataFrame every bar from a growing list)
is O(N^2) — for a 2516-bar * 5-asset backtest this becomes the dominant cost.
We now maintain per-symbol columnar NumPy buffers that grow append-only; we
only materialize a DataFrame when actually handed to the strategy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from quantforge.backtest.broker import SimulatedBroker
from quantforge.core.event import SignalEvent
from quantforge.core.order import Order, OrderSide, OrderType
from quantforge.core.portfolio import Portfolio


class Strategy(Protocol):
    """Duck-typed strategy interface."""
    def on_bar(self, symbol: str, bar: pd.Series, history: pd.DataFrame) -> List[SignalEvent]: ...
    def warmup(self) -> int: ...


_OHLCV_COLS = ("open", "high", "low", "close", "volume")


class _SymbolBuffer:
    """Append-only numeric buffer for per-symbol OHLCV + timestamps."""

    __slots__ = ("_ts", "_cols", "_n", "_cap")

    def __init__(self, initial_cap: int = 256) -> None:
        self._ts: List[pd.Timestamp] = []
        self._cols: Dict[str, np.ndarray] = {c: np.empty(initial_cap, dtype=np.float64) for c in _OHLCV_COLS}
        self._n = 0
        self._cap = initial_cap

    def _grow(self, new_cap: int) -> None:
        for c in _OHLCV_COLS:
            buf = np.empty(new_cap, dtype=np.float64)
            buf[: self._n] = self._cols[c][: self._n]
            self._cols[c] = buf
        self._cap = new_cap

    def append(self, ts: pd.Timestamp, o: float, h: float, low: float, c: float, v: float) -> None:
        if self._n == self._cap:
            self._grow(self._cap * 2)
        i = self._n
        self._cols["open"][i] = o
        self._cols["high"][i] = h
        self._cols["low"][i] = low
        self._cols["close"][i] = c
        self._cols["volume"][i] = v
        self._ts.append(ts)
        self._n += 1

    def __len__(self) -> int:
        return self._n

    def as_frame(self, tail: Optional[int] = None) -> pd.DataFrame:
        """Return a DataFrame view. Zero-copy where possible via np.asarray slicing."""
        if self._n == 0:
            return pd.DataFrame(columns=list(_OHLCV_COLS))
        start = 0 if tail is None else max(0, self._n - tail)
        idx = pd.DatetimeIndex(self._ts[start:self._n])
        return pd.DataFrame(
            {c: self._cols[c][start:self._n].copy() for c in _OHLCV_COLS},
            index=idx,
        )


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    positions_over_time: pd.DataFrame
    initial_capital: float

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().fillna(0.0)

    @property
    def total_return(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1)

    @property
    def turnover(self) -> float:
        """Total traded notional / (avg equity)."""
        if self.trades.empty or len(self.equity_curve) == 0:
            return 0.0
        notional = (self.trades["qty"].abs() * self.trades["price"]).sum()
        return float(notional / self.equity_curve.mean())

    @property
    def total_cost(self) -> float:
        if self.trades.empty:
            return 0.0
        slip_cost = (self.trades["qty"].abs() * self.trades["slippage"]).sum()
        return float(self.trades["commission"].sum() + slip_cost)

    def to_dict(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "final_equity": float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.initial_capital,
            "total_return": self.total_return,
            "n_trades": len(self.trades),
            "turnover": self.turnover,
            "total_cost": self.total_cost,
        }


@dataclass
class BacktestEngine:
    strategy: Strategy
    data: Dict[str, pd.DataFrame]
    initial_capital: float = 100_000.0
    broker: Optional[SimulatedBroker] = None
    sizing_fraction: float = 0.1
    target_weights: bool = False
    rebalance: str = "bar"  # "bar" | "weekly" | "monthly"
    history_tail: Optional[int] = None  # if set, strategies see only the last N bars

    portfolio: Portfolio = field(init=False)
    _buffers: Dict[str, _SymbolBuffer] = field(init=False)

    def __post_init__(self) -> None:
        self.portfolio = Portfolio(initial_capital=self.initial_capital)
        self.broker = self.broker or SimulatedBroker()
        self._buffers = {s: _SymbolBuffer() for s in self.data}

    def _size_order(self, symbol: str, direction: int, strength: float, price: float) -> float:
        if price <= 0:
            return 0.0
        current_qty = self.portfolio.position(symbol).quantity
        if direction == 0:
            return -current_qty
        if self.target_weights:
            target_value = direction * strength * self.portfolio.equity
        else:
            alloc = self.portfolio.equity * self.sizing_fraction * max(0.1, min(1.0, strength))
            target_value = direction * alloc
        target_qty = target_value / price
        return target_qty - current_qty

    def _should_rebalance(self, ts: pd.Timestamp, prev_ts: Optional[pd.Timestamp]) -> bool:
        if self.rebalance == "bar":
            return True
        if prev_ts is None:
            return True
        if self.rebalance == "weekly":
            return ts.isocalendar().week != prev_ts.isocalendar().week
        if self.rebalance == "monthly":
            return (ts.year, ts.month) != (prev_ts.year, prev_ts.month)
        return True

    def run(self) -> BacktestResult:
        all_index = sorted({ts for df in self.data.values() for ts in df.index})
        warmup = self.strategy.warmup()

        trades: List[dict] = []
        positions_snapshots: List[dict] = []
        last_rebalance_ts: Optional[pd.Timestamp] = None

        # Pre-extract numpy columns once — avoids df.loc[ts] in hot loop
        col_cache: Dict[str, Dict[str, np.ndarray]] = {}
        ts_idx_cache: Dict[str, Dict[pd.Timestamp, int]] = {}
        for s, df in self.data.items():
            col_cache[s] = {c: df[c].values for c in _OHLCV_COLS if c in df.columns}
            # fill any missing OHLCV columns with close
            if "volume" not in col_cache[s]:
                col_cache[s]["volume"] = np.ones(len(df))
            ts_idx_cache[s] = {ts: i for i, ts in enumerate(df.index)}

        for i, ts in enumerate(all_index):
            current_prices: Dict[str, float] = {}
            for symbol in self.data:
                idx = ts_idx_cache[symbol].get(ts)
                if idx is None:
                    continue
                cc = col_cache[symbol]
                o, h, low, c, v = cc["open"][idx], cc["high"][idx], cc["low"][idx], cc["close"][idx], cc["volume"][idx]
                current_prices[symbol] = float(c)
                self._buffers[symbol].append(ts, o, h, low, c, v)

                fills = self.broker.on_bar(ts, symbol, {"open": o, "high": h, "low": low, "close": c, "volume": v})
                for f in fills:
                    signed_qty = f.quantity * f.direction
                    self.portfolio.apply_fill(symbol, signed_qty, f.fill_price, f.commission)
                    trades.append({
                        "timestamp": f.timestamp, "symbol": symbol, "qty": signed_qty,
                        "price": f.fill_price, "commission": f.commission, "slippage": f.slippage,
                    })

            self.portfolio.mark_to_market(current_prices)

            if i >= warmup and self._should_rebalance(ts, last_rebalance_ts):
                for symbol in self.data:
                    idx = ts_idx_cache[symbol].get(ts)
                    if idx is None:
                        continue
                    cc = col_cache[symbol]
                    # Construct bar as a pandas Series with ts as .name (strategies rely on this)
                    bar = pd.Series({c: cc[c][idx] for c in _OHLCV_COLS}, name=ts)
                    history_df = self._buffers[symbol].as_frame(tail=self.history_tail)
                    signals = self.strategy.on_bar(symbol, bar, history_df)
                    for sig in signals:
                        px = current_prices.get(sig.symbol, current_prices.get(symbol, float(cc["close"][idx])))
                        qty = self._size_order(sig.symbol, sig.direction, sig.strength, px)
                        if abs(qty) < 1e-8:
                            continue
                        order = Order(
                            symbol=sig.symbol,
                            quantity=abs(qty),
                            side=OrderSide.BUY if qty > 0 else OrderSide.SELL,
                            order_type=OrderType.MARKET,
                        )
                        self.broker.submit(order)
                last_rebalance_ts = ts

            self.portfolio.record_equity(ts)
            snap = {s: p.quantity for s, p in self.portfolio.positions.items()}
            snap["_ts"] = ts
            positions_snapshots.append(snap)

        eq_ts = [t for t, _ in self.portfolio.equity_curve]
        eq_v = [v for _, v in self.portfolio.equity_curve]
        equity_curve = pd.Series(eq_v, index=pd.DatetimeIndex(eq_ts), name="equity")
        trades_df = pd.DataFrame(trades)
        positions_df = pd.DataFrame(positions_snapshots).set_index("_ts").fillna(0.0)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            positions_over_time=positions_df,
            initial_capital=self.initial_capital,
        )
