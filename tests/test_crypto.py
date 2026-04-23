"""Tests for crypto data helpers (no network — monkeypatched DataLoader)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantforge.data import crypto as crypto_mod


class TestNormalization:
    def test_normalize_aliases(self):
        assert crypto_mod.normalize_crypto_ticker("BTC") == "BTC-USD"
        assert crypto_mod.normalize_crypto_ticker("bitcoin") == "BTC-USD"
        assert crypto_mod.normalize_crypto_ticker("ETH") == "ETH-USD"
        assert crypto_mod.normalize_crypto_ticker("SOL") == "SOL-USD"

    def test_normalize_already_qualified(self):
        assert crypto_mod.normalize_crypto_ticker("BTC-USD") == "BTC-USD"
        assert crypto_mod.normalize_crypto_ticker("ETH-EUR") == "ETH-EUR"

    def test_normalize_bare_symbol_appends_usd(self):
        # Not in aliases but valid format → append -USD
        assert crypto_mod.normalize_crypto_ticker("XYZ") == "XYZ-USD"

    def test_normalize_lowercase(self):
        assert crypto_mod.normalize_crypto_ticker("eth-usd") == "ETH-USD"

    def test_invalid_rejected(self):
        with pytest.raises(ValueError):
            crypto_mod.normalize_crypto_ticker("bad ticker!")
        with pytest.raises(ValueError):
            crypto_mod.normalize_crypto_ticker("")


class TestLoadCrypto:
    def test_load_uses_dataloader(self, monkeypatch):
        calls = {}
        class FakeLoader:
            def __init__(self, cache_dir=None): pass
            def yfinance(self, symbol, start, end, interval="1d"):
                calls["symbol"] = symbol
                # return a tiny fake frame
                idx = pd.date_range(start, periods=3, freq="D")
                return pd.DataFrame(
                    {"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
                     "close": [1, 2, 3], "volume": [100, 200, 300]}, index=idx,
                )
        monkeypatch.setattr(crypto_mod, "DataLoader", FakeLoader)
        df = crypto_mod.load_crypto("BTC", start="2024-01-01")
        assert calls["symbol"] == "BTC-USD"
        assert len(df) == 3

    def test_load_panel_filters_empty(self, monkeypatch):
        class FakeLoader:
            def __init__(self, cache_dir=None): pass
            def yfinance(self, symbol, start, end, interval="1d"):
                if symbol == "ETH-USD":
                    return pd.DataFrame()
                idx = pd.date_range(start, periods=2, freq="D")
                return pd.DataFrame({"close": [1, 2]}, index=idx)
        monkeypatch.setattr(crypto_mod, "DataLoader", FakeLoader)
        panel = crypto_mod.load_crypto_panel(["BTC", "ETH"], start="2024-01-01")
        assert "BTC-USD" in panel
        assert "ETH-USD" not in panel  # filtered out empty result


class TestCryptoVol:
    def test_vol_uses_365_periods(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 1000))
        vol = crypto_mod.crypto_volatility_24_7(r)
        # crypto annualization ~ daily_std * sqrt(365)
        expected = r.std(ddof=1) * np.sqrt(365)
        assert abs(vol - expected) < 1e-9

    def test_vol_nan_on_empty(self):
        assert np.isnan(crypto_mod.crypto_volatility_24_7(pd.Series(dtype=float)))
