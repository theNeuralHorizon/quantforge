"""/v1/market routes — ticker data pass-through with cache."""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query

from quantforge.api.auth import verify_api_key
from quantforge.api.cache import cache_backend, make_key
from quantforge.api.schemas import MarketDataResponse
from quantforge.data.loader import DataLoader


router = APIRouter(prefix="/v1/market", tags=["market"])

_TICKER_RX = re.compile(r"^[A-Z0-9\-.]{1,10}$")


@router.get("/data/{ticker}", response_model=MarketDataResponse)
def market_data(
    ticker: str,
    start: str = Query("2020-01-01"),
    end: str = Query("2025-01-01"),
    sample: int = Query(200, ge=0, le=5000),
    _key: str = Depends(verify_api_key),
) -> MarketDataResponse:
    ticker = ticker.strip().upper()
    if not _TICKER_RX.match(ticker):
        raise HTTPException(status_code=400, detail="invalid ticker")

    cache = cache_backend()
    ck = make_key("market_data", t=ticker, s=start, e=end, n=sample)
    hit = cache.get(ck)
    if hit:
        import json
        return MarketDataResponse(**json.loads(hit))

    try:
        df = DataLoader(cache_dir="data/cache").yfinance(ticker, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"data fetch failed: {e}")
    if df.empty:
        raise HTTPException(status_code=404, detail="no data")

    step = max(1, len(df) // sample) if sample else 1
    sampled = df.iloc[::step]
    bars = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        for idx, row in sampled.iterrows()
    ]
    ytd = None
    last = df["close"].iloc[-1]
    last_year = df.index[-1].year
    ytd_df = df[df.index.year == last_year]
    if len(ytd_df) > 2:
        ytd = float(last / ytd_df["close"].iloc[0] - 1)

    payload = MarketDataResponse(
        ticker=ticker, n_bars=len(df),
        start_date=df.index[0].strftime("%Y-%m-%d"),
        end_date=df.index[-1].strftime("%Y-%m-%d"),
        last_close=float(last), ytd_return=ytd, bars=bars,
    )
    import json
    cache.set(ck, json.dumps(payload.model_dump(), default=str).encode(), ttl_seconds=900)
    return payload
