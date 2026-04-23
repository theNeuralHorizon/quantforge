"""Thin, stdlib-only (httpx) QuantForge REST client."""
from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urljoin

import httpx


class QuantForgeError(Exception):
    """Raised on HTTP errors. Exposes the parsed JSON body if available."""
    def __init__(self, status: int, detail: Any, url: str = ""):
        self.status = status
        self.detail = detail
        self.url = url
        super().__init__(f"HTTP {status} {url}: {detail}")


class _BaseClient:
    def __init__(
        self, base_url: str, api_key: Optional[str] = None, *,
        timeout: float = 30.0, verify_tls: bool = True,
        bearer_token: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.bearer_token = bearer_token
        self._timeout = timeout
        self._verify = verify_tls

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        if self.bearer_token:
            h["Authorization"] = f"Bearer {self.bearer_token}"
        return h

    def _url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))


class QuantForge(_BaseClient):
    """Synchronous client."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = httpx.Client(timeout=self._timeout, verify=self._verify)

    # ------------------------- low-level ------------------------------------
    def _request(self, method: str, path: str, **kw) -> Any:
        url = self._url(path)
        r = self._client.request(method, url, headers=self._headers(), **kw)
        if 200 <= r.status_code < 300:
            if r.headers.get("content-type", "").startswith("application/json"):
                return r.json()
            return r.text
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise QuantForgeError(r.status_code, detail, url)

    def get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, json: Optional[dict] = None) -> Any:
        return self._request("POST", path, json=json or {})

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------- meta -----------------------------------------
    def health(self) -> dict:
        return self.get("/healthz")

    def readiness(self) -> dict:
        return self.get("/readyz")

    # ------------------------- auth -----------------------------------------
    def issue_token(self, subject: str, scopes: Optional[Iterable[str]] = None,
                     ttl_seconds: int = 3600) -> dict:
        body = {"subject": subject, "ttl_seconds": ttl_seconds}
        if scopes:
            body["scopes"] = list(scopes)
        return self.post("/v1/auth/token", body)

    # ------------------------- options --------------------------------------
    def price_option(self, S: float, K: float, T: float, sigma: float,
                      r: float = 0.04, option: str = "call", **kw) -> dict:
        return self.post("/v1/options/price", {
            "S": S, "K": K, "T": T, "sigma": sigma, "r": r, "option": option, **kw,
        })

    def implied_vol(self, price: float, S: float, K: float, T: float,
                     r: float = 0.04, option: str = "call", **kw) -> dict:
        return self.post("/v1/options/iv", {
            "price": price, "S": S, "K": K, "T": T, "r": r, "option": option, **kw,
        })

    # ------------------------- backtests ------------------------------------
    def run_backtest(self, strategy: str, tickers: Iterable[str],
                      start: str = "2015-01-01", end: str = "2025-01-01",
                      capital: float = 100_000, sizing_fraction: float = 0.25,
                      rebalance: str = "monthly", params: Optional[dict] = None) -> dict:
        body = {
            "strategy": strategy, "tickers": list(tickers),
            "start": start, "end": end, "capital": capital,
            "sizing_fraction": sizing_fraction, "rebalance": rebalance,
            "params": params or {},
        }
        return self.post("/v1/backtest", body)

    def walk_forward(self, strategy: str, tickers: Iterable[str],
                      start: str, end: str, n_folds: int = 5,
                      capital: float = 100_000, sizing_fraction: float = 0.25,
                      rebalance: str = "monthly", params: Optional[dict] = None) -> dict:
        body = {
            "strategy": strategy, "tickers": list(tickers),
            "start": start, "end": end, "n_folds": n_folds,
            "capital": capital, "sizing_fraction": sizing_fraction,
            "rebalance": rebalance, "params": params or {},
        }
        return self.post("/v1/backtest/walk-forward", body)

    def compare(self, strategies: List[dict]) -> dict:
        return self.post("/v1/backtest/compare", {"strategies": strategies})

    # ------------------------- portfolio ------------------------------------
    def optimize(self, tickers: Iterable[str], start: str = "2015-01-01",
                  end: str = "2025-01-01", objective: str = "max_sharpe",
                  risk_free: float = 0.02) -> dict:
        return self.post("/v1/portfolio/optimize", {
            "tickers": list(tickers), "start": start, "end": end,
            "objective": objective, "risk_free": risk_free,
        })

    # ------------------------- risk -----------------------------------------
    def var(self, returns: Iterable[float], confidence: float = 0.95,
             method: str = "historical") -> dict:
        return self.post("/v1/risk/var", {
            "returns": list(returns), "confidence": confidence, "method": method,
        })

    # ------------------------- ml -------------------------------------------
    def train_ml(self, ticker: str = "SPY", start: str = "2010-01-01",
                  end: str = "2025-01-01", **kw) -> dict:
        return self.post("/v1/ml/train", {
            "ticker": ticker, "start": start, "end": end, **kw,
        })

    # ------------------------- market data ----------------------------------
    def market_data(self, ticker: str, start: str = "2020-01-01",
                     end: str = "2025-01-01", sample: int = 200) -> dict:
        return self.get(f"/v1/market/data/{ticker}",
                         params={"start": start, "end": end, "sample": sample})

    # ------------------------- async jobs -----------------------------------
    def submit_backtest(self, strategy: str, tickers: Iterable[str],
                         start: str = "2015-01-01", end: str = "2025-01-01",
                         capital: float = 100_000, sizing_fraction: float = 0.25,
                         rebalance: str = "monthly", params: Optional[dict] = None) -> dict:
        body = {
            "strategy": strategy, "tickers": list(tickers),
            "start": start, "end": end, "capital": capital,
            "sizing_fraction": sizing_fraction, "rebalance": rebalance,
            "params": params or {},
        }
        return self.post("/v1/jobs/backtest", body)

    def get_job(self, job_id: str) -> dict:
        return self.get(f"/v1/jobs/{job_id}")

    def cancel_job(self, job_id: str) -> dict:
        return self.delete(f"/v1/jobs/{job_id}")

    def list_jobs(self, limit: int = 50) -> list:
        return self.get("/v1/jobs", params={"limit": limit})

    def wait(self, job_id: str, poll_interval: float = 1.0, timeout: float = 300.0) -> dict:
        """Block until a job reaches a terminal state (completed/failed/cancelled)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            j = self.get_job(job_id)
            if j.get("status") in ("completed", "failed", "cancelled"):
                return j
            time.sleep(poll_interval)
        raise TimeoutError(f"job {job_id} did not finish in {timeout}s")

    # ------------------------- alerts ---------------------------------------
    def create_alert_rule(self, name: str, metric: str, threshold: float,
                           direction: str = "above", severity: str = "warning") -> dict:
        return self.post("/v1/alerts/rules", {
            "name": name, "metric": metric, "threshold": threshold,
            "direction": direction, "severity": severity,
        })

    def list_alert_rules(self) -> list:
        return self.get("/v1/alerts/rules")

    def delete_alert_rule(self, name: str) -> dict:
        return self.delete(f"/v1/alerts/rules/{name}")

    def evaluate_alerts(self, context: Dict[str, float]) -> dict:
        return self.post("/v1/alerts/evaluate", {"context": context})

    def recent_alerts(self, limit: int = 50) -> list:
        return self.get(f"/v1/alerts/events?limit={limit}")

    # ------------------------- audit ----------------------------------------
    def audit(self, limit: int = 100) -> list:
        return self.get(f"/v1/audit?limit={limit}")


class AsyncQuantForge(_BaseClient):
    """Asyncio version for concurrent fan-out."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = httpx.AsyncClient(timeout=self._timeout, verify=self._verify)

    async def _request(self, method: str, path: str, **kw) -> Any:
        url = self._url(path)
        r = await self._client.request(method, url, headers=self._headers(), **kw)
        if 200 <= r.status_code < 300:
            if r.headers.get("content-type", "").startswith("application/json"):
                return r.json()
            return r.text
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise QuantForgeError(r.status_code, detail, url)

    async def health(self) -> dict:
        return await self._request("GET", "/healthz")

    async def price_option(self, S: float, K: float, T: float, sigma: float, **kw) -> dict:
        return await self._request("POST", "/v1/options/price",
                                    json={"S": S, "K": K, "T": T, "sigma": sigma, **kw})

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()
