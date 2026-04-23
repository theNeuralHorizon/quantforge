"""QuantForge Python client — thin wrapper over the REST + WebSocket API.

Install the same repo and import:

    from quantforge_client import QuantForge

    qf = QuantForge("http://localhost:8000", api_key="dev_test_key_12345")
    print(qf.health())
    print(qf.price_option(S=100, K=100, T=1.0, sigma=0.2))
    job = qf.submit_backtest(strategy="buy_and_hold", tickers=["SPY"])
    print(qf.wait(job["job_id"]))
"""
from quantforge_client.client import QuantForge, AsyncQuantForge, QuantForgeError

__all__ = ["QuantForge", "AsyncQuantForge", "QuantForgeError"]
