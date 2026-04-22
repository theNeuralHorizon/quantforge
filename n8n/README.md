# n8n workflows for QuantForge

These are ready-to-import workflow JSON files for [n8n](https://n8n.io/).
They hit the QuantForge REST API to automate research and alerting.

## Prerequisites

1. Running QuantForge API (`docker compose up -d api`) reachable from n8n
2. An `X-API-Key` header value (set `QUANTFORGE_API_KEYS` and use the raw key)
3. Create an n8n credential of type **Header Auth** named `QuantForge API Key`
   with Name=`X-API-Key` and Value=`<your raw key>`

## Import

In n8n: **Workflows → Import from File** → pick any JSON in this folder.

## Workflows

| File | What it does |
|---|---|
| `daily-risk-check.json` | Every weekday at 8am: pull SPY data, compute VaR, Slack on breach |
| `weekly-tournament.json` | Every Monday: run 6 strategies on 10y data, email the results |
| `ml-retrain-monthly.json` | 1st of each month: retrain the SPY gradient booster, post metrics to a webhook |
| `options-alert.json` | When a webhook fires with S/K/T/sigma, price + Greeks are returned |

## Security note

The n8n container should sit on the same private network as the API (e.g.
docker compose `qfnet` or K8s `quantforge` namespace). Do NOT expose the
API public without TLS and rate limiting at an ingress.
