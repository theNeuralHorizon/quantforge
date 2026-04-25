# Deployment — Vercel (frontend) + Render (backend)

> **Live deployment** (as of last verified run):
> - API: `https://quantforge-api.onrender.com` (Render free plan, sleeps after 15 min idle, no persistent disk).
> - UI: deploy via Vercel — see step 2 below.
> - Mode: **demo** (`QUANTFORGE_ALLOW_UNAUTH=true`). The `/v1/*` routes
>   accept unauthenticated requests so visitors can play with the terminal
>   without an API key. To harden, flip the env var to `false` in the
>   Render dashboard and set `EXPECT_UNAUTH=false` (or unset) in the GitHub
>   repo variables — the `smoke` job will start enforcing 401/403.

The production topology splits the UI and the API across two platforms:

```
                    ┌────────────────────┐
 Browser ─ HTTPS ─→ │  Vercel (edge)     │
                    │  web/index.html    │  ← static assets, CDN-cached
                    │  /v1/* rewrite ──┐ │
                    └──────────────────┼─┘
                                       │  HTTPS
                                       ▼
                            ┌───────────────────────┐
                            │  Render (Oregon)      │
                            │  FastAPI + uvicorn    │  ← auth, rate limit,
                            │  /app/data (10 GB disk│     audit log, jobs
                            └───────────────────────┘
```

The browser only ever talks to the Vercel origin. Vercel proxies API paths
(`/v1/*`, `/healthz`, `/docs`, `/ws/*`) to Render. This gives you:

- **No CORS headaches** — single origin for the browser.
- **Cheap edge caching** for the static UI.
- **Hardened API** — the public surface on Render is just FastAPI, behind
  a real WAF (Cloudflare/Render's own), not `localhost:8000`.
- **Independent redeploy** cadence (push-to-main redeploys both, but they
  scale/roll back separately).

---

## One-time setup

### 0. Prerequisites

- GitHub repo on `main` with CI green. (The `CI`, `gitleaks`, and `CodeQL`
  workflows must pass before deploy.)
- A Vercel account and a Render account (both have free tiers).
- `gh` CLI authenticated locally (for setting GitHub secrets).

### 1. Deploy the backend to Render

1. In Render: **New** → **Blueprint** → connect this GitHub repo.
2. Render reads [`render.yaml`](./render.yaml) and provisions:
   - a `quantforge-api` web service on the `starter` plan
   - a 10 GB persistent disk at `/app/data`
3. Fill in the **three secrets** Render prompts you for (marked
   `sync: false` in the blueprint):
   - `QUANTFORGE_API_KEYS` — SHA-256 digests of the raw keys you hand out.
     Generate one per client:
     ```bash
     raw=$(openssl rand -hex 32)
     hash=$(printf '%s' "$raw" | sha256sum | awk '{print $1}')
     echo "raw (give to client): $raw"
     echo "hash (paste into Render): $hash"
     ```
     Multiple hashes → comma-separated.
   - `QUANTFORGE_JWT_SECRET` — 48+ random URL-safe chars:
     ```bash
     python -c "import secrets; print(secrets.token_urlsafe(48))"
     ```
   - `QUANTFORGE_CORS_ORIGINS` — the origin of your Vercel deploy, e.g.
     `https://quantforge.vercel.app`. You'll know this after step 2.
4. Wait for the first deploy to finish. `curl https://<your-service>.onrender.com/healthz`
   should return `{"status":"ok",...}`.
5. Copy the service URL (e.g. `https://quantforge-api.onrender.com`) —
   you'll need it in the next step.
6. Under **Settings** → **Deploy Hook**, click **Generate**. Copy the URL.

> **Note on Render service name.** The default [`vercel.json`](./vercel.json)
> proxies to `https://quantforge-api.onrender.com`. If you named the service
> differently, update the `destination` URLs in `vercel.json` before the
> Vercel step.

### 2. Deploy the frontend to Vercel

The WebSocket host is already wired in `web/index.html`:
```html
<meta name="qf-api-host" content="quantforge-api.onrender.com" />
```
Plain HTTP calls go through Vercel's rewrites; `/ws/*` reads this meta
and connects straight to Render (Vercel rewrites don't carry WS
upgrades). If you change the Render service name, update this line and
the `destination` URLs in `vercel.json` together.

#### Option A — Vercel CLI (one-time login)
```bash
npm i -g vercel        # if not installed
vercel login           # opens browser once
vercel link            # pick the team + project (creates .vercel/project.json)
vercel --prod          # deploys
```
After the first `vercel link`, every subsequent `vercel --prod` is a
single command — no flags needed because `vercel.json` is in the repo.

#### Option B — Vercel dashboard (no CLI)
1. **Add New** → **Project** → import `theNeuralHorizon/quantforge`.
2. **Framework Preset**: Other (Vercel will read `vercel.json` directly).
3. **Root Directory**: `.` (default).
4. **Build / Output Settings**: leave defaults — `vercel.json`'s
   `outputDirectory: "web"` handles it.
5. **Environment Variables**: none required.
6. Click **Deploy**. Visit the URL Vercel gives you (typically
   `<project>.vercel.app`).

#### After either path

7. Smoke-test through the browser:
   - Hit `/healthz` → should return `{"status":"ok",...}` (proxied to
     Render).
   - Open the terminal UI, click **Health** → green pill.
   - Open the **Alerts** page, watch the WS pill go green.
8. If your Vercel URL isn't `https://quantforge.vercel.app` or
   `https://quantforge-terminal.vercel.app`, append it to
   `QUANTFORGE_CORS_ORIGINS` in the Render dashboard (comma-separated).
9. (Optional) **Settings** → **Git** → **Deploy Hooks** → generate a
   hook URL for CI.

> **Re-using an existing `quantforge.vercel.app` slug.** If that name is
> already taken by an unrelated project (e.g. a marketing landing page),
> import this repo under a different name (e.g. `quantforge-terminal`).
> Vercel only allows one project per slug per account.

### 3. Wire Render's URL back into CORS

Now that you know the Vercel URL, go back to Render:

- **Environment** → edit `QUANTFORGE_CORS_ORIGINS` → set it to
  `https://<your-vercel-app>.vercel.app` (add the production domain too if
  you have one: `,https://quantforge.com`).
- Save. Render redeploys automatically.

Verify CORS is locked down:

```bash
curl -I -H "Origin: https://evil.example.com" \
     https://<your-service>.onrender.com/v1/meta/version
# → the Access-Control-Allow-Origin header MUST be absent or not echo evil.example.com
```

### 4. Tell GitHub Actions about the hooks

Set GitHub repo secrets and vars:

```bash
# Replace the URLs with the ones you generated in steps 1.6 and 2.7.
gh secret set RENDER_DEPLOY_HOOK_URL -b '<render-hook-url>'
gh secret set VERCEL_DEPLOY_HOOK_URL -b '<vercel-hook-url>'   # optional

# Variables (not secrets) — used by the smoke-test job. Safe to expose.
gh variable set PROD_API_URL    -b 'https://quantforge-api.onrender.com'

# Tell the smoke job whether prod should reject unauthenticated /v1/*
# requests. Set to `true` for demo deploys (matches the live config in
# this repo); set to `false` (or leave unset) once you flip
# QUANTFORGE_ALLOW_UNAUTH=false on Render.
gh variable set EXPECT_UNAUTH   -b 'true'
```

The next push to `main` will trigger:

1. `test` + `lint` + `security-scan` (unchanged)
2. `docker-build` (pushes image to GHCR — still happens, independent of deploy)
3. `deploy` — POSTs to both deploy hooks
4. `smoke` — polls `/healthz` until the new build is live (free-plan cold
   start can take 30–60 s), asserts the three security headers are
   present (`Strict-Transport-Security`, `X-Content-Type-Options`,
   `X-Frame-Options`), then enforces the `EXPECT_UNAUTH` contract:
   - `EXPECT_UNAUTH=true` → `/v1/meta/version` must return **200** *and*
     `/v1/meta/dev-key` must report `dev_mode=false` so demo mode
     isn't accidentally leaking a dev key.
   - `EXPECT_UNAUTH=false` (or unset) → `/v1/meta/version` must return
     **401/403**.

   Either way, anything else (5xx, network error, missing header, leaked
   dev key) fails the job and surfaces a clear error.

---

## Ongoing operations

### Rolling deploy
- Push to `main` → both services redeploy in parallel (~2-3 min each).
- Render does a zero-downtime roll if health check passes for the new
  instance before the old one is drained.

### Rollback
- **Render**: service → **Deploys** → pick a prior green deploy → **Redeploy**.
- **Vercel**: **Deployments** → pick a prior deploy → **Promote to
  Production**.

### Secret rotation
- API keys: append the new hash to `QUANTFORGE_API_KEYS`, hand out the new
  raw key, wait ≥ 7 days for clients to migrate, remove the old hash.
- JWT secret: rotating invalidates all in-flight tokens. Only rotate when
  you suspect compromise; otherwise every ~90 days.

### Scaling
- Vercel: automatic, no action needed.
- Render: **Scaling** → bump the plan or add a second instance. If you add
  a second instance, **set `REDIS_URL`** first — the in-memory cache and
  rate-limit buckets don't sync across instances otherwise.

---

## Staging environment (optional)

Duplicate the Render service and point it at a non-`main` branch:

```yaml
# render-staging.yaml (second blueprint, committed to a different branch)
services:
  - type: web
    name: quantforge-api-staging
    branch: staging
    plan: free   # sleeps when idle — fine for staging
    ...
```

Create a Vercel preview alias `quantforge-staging.vercel.app` pointing at
the same branch, and update `QUANTFORGE_CORS_ORIGINS` on the staging
Render service to allow it.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| UI loads, every API call returns a CORS error | `QUANTFORGE_CORS_ORIGINS` on Render doesn't include the Vercel URL | Add it, save, wait for redeploy |
| UI loads, API calls return 502 | Render service is sleeping (free plan) | Upgrade to `starter` or wait 30 s for wake |
| `/healthz` returns `{"status":"ok"}` locally, 500 in prod | Missing env var (most often `QUANTFORGE_API_KEYS` unset *and* `ALLOW_UNAUTH=false`, causing auth to reject everything including its own probe) | Set `QUANTFORGE_API_KEYS` in Render |
| Smoke job fails with "prod allows unauth access" | `QUANTFORGE_ALLOW_UNAUTH=true` slipped into prod | Set to `false` in Render env |
| Vercel shows a 404 on `/v1/...` | `vercel.json` rewrites point at the wrong Render URL | Edit `vercel.json`, push, redeploy |
| Alerts/Live Signals page stuck on "connecting…" | WebSocket host not set in `web/index.html` (`<meta name="qf-api-host">` is empty) and Vercel can't proxy WS | Set the meta `content` to the Render host (step 2.1 above) and redeploy |
| Audit log resets on every deploy | Render disk not mounted at `/app/data` | Confirm the `disk:` block in `render.yaml`, redeploy |

---

## Security posture

See [`SECURITY.md`](./SECURITY.md) for the full policy. The deployment
itself enforces:

- **HTTPS-only** — Render and Vercel both force TLS; `Strict-Transport-Security`
  is set at the Vercel edge (see `vercel.json` `headers`).
- **Auth mandatory** — `QUANTFORGE_ALLOW_UNAUTH=false` in prod; the CI
  `smoke` job asserts this holds after every deploy.
- **CORS allowlist** — only the Vercel frontend origin(s) are accepted.
- **Secrets never in git** — `gitleaks` scans every push. The only
  `.env.example` files committed are templates.
- **Container is non-root** — the `Dockerfile` runs uvicorn as UID 10001.
- **SBOM + CVE scan** — Trivy scans the image on every `main` push
  (`docker-build` job). Findings upload to GitHub Security tab.
- **Static analysis** — `bandit` (Python), `CodeQL` (Python + JS), and
  `ruff` gate every PR.
