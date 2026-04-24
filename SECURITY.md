# Security Policy

## Reporting a vulnerability

If you discover a security issue, email the maintainer directly (see GitHub
profile) or use **[GitHub's private security advisory](https://github.com/theNeuralHorizon/quantforge/security/advisories/new)**. **Do not** open a public issue.

A response will follow within 48 hours. Please include:
- affected version / commit
- reproduction steps
- expected vs observed behaviour
- any proof-of-concept code

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅         |

## Security architecture

### API authentication
- Header: `X-API-Key` required for all `/v1/*` routes
- Keys stored as SHA-256 hex digests — raw keys never touch disk
- Constant-time comparison (`hmac.compare_digest`)
- Dev-only auto-key printed at startup when `QUANTFORGE_API_KEYS` is unset

### Rate limiting
- `slowapi` with per-API-key (or per-IP) limits
- Default: **120 requests/minute**, **2000 requests/hour**
- Exceeded → `429 Too Many Requests` with `Retry-After` header

### Request hardening
- **256 KiB** max body (DoS defense)
- `X-Request-ID` correlation (auto-generated or echoed)
- Strict input validation via Pydantic (bounds + regex on every field)
- Content-Length pre-check rejects oversized payloads before body read

### HTTP security headers
Every response carries:
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Content-Security-Policy` with strict allowlist
- `Referrer-Policy: no-referrer`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Resource-Policy: same-site`

### CORS
Strict allowlist only — no wildcard with credentials. Configure additional
origins via `QUANTFORGE_CORS_ORIGINS` (comma-separated).

### Container hardening
- Runs as non-root UID 10001
- `readOnlyRootFilesystem: true`
- All Linux capabilities dropped
- `seccompProfile: RuntimeDefault`
- `allowPrivilegeEscalation: false`
- Network isolation via `NetworkPolicy` (default deny)

### CI/CD checks
- `bandit` for Python security lint (MEDIUM+ fails the build)
- `safety` for CVE scanning of dependencies
- `CodeQL` for static analysis (Python + JavaScript)
- `gitleaks` for committed-secret detection (fails PRs with leaks)
- `Trivy` for container image vulnerability scan (CRITICAL/HIGH surfaced to
  GitHub Security tab)
- Dependabot weekly PRs for pip, docker, and GitHub Actions

### Deploy pipeline (Render + Vercel)
- Deploys only fire after **test + lint + security-scan** are all green.
- CI posts to Render/Vercel deploy hooks via URLs stored as GitHub secrets
  (`RENDER_DEPLOY_HOOK_URL`, `VERCEL_DEPLOY_HOOK_URL`). The URLs themselves
  are not logged.
- A post-deploy `smoke` job polls `/healthz` and asserts that an
  unauthenticated request to `/v1/meta/version` returns **401/403**. If a
  deploy accidentally ships `QUANTFORGE_ALLOW_UNAUTH=true`, the job fails
  and flags the regression before traffic hits it.

## Production deployment checklist

Run through this before pointing real clients at a fresh deploy. The
companion guide in [`DEPLOYMENT.md`](./DEPLOYMENT.md) walks through the
mechanics.

### Secrets
- [ ] `QUANTFORGE_API_KEYS` set to comma-separated **SHA-256 digests** (not
      raw keys). At least one key per distinct client so you can revoke
      granularly.
- [ ] `QUANTFORGE_JWT_SECRET` is ≥ 32 random bytes if any route uses JWT
      auth. Generated via `secrets.token_urlsafe(48)` or `openssl rand`.
- [ ] `QUANTFORGE_ALLOW_UNAUTH=false` — confirm with the smoke job.
- [ ] No secrets in repo. `gitleaks` is green on `main`.

### Network
- [ ] `QUANTFORGE_CORS_ORIGINS` contains **only** your frontend origin(s).
      No wildcards, no `http://`, no trailing paths.
- [ ] Vercel enforces HSTS (see `vercel.json` headers block).
- [ ] Render serves only over HTTPS (default; don't disable).

### Persistence
- [ ] Render disk is mounted at `/app/data` and `QUANTFORGE_AUDIT_DB`
      points inside it. Without this, audit history resets on every deploy.
- [ ] Audit log rotation threshold (`MAX_ROWS=200_000`) is acceptable for
      your compliance retention; bump the disk size if you need longer.

### Runtime
- [ ] Rate limits appropriate for the caller base (default 120/min, 2000/hr
      per key). Tighten for public deploys, loosen for trusted internal.
- [ ] `REDIS_URL` set if you run >1 instance of the API — otherwise the
      in-memory cache and rate-limit counters diverge between instances.
- [ ] Trivy SARIF uploads are visible under **Security → Code scanning**.
- [ ] At least one secondary on-call has access to rotate secrets.
