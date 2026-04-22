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
- `bandit` for Python security lint (HIGH+ fails the build)
- `safety` for CVE scanning of dependencies
- `CodeQL` for static analysis
- `Trivy` for container image vulnerability scan
- Dependabot weekly PRs for pip, docker, and GitHub Actions
