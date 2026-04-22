# Kubernetes manifests

Hardened production manifests. Apply in order:

```bash
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
# Real secrets:
kubectl create secret generic quantforge-secrets -n quantforge \
  --from-literal=QUANTFORGE_API_KEYS=$(echo -n 'your-real-key' | sha256sum | awk '{print $1}')
kubectl apply -f redis.yaml
kubectl apply -f api-deployment.yaml
kubectl apply -f networkpolicy.yaml
kubectl apply -f ingress.yaml   # requires nginx-ingress + cert-manager
```

## Security posture

- All pods run as non-root UID 10001 with `readOnlyRootFilesystem: true`
- All pods drop all Linux capabilities, use `RuntimeDefault` seccomp
- `NetworkPolicy` denies all ingress/egress by default, then explicitly
  allows: nginx-ingress → api:8000, api → redis:6379, api → DNS, api → HTTPS/443
- HPA scales 2 → 10 replicas on CPU 70% / mem 80%
- PDB ensures at least 1 pod available during node drains
- Ingress forces HTTPS, rate-limits to 20 rps per IP, caps body at 1 MiB
- Redis: no persistence beyond PVC, LRU eviction at 256 MiB
