"""Hierarchical Risk Parity (Lopez de Prado)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def _correl_dist(corr: np.ndarray) -> np.ndarray:
    return np.sqrt(0.5 * (1 - corr))


def _quasi_diag(link: np.ndarray) -> list:
    link = link.astype(int)
    sort_ix = [link[-1, 0], link[-1, 1]]
    n = link.shape[0] + 1
    while max(sort_ix) >= n:
        new_sort = []
        for ix in sort_ix:
            if ix >= n:
                l0, l1 = link[ix - n, 0], link[ix - n, 1]
                new_sort.extend([int(l0), int(l1)])
            else:
                new_sort.append(int(ix))
        sort_ix = new_sort
    return sort_ix


def _ivp(cov: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def _cluster_var(cov: np.ndarray, idx: list) -> float:
    sub = cov[np.ix_(idx, idx)]
    w = _ivp(sub).reshape(-1, 1)
    return float((w.T @ sub @ w)[0, 0])


def _recursive_bisection(cov: np.ndarray, sorted_ix: list) -> np.ndarray:
    w = np.ones(cov.shape[0])
    clusters = [sorted_ix]
    while clusters:
        new_clusters = []
        for c in clusters:
            if len(c) <= 1:
                continue
            mid = len(c) // 2
            left, right = c[:mid], c[mid:]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1 - var_l / (var_l + var_r)
            w[left] *= alpha
            w[right] *= 1 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    return w


def hierarchical_risk_parity(cov: pd.DataFrame | np.ndarray) -> np.ndarray:
    S = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)
    std = np.sqrt(np.diag(S))
    corr = S / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    dist = _correl_dist(corr)
    link = linkage(squareform(dist, checks=False), method="single")
    sorted_ix = _quasi_diag(link)
    w = _recursive_bisection(S, sorted_ix)
    return w / w.sum()
