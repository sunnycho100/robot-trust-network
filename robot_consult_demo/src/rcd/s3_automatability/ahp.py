"""Analytic Hierarchy Process (Saaty 1980): weights = normalised principal eigenvector of a reciprocal pairwise
comparison matrix; consistency index CI = (lambda_max - n)/(n - 1); consistency ratio CR = CI / RI(n),
accepted when CR < 0.10. Random indices RI from Saaty (1980), n = 1..10."""
from __future__ import annotations

import numpy as np

RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def matrix_from_judgements(keys: list[str], judgements: dict[tuple[str, str], float]) -> np.ndarray:
    n = len(keys)
    A = np.ones((n, n))
    idx = {k: i for i, k in enumerate(keys)}
    for (a, b), v in judgements.items():
        A[idx[a], idx[b]] = v
        A[idx[b], idx[a]] = 1.0 / v
    for i in range(n):
        for j in range(n):
            if i != j and (keys[i], keys[j]) not in judgements and (keys[j], keys[i]) not in judgements:
                raise ValueError(f"missing judgement {keys[i]} vs {keys[j]}")
    return A


def ahp(A: np.ndarray) -> dict:
    n = A.shape[0]
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    lam = float(vals[k].real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    ci = (lam - n) / (n - 1) if n > 2 else 0.0
    cr = ci / RI[n] if RI.get(n) else 0.0
    return {"weights": w, "lambda_max": lam, "ci": ci, "cr": cr}
