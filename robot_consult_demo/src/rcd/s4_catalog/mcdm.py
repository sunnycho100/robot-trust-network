"""AHP criterion weights (Saaty eigenvector method, consistency ratio) + TOPSIS ranking (Hwang & Yoon 1981)."""
from __future__ import annotations

import numpy as np

# Criteria: (key, direction, description)
CRITERIA = [
    ("cost", "cost", "robot list price mid-point, USD"),
    ("conveyor_tracking", "benefit", "vendor conveyor/line-tracking feature (1/0)"),
    ("reach_margin", "benefit", "(usable reach - required reach) / required reach"),
    ("speed", "benefit", "mean datasheet max speed of J1-J3, deg/s (proxy for free-move time)"),
    ("ecosystem", "benefit", "accessory/integrator ecosystem, analyst ordinal score 1-5 (assumption)"),
    ("payload_margin", "benefit", "payload / (1.25 x required) - 1"),
    ("footprint", "cost", "cell floor area total_m2 (ISO 13855 + aisle allowance)"),
    ("repeatability", "cost", "pose repeatability, mm"),
]

# Saaty 1-9 pairwise judgments a[i][j] = importance of criterion i over j (upper triangle; lower = reciprocal).
# Order as in CRITERIA. Rationale: an SME buyer weighs price first; on a moving line tracking is near-mandatory;
# reach margin protects against placement error; speed drives cycle time; ecosystem lowers integration risk;
# payload/footprint/repeatability matter less because every passed model already meets the screw/part needs.
JUDGMENTS = {
    (0, 1): 1, (0, 2): 2, (0, 3): 2, (0, 4): 3, (0, 5): 4, (0, 6): 4, (0, 7): 5,
    (1, 2): 2, (1, 3): 2, (1, 4): 3, (1, 5): 4, (1, 6): 4, (1, 7): 5,
    (2, 3): 1, (2, 4): 2, (2, 5): 2, (2, 6): 3, (2, 7): 3,
    (3, 4): 2, (3, 5): 2, (3, 6): 2, (3, 7): 3,
    (4, 5): 1, (4, 6): 2, (4, 7): 2,
    (5, 6): 1, (5, 7): 2,
    (6, 7): 1,
}
RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}  # Saaty random index

# Ecosystem ordinal score (analyst assumption, documented in output): size of certified accessory ecosystem
# (screwdrivers, grippers, vision) and integrator base.
ECOSYSTEM = {"Universal Robots": 5, "FANUC": 4, "ABB": 4, "Yaskawa": 4, "KUKA": 3, "Doosan Robotics": 3,
             "Techman Robot": 3, "Franka Robotics": 2, "UFACTORY": 2}


def ahp() -> dict:
    n = len(CRITERIA)
    A = np.ones((n, n))
    for (i, j), v in JUDGMENTS.items():
        A[i, j] = v
        A[j, i] = 1.0 / v
    vals, vecs = np.linalg.eig(A)
    k = int(np.argmax(vals.real))
    w = np.abs(vecs[:, k].real)
    w /= w.sum()
    lam = float(vals[k].real)
    ci = (lam - n) / (n - 1)
    cr = ci / RI[n]
    return {"criteria": [{"key": c, "direction": d, "description": s} for c, d, s in CRITERIA],
            "pairwise_matrix": A.round(4).tolist(), "lambda_max": round(lam, 4), "consistency_index": round(ci, 4),
            "random_index": RI[n], "consistency_ratio": round(cr, 4),
            "weights": {c[0]: round(float(x), 4) for c, x in zip(CRITERIA, w)}}


def topsis(X: np.ndarray, w: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Closeness coefficient C_i = D-_i / (D+_i + D-_i), vector normalisation."""
    X = np.asarray(X, float)
    if len(X) == 1:
        return np.array([1.0])
    norm = np.sqrt((X ** 2).sum(0))
    norm[norm == 0] = 1.0
    V = X / norm * w
    ideal = np.where(benefit, V.max(0), V.min(0))
    anti = np.where(benefit, V.min(0), V.max(0))
    dp = np.sqrt(((V - ideal) ** 2).sum(1))
    dm = np.sqrt(((V - anti) ** 2).sum(1))
    return np.where(dp + dm > 0, dm / (dp + dm + 1e-15), 1.0)
