"""SOP-only work-element segmentation: two-level hierarchical HSMM over the whole clip, no annotated timestamps.

Inputs per cycle: human joints (any source; video pose may carry a validity mask) and the station's ordered SOP
steps with their activity text (customer intake). Output: elements with `subgoal` = SOP step name.

Model (Rabiner 1989 sec. IV-D explicit-duration HMM; Durbin et al. 1998 profile HMM; Murphy & Paskin 2001
hierarchical HMM as a two-level semi-Markov decomposition; Bourlard & Morgan 1994 hybrid scaled likelihoods):

  top level   ordered SOP steps as semi-Markov segments. A step may be performed or skipped (symmetric prior,
              STEP_SKIP / CLIP_SKIP); the first observed step may be entered mid-script (START_MID, the recording
              started during it) and the last one left mid-script (END_MID). Non-cycle time before / after the SOP
              is a background segment with an exponential prior whose mean is how much longer the recording is
              than one takt (OUTSIDE_*). Small deviations from the written order: Mallows prior (ORDER_LAMBDA per
              Kendall inversion, <= MAX_INVERSIONS). Step durations: weak log-normal prior, median = r x MTM-UAS
              synthetic time of the step's activities (sop_lexicon.mtm_nominal_s), sigma DUR_SIGMA, r = clip pace
              factor with a log-normal hyper-prior (PACE_*); an intake standard time per step (optional
              `std_time_s`) replaces it with sigma STD_TIME_SIGMA. Boundary prior: steps change at line-side visits
              (fetching the next part / tool) or at posture/location change points; elsewhere BOUNDARY_OFF.
  bottom level inside a step, the left-to-right element script from its activity text (sop_lexicon), optional
              WALK and IDLE (waiting) between elements, profile-HMM deletions (SKIP_PENALTY per skipped element,
              at most MAX_SKIP in a row), explicit minimum durations as chains of sub-states. An activity line the
              lexicon cannot classify becomes an ANY state and is relabelled from the motion afterwards.
  evidence    the kinematic element rules of segment.evidence turned into scaled likelihoods (calibrate), tempered
              by their integrated autocorrelation time so that structural priors keep their meaning.

Inference: (1) for every step one batched Viterbi pass over all candidate start boundaries (1 s grid, 5 Hz) gives
the inner score of every (start, end) pair; (2) exact top-level DP over boundaries for every candidate order and
pace factor; (3) the chosen step spans are re-aligned at 10 Hz (standard forced alignment, as in the annotated
mode) to produce the elements; waiting is carved out as IDLE.

Tested and switched off (kept for the record, see the constants): forward (sum-over-alignments) inner scores,
Z-normalised script scores, an identity-free appearance change-point likelihood, Occam element-switch costs,
pause-and-resume loops.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.special import log_ndtr

from . import segment as SG
from .features import RATE_HZ, extract, path_length, work_anchor
from .sop_lexicon import ANY, mtm_nominal_s, script_from_lines

MIN_DUR_S = {**SG.MIN_DUR_S, ANY: 1.0}
STATION_CODES = ["GET_PART", "INSERT", "GET_TOOL", "REACH_FASTENER", "FASTEN", "INSPECT", "SETUP", "IDLE"]
WALK_ENTRY = SG.WALK_ENTRY_PENALTY
SKIP_PENALTY = SG.SKIP_PENALTY
MAX_SKIP = SG.MAX_SKIP
# top-level costs are log-probabilities (nats) on the tempered scale (see decode_steps)
STEP_SKIP = math.log(20.0)      # an SOP step is mandatory work: not performing it has p = 0.05
CLIP_SKIP = 0.0                 # steps cropped away by a recording shorter than one takt: no cost
START_MID = math.log(10.0)      # first observed step entered mid-script (recordings usually start at a step, p = 0.1)
END_MID = math.log(2.0)         # last observed step left mid-script (clip ends during the step, p = 0.5)
ORDER_LAMBDA = -math.log(0.1)   # Mallows dispersion: cost per Kendall inversion w.r.t. the written order
MAX_INVERSIONS = 1
DUR_SIGMA = 0.7           # log-normal sigma of step duration (weak prior)
STD_TIME_SIGMA = 0.20     # sigma with an intake standard time per step: Westinghouse consistency class D (CV <= 20 %)
PACE_GRID = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
PACE_PRIOR = (math.log(1.5), 0.5)   # hyper-prior on observed/PMTS pace: log-normal, median 1.5 (allowances, regrasps)
GRID_S = 1.0              # candidate step-boundary grid
TOP_RATE = 5.0            # analysis rate of the top-level search
ELEMENT_SWITCH = 0.0             # Occam cost per element transition (tested ln(25): collapses steps)
SCRIPT_WEIGHT = 1.0
FETCH_START = True              # boundary prior: steps change at line-side visits / posture-location change points
VISIT_TOL_S = 2.0
OUTSIDE_COST = math.log(2.0)    # non-cycle time present before / after the SOP (p = 0.5 each)
CYCLE_SIGMA = 0.10              # paced line: cycle length ~ takt +/- 10 % (only for recordings longer than a takt)
LOW_MARGIN = 1.0                # boundary margin (tempered nats) below which a boundary is flagged low_margin
OUTSIDE_MIN_MEAN_S = 2.0
OUTSIDE_FILLER = True           # non-cycle time explained by an ergodic filler model (not by an empty background)
OUTSIDE_SWITCH = math.log(9.0)  # uniform choice of the next code among the 9 element codes        # recordings no longer than a takt: non-cycle time ~ Exp(mean 2 s)
INNER_FORWARD = False           # forward (sum) inner scores: tested, scripts then score below background (weak evidence)
ELEM_MEAN_S = 4.0               # mean element duration (geometric tail) for the forward variant
REPEAT_COST = None              # pause-and-resume loops (tested, no gain)
BOUNDARY_OFF = math.log(20.0)   # a step boundary away from any line-side visit / change point (p = 0.05)
APPEARANCE = False              # identity-free change-point likelihood added to every step (tested, hurts identity)
SCORE_NORM = False              # Z-norm of script scores by their whole-clip rate (tested, hurts)
INVISIBLE_WALK_COST = 0.0       # frames without a valid pose carry no evidence (the HSMM carries its state through)
UNOBS_MIN_S = 1.0               # pose gaps at least this long are unobserved time (shorter ones are interpolated)
TOP_CALIBRATE = True      # firing-rate normalisation of the evidence in the top-level search
NEG = -1e18


# --------------------------------------------------------------------------------------------- evidence
def _speed_floor(f: dict, pct: float = 25.0, walk_window_s: float = 2.0) -> dict:
    """Noise-robust speeds for noisy pose (video), applied only to non-mocap joints:
    * pelvis: net displacement over a centred walk_window_s window / its duration - jitter and slow pose offsets
      do not accumulate into net travel, real walking does;
    * wrists: subtract the clip's lower-quartile speed (a still hand reads as moving at the noise speed)."""
    p = f["pelvis_xy"]
    h = max(1, int(round(walk_window_s * RATE_HZ / 2)))
    idx = np.arange(len(p))
    a, b = np.clip(idx - h, 0, len(p) - 1), np.clip(idx + h, 0, len(p) - 1)
    f["pelvis_v"] = np.linalg.norm(p[b] - p[a], axis=1) / np.maximum(1, b - a) * RATE_HZ
    fl = {}
    for k in ("wrist_vR", "wrist_vL"):
        fl[k] = float(np.percentile(f[k], pct))
        f[k] = np.maximum(0.0, f[k] - fl[k])
    f["act"] = np.maximum(f["wrist_vR"], f["wrist_vL"])
    f["dwell"] = np.minimum(f["wrist_vR"], f["wrist_vL"])
    f["speed_floor"] = {k: round(v, 3) for k, v in fl.items()}
    return f


def prepare(J: np.ndarray, names: list[str], fps: float, valid: np.ndarray | None = None, denoise: bool = False,
            calib: dict | None = None, clf: dict | None = None, line_side_xy=None):
    """Features + evidence at RATE_HZ. Frames without a valid pose are interpolated for the kinematics and get a
    neutral (code-independent) emission so they never bias the decoding."""
    J = np.asarray(J, float).copy()
    if valid is not None and (~valid).any() and valid.any():
        t = np.arange(len(J))
        for j in range(J.shape[1]):
            for d in range(3):
                J[~valid, j, d] = np.interp(t[~valid], t[valid], J[valid, j, d])
    f = extract(J, names, fps)
    if calib:                          # pose-source calibration (quantile maps onto the reference instrument)
        from .generic_calib import apply as _apply_calib
        f = _apply_calib(f, calib)
    elif denoise and clf is None:
        f = _speed_floor(f)
    anchor = work_anchor(f)
    if clf is not None:                # learned evidence (scaled likelihoods of a frame classifier)
        from . import generic_clf as GCL
        X, _ = GCL.frame_features(f, anchor, line_side_xy)
        f["d_anchor"] = np.linalg.norm(f["pelvis_xy"] - anchor, axis=1)
        E = GCL.evidence(clf, X, ["WALK"] + STATION_CODES)
    else:
        E = SG.evidence(f, anchor)
    M = np.stack([E[c] for c in STATION_CODES])
    E[ANY] = np.logaddexp.reduce(M, axis=0) - np.log(len(STATION_CODES))
    v10 = np.ones(f["n"], bool)
    if valid is not None:
        step = max(1, int(round(fps / RATE_HZ)))
        n = f["n"]
        vv = valid[: n * step].reshape(n, step).mean(1) if len(valid) >= n * step else np.ones(n)
        v10 = vv >= 0.5
        for c in E:
            E[c] = np.where(v10, E[c], 0.0)
        # an operator who stays unobserved for long is far more likely occluded at the workpiece than walking
        # (walks are short trips): a small per-frame cost keeps WALK from absorbing long pose gaps
        E["WALK"] = np.where(v10, E["WALK"], -INVISIBLE_WALK_COST)
    f["valid10"] = v10
    # unobserved stretches (no valid pose for >= UNOBS_MIN_S): no element is emitted there; the time is counted as
    # unobserved (like the part of the cycle outside a cropped clip), never as IDLE or work
    un = np.zeros(f["n"], bool)
    i = 0
    while i < f["n"]:
        if not v10[i]:
            j = i
            while j < f["n"] and not v10[j]:
                j += 1
            if j - i >= UNOBS_MIN_S * RATE_HZ:
                un[i:j] = True
            i = j
        else:
            i += 1
    f["unobserved10"] = un
    return f, E


def calibrate(E: dict) -> dict:
    """Scaled likelihoods (hybrid HMM, Bourlard & Morgan 1994): the rule scores are turned into per-frame code
    posteriors P(c|x_t) (softmax over the codes, so codes compete) and divided by the clip-level code prior
    P(c) = mean_t P(c|x_t): log p(x_t|c) + const = log P(c|x_t) - log P(c). Broad rules that fire everywhere no
    longer act as garbage collectors in the top-level search. Unsupervised, per clip."""
    codes = [c for c in E if c != ANY]
    M = np.stack([E[c] for c in codes])
    L = M - np.logaddexp.reduce(M, axis=0)
    prior = np.logaddexp.reduce(L, axis=1) - math.log(M.shape[1])
    out = {c: L[i] - prior[i] for i, c in enumerate(codes)}
    S = np.stack([out[c] for c in codes if c != "WALK"])
    out[ANY] = np.logaddexp.reduce(S, axis=0) - math.log(len(S))
    return out


def autocorr_frames(E: dict, max_lag: int = 200) -> float:
    """Integrated autocorrelation time (frames) of the frame-level evidence (mean over codes, Sokal window: sum
    of the autocorrelation up to its first non-positive value). Used to temper the composite likelihood."""
    taus = []
    for x in E.values():
        x = np.asarray(x, float) - np.mean(x)
        v = float(np.dot(x, x))
        if v <= 0 or len(x) < 10:
            continue
        tau = 1.0
        for lag in range(1, min(max_lag, len(x) - 1)):
            r = float(np.dot(x[:-lag], x[lag:])) / v
            if r <= 0:
                break
            tau += 2 * r
        taus.append(tau)
    return float(np.mean(taus)) if taus else 1.0


def _downsample(E: dict, k: int) -> dict:
    out = {}
    for c, x in E.items():
        n = len(x) // k
        y = x[: n * k].reshape(n, k).sum(1)
        if len(x) > n * k:
            y = np.append(y, x[n * k:].sum())
        out[c] = y
    return out


# --------------------------------------------------------------------------------------------- inner graph
def _graph(script: list[dict], rate: float, idle_gaps: bool = False, switch_cost: float = 0.0,
           repeat_cost: float | None = None) -> dict:
    """Left-to-right profile HSMM of one step script. Between elements (and before the first / after the last) is
    an optional *gap*: a WALK, and with idle_gaps also an unavoidable-delay IDLE after it (therblig UD: waiting for
    an actuator or the robot can happen anywhere). Elements can be deleted (SKIP_PENALTY each, <= MAX_SKIP)."""
    gap_codes = ["WALK", "IDLE"] if idle_gaps else ["WALK"]
    segs, el_seg, gap_segs = [], [], []
    n = len(script)
    for i in range(n + 1):
        gap_segs.append([])
        for c in gap_codes:
            gap_segs[-1].append(len(segs))
            segs.append({"code": c, "optional": True, "step": None})
        if i < n:
            el_seg.append(len(segs))
            segs.append({"code": script[i]["code"], "optional": False, "step": script[i]})
    for g in segs:
        g["m"] = max(1, int(math.floor(MIN_DUR_S[g["code"]] * rate)))
    first_of, last_of, st_seg = [], [], []
    for j, g in enumerate(segs):
        first_of.append(len(st_seg))
        st_seg += [j] * g["m"]
        last_of.append(len(st_seg) - 1)
    S = len(st_seg)
    self_ok = np.zeros(S, bool)
    self_ok[last_of] = True
    entries: list[list[tuple[int, float]]] = [[] for _ in range(S)]

    def exits(gi: int) -> list[int]:
        return [last_of[x] for x in gap_segs[gi]] + ([last_of[el_seg[gi - 1]]] if gi >= 1 else [])

    for i, j in enumerate(el_seg):
        for k in range(MAX_SKIP + 1):
            if i - k < 0:
                break
            entries[first_of[j]] += [(p, -SKIP_PENALTY * k - switch_cost) for p in exits(i - k)]
    if repeat_cost is not None and idle_gaps:   # pause and resume: element -> waiting (UD) -> same element
        for i, j in enumerate(el_seg):
            d_ = gap_segs[i + 1][-1]
            entries[first_of[j]].append((last_of[d_], -repeat_cost - switch_cost))
    for gi, lst in enumerate(gap_segs):
        for r, j in enumerate(lst):
            src = ([last_of[el_seg[gi - 1]]] if gi >= 1 else []) + [last_of[x] for x in lst[:r]]
            entries[first_of[j]] += [(p, -WALK_ENTRY - switch_cost) for p in src]
    for j, g in enumerate(segs):
        for k in range(1, g["m"]):
            entries[first_of[j] + k].append((first_of[j] + k - 1, 0.0))
    M = max(1, max(len(e) for e in entries))
    Pm = np.full((M, S), -1)
    Cm = np.zeros((M, S))
    for s_, lst in enumerate(entries):
        for m_, (p, c) in enumerate(lst):
            Pm[m_, s_], Cm[m_, s_] = p, c
    init = np.full(S, NEG)
    for x in gap_segs[0]:
        init[first_of[x]] = 0.0
    for k, j in enumerate(el_seg[:MAX_SKIP + 1]):
        init[first_of[j]] = max(init[first_of[j]], -SKIP_PENALTY * k)
    init_mid = init.copy()
    for j in range(len(segs)):
        init_mid[first_of[j]] = max(init_mid[first_of[j]], 0.0)       # cost added at the top level
    endv = np.full(S, NEG)
    for k in range(MAX_SKIP + 1):
        if n - k < 0:
            break
        for p in exits(n - k):
            endv[p] = max(endv[p], -SKIP_PENALTY * k)
    endv_mid = endv.copy()
    for j in range(len(segs)):
        endv_mid[last_of[j]] = max(endv_mid[last_of[j]], 0.0)
    return {"segs": segs, "st_seg": np.array(st_seg), "first_of": first_of, "last_of": last_of, "S": S,
            "self_ok": self_ok, "Pm": Pm, "Cm": Cm, "valid": Pm >= 0, "Pm0": np.maximum(Pm, 0),
            "init": init, "init_mid": init_mid, "end": endv, "end_mid": endv_mid}


def _emissions(G: dict, E: dict, a: int, b: int) -> np.ndarray:
    return np.stack([E[g["code"]][a:b] for g in G["segs"]])[G["st_seg"]]          # (S, T)


def _step(G: dict, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One Viterbi transition for a batch of rows delta (B, S); returns (best (B,S), argmax predecessor (B,S))."""
    best = np.where(G["self_ok"], delta, NEG)
    arg = np.broadcast_to(np.arange(G["S"]), delta.shape).copy()
    cand = np.where(G["valid"], delta[:, G["Pm0"]] + G["Cm"], NEG)                 # (B, M, S)
    mi = np.argmax(cand, axis=1)
    cb = np.take_along_axis(cand, mi[:, None, :], 1)[:, 0, :]
    better = cb > best
    best = np.where(better, cb, best)
    arg = np.where(better, G["Pm0"][mi, np.arange(G["S"])[None, :]], arg)
    return best, arg


def forward_graph(G: dict, rate: float) -> dict:
    """Proper transition probabilities for the forward (sum-over-alignments) score: every state's outgoing
    probabilities sum to one. Min-duration chain states move on with probability 1; the last sub-state of a segment
    stays with 1 - 1/L (geometric extra duration, mean L frames) and leaves with 1/L, shared among its successors
    in proportion to exp(-cost) of the structural costs (walk entry, deletions). Because probability mass is
    shared, a flexible script is not rewarded for its flexibility (Bayesian Occam factor), unlike the Viterbi max."""
    S, Pm, Cm, valid = G["S"], G["Pm"], G["Cm"], G["valid"]
    outw = np.full(S, NEG)
    for m_ in range(Pm.shape[0]):
        for s_ in range(S):
            if valid[m_, s_]:
                p = Pm[m_, s_]
                outw[p] = np.logaddexp(outw[p], Cm[m_, s_])
    Cf = np.full_like(Cm, NEG)
    self_lp = np.full(S, NEG)
    for j, g in enumerate(G["segs"]):
        L = max(1.0, (ELEM_MEAN_S - MIN_DUR_S[g["code"]]) * rate)
        self_lp[G["last_of"][j]] = math.log(1.0 - 1.0 / L) if L > 1 else NEG
    for m_ in range(Pm.shape[0]):
        for s_ in range(S):
            if valid[m_, s_]:
                p = Pm[m_, s_]
                if G["self_ok"][p]:
                    L_leave = -math.log(max(1.0, 1.0 / max(1e-9, 1 - math.exp(self_lp[p])))) if self_lp[p] > NEG / 2 else 0.0
                    Cf[m_, s_] = L_leave + Cm[m_, s_] - outw[p]
                else:
                    Cf[m_, s_] = Cm[m_, s_] - outw[p]
    H = dict(G)
    H["Cm"], H["self_lp"] = Cf, self_lp
    H["init"] = G["init"] - np.logaddexp.reduce(G["init"])
    H["init_mid"] = G["init_mid"] - np.logaddexp.reduce(G["init_mid"])
    H["forward"] = True
    return H


def _step_fwd(G: dict, delta: np.ndarray) -> np.ndarray:
    stay = np.where(G["self_ok"], delta + G["self_lp"], NEG)
    cand = np.where(G["valid"], delta[:, G["Pm0"]] + G["Cm"], NEG)
    return np.logaddexp(stay, np.logaddexp.reduce(cand, axis=1))


def inner_scores(G: dict, E: dict, bounds: np.ndarray, T: int) -> dict:
    """Batched Viterbi of one step script for every start boundary: S[i, j] = best score of the step occupying
    frames [bounds[i], bounds[j]) (normal start and end). Also mid-script variants at the clip edges."""
    em = _emissions(G, E, 0, T)
    nb = len(bounds)
    starts = bounds[:-1]
    NBs = len(starts)
    out = np.full((nb, nb), NEG)
    out_end_mid = np.full(nb, NEG)              # start normal at bounds[i], end mid-script at T
    delta = np.full((NBs, G["S"]), NEG)
    # extra row 0': mid-script start at frame 0
    dmid = np.full((1, G["S"]), NEG)
    mid_end_normal = np.full(nb, NEG)           # mid start at 0, normal end at bounds[j]
    mid_mid = NEG
    end_idx = {int(b): j for j, b in enumerate(bounds)}
    fwd = G.get("forward", False)
    red = (lambda x, axis=None: np.logaddexp.reduce(x, axis=axis)) if fwd else (lambda x, axis=None: np.max(x, axis=axis))
    for t in range(T):
        if t > 0:
            if fwd:
                delta, dmid = _step_fwd(G, delta), _step_fwd(G, dmid)
            else:
                delta, _ = _step(G, delta)
                dmid, _ = _step(G, dmid)
            delta = delta + em[:, t]
            dmid = dmid + em[:, t]
        k = np.searchsorted(starts, t)
        if k < NBs and starts[k] == t:
            delta[k] = G["init"] + em[:, t]
        if t == 0:
            dmid[0] = G["init_mid"] + em[:, 0]
        j = end_idx.get(t + 1)
        if j is not None:
            out[:NBs, j] = red(delta + G["end"], axis=1)
            mid_end_normal[j] = float(red(dmid[0] + G["end"]))
            if t + 1 == T:
                out_end_mid[:NBs] = red(delta + G["end_mid"], axis=1)
                mid_mid = float(red(dmid[0] + G["end_mid"]))
    # a row only counts for ends after its start
    for i in range(NBs):
        out[i, : i + 1] = NEG
    return {"full": out, "end_mid": out_end_mid, "start_mid": mid_end_normal, "mid_mid": mid_mid}


def align_span(G: dict, E: dict, a: int, b: int, mid_start: bool, mid_end: bool) -> list[tuple[dict, int, int]]:
    """Viterbi with back-pointers of one step script on frames [a, b) -> [(segment, start, end)]."""
    T = b - a
    em = _emissions(G, E, a, b)
    delta = ((G["init_mid"] if mid_start else G["init"]) + em[:, 0])[None, :]
    back = np.zeros((T, G["S"]), np.int32)
    for t in range(1, T):
        delta, arg = _step(G, delta)
        delta = delta + em[:, t]
        back[t] = arg[0]
    fin = delta[0] + (G["end_mid"] if mid_end else G["end"])
    s = int(np.argmax(fin))
    path = np.zeros(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s = back[t, s]
    seg_path = G["st_seg"][path]
    out, x = [], 0
    for t in range(1, T + 1):
        if t == T or seg_path[t] != seg_path[x]:
            out.append((G["segs"][int(seg_path[x])], a + x, a + t))
            x = t
    return out


# --------------------------------------------------------------------------------------------- top level
def _lognorm_logpdf(d: np.ndarray, mu: float, sigma: float = DUR_SIGMA) -> np.ndarray:
    d = np.maximum(d, 1e-3)
    z = (np.log(d) - math.log(mu)) / sigma
    return -np.log(d * sigma * math.sqrt(2 * math.pi)) - 0.5 * z * z


def _lognorm_logsf(d: np.ndarray, mu: float, sigma: float = DUR_SIGMA) -> np.ndarray:
    d = np.maximum(d, 1e-3)
    return log_ndtr(-(np.log(d) - math.log(mu)) / sigma)


def _orders(K: int) -> list[tuple[tuple[int, ...], int]]:
    """Candidate step orders within MAX_INVERSIONS Kendall inversions of the written order (cheap for small K;
    adjacent transpositions only for long SOPs)."""
    base = tuple(range(K))
    if K <= 6:
        res = []
        for p in itertools.permutations(base):
            inv = sum(1 for i in range(K) for j in range(i + 1, K) if p[i] > p[j])
            if inv <= MAX_INVERSIONS:
                res.append((p, inv))
        return res
    res = [(base, 0)]
    for i in range(K - 1):
        p = list(base)
        p[i], p[i + 1] = p[i + 1], p[i]
        res.append((tuple(p), 1))
    return res


def _top_dp(order, inner, prior_full, prior_part, nb, pre=None, suf=None, clip_skip=None, end_prior=None):
    """Exact DP over boundaries for one step order. Returns (score, [(step, i, j, mid_start, mid_end)]).
    pre[i] / suf[j]: score of non-cycle time [0, b_i) before the first step / [b_j, T) after the last step."""
    K = len(order)
    last = nb - 1
    clip_skip = CLIP_SKIP if clip_skip is None else clip_skip
    # steps before the first / after the last observed one: cropped by the recording (clip_skip) if the cycle
    # touches the clip edge, otherwise really not performed (STEP_SKIP)
    skip_pre = np.where(np.arange(nb) == 0, clip_skip, STEP_SKIP)
    skip_suf = np.where(np.arange(nb) == last, clip_skip, STEP_SKIP)
    if pre is None:
        pre = np.full(nb, NEG)
        pre[0] = 0.0
    if suf is None:
        suf = np.full(nb, NEG)
        suf[last] = 0.0
    if end_prior is not None:
        suf = suf + end_prior
    dp = np.full((K, nb), NEG)
    bp: dict = {}
    for m, k in enumerate(order):
        I = inner[k]
        # first observed step (steps before it skipped at CLIP_SKIP each): normal start at 0 or mid-script start
        sc0 = (pre - skip_pre * m)[:, None] + I["full"] + prior_full[k]
        i0 = np.argmax(sc0, axis=0)
        a0 = sc0[i0, np.arange(nb)]
        a1 = I["start_mid"] + prior_part[k][0] - clip_skip * m - START_MID
        cand = np.where(a0 >= a1, a0, a1)
        for j in range(1, nb):
            if cand[j] > dp[m, j]:
                dp[m, j] = cand[j]
                bp[(m, j)] = (None, 0, True) if a1[j] > a0[j] else (None, int(i0[j]), False)
        for mp in range(m):
            sc = dp[mp][:, None] + I["full"] + prior_full[k] - STEP_SKIP * (m - mp - 1)     # (i, j)
            ib = np.argmax(sc, axis=0)
            v = sc[ib, np.arange(nb)]
            upd = v > dp[m]
            for j in np.nonzero(upd)[0]:
                dp[m, j] = v[j]
                bp[(m, int(j))] = (mp, int(ib[j]), False)
    # finish: last observed step ends at T (normal end, already in dp[:, last]) or mid-script
    best, best_end = NEG, None
    ep_last = 0.0 if end_prior is None else float(end_prior[last])
    for m, k in enumerate(order):
        tail = -clip_skip * (K - 1 - m)
        fin = dp[m] + suf - skip_suf * (K - 1 - m)
        jb = int(np.argmax(fin))
        if fin[jb] > best:
            best, best_end = fin[jb], ("normal", m, jb)
        I = inner[k]
        # mid-script end: previous step m' ended at i (or none: whole clip inside one step)
        for mp in range(m):
            sc = dp[mp] + I["end_mid"] + prior_part[k][1] - STEP_SKIP * (m - mp - 1) - END_MID + tail + ep_last
            i = int(np.argmax(sc))
            if sc[i] > best:
                best, best_end = sc[i], ("mid", m, mp, i)
        sc0 = I["end_mid"][0] + prior_part[k][1][0] - clip_skip * m - END_MID + tail + ep_last
        if sc0 > best:
            best, best_end = sc0, ("mid0", m)
        scmm = I["mid_mid"] + prior_part[k][2] - clip_skip * m - START_MID - END_MID + tail + ep_last
        if scmm > best:
            best, best_end = scmm, ("midmid", m)
    if best_end is None:
        return NEG, []
    spans = []
    kind = best_end[0]
    if kind == "normal":
        m, j = best_end[1], best_end[2]
        mid_e = False
    elif kind == "mid":
        _, m, mp, i = best_end
        spans.append((order[m], i, last, False, True))
        m, j, mid_e = mp, i, False
    elif kind == "mid0":
        return best, [(order[best_end[1]], 0, last, False, True)]
    else:
        return best, [(order[best_end[1]], 0, last, True, True)]
    while True:
        mp, i, mid_s = bp[(m, j)]
        spans.append((order[m], i, j, mid_s, mid_e))
        if mp is None:
            break
        m, j, mid_e = mp, i, False
    return best, spans[::-1]


def appearance_features(f: dict) -> np.ndarray:
    """Slow posture/location descriptors at RATE_HZ (operator side of the line, hand height, hand separation, reach,
    trunk flexion, walking speed), smoothed over 2 s: SOP steps are performed at different places on the
    workpiece, so their descriptors shift between steps."""
    from scipy.ndimage import uniform_filter1d
    wz = np.maximum(f["wrist_zR"], f["wrist_zL"])
    X = np.stack([f["pelvis_xy"][:, 1], wz, f["hand_dist"], f["reach_max"], f["trunk_flex"] / 30.0, f["pelvis_v"]], 1)
    return uniform_filter1d(X, int(2.0 * RATE_HZ), axis=0, mode="nearest")


def appearance_inner(X5: np.ndarray, bounds: np.ndarray, sigma: float = 0.6) -> np.ndarray:
    """Identity-free segmental change-point score of a piecewise-stationary Gaussian (segment mean fitted, BIC
    penalty for it), tempered by the descriptors' correlation time: ll[i, j] for a segment [bounds[i], bounds[j])."""
    T, d = X5.shape
    C1 = np.vstack([np.zeros(d), np.cumsum(X5, 0)])
    C2 = np.vstack([np.zeros(d), np.cumsum(X5 ** 2, 0)])
    B = bounds
    n = (B[None, :] - B[:, None]).astype(float)
    S1 = C1[B][None, :, :] - C1[B][:, None, :]
    S2 = C2[B][None, :, :] - C2[B][:, None, :]
    sse = (S2 - np.where(n[..., None] > 0, S1 ** 2 / np.maximum(n[..., None], 1), 0)).sum(-1)
    kap = autocorr_frames({i: X5[:, i] for i in range(d)})
    return np.where(n > 0, -sse / (2 * sigma ** 2) / kap - 0.5 * d * np.log(np.maximum(n / kap, 1.0)), NEG)


def line_side_visits(f: dict, d_away: float = 1.0, min_s: float = 1.0) -> list[int]:
    """Mid-points (RATE_HZ frames) of excursions away from the work anchor (> d_away for >= min_s): the moments
    the operator is at the line side (cart, tool station, rack). A step that starts by fetching its part or tool
    starts at such a visit."""
    away = f["d_anchor"] > d_away
    out, i, n = [], 0, f["n"]
    while i < n:
        if away[i]:
            j = i
            while j < n and away[j]:
                j += 1
            if j - i >= min_s * RATE_HZ:
                k = i
                while k > 0 and f["pelvis_v"][k - 1] > 0.2:      # include the walk out
                    k -= 1
                out.append((k + j) // 2)
            i = j
        else:
            i += 1
    return out


def change_points(X5: np.ndarray, bounds: np.ndarray, pen: float = 10.0, min_s: float = 10.0) -> list[int]:
    """Optimal partition of the appearance descriptors (penalised piecewise-constant Gaussian, >= min_s per piece);
    returns the internal cut frames (TOP_RATE)."""
    ll = appearance_inner(X5, bounds)
    nb = len(bounds)
    dp = np.full(nb, NEG)
    dp[0] = 0.0
    bp = np.zeros(nb, int)
    for j in range(1, nb):
        cand = dp[:j] + ll[:j, j] - pen
        cand[(bounds[j] - bounds[:j]) < min_s * TOP_RATE] = NEG
        i = int(np.argmax(cand))
        dp[j], bp[j] = cand[i], i
    cuts, j = [], nb - 1
    while j > 0:
        cuts.append(int(bounds[bp[j]]))
        j = bp[j]
    return sorted(c for c in cuts if c > 0)


def _filler_scores(E5: dict, switch: float) -> tuple[np.ndarray, np.ndarray]:
    """Filler ("garbage") model for non-cycle time: an ergodic loop over all element codes with a uniform switch
    probability (cost `switch` per change), on the same tempered evidence as the step scripts, so that non-SOP
    activity is not forced into a script. Returns prefix[t] = best score of frames [0, t) and suffix[t] = best score
    of [t, T)."""
    codes = [c for c in E5 if c != ANY]
    M = np.stack([E5[c] for c in codes])
    K, T = M.shape

    def run(X):
        out = np.zeros(T + 1)
        d = np.zeros(K)
        for t in range(T):
            d = np.maximum(d, d.max() - switch) + X[:, t]
            out[t + 1] = d.max()
        return out
    pre = run(M)
    suf = run(M[:, ::-1])[::-1]
    return pre, suf


def _boundary_margins(spans, inner, prior_full, prior_part, pre, suf, nb, win_s=10.0, excl_s=2.0):
    """Confidence of every decoded step boundary: score of the chosen boundary minus the best alternative boundary
    position within +/- win_s (but >= excl_s away), with the neighbouring segments re-scored (tempered nats)."""
    last = nb - 1
    w = int(round(win_s * TOP_RATE / (GRID_S * TOP_RATE)))
    ex = int(round(excl_s / GRID_S))

    def seg(kk, i, j, ms, me):
        I = inner[kk]
        if j <= i:
            return NEG
        if ms and me:
            return I["mid_mid"] + prior_part[kk][2] if (i == 0 and j == last) else NEG
        if ms:
            return I["start_mid"][j] + prior_part[kk][0][j] if i == 0 else NEG
        if me:
            return I["end_mid"][i] + prior_part[kk][1][i] if j == last else NEG
        return I["full"][i, j] + prior_full[kk][i, j]

    def margin(f, b0):
        cand = [(b, f(b)) for b in range(max(0, b0 - w), min(last, b0 + w) + 1)]
        chosen = f(b0)
        alt = [v for b, v in cand if abs(b - b0) >= ex and v > NEG / 2]
        if not alt or chosen <= NEG / 2:
            return None
        return round(float(chosen - max(alt)), 3)

    res = []
    for n_, (kk, i, j, ms, me) in enumerate(spans):
        if n_ == 0:
            m0 = margin(lambda b: pre[b] + seg(kk, b, j, ms and b == 0, me), i)
        else:
            pk, pi, _, pms, _ = spans[n_ - 1]
            m0 = margin(lambda b: seg(pk, pi, b, pms, False) + seg(kk, b, j, False, me), i)
        if n_ == len(spans) - 1:
            m1 = margin(lambda b: seg(kk, i, b, ms, me and b == last) + suf[b], j)
        else:
            m1 = None                   # = the next step's start margin
        res.append([m0, m1])
    for n_ in range(len(res) - 1):
        res[n_][1] = res[n_ + 1][0]
    return res


def decode_steps(E10: dict, steps: list[dict], X10: np.ndarray | None = None, visits: list[int] | None = None,
                 outside: np.ndarray | None = None, takt_s: float | None = None, paced: bool = True) -> dict:
    """Top-level search: returns the chosen step spans (10 Hz frame indices) and diagnostics."""
    k = int(round(RATE_HZ / TOP_RATE))
    E5 = _downsample(calibrate(E10) if TOP_CALIBRATE else E10, k)
    T = len(E5["WALK"])
    g = int(round(GRID_S * TOP_RATE))
    bounds = np.unique(np.append(np.arange(0, T, g), T))
    nb = len(bounds)
    dur = (bounds[None, :] - bounds[:, None]) / TOP_RATE                           # (i, j) seconds
    graphs = [_graph(s["script"], TOP_RATE, idle_gaps=True, switch_cost=ELEMENT_SWITCH, repeat_cost=REPEAT_COST)
              for s in steps]
    # tempering: frame-level evidence is serially correlated; dividing the emissions by their integrated
    # autocorrelation time turns the composite likelihood into roughly independent evidence units, so that the
    # structural priors (deletions, step skips, order, durations) keep their meaning as log-probabilities
    kappa = autocorr_frames(E5)
    E5 = {c: x * SCRIPT_WEIGHT / kappa for c, x in E5.items()}
    if INNER_FORWARD:
        graphs = [forward_graph(G, TOP_RATE) for G in graphs]
    inner = [inner_scores(G, E5, bounds, T) for G in graphs]
    if SCORE_NORM:
        # Z-norm style score normalisation: a flexible script fits *any* motion; subtract the evidence rate the step
        # script achieves on the whole clip, so a step only gains where it fits better than it fits on average
        fr = (bounds[None, :] - bounds[:, None]).astype(float)
        for I in inner:
            g_rate = I["mid_mid"] / T
            I["full"] = np.where(fr > 0, I["full"] - g_rate * fr, NEG)
            I["start_mid"] = I["start_mid"] - g_rate * bounds
            I["end_mid"] = I["end_mid"] - g_rate * (T - bounds)
            I["mid_mid"] = 0.0
    X5n = None
    if X10 is not None:
        n5 = len(X10) // k
        X5 = X10[: n5 * k].reshape(n5, k, -1).mean(1)
        if len(X5) < T:
            X5 = np.vstack([X5, np.repeat(X5[-1:], T - len(X5), 0)])
        X5 = (X5 - X5.mean(0)) / (X5.std(0) + 1e-6)
        X5n = X5[:T]
        ll = appearance_inner(X5n, bounds)
        for I in (inner if APPEARANCE else []):
            I["full"] = I["full"] + ll
            I["start_mid"] = I["start_mid"] + ll[0]
            I["end_mid"] = I["end_mid"] + ll[:, -1]
            I["mid_mid"] = I["mid_mid"] + ll[0, -1]
    if visits is not None and FETCH_START:
        # boundary prior: steps change at line-side visits (fetching the next part / tool) or at a marked change
        # of working posture/location (appearance change point); other boundaries cost BOUNDARY_OFF
        cands = [v // k for v in visits]
        if X10 is not None:
            cands += list(change_points(X5n, bounds))
        vv = np.array(cands) if cands else np.array([-10 ** 6])
        near = np.abs(bounds[:, None] - vv[None, :]).min(1) <= VISIT_TOL_S * TOP_RATE
        off = np.where(near, 0.0, -BOUNDARY_OFF)
        off[0] = 0.0                                                  # clip start is always a valid start
        for st_, I in zip(steps, inner):
            I["full"] = I["full"] + off[:, None]
            I["end_mid"] = I["end_mid"] + off
    # non-cycle time before the first / after the last SOP step (recording longer than the cycle): background
    # emission (scaled likelihood of the marginal = 1, i.e. 0 per frame) and an entry cost OUTSIDE_COST
    # exponential prior on the amount of non-cycle time, mean = how much longer the recording is than one takt
    # expected cycle length: the sum of the standard-work-sheet step times when the sheet covers the SOP, else takt
    stds = [s_.get("std_time_s") for s_ in steps]
    cyc_len = float(sum(stds)) if all(stds) else takt_s
    mean_out = max(OUTSIDE_MIN_MEAN_S, T / TOP_RATE - cyc_len) if cyc_len else OUTSIDE_MIN_MEAN_S
    pre = np.where(bounds > 0, -OUTSIDE_COST - (bounds / TOP_RATE) / mean_out, 0.0)
    suf = np.where(bounds < T, -OUTSIDE_COST - ((T - bounds) / TOP_RATE) / mean_out, 0.0)
    if outside is not None:
        o5 = np.concatenate([[0.0], np.cumsum(_downsample({"o": outside}, k)["o"][:T] * SCRIPT_WEIGHT / kappa)])
        pre = pre + o5[bounds]
        suf = suf + (o5[T] - o5[bounds])
    if OUTSIDE_FILLER:
        fp, fs = _filler_scores(E5, OUTSIDE_SWITCH)
        pre = pre + fp[bounds]
        suf = suf + fs[bounds]
    # a recording shorter than one takt is a crop of the cycle: steps beyond its edges are simply not recorded;
    # a recording covering more than a takt holds the whole cycle, so a missing step was really not performed
    clip_skip = CLIP_SKIP if (not takt_s or T / TOP_RATE < takt_s) else STEP_SKIP
    nominal = [max(1.0, mtm_nominal_s(s["script"])) for s in steps]
    D = T / TOP_RATE
    def priors_for(r):
        prior_full, prior_part, mus = [], [], []
        for kk in range(len(steps)):
            std = steps[kk].get("std_time_s")
            # customer standard time of the step (intake, optional) is a much stronger prior than PMTS x pace
            mu, sg = (float(std), STD_TIME_SIGMA) if std else (r * nominal[kk], DUR_SIGMA)
            pf = np.where(dur > 0, _lognorm_logpdf(np.abs(dur), mu, sg), NEG)
            pp = (_lognorm_logsf(bounds / TOP_RATE, mu, sg),                         # mid start at 0, end at j
                  _lognorm_logsf(D - bounds / TOP_RATE, mu, sg),                     # start at i, mid end at T
                  float(_lognorm_logsf(np.array([D]), mu, sg)[0]))
            prior_full.append(pf)
            prior_part.append(pp)
            mus.append((mu, sg))
        return prior_full, prior_part, mus

    def search(end_prior):
        best = (NEG, None, None, None)
        for r in PACE_GRID:
            prior_full, prior_part, _ = priors_for(r)
            for order, inv in _orders(len(steps)):
                sc, spans = _top_dp(order, inner, prior_full, prior_part, nb, pre, suf, clip_skip, end_prior)
                sc -= ORDER_LAMBDA * inv
                sc += -0.5 * ((math.log(r) - PACE_PRIOR[0]) / PACE_PRIOR[1]) ** 2
                if sc > best[0]:
                    best = (sc, spans, order, r)
        return best

    best = search(None)
    paced_cycle = bool(cyc_len) and (paced or all(stds)) and D > cyc_len
    end_prior = None
    if paced_cycle and best[1]:
        # recording longer than one cycle: the cycle (first step start -> last step end) lasts about cyc_len (takt on
        # a paced line, or the sheet total); second pass with a prior on the cycle end given the first pass's cycle start
        w0 = bounds[best[1][0][1]] / TOP_RATE
        z = (bounds / TOP_RATE - w0 - cyc_len) / (CYCLE_SIGMA * cyc_len)
        end_prior = -0.5 * z * z
        best = search(end_prior)
    sc, spans, order, r = best
    margins = _boundary_margins(spans or [], inner, *priors_for(r)[:2], pre,
                                suf + (end_prior if end_prior is not None else 0.0), nb)
    durz = []
    _, _, mus = priors_for(r)
    for kk, i, j, ms, me in spans or []:
        d = (bounds[j] - bounds[i]) / TOP_RATE
        mu, sg = mus[kk]
        z_ = (math.log(max(d, 1e-3)) - math.log(mu)) / sg
        durz.append(None if (ms or me) and z_ < 0 else round(float(z_), 2))
    out = []
    n10 = len(E10["WALK"])
    for kk, i, j, ms, me in spans:
        a = int(bounds[i]) * k
        b = n10 if j == nb - 1 else int(bounds[j]) * k
        out.append({"step": kk, "a": a, "b": b, "mid_start": bool(ms), "mid_end": bool(me)})
    for o, (m0, m1), zz in zip(out, margins, durz):
        o["start_margin"], o["end_margin"], o["duration_z"] = m0, m1, zz
    return {"spans": out, "score": float(sc), "order": list(order), "pace_factor": r, "nominal_s": nominal,
            "kappa_s": round(kappa / TOP_RATE, 2), "paced_cycle_prior": paced_cycle}


# --------------------------------------------------------------------------------------------- per cycle
def build_steps(sop: list[dict]) -> list[dict]:
    steps = []
    for k, s in enumerate(sop):
        lines = list(s.get("activities") or []) or [s.get("text") or s["step"]]
        steps.append({"name": s["step"], "text": s.get("text", s["step"]), "script": script_from_lines(lines, f"{k + 1}."),
                      "std_time_s": s.get("std_time_s")})
    return steps


def _observed_parts(un: np.ndarray, a: int, b: int) -> list[tuple[int, int]]:
    """[a, b) minus the unobserved frames."""
    out, i = [], a
    while i < b:
        if un[i]:
            i += 1
            continue
        j = i
        while j < b and not un[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def segment_cycle_sop(station: str, cycle_id: str, J: np.ndarray, names: list[str], fps: float, sop: list[dict],
                      valid: np.ndarray | None = None, joint_source: str = "", video: dict | None = None,
                      takt_s: float | None = None, denoise: bool = False, calib: dict | None = None,
                      clf: dict | None = None, line_side_xy=None):
    f, E = prepare(J, names, fps, valid, denoise, calib, clf, line_side_xy)
    post = SG.posterior({c: E[c] for c in E if c != ANY})
    n = f["n"]
    dur = n / RATE_HZ
    steps = build_steps(sop)
    top = decode_steps(E, steps, appearance_features(f), line_side_visits(f), takt_s=takt_s)
    raw = []
    diag = []
    for sp in top["spans"]:
        st = steps[sp["step"]]
        a, b = sp["a"], sp["b"]
        if b - a < 2:
            continue
        G = _graph(st["script"], RATE_HZ)
        for g, x, y in align_span(G, E, a, b, sp["mid_start"], sp["mid_end"]):
            pieces = [(g["code"], x, y)]
            if g["code"] == ANY:              # motion-only fallback: relabel the span from the kinematics
                pieces = SG.ergodic(E, x, y, STATION_CODES)
            pieces = [(c_, a_, b_) for c_, x_, y_ in pieces for a_, b_ in _observed_parts(f["unobserved10"], x_, y_)]
            for code, x1, y1 in pieces:
                src = "mocap_rules" if (g["code"] in ("WALK", ANY)) else "merged"
                for code2, x2, y2 in SG.carve_idle(f, code, x1, y1):
                    keep = code2 == code and g["code"] != ANY
                    raw.append((code2, x2, y2, st["name"], src if keep else "mocap_rules", g["step"] if keep else None))
        nexp = sum(s.get("count", 1) for s in st["script"] if s["code"] == "FASTEN")
        if nexp:
            diag.append({"subgoal": st["name"], "expected": nexp, "detected_dwells": SG.count_screw_cycles(f, a, b)})
    els = []
    seen = set()
    for i, (code, a, b, sgname, src, step) in enumerate(raw):
        t0, t1 = a / RATE_HZ, b / RATE_HZ
        conf = float(np.mean(post[code][a:b])) if b > a else 0.0
        e = {"id": f"{station}_{cycle_id}_{i:03d}", "code": code, "start_s": round(t0, 2), "end_s": round(t1, 2),
             "duration_s": round(t1 - t0, 2), "source": src, "confidence": round(conf, 3), "subgoal": sgname}
        e.update(SG.element_attrs(f, code, a, b))
        if step:
            e["activity"] = step["activity"]
            e["activity_text"] = step["text"]
            key = (sgname, step["activity"], step.get("rep"))
            first = key not in seen
            seen.add(key)
            e["step_first_piece"] = first
            if code in ("FASTEN", "REACH_FASTENER"):
                e["fastener_count"] = int(step.get("count", 1)) if first else 0
            if "rep" in step:
                e["repetition"] = f"{step['rep']}/{step['of']}"
        els.append(e)
    for i, e in enumerate(els):
        if e["code"] == "WALK":
            prev = next((x for x in reversed(els[:i]) if x["code"] != "WALK"), None)
            e["loaded"] = bool(prev and prev["code"] in ("GET_PART", "GET_TOOL"))
    for e in els:
        e["duration_s"] = round(e["end_s"] - e["start_s"], 2)
    subgoals = [{"name": steps[sp["step"]]["name"], "refined": [round(sp["a"] / RATE_HZ, 2), round(sp["b"] / RATE_HZ, 2)],
                 "mid_start": sp["mid_start"], "mid_end": sp["mid_end"]} for sp in top["spans"] if sp["b"] - sp["a"] >= 2]
    spans_ok = [sp for sp in top["spans"] if sp["b"] - sp["a"] >= 2]
    review = []
    prev = None
    for sp in spans_ok:
        k = sp["step"]
        flags = []
        if (prev is not None and k > prev + 1) or (prev is None and k > 0 and sp["a"] > 0):
            flags.append("skipped")
        if prev is not None and k < prev:
            flags.append("order_deviation")
        if sp.get("duration_z") is not None and abs(sp["duration_z"]) > 2.0:
            flags.append("duration_outlier")
        ms = [m for m in (sp.get("start_margin"), sp.get("end_margin")) if m is not None]
        conf = min(ms) if ms else None
        if conf is not None and conf < LOW_MARGIN:
            flags.append("low_margin")
        review.append({"cycle_id": cycle_id, "station": station, "step": steps[k]["name"],
                       "start_s": round(sp["a"] / RATE_HZ, 2), "end_s": round(sp["b"] / RATE_HZ, 2),
                       "start_margin": sp.get("start_margin"), "end_margin": sp.get("end_margin"),
                       "confidence": conf, "duration_z": sp.get("duration_z"), "flags": flags})
        prev = k
    w0 = spans_ok[0]["a"] / RATE_HZ if spans_ok else 0.0
    w1 = spans_ok[-1]["b"] / RATE_HZ if spans_ok else dur
    a10, b10 = int(round(w0 * RATE_HZ)), int(round(w1 * RATE_HZ))
    un_s = float(f["unobserved10"][a10:b10].sum()) / RATE_HZ
    cyc = {"station": station, "cycle_id": cycle_id, "duration_s": round(w1 - w0 - un_s, 3),
           "unobserved_in_clip_s": round(un_s, 2),
           "clip_duration_s": round(dur, 3), "cycle_window_s": [round(w0, 2), round(w1, 2)],
           "outside_cycle_s": round(dur - (w1 - w0), 2), "joint_source": joint_source,
           "video": video or {}, "long_dwells_s": f.get("dwells", []), "fastener_check": diag, "elements": els, "review": review,
           "subgoals": subgoals,
           "sop_decoding": {"order": [steps[k]["name"] for k in top["order"]], "pace_factor": top["pace_factor"],
                            "evidence_correlation_time_s": top["kappa_s"],
                            "score": round(top["score"], 2),
                            "step_mtm_nominal_s": {steps[k]["name"]: round(v, 2) for k, v in enumerate(top["nominal_s"])},
                            "valid_pose_share": round(float(np.mean(f["valid10"])), 4),
                            "speed_noise_floor_m_s": f.get("speed_floor")}}
    return cyc, f
