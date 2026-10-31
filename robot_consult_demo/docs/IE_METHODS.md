# Industrial-engineering methods (stages s2, s3)

Code: `src/rcd/s2_elements/`, `src/rcd/s3_automatability/`. Outputs: `out/s2/{elements,time_study,ergonomics}.json`,
`out/s2/figures/*.png`, `out/s3/automatability.json`. Every number below is computed from the CarDA data at run time
unless marked *assumption*.

## 1. Work-element taxonomy (s2 `taxonomy.py`)
Elements are *minimum rational work elements* (ISyE 510 lecture06), each one an MTM-UAS basic-operation group
mapped to Gilbreth therbligs:

| code | therbligs | MTM-UAS | lean |
|---|---|---|---|
| WALK | TE / TL | KA (25 TMU/m) | NVA - Motion (loaded share = Transportation) |
| GET_PART | TE, S, SE, G, TL | AB (<=1 kg) / AJ (1-8 kg) | NNVA |
| INSERT | P, A, RL | PC (place tight) | VA |
| GET_TOOL | TE, G, TL, RL | HA (handle tool) | NNVA |
| REACH_FASTENER | TE, SE, G, TL, PP | AF (difficult grasp) | NNVA |
| FASTEN | TL, P, U | PC + BA (place tool, actuate) | VA |
| INSPECT | I | VA (15 TMU) | NNVA |
| SETUP | TL, P, PP | BB + KA | NNVA |
| IDLE | UD, AD, H | - | NVA - Waiting |
| UNOBS (pseudo) | - | - | manual block, never automated |

A keyword lexicon turns each CarDA activity line into a code. "Fix N screws" with sub-steps is expanded N times;
template steps that were not performed may be skipped (see 2).

## 2. Segmentation from mocap (s2 `features.py`, `segment.py`, `elements.py`)
The signals come from Xsens joints at 10 Hz: pelvis horizontal speed, wrist speeds and dwell (the slower hand),
reach (horizontal wrist-to-pelvis distance), hand separation and height difference, and trunk flexion.
* **Time-local work anchor**: the centroid of the nearest-in-time long dwell (pelvis stationary for at least 4 s).
  The anchor moves with the carrier. Short stops more than 0.6 m from it are line-side picks.
* **Evidence rules** (fixed constants): WALK if pelvis speed > 0.3 m/s. GET_* if the operator slows down away
  from the anchor. INSERT if the hands are level and close together at the anchor. FASTEN if one hand dwells
  (< 0.08 m/s) on an extended arm (> 0.33 m). REACH_FASTENER if a hand is near the body or the hands come
  together while moving. IDLE if both hands are still.
* **Standard-work forced alignment**: inside each annotated subgoal, a left-to-right hidden semi-Markov model is
  decoded with Viterbi. It has explicit minimum durations (chained sub-states, Rabiner 1989 sec. IV-D), an
  optional WALK between every pair of steps, and profile-HMM deletions (skip cost 6). The written standard
  work fixes the order of elements; the motion fixes every boundary.
* Outside the annotated subgoals, an ergodic sticky HMM (switch penalty 4) is used, followed by a
  minimum-segment-length merge.
* **IDLE carving**: runs of 3 s or more with both hands still inside a long element become IDLE (for example,
  waiting for the robot hand-over).
* **Subgoal boundaries** are snapped to the nearest motion pause within ±1.5 s (median error vs. CarDA: 0.45 s).
* **Fastener plausibility**: an independent count of fastening dwells is compared with the N stated in the
  activity list (`elements.json -> cycles[].fastener_check`).

## 3. Time study (s2 `timestudy.py`)
* **Unit**: the per-cycle total of a code in a station (`element_key = <station>:<code>`), giving mean,
  standard deviation (ddof = 1) and n.
* **Westinghouse leveling** (Niebel & Freivalds), with rating = 1 + skill + effort + conditions + consistency.
  Table values are in the code.
  * Skill D (0) and effort D (0). The rigidly paced conveyor sets the pace (ISyE 510 lecture06).
  * Conditions C (+0.02): a clean, well-lit pilot line.
  * Consistency is derived from the time-weighted coefficient of variation (CV) of the same subgoal's duration
    across cycles: CV <= 2 % A, <= 5 % B, <= 10 % C, <= 20 % D, <= 30 % E, otherwise F.
  * Normal = mean x rating. IDLE normal = 0, because delay is in the PFD allowance.
  * UNOBS is counted at its elapsed time: normal = mean/(1+A), so that standard = elapsed residual.
* **ILO allowances**:
  * Constant: personal 5 %, basic fatigue 4 %.
  * Standing 2 %.
  * Abnormal position, from the measured EAWS posture shares: 2 % if bent or elbow-at-shoulder >= 10 % of the
    time; 7 % if hands above head >= 10 %.
  * Use of force, from the ILO kg steps applied to the heaviest part.
  * Close attention 2 % and monotony 1 % (analyst judgement).
  * The total is derived, with no balancing item: 0.16 at line level (`params.allowance_pfd`), and per-station
    totals are given in `allowance_detail.by_station`.
  * Standard = normal x (1 + A).
* **Sample size**: n' = ceil((z s / (a x̄))²) with z = 1.96 and a = 0.05.
* **MTM-UAS (PMTS)**: each occurrence is built from UAS blocks. The distance class (<= 20 / 20-50 / 50-80 cm)
  comes from the active wrist's maximum displacement in mocap. Walking uses KA x metres of pelvis travel, and KB
  is added when trunk flexion >= 60°. pmts_s = TMU x 0.036. The observed/PMTS gaps are explained per code in
  `pmts.rows[].gap_explanation`. The synthetic rating (Morrow) PMTS/observed for handling elements is reported
  but not applied.
* **SPC (Individuals and Moving Range, I-MR)** (Montgomery; ISyE 512 Ch. 6) on per-cycle station work content
  (non-idle observed time + UNOBS):
  * UCL = x̄ + 2.66 MR̄ and σ̂ = MR̄/1.128.
  * Cpu = (takt − x̄)/(3σ̂).
  * `basis` flags stations where UNOBS dominates.
* **Lean**: each code is classified VA / NNVA / NVA and NVA is mapped to TIMWOOD. VA ratio = VA time / observed
  time. The spaghetti distance is the pelvis path with sway below 0.1 m/s removed (`figures/spaghetti_*.png`).
* **Overload honesty**: `station_stats` gives observed_fraction = observed_mean_s / full_cycle_s,
  overload_basis (measured if >= 0.85, otherwise assumption) and utilisation = standard_s / takt.

## 4. Ergonomics (s2 `ergonomics.py`)
* **EAWS section-1 postures** (Schaub et al. 2013):
  * P3 bent 20-60° and P4 > 60°, using Hips→Neck flexion in the pelvis sagittal plane.
  * P5 elbow at or above the shoulder, and P6 wrist above the head.
  * TR: shoulder line vs. hip line > 10°. LB: frontal-plane inclination > 10° (ISO 11226).
  * A posture is counted only in bouts of 4 s or more (gaps under 1 s are bridged first). This is the EAWS
    static rule, and CarDA's annotators used the same rule.
* **RULA upper-arm score** (1-4) and **REBA trunk score** (1-4, +1 if twisted or side-bent) are reported per
  cycle and per subgoal.
* **CarDA column mapping** (headers missing) is inferred by an order-preserving assignment of columns G..N to
  (P1, P3, P4, P5, P6) and {O, Q} to {TR, LB}. The assignment minimises MAE, and a column is mapped only if it
  beats the "0 s" predictor. Result: G = P1, H = P3, K = P5, O = TR, Q = LB. Columns I, J and L have data but
  no skill, so they stay unmapped.
* **Validation**: the headline excludes trivial P1. Spearman 0.57 over 124 (cycle, subgoal, posture) pairs,
  MAE 7.1 s. By posture: P3 0.63, P5 0.76, LB 0.50, TR −0.06 (not reproduced). With a 2-s holding rule the
  value is 0.60 (sensitivity only).

## 5. Automatability (s3)
* **Criteria** (8, all scores in [0,1]): repetitiveness, handling simplicity, tolerance slack, force
  feasibility, motion consistency (mocap dispersion), ergonomic burden (s2), cycle-time share (Pareto), and
  transport content.
* **Sources**: Fraunhofer IPA Automation Potential Analysis; MTM-based automation suitability (Eng 2024 5(2) 53);
  HRC task-allocation review (IJCIM 2023).
* **AHP weights** (Saaty): principal eigenvector, CR = 0.011 (RI = 1.41). The pairwise judgements and their
  rationale are in `ahp`.
* **Scoring and allocation**:
  * total = Σ w·s.
  * robot if total >= 0.60 and handling and force >= 0.5; human_robot if total >= 0.45; otherwise human.
  * WALK and SETUP go to layout_change. IDLE goes to human, because waiting is removed by balancing.
  * GET_TOOL and REACH_FASTENER follow the station's FASTEN allocation: the tool becomes the robot's
    end-of-arm tool (EOAT) and screws are auto-fed.
* **Requirements per element_key**, plus `groups[]` keyed by `<station>:<category>`:
  * Payload = part + EOAT (OnRobot Screwdriver 2.5 kg, Robotiq 2F-85 0.925 kg; part masses are
    *assumptions* in `parts.py`).
  * Reach = p95 3D wrist distance from a base at 0.85 m beside the operator, plus half the conveyor travel per
    occurrence (s1 speed of about 24 mm/s).
  * Precision, force/torque, conveyor_tracking = true, and the s1 workpiece_contact points.
* **Sensitivity**: 1000 draws with each weight multiplied by U(0.8, 1.2) and renormalised. Rank stability is the
  mean Spearman correlation against the baseline ranking. The allocation flip rate and the top-5 probability are
  also reported.

## Known limitations
* Element labels are not hand-verified frame by frame. The CarDA activity lists are templates; for example, WS10
  and WS20 list both tool-station and robot hand-over variants. Some forced-aligned elements are therefore long
  (WS20 FASTEN up to 46 s, which includes hand-over waiting).
* Part masses, force levels and handling scores are assumptions.
* With only 4-5 cycles per station, n' far exceeds n for most elements.
* WS10 and WS30 cover only 22-39 % of the 240 s cycle, so their overload and Cpu rest on the UNOBS assumption.

## 6. Generic mode: SOP-only segmentation, intake parts (`mode.segmentation: sop`, `mode.parts: intake`)
Code: `s2_elements/sop_lexicon.py`, `sop_segment.py`, `sop_run.py`, `generic_io.py`, `s3_automatability/generic_parts.py`.
Default mode is unchanged (byte-identical s2/s3 outputs).
* **Inputs**: joints from `<out_root>/s1/scene_graph.json` (any joint set; names canonicalised, missing pelvis /
  neck / head synthesised, `valid` mask -> neutral evidence); the intake's ordered SOP steps (text only). The
  annotation workbook is read only to validate EAWS postures on the annotators' spans, never for segmentation.
* **Lexicon** (dataset-neutral): fasten/tighten/screw/bolt/rivet -> FASTEN, pick/take/get/grab -> GET_PART,
  place/insert/mount/fit/attach/install -> INSERT, inspect/check/verify/test -> INSPECT, walk/go/move to -> WALK,
  adjust/set up/position cart -> SETUP, tool handling -> GET_TOOL, fastener acquisition -> REACH_FASTENER,
  wait / robot-subject sentences -> IDLE; unmatched text -> motion-only state. Group headers ("Fasten 4 bolts" +
  acquire/drive lines) are expanded N times from the flat list.
* **Two-level HSMM** over the whole clip (`sop_segment.py` docstring): SOP steps (skippable, clip may start/end
  mid-SOP, non-cycle time outside the SOP, Mallows order prior, weak MTM-UAS duration prior with a pace factor,
  boundary prior at line-side visits / posture change points) x per-step element scripts (profile HSMM).
  Hybrid scaled likelihoods of the kinematic rules, tempered by their autocorrelation time.
* **Result on CarDA subset A (mocap)**: element frame agreement with the annotation-guided reference ~77%, but
  step identity/boundaries are often wrong (median boundary error ~34 s). With the annotated spans given, the
  true step's script fits its own span best in only 11/27 spans (chance 5.4/27): the activity text + kinematic
  element rules do not discriminate SOP steps well, and MTM durations miss machine/process waiting (e.g. AWS height
  adjustment ~25-35 s vs. 5 s PMTS), which makes a one-step-shifted labelling of WS20 the more likely one.
  An intake standard time per step (`std_time_s`) is supported and would make the duration prior informative.
* **Standard-work-sheet times** (optional intake field `std_time_s` per SOP step; CarDA emulation leave-one-cycle-out
  by `scripts/make_std_times_loo.py`): log-normal step-duration prior with sigma 0.20 (Westinghouse consistency
  class D); the sheet total replaces takt as the expected cycle length. Outputs `elements_sop_std.json`.
* **IE review queue** (`s2/review_queue.json`): per decoded step, boundary margins (score gap to the best
  alternative boundary >= 2 s away within +/-10 s), flags skipped / order_deviation / duration_outlier / low_margin;
  the lowest-confidence 20 % are `needs_review`.
* **Pose-source calibration** (`generic_calib.py`, `scripts/fit_pose_calib.py`, `models/pose_calib/video.json`):
  the video pose pipeline is a measurement instrument calibrated against a reference instrument (mocap), like a
  gauge study / calibration curve. On video, jitter inflates every speed-type feature 4-8x (median wrist speed
  0.34 vs 0.09 m/s, pelvis 0.17 vs 0.05 m/s) while reach stays close, so the absolute evidence thresholds misfire.
  A monotone empirical quantile map from test to reference (201 quantiles) is fitted for each feature group
  (pelvis speed, wrist speed, hand distance, hand height difference) on paired recordings. For evaluation it is
  fitted leave-one-station-out, so the map applied to station S never saw S; held-out lines use the map fitted on
  all reference stations. It is fitted once per pose model and camera type, not per customer, and stored with its
  provenance. It is applied only when `mode.human_source = video`.
  Result on CarDA: the marginal distributions are matched, but frame agreement with the reference elements stays
  at about 39% and the top-3 automation overlap is 1/3. Jitter also destroys the temporal structure of the signals
  (spurious short motions), which a marginal map cannot restore.
* **Unobserved time** (video): pose gaps of at least 1 s (`valid == False`, e.g. the operator out of the only camera)
  carry no evidence and get no element. They are counted as unobserved time, like the part of the cycle outside a
  cropped clip: `duration_s` excludes them, and UNOBS = full cycle - observed time. Before this fix, the
  interpolated pose in these gaps looked motionless and was labelled IDLE (WS20 video IDLE was 32% vs 11% in the
  reference; it is 9-10% now).
* **Learned evidence** (`generic_clf.py`, `scripts/train_element_clf.py`, `models/pose_calib/elements_clf_*`):
  * Model: a HistGradientBoosting frame classifier. Input: multi-scale (0.5 / 1.5 / 4 s) rolling mean, std, p10 and
    p90 of the speed, reach, hand-distance and wrist-height signals, plus trunk flexion and the distances to the
    work anchor and to the line side. Labels: reference element codes (mocap + annotated subgoals).
  * Evidence: the classifier's log P(code|x) - log P(code) replaces the threshold rules (a hybrid HMM). The HSMM
    and SOP structure are unchanged.
  * Training is strictly leave-one-station-out: the model applied to station S never saw S. Held-out lines use
    the model trained on all stations. LOSO frame accuracy before the HSMM is 0.34-0.41.
  * Three variants were compared with one pass each: rules + quantile map, learned, and learned + quantile map.
    The primary is learned + quantile map, chosen by G3 top-3 (the only variant with 2/3), then frame agreement.
  * Caveat: leave-one-station-out with only three stations is a small-sample estimate of transfer to a new line.
