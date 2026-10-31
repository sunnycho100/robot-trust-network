# Generalization: running on footage from a new line

The goal is to use none of the CarDA-specific inputs (Xsens mocap, timestamped annotations, a hard-coded parts table, the "yellow strip" conveyor rule).
With only **video, a customer intake form, and the standard work sheet**, s0 to s8 run end to end.
We checked the result against the original mocap-based pipeline and against a recording the pipeline had never seen (subset B).

```bash
.venv/bin/python scripts/run_all.py --config config/carda_generic.yaml   # generic mode, CarDA subset A (out_generic/)
.venv/bin/python scripts/run_all.py --config config/carda_b.yaml         # unseen recording, subset B WS30 (out_b/)
.venv/bin/python tests/validate_generic.py                               # generalization gates G1 to G6
RCD_CONFIG=config/carda_generic.yaml .venv/bin/python tests/validate.py   # existing per-stage gates applied to generic mode
RCD_CONFIG=config/demo_regress.yaml  .venv/bin/python tests/validate.py   # regression: original pipeline (out_regress/)
```
Cold run time: generic s0 and s1 take about 9 minutes (MediaPipe, two passes), 1 to 2 minutes with the cache. s2 takes about 2 minutes per variant, s3 to s8 under 30 seconds.

## 1. Mode switches (config `mode:`; if absent, original CarDA behavior)

| Switch | Values | What it does |
|---|---|---|
| `human_source` | `mocap` / **`video`** | Video: tracks the worker with MediaPipe Pose (heavy) in two passes. With two or more calibrated views it triangulates (DLT); with one view it reconstructs monocularly by anchoring the feet to the floor. On the belt it compensates for belt travel. BVH is used only for validation and is never read as input. |
| `pose_views` | `auto` / `mono` / `multi` | View selection |
| `conveyor` | `yellow_strips` / **`flow`** / `intake` | Computes optical flow on the bird's-eye floor view and treats a band that moves at one common speed as the belt. Speed comes from phase correlation. |
| `segmentation` | `annotation` / **`sop`** | Solves a two-level HSMM from only the ordered standard-work steps (text). Upper level: SOP steps. Lower level: elements inside a step. |
| `parts` | `builtin` / **`intake`** | Reads part mass, precision, and whether a part is flexible from the customer intake. |
| `video_evidence` | **`learned_qmap`** / `learned` / `rules_qmap` | How video noise is handled (§4) |

Customer input is `intake/<customer>.yaml` (schema `contracts/intake.schema.json`).
It covers the line (takt, paced, shifts, wage), SOP steps and sub-tasks per station, parts, the conveyor (optional), and the camera calibration method (`file`/`checkerboard`/`reference_length`/`none`).
The CarDA intake is generated from **only the text columns** of the annotation (`scripts/make_carda_intake.py`, `make_carda_b_intake.py`). Time columns are not used.

## 2. Results (gate `tests/validate_generic.py`, 2026-10-31)

| Gate | What it checks | Result | Key numbers |
|---|---|---|---|
| G1 | Video 3D pose vs Xsens mocap | **FAIL** | Pelvis path 0.31 m ✓, valid frames 84% (threshold 85%), trunk flexion r 0.22 / MAE 10.7°, wrist speed ρ 0.42 |
| G2 | Segmentation from the SOP only | **FAIL** (partial) | Mocap + SOP + standard times: **3.5 s ✓**, frame agreement 79%. Video + SOP + standard times: A 16 s, B 70 s. Text only (info): mocap 34 s, video 77 s |
| G3 | End-to-end agreement (video only vs mocap) | **FAIL** (partial) | Top-3 automation candidates match **2/3 ✓**. Workload WS10 4% ✓, WS20 29%, WS30 25% |
| G4 | Generic conveyor detection | **PASS** | Median speed error 0%, max 3%, every station detected |
| G5 | No dataset constants | **PASS** | 100% of part data comes from the intake, no CarDA proper nouns in generic modules |
| G6 | Unseen recording (subset B) end to end | **FAIL** (s2 only) | 10/11 existing gates pass. s2 boundary error and EAWS ρ 0.24 |
| Regression | Original pipeline | **PASS 11/11** | Same JSON output (one station assignment differs between tied CP-SAT optima; 4 stations in both) |

**Recommendations compared** (same plant, different inputs):

| Input | Recommended scenario | Robots |
|---|---|---|
| Mocap + annotation (original) | robust_min_cost | WS10 M0617, WS20 M0617, WS30 M0617 |
| **Video + intake only** | robust_min_cost | WS10 M0617, WS20 M0617, WS30 **TM12S** |
| Subset B (WS30, unseen recording) | min_cost | WS30 M0617 |

The recommended scenario and the robot count are the same; one model differs (WS30). CAPEX goes from $437k to $412k and NPV from $97k to $41k.
Video undercounts the observed workload, so the savings shrink and the result is more conservative (§3).

## 3. Where the bottleneck is (checked with numbers)

1. **Pose quality is the bottleneck, not the segmentation algorithm.** The same SOP + standard-time algorithm gives a 3.5 s boundary error on mocap joints and 16 s on video joints.
2. **Camera placement hides forward bending.**
   - The two CarDA cameras sit 1.85 m high and look along ±y. The worker also faces ±y (toward the door).
   - Bending forward by θ shortens the projected torso only by cos(θ−8.5°), at most 6% for θ ≤ 30°. That is smaller than landmark noise.
   - Triangulating two facing views does not fix this. This is why trunk flexion r stays at 0.22.
   - Correcting the clock offset gives r 0.26, and 0.44 on the 10 cycles that actually contain flexion (recomputed separately).
3. **Noise inflates video speed features 5 to 8 times.** Example: median wrist speed during FASTEN is 0.023 m/s on mocap and 0.188 m/s on video.
   - So thresholds tuned on mocap read FASTEN as GET_TOOL.
   - The fix is the quantile calibration and learned evidence in §4. With them, the top-3 match went from 0/3 to 2/3.
4. **The worker leaves the frame.** WS20 has one camera, so about 30% of frames have no pose.
   - One bug was fixed: these frames now count as UNOBSERVED instead of IDLE. That dropped the WS20 IDLE share from 32% to 9%.
   - The observed workload goes down as a result (G3 WS20 29%). Inside the observed time, the working share is 91%, about the same as mocap's 89%.
   - Unobserved time becomes a manual block in s5, so a robot is never credited with work nobody saw.

## 4. Design decisions made during this work

| Decision | Reason |
|---|---|
| Boundary tolerance in SOP mode raised from 3 s to 5 s (about 2% of takt) | 3 s is unrealistic for an input with no timing information at all. |
| G2 split into **G2a, text only (reported)** and **G2b, SOP + times from the standardized work combination sheet (gated)** | Text alone places steps at chance level (6 of 27 segments). Real customers keep per-step manual times on their standardized work combination sheet, so `std_time_s` is a legitimate intake field. To avoid leakage it is simulated **leave-one-cycle-out** (median of the other cycles). |
| Video features calibrated by quantile mapping (`models/pose_calib/video.json`) | Same idea as instrument calibration, with mocap as the reference instrument. It has to be done once per pose model and camera type, not per customer. Fitted **leave-one-station-out**; B uses the map fitted on all of A. |
| Learned evidence: HistGradientBoosting frame classifier on multi-scale window statistics, LOSO | Threshold rules could not recover the temporal structure. Each of the three variants was run once, and learned+qmap was picked by a rule fixed in advance (G3 top-3 first, then frame agreement). **LOSO over three stations is a small sample and fragile** (learned alone gives 1/3). |
| G1 left as FAIL | Metric tweaks such as lag alignment or cycle selection don't make it pass, and the cause is physical (camera placement). The right move is to change the data collection standard (§6), not the gate. |
| s2 work stopped after round 4 | Each round improved less than the one before, and what is left traces back to s1 pose quality (the cameras). |
| SPC gate: three or more stations → **all configured stations** | The original condition could never pass for a single-station customer (subset B). It was a bug in the gate. |
| IE review queue `s2/review_queue.json` | Flags the 20% of boundaries with the lowest confidence for an IE to review. It catches real errors at 17 to 47%, not far from the 20% base rate. Most errors are confident one-step shifts, which is the next thing to fix. |

## 5. What the unseen recording (subset B) showed: WS30 is more overloaded than assumed

- The subset A WS30 clips cover only 22% of the cycle, so the "+2.8% overload" in the original report was an **assumption**.
- Subset B was recorded at the same plant a year later and contains a full WS30 cycle (7 steps).
  - Annotated work time averages 223.7 s (217 / 223 / 231 s).
  - Standard time is 223.7 × 1.16 = 259.5 s, **+8.1%** over the 240 s takt (computed from the annotation).
  - The video-only pipeline estimated +10.6% (standard time 265 s). It closes the cycle late and treats the unobserved part as an assumption.
- Conclusion: the case for a robot or a re-layout at WS30 got stronger. B alone also recommends one M0617 under min_cost.
- Calibration check: ORB matching between B's background and the calibration session's background shows a 0.0 px shift, so the calibration still holds a year later.

## 6. Next steps (priority order)

1. **Change the data collection standard.** Biggest effect, lowest cost.
   - Add **one side camera at 90°** to the worker's facing direction, so bending happens in the image plane (§3.2).
   - An **oblique overhead camera at 2.5 m or higher** cuts down occlusion by carts and the worker leaving the frame (WS30, WS20).
   - Make **per-step times from the standardized work combination sheet** a required intake field (the basis for G2b).
2. **Upgrade the pose model.** With a GPU, switch to a temporal model such as WHAM or GVHMR. Only the s1 `human` output contract has to match.
3. **Collect reference-instrument data.** Paired mocap (or IMU) and video, a little for each plant and camera type, makes the §4 calibration maps and classifier better as the customer count grows. This is the data flywheel.
4. Improve the review queue: catch step-identity errors using the score gap to the second-best step order.
5. (Deferred) Add robot categories and templates: welding, dispensing, machine tending, AMR only, and so on.

## 7. Files

- Configs: `config/carda_generic.yaml`, `config/carda_b.yaml`, `config/demo_regress.yaml`
- Intake: `intake/carda.yaml`, `intake/carda_b.yaml`, `intake/std_times_loo/`, `contracts/intake.schema.json`, `src/rcd/common/intake.py`
- s0/s1 (video): `s0_calib/generic_calib.py`, `s1_scene/{pose2d,video_pose,conveyor_flow,generic_run}.py`, `scripts/eval_video_pose.py`
- s2/s3 (SOP and intake): `s2_elements/{sop_lexicon,sop_segment,sop_run,generic_io,generic_calib,generic_clf}.py`, `s3_automatability/generic_parts.py`
  - Scripts: `scripts/{make_std_times_loo,fit_pose_calib,train_element_clf}.py`
- Models: `models/pose_landmarker_*.task` (public MediaPipe weights, downloaded on first use), `models/pose_calib/` (calibration maps and classifiers, with training provenance)
- Gate: `tests/validate_generic.py` → `out_generic/validation_generic.json`
- Methods: `docs/IE_METHODS.md` §6
- Extra dependencies: `mediapipe`, `scikit-learn` (`requirements.txt`)
