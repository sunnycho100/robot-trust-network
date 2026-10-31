# robot_consult_demo: from factory video to a robot purchase decision

This pipeline takes the CarDA dataset (car door assembly line at the Stellantis Melfi plant, 3 stations, 13 cycles, CC BY 4.0) and carries it all the way to **which tasks to automate, with which off-the-shelf robots, how many, why, and at what cost**.
Every stage runs locally on a laptop without a GPU. A full run of the original mode takes about 2 minutes.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
bash scripts/fetch_carda.sh                # CarDA subset A, about 5 GB (.svo stereo files are skipped)
```

MediaPipe pose weights are downloaded into `models/` on first use.

## Run

```bash
.venv/bin/python scripts/run_all.py        # full pipeline, writes out/
.venv/bin/python tests/validate.py         # per-stage acceptance gates (currently 12/12 PASS)
```

- Report: `out/s8/report.html` (one self-contained file). Supplier RFQs: `out/s8/rfq/WS*.md`.
- Package for the simulation team: `python scripts/export_sim_package.py` writes `out/sim_package/`. See `docs/SIM_HANDOFF.md`.
- Video-only mode (no mocap, no timestamped annotations): `--config config/carda_generic.yaml`. See `docs/GENERALIZATION.md`.

## Stages and owners

| Stage | What it does | Key methods | Owner |
|---|---|---|---|
| s0 | Camera calibration | Provided K, R, t plus checkerboard/ArUco reprojection check (RMSE 0.12 to 0.49 px) | 3D |
| s1 | 3D scene and 4D human | Xsens forward kinematics, conveyor speed from phase correlation on the floor plane (about 24 mm/s), work zones and contact points | 3D |
| **s2** | **Work-element breakdown and time study** | Therblig/MTM-UAS taxonomy, HSMM forced alignment (Viterbi), Westinghouse rating, ILO allowances, standard time, required sample size, MTM-UAS comparison, I-MR chart and Cpu, lean VA/NVA (TIMWOOD), spaghetti diagram, EAWS/RULA/REBA | **IE** |
| **s3** | **Automatability score** | Fraunhofer IPA APA and MTM suitability criteria, AHP weights (CR 0.011), ±20% sensitivity analysis | **IE** |
| s4 | Robot catalog and candidates | 15 commercial models (source URL, price range), IK-based placement search for reach, AHP+TOPSIS, ISO 13855 safety distance and aisle allowance | robot sim |
| s6a | Robot motion simulation | MuJoCo, 8 manufacturer URDF kinematics plus Menagerie, DLS IK, conveyor tracking, trapezoidal velocity profile | robot sim |
| **s5** | **Allocation and station count** | RPW/LCR/COMSOAL heuristics and exact SALBP-1/2 (CP-SAT), human-robot allocation MILP, worker-machine chart, machine clusters, RCA | **IE** |
| **s5x** | **Design space** | Full factorial over robot choices, Pareto screening, SMAA-2 preference analysis, three options for the customer (`docs/DESIGN_SPACE.md`) | **IE** |
| **s6b** | **Line simulation** | SimPy paced line, 30 replications with 95% CI, Little's law check, comparison against the analytic solution | **IE** |
| s7 | Cost and ROI | A/P capital recovery factor, NPV/IRR/payback, BLS wages, tornado sensitivity | econ |
| s9 | Robot task plan | BehaviorTree.CPP v4 generated from the element table (slot for SeeDo and Video-to-BT style work) | robot sim |
| s8 | Report and RFQ | Every number is read from the JSON files in `out/` and rendered | all |

IE formulas and their sources are in `docs/IE_METHODS.md` (ISyE 510 lectures 03/04/06/07, RCA lab, ISyE 512 Ch. 6).

## Replacing a stage

- Data contracts between stages are in `contracts/*.schema.json` and `docs/CONTRACTS.md`.
- To replace a stage with a real implementation, swap `run(cfg)` in `src/rcd/<stage>/run.py`. If it writes the same JSON, everything downstream keeps working.
  - Example: the 3D owner replaces Xsens in s1 with video-based pose (WHAM/GVHMR) and 3DGS.
  - Example: the robot simulation owner replaces s6a with Isaac Sim or cuRobo.
- After a swap, `tests/validate.py` has to pass. The gates do not trust reported numbers. They recompute them from raw data and formulas.
  - Examples: standard time = normal time × (1 + allowance), line balance efficiency, Cpu, Little's law, A/P factor, ISO 13855 S = K·T + C.

## Results on CarDA

- **Current state (standard time)**: WS20 is **9.4% over** the 240 s takt (measured). WS10 +6.1% and WS30 +2.8% depend on parts of the cycle the clips don't show, so they are assumptions.
- **Recommendation**: one Doosan M0617 per station (linear track synced to the conveyor) plus AMR kitting.
  - Max station cycle 262.5 s → 211.1 s
  - Awkward-posture exposure 83.1 s → 13.1 s per unit
  - Takt overrun in the DES 65% → 5%
- **Economics**: CAPEX $437k, net annual savings $134k, payback 3.3 years, NPV $97k (5 years, 8%). Counting headcount savings only, NPV is negative. The inputs that move the result most are shifts per day and wage.
- **Options for the customer**: s5x evaluates 1,219 designs and shows three (layout only / one robot / three robots) that together cover 88% of possible priority weightings. See `docs/DESIGN_SPACE.md` and `docs/SIM_OPTIONS_ABC.md`.

## Limitations (also stated in the report)

- CarDA clips cover only part of the roughly 4-minute cycle (WS10 39%, WS30 22%), so the WS10 and WS30 overloads are assumptions.
- There is no frame-level ground truth for work elements, so element labels are motion-based estimates. Ergonomics validation reached Spearman 0.57 and did not reproduce trunk twisting.
- Part weights, track and tool changer costs, and the integration cost multiplier are assumptions with cited sources. Manufacturers don't publish robot accelerations, so a common value is used.
- No VLA was run (no GPU). s9 is a rule-based plan that marks where a VLA/VLM policy would plug in.

## Data

`data/` is not tracked. `bash scripts/fetch_carda.sh` downloads CarDA subset A from Zenodo ([10.5281/zenodo.14644367](https://zenodo.org/records/14644367), CC BY 4.0) into `data/carda/`.
Subset B (the held-out WS30 recordings used in `docs/GENERALIZATION.md`) is in the same record. Put it under `data/carda_b/ws30/` with the same layout.

`models/pose_calib/` holds the video feature calibration map and the frame classifiers trained on CarDA (`scripts/fit_pose_calib.py`, `scripts/train_element_clf.py`).

## Layout

```
config/      run configs (demo, demo_regress, carda_generic, carda_b)
intake/      customer intake forms (line, SOP steps, parts, cameras)
contracts/   JSON schemas for every stage output
src/rcd/     one folder per stage, each with run.py
scripts/     run_all, data fetch, packaging, model fitting
tests/       acceptance gates
docs/        methods, contracts, generalization, design space, simulation requests
models/      pose calibration and element classifiers
```
