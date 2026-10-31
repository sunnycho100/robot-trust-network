# Design space exploration (s5x): every option worth choosing, not one answer

## Why
The old s5 solved 8 named scenarios (min_cost, robust_min_cost, ...) and used rules to recommend **one**.
Customers actually ask "what if we use that robot instead?", "what if we only buy 2?", "what if the budget is $300k?".
s5x builds **every combination** the line allows, scores them, keeps only the ones nothing else beats, and then shows which one fits a customer's priorities. The existing recommendation (s5) is left as it is, and s5x shows where it sits in the full space.

## Algorithm (IE methods)
| Step | Method | Source |
|---|---|---|
| 1. Full enumeration | Full factorial design: robot per station (none, plus every model that passed s4 and was simulated in s6a) × layout improvement on/off × 3 task-split policies (min_cost / max_automation / ergonomics_first, i.e. what the robot does) × planning cycle (takt / 95 % of takt). For each combination the s5 MILP optimizes the task split with the robots fixed. Combinations that can't meet takt are retried with a relief worker allowed | Montgomery 2017 ch.5 |
| 2. First screen | Pareto non-dominated set: NPV↑, installed cost↓, awkward posture s/door↓, max station cycle↓, robot count↓. For every dominated design, the ID of a design that beats it on every criterion is recorded | Deb 2001 |
| 3. Simulation | Screened designs plus the s5 named scenarios run through the s6b DES with **common random numbers (CRN)**: every design sees the same random stream, so the difference between two designs is a paired comparison (variance reduction) | Law 2015 §11.2 |
| 4. Final Pareto | DES takt overrun rate replaces max station cycle | |
| 5. Preference analysis | SMAA-2: 20,000 weight vectors sampled uniformly from the simplex → share of weights where each design ranks 1st, 2nd, 3rd (rank acceptability) and the typical weights of a customer who would pick it (central weight). 5 stakeholder profiles (CFO, EHS, plant manager, pilot first, equal) | Lahdelma & Salminen 2001 |
| 6. A vs B | Winner per criterion, dominance, "share of all weights where A is preferred", CRN paired throughput difference ±95 % CI, trade-off price (extra installed cost per second of awkward posture removed) | |

## Results (CarDA mocap, `config/demo_regress.yaml`)
- 162 robot combinations → 1,936 factorial runs (3,361 MILP solves) → **1,219 distinct designs** → first Pareto 40 → simulated 41 → **final Pareto 31**. Runtime 42 s.
- The existing recommendation robust_min_cost = D713 (M0617 ×3) is on the final Pareto front and is the first choice of the plant manager (stability) profile.
- First choice by profile: CFO and equal → D059 (one M0617 at WS20 only, NPV $238k, installed cost $191k, takt overrun 28 %);
  EHS → D715 (M0617 ×3, ergonomics-first split); pilot first → D003 (layout only, no robot).
- Middle option: D085 (M0617 at WS20 and WS30, 2 robots): NPV $165k, installed cost $314k, takt overrun 15 %, awkward posture 22 s/door.
- Model main effect: only M0617 (every station) and TM12S (WS30, 3 designs) reach the final Pareto.
  UR10e, UR15, CRX, HC10DTP and M1013 lose to another design on every criterion in every combination. The reason: for screwdriving,
  only M0617 passed reach and payload in s4 (WS20, WS30). The other models can only take inspection and picking, so the savings are small.

## The three options shown to the customer (headline)
31 is too many for a customer to choose from. `pick_headline` puts 3 up front (the rest go to the appendix and the explorer).
1. **Realistic candidates**: final Pareto designs that need no relief worker and have NPV > 0 → 10 of 31
   (14 need extra hires, 7 have NPV ≤ 0).
2. **Archetype = robot count** (0, 1, 2, 3). Near-identical variants of one archetype are merged into one card.
   The representative is the design with the highest expected utility over all weights.
3. **Maximize preference coverage**: for each of the 20,000 weight samples, find which archetype holds the best realistic candidate,
   and pick the 3 archetypes that hold that best design most often (ties: include the s5 recommendation, then the cheaper one).

| Result (mocap) | Design | Share of weights where it is best | Profiles that pick it first |
|---|---|---|---|
| A. Layout only (no robot) | D003 | 24 % | pilot first |
| B. One robot (M0617 at WS20) | D059 | 38 % | CFO, equal |
| C. Three robots (M0617 ×3, existing recommendation) | D713 | 26 % | EHS, plant manager |

Together the three options hold the best design for **88 %** of all possible priorities. The 2-robot archetype (D085) is best only 12 % of the time and goes to the explorer.
The video-only run (carda_generic) gives the same three archetypes (on the 3-robot side WS30 is TM12S), with 87 % coverage.
Subset B has one station and only two archetypes, so it shows 2 options.

## Report
**Design space** section in `out*/s8/report.html`:
scatter plot (installed cost vs NPV; dominated, screened, final Pareto) · SMAA rank acceptability · choice per profile · head-to-head table against the recommendation ·
**interactive explorer** (weight sliders, budget cap, relief worker allowed or not → live re-ranking; pick any two designs for an A vs B card:
bars and winner per criterion, tasks per robot, station cycles, DES throughput) · full final Pareto table · "if you must use this model" main-effect table.

## Validation gate (`tests/validate.py --stage s5x`, 21 checks)
Recount of factorial runs, every design meets takt, first and final Pareto recomputed with matching flags, every "dominated_by" claim verified,
SMAA first-place shares sum to 1, profile weights sum to 1, A vs B per-criterion winners recomputed, all 7 s5 scenarios located,
recommended design NPV = s7 NPV, recommended design DES throughput agrees within CI with s6b run on different random numbers.

## Limitations
- At most one robot per station (two robots working together in one station is outside the model). Sharing a robot across stations (7th-axis rail) appears only in the s5 machine_cluster analysis.
- Weights are normalized with a linear min-max scale over the simulated candidates, so scores change when the candidate set changes (rank reversal is possible, a known property of MCDA).
- Ergonomic and quality benefits are not converted to money (they are used only as criteria).

## Run
```
.venv/bin/python scripts/run_all.py --config config/demo_regress.yaml --stage s5x s8
RCD_CONFIG=config/demo_regress.yaml .venv/bin/python tests/validate.py --stage s5x s8
```
Files: `src/rcd/s5_optimize/explore.py`, `src/rcd/s5_optimize/allocation.py` (`fix_models` argument),
`src/rcd/s8_report/ds_app.py`, `src/rcd/s8_report/figures.py` (`design_space_chart`), `contracts/design_space.schema.json`.
