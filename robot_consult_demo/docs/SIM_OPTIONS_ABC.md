# Simulation request v3: validate each of the three customer options (A, B, C)

Author: industrial engineering · Data: CarDA (Stellantis Melfi experimental line, CC BY 4.0) · Basis: run of `config/demo_regress.yaml` (`out_regress/`), packages in `out_options/`

> **What changed since v2**
> - v2 (`docs/SIM_HANDOFF.md`) asked you to validate one recommended option (three M0617 robots).
> - This time the pipeline scored all 1,219 robot combinations and then picked **three options** to show the customer.
> - Each option is the pick of a customer with different priorities. For a fair comparison, **all three** must be simulated against the same criteria.
> - Coordinate frames, time conventions, the SSM formula, and KPI definitions are the same as v2 (see v2 sections 2 and 7). The only new parts are the **per-option packages and the run matrix**.

---

## 0. Questions this simulation must answer

1. **Option A (no robot, layout only)**: Is the assumption right that AMR kitting cuts worker walking time in half? If not, option A misses takt (WS30 is already at 239.9 s against a 240 s takt).
2. **Option B (one M0617 at WS20)**: Does one robot clear the WS20 bottleneck? And do WS10 and WS30, which have no robot (233.6 s and 239.9 s), actually hold?
3. **Option C (three M0617, earlier recommendation)**: The main v2 issue sits here. SSM violation is 71% at WS10, and 100% at WS30 if the worker stays. Does it stay within the 228 s cap with the safety policies (P1, P2, P3) applied?
4. **Does the ranking of the three options change?** When you send results back, industrial engineering will swap in the station cycle distributions and compare the three options again.

---

## 1. The three options

Takt is 240 s, and the planning cap is 95% of takt, 228 s. The station cycles below are IE predictions (s5) based on standard time. Throughput and takt overrun are predictions from the line DES (s6b, 30 replications).

| | **Option A: layout only** | **Option B: 1 robot** | **Option C: 3 robots (earlier recommendation)** |
|---|---|---|---|
| Robots | None | M0617 at WS20 (linear track, tool changer) | M0617 at WS10, WS20, WS30 (linear track) |
| Layout | AMR kitting (all stations, walking ×0.5 assumed) | Same | Same |
| WS10 cycle | 233.6 s ⚠ | 233.6 s ⚠ | 201.7 s |
| WS20 cycle | 233.2 s ⚠ | 160.2 s (robot 150.2 s) | 160.2 s (robot 150.2 s) |
| WS30 cycle | 239.9 s ⚠ | 239.9 s ⚠ | 211.1 s |
| DES throughput / takt overrun | 13.58 units/h · 41% | 14.14 units/h · 28% | 14.86 units/h · 5% |
| Capex / NPV (5 yr) | $54k / $41k | $191k / $238k | $437k / $97k |
| Awkward posture (s/door) | 80.0 | 36.8 | 13.1 |
| Chosen by | Customers who want to start with a small pilot | CFO, equal weights | Safety (EHS), plant manager |
| Share of priority weightings where this option is best | 24% | 38% | 26% |

⚠ = above the 228 s cap (still within the 240 s takt).

> Taken together, the three options contain the best option for **88%** of 20,000 possible priority weightings. The type left out (two robots) is best in only 12%, so it is not part of this request.

---

## 2. Packages and how to run

```bash
python scripts/export_option_packages.py      # regenerate all of out_options/ (about 20 s)
cd out_options/C/sim_package
python replay.py WS10 --video                  # same baseline replay as v2 (KPI, trace CSV, preview, mp4)
python replay.py WS30 --after away             # assume the worker leaves after the motion data ends
```

| Path | Contents |
|---|---|
| `out_options/options_matrix.yaml` | **Run matrix for this request** (per-option variables, replication counts, IE predictions). The `scenario_matrix.yaml` inside each package is the v2 file, so the names differ. v2's A_recommended is option **C** here, and v2's B_min_cost is option **B** here |
| `out_options/manifest.json` | Per option: design ID, s5 scenario name, metrics, station times, robot tasks, baseline replay KPIs |
| `out_options/A/sim_package/WS10·WS20·WS30/` | **Worker-only packages**<br>`human_asis.npz`, `human_tobe.npz` (walking compressed by kitting), `human_tobe_timeline.json` (station times, including layout levers)<br>For the scene, take `scene.xml` from the same station in option C and remove the robot, track, and trays |
| `out_options/B/sim_package/WS20/` | Same layout as v2: `scene.xml`, `targets.json`, `robot_reference.json`, `motion.json`, worker paths, baseline KPIs |
| `out_options/C/sim_package/WS10·WS20·WS30/` | Same layout as v2. Replay KPIs confirm it matches v2's `out/sim_package` |

The option B robot cell at WS20 is the same as WS20 in option C (same tasks, same robot program). So what is new in option B is **the whole line, not the WS20 cell**.

---

## 3. What to do for each option

### Option A: worker-only line (no robot cell simulation)
- **Assumption to test**: AMR kitting scales walking (WALK) time by ×0.5. Option A as a whole stands or falls on this one assumption.
- **Runs**
  - For each of the 3 stations, walk factor {0.5, 0.7, 1.0} × 30 randomized draws = **270 runs**
  - Worker element times use the lognormal distributions in `out/s2/time_study.json` (same as v2)
- **Report**
  - p50 and p95 of station cycle
  - Share of cycles above 228 s
  - AMR delivery loop: can one AMR serve all 3 stations within takt
  - Points where workers and the AMR meet (the line-side areas are the zones in `scene.xml`)
- **Pass criterion**: if p95 ≤ 228 s still holds at walk factor 0.7, option A works. If it holds only at 0.5, flag it to the customer as "sensitive to the assumption".

### Option B: one robot at WS20 + worker-only WS10 and WS30
- **WS20 cell**
  - Policies P1, P2, P3 × worker {stays, leaves} × 30 runs = **180 runs**
  - Baseline replay: SSM violation 10%, minimum distance −0.18 m. The only overlap is the first ~27 s
  - Please check the 8 mm reach margin on pick-and-place (v2 section 4)
- **Line impact**
  - WS10 (233.6 s) and WS30 (239.9 s) exceed the 228 s cap with workers alone
  - Feed the cell results into the line DES to confirm the actual takt overrun rate of option B (IE prediction 28%)
  - For this part, reuse the option A results as they are (WS10 and WS30 distributions per walk factor)
- **Pass criterion**: WS20 robot cycle p95 ≤ 218 s, 0 collisions, 0 IK failures. The weak spot of option B is **the two worker-only stations**, not the cell; keep them separate in the report.

### Option C: three robots (same cells as v2, now compared on the same basis as A and B)
- **Runs**: 3 stations × P1, P2, P3 × worker {stays, leaves} × 30 = **540 runs**
- **Main issues** (baseline replay)
  - WS10 has 71% SSM violation and a minimum distance of −0.17 m. The robot drives screws into the same door while the worker is inserting the glass, so they overlap
  - WS30 is 100% if the worker stays and 0% if the worker leaves
- **With P3 (sequential)**: if the unobserved blocks are also put in series, WS10 reaches 249 s and WS30 260 s, both above the cap. So **which tasks must be sequential** decides the outcome.
- **Pass criterion**: under the best policy, station cycle p95 ≤ 228 s, 0 collisions, tracking TCP error ≤ 1 mm.

**990 runs in total** (A 270 + B 180 + C 540). Results for the same cell (WS20 in B and C) can come from a single shared run.

---

## 4. Common rules

- **Randomization**: same as v2. Conveyor speed ±5%, door position ±10 mm and ±1°, lognormal worker times, joint acceleration 1.0–2.0 rad/s², screwdriving 2.0–3.5 s.
- **Common random numbers**: use **the same list of random seeds** for all three options (draw k uses the same random numbers in every option). Then differences between options come from the design, not from luck, and paired comparison shrinks the confidence intervals a lot. The IE DES works the same way, which is why the A−B throughput difference comes out at ±0.02 units/h.
- **SSM**: ISO/TS 15066 protective separation distance. Parameters are under `ssm` in `scenario_matrix.yaml`, same as v2.
- **KPI** (per run, per station): `station_cycle_s`, `robot_cycle_s`, `robot_wait_s`, `ssm_slowdown_s`, `min_robot_worker_distance_m`, `ik_failures`, `collisions`, `tracking_tcp_error_mm`, `takt_overrun_share`

---

## 5. Return format

| File | Contents |
|---|---|
| `results_<A·B·C>.csv` | One row = one run<br>`option, station, policy, worker_after, draw, seed, walk_factor, station_cycle_s, robot_cycle_s, robot_wait_s, ssm_slowdown_s, min_dist_m, collisions, ik_failures, tcp_err_mm` |
| `summary.json` | Per (option, station, policy): p50 and p95 of `station_cycle_s`, share above 228 s, and the best policy within that option |
| `robot_task_times.json` | Same contract as v2 (`contracts/robot_task_times.schema.json`). Send it only if the robot cell cycle changed |

> Receiving side: the IE team feeds the station cycle distributions into the line DES (s6b) and the design-space evaluation (s5x) and compares the three options again. The code that reads these results will be added once the result format is fixed.

---

## 6. What changes depending on the results

| Simulation result | Effect on the customer proposal |
|---|---|
| Option A p95 > 228 s at walk factor 0.7 | Mark option A as "sensitive to the AMR effect" and cut the pilot scope to one station |
| Option B WS20 cell passes, but the worker-only WS30 overruns | Look at a variant of option B that adds a WS30 improvement (layout or jig). It is already in the design space |
| Option C WS10 is safe only under P2 or P3, and then exceeds 228 s | The advantage of option C (5% takt overrun) shrinks and the gap to option B narrows. In that case, redesign the WS10 task sequence |
| Option C WS30 worker can leave | WS30 is fine (SSM 0%). Propose it to the customer as a worker movement rule |

---

## 7. Limitations (same as v2)
- **Door geometry**: there is no CAD, so door geometry and screw positions are approximated from where the worker's wrist dwelt (±5 cm). A real cell needs CAD.
- **Unobserved blocks**: the data does not show where the worker is during this time. So the bounds come from two cases, stays and leaves.
- **One robot per station**: two robots working together on one process are not part of this request.
