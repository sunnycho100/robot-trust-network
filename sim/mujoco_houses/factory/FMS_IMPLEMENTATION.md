# Flexible manufacturing upgrade — implementation record

This is an incremental upgrade of the current project. The existing showcase,
G1 logistics, factory executor and original cooperative lift commands remain.
The complete requested production system is **not yet commissioned**.

## Current scene-composition work

The latest user direction prioritizes one attractive, densely populated scene
and explicitly defers video montage. `plant_view` now previews the larger
configured plant, preserving the existing carriers and adding four fixed cells
(20 robots total, including two TurtleBots, two Jackals, two Huskys and two Spots). See
[Dense layout](DENSE_LAYOUT.md). The cleaning cell and single humanoid have
exchanged sides so robot types are mixed spatially. Added arm cells now run
finite measured work orders through the existing IK/servo controller and assisted
grasp. QC starts from a physical TurtleBot arrival. The stationary conveyor and
local work orders do not claim full production scheduling or dynamic dispatch.

Spots use an original contact-force crawl controller, with a floating base,
three stance feet and one swing foot. No downloaded locomotion policy or base
forces are used. `spot_walk_park.json` validates a 90-second walk/park probe:
32 steps, 0.828 m net progress, 0.078 m maximum foot lift, no body-ground
contact and four-foot support after parking. Shared-world validation is reported
separately in `factory_spots_walking.json`.

The current brisk gait is commissioned in `factory_fast_final.json`: both
Spots travel approximately 1.52 m. The two-G1 crate route is approximately
0.89 m and the three-G1 fixture route 2.03 m; single-G1 side carrying retains
the proven short route after longer candidates failed grasp/placement tests.
The existing packing arm now teams with an additional UR10e after completing
its box jobs. `dual_arm_verified.json` validates a shared physical panel with
synchronized soft assisted grasps, lift, transfer, contact-gated release and
retraction. This is not contact-only finger grasping.

Two long-route TurtleBots are replaced by existing local Jackal models to
increase actual travel speed within each model's actuator limits. Native
tyre/floor contacts and skid-steer yaw calibration are restored when attaching
Jackal/Husky. `factory_fast_commissioned.json` records the final combined run.
The 75-second run passes all three G1 transport/placement checks, four fixed-cell
orders, both Spot walking checks and the shared-panel task. All six mobile
deliveries arrive; robot/obstacle contact records and MuJoCo warnings are empty.
Assembly replenishment finishes at 48.67 s, rework transport at 33.99 s and the
shipping Husky at 34.68 s. The original executor's 100-second regression also
reaches SHIPPED with five handoffs (`factory_legacy_regression.json`).

Subsequent measured placement controllers resolved the old long-carry probes
below for the standalone transport/place/release/return lifecycle. The earlier
failed continuous-carry tests remain historical evidence, not current placement
results. Source pickup is still initialized with held loads.

## Physical gates

- Phase 1: eight-section layout added through an optional scene hook, retaining
  the original world and controllers. `outputs/fms_phase01_regression.json`:
  three carries pass, both couriers arrive, Panda picks and places the same
  delivered part, `completion_pass=true`.
- A standalone command regression introduced by extracting `build_group` was
  found and fixed (`solo` scope and per-actor prefix).
- 30-second probes were run before enabling continuous operation:
  `fms_flex_30s_probe.json`, `fms_team_30s_probe.json`,
  `fms_sheet_30s_probe.json`. All fail extended carry validation. Consequently
  no continuous-production or completed pickup/place/team-return claim is made.
- Original eight-second benchmarks are preserved. A resource reservation is
  not equivalent to a measured pickup, placement, or completed mission.

## Production foundation

Separate modules now define configuration, jobs, buffers, stations, events,
capability-based resource selection, priority dispatch and production logic.
They do not write qpos or move physical parts.

Tested contracts cover buffer capacity/reservation, single material custody,
priority/compatibility, atomic team reservation and release, blocked CNC output,
QC/rework routing and mandatory component handling during assembly.
`ManualBackend` exists only in tests; it is not a MuJoCo production backend.

Implemented logic must still be connected to measured motion before it is
presented as a running manufacturing feature. In particular:

| Requested phase | Current gate |
|---|---|
| 1 Layout | Headless physical regression passed; user-rendered review pending |
| 2 Robot roles | Capability registry implemented; production missions not yet bound |
| 3 Three carries | Reference behaviors preserved; full pickup/place lifecycle pending |
| 4 Buffers | Transactional core tested; physical multi-slot inventory not yet bound |
| 5 Station FSMs | Scheduling/state foundation tested; CNC/deburr/assembly backends pending |
| 6 Conveyor | Existing motorized conveyor retained in factory executor; new layout binding pending |
| 7 Product routes | A/B/C/D configuration implemented |
| 8 Concurrent production | Independent logical scheduling supported; physical multi-product run pending |
| 9 Blocked/starved | Core tested; hardware recovery not yet commissioned |
| 10 Priority dispatch | Core tested, non-preemptive, compatible resources only |
| 11 Flex reassignment | Requests/capabilities exist; extended G1 mission controller pending |
| 12 QC/rework | Deterministic logic tested; physical return loop pending |
| 13 Temporary teams | Atomic reservation/release tested; physical rendezvous/pick/place pending |
| 14 HUD/visual refinement | Section labels and meaningful floor marks added; live production HUD pending |

## Review commands

```bash
# Eight-second layout regression in the existing environment:
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/python -m unitree_demo.flexible_factory \
  --duration 8 --require-complete --report outputs/fms_phase01_regression.json

# User-run graphical review; fixed overview, no automatic camera cuts:
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython -m unitree_demo.flexible_factory \
  --duration 8 --shot overview --require-complete \
  --video outputs/fms_layout_review.mp4 --report outputs/fms_layout_review.json

PYTHONPATH=src ../coohoi_mujoco/.venv/bin/python -m pytest \
  tests/test_flexible_factory.py tests/test_factory_scene_labels.py -q
```

The full legacy test suite currently cannot collect in this environment because
`cryptography` is not installed; this affects MQTT/trust/house tests, not the
new isolated factory tests. It is already listed in the project requirements.

Next physical work: measured, supported G1 placement and safe idle/return;
station manipulators and shared inventory bindings; then conveyor/QC/packaging
and longer concurrent runs. Do not expose a simulated success counter until
these backends acknowledge actual physical completion.
