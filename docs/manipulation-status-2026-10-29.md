# Manipulation implementation review — 2026-10-29

## Decision and project scope

Publish the current work on **`sim/manipulation-review-2026-10-29`**, not `main`.
The fetched `main` and `origin/main` both pointed to `9c55e56` at review time;
there were no remote changes to reconcile. The entire `sim/sam-g1-manipulation/`
subsystem was untracked, alongside an edited `sim/README.md`. This is a first
publication of that subsystem, not an incremental change already on main.

The runtime is isolated and does not modify the existing network protocol or
household demo. Nevertheless, merging it directly into main would introduce an
experimental controller with failing physical acceptance tests, a separate
external dependency setup, and previously overstated documentation. A branch
preserves the existing baseline while making all project source reviewable.

The original project is **trust-mediated cooperation between robots owned by
different households**: verify identity, apply owner policy/trust, execute only
approved actions, and report evidence to the owner. See [architecture](architecture.md)
and [the shared message contract](message-format.md). The manipulation lab is
an execution prototype for C1 and part of C6/C7; it does not replace the C3–C5
identity, gatekeeper and trust contribution.

## Where the previous work stopped

The prior Codex implementation had camera-grounded manipulation, episodic
location memory, height-aware routes and arm approaches, bounded fresh-camera
replanning, and a larger regression suite (196 tests at that checkpoint).
Local historical acceptance artifacts showed:

- A red-on-blue then return-only-red sequence passed on that older revision:
  return error 3.949 mm, lower blue object displacement 0.190 mm during return.
- A ball transport/placement trial passed after one fresh-map route replan.
- The latest yellow-cylinder-to-orange-vicinity trial still failed: the
  cylinder was lifted about 17.3 cm but remained held about 59.5 cm from its goal.

These are historical local observations, **not results for the current source**.
The previous source and remaining files were archived locally as
`runs/backups/codex-build-20261024.tar.gz` and
`runs/backups/codex-leftovers-20261025.tar.gz`. Those ignored archives and the raw
acceptance traces are not distributed here. They explain the comparison, but
should not be used as reproducible published benchmarks.

The current workspace contains a substantial rewrite consistent with the
subsequent Claude Code work. Because these files were untracked, Git cannot
establish per-change authorship or an exact commit-by-commit timeline.

## What is implemented now

| Layer | Implementation | Current boundary |
| --- | --- | --- |
| Native interface | `desktop_window.py`, native launchers, `view.py agent` combine rendered simulation, input, Activity and Full debug trace | Local Mac interface; not an owner trust dashboard |
| Model loop | `agent.py`, `google_brain.py`: Gemini reads head/overview images, chooses one tool, gets results and fresh views | VLM tool use; no trained VLA that emits joint-action chunks |
| Perception | `perception.py`: RGB-D point clouds, object dimensions, support surfaces, obstacle geometry, frame identifiers and reacquisition | Geometry can be wrong despite a correct image box |
| Planning | `planner.py`: jointly ranks stance/posture, checks reach, measured body envelope, arm sweeps and walking paths | Candidate feasibility does not guarantee the physical robot reaches the same pose |
| Body control | `body.py`: turning, forward walking, docking, posture and ordered arm/body movement | No demonstrated general mobile manipulation or human-equivalent movement |
| Manipulation | `skills.py`: generic pickup, placement relations, movement, release, gripper and posture tools with contact feedback | Physical trials still fail; generic tools are not object-specific scripts, but are still programmed control algorithms |
| Low-level control | GR00T-WBC locomotion through BiGym2; Mink arm IK, Ruckig smoothing and gravity compensation | Inherits the policy's balance/reach limits |
| Joint feasibility | `controller.py`: IK range intersects joint and actuator limits, including the right shoulder-roll mismatch | Regression covered; not a fix for every failed reach |
| Memory | `scene_memory.py`: camera-derived pre-move observations and explicit memory IDs for former locations | Identity ambiguity and stacked-object isolation need end-to-end coverage |
| Runtime | `agent_server.py`: physics queue, interruption epochs, cameras, monitoring and traces | Local HTTP origin checks are not robot identity verification |
| Calibration | Posture/gait builders and JSON tables | Must regenerate and validate against changed assets/controller |
| Trials | `skill_trials.py`, `agent_trials.py`, 38 pre-review tests | Historical grounded validation suites were removed from active code |

The rewrite replaces the large `grounded_motion.py`, earlier walking modules,
`visual_actions.py` and earlier grounded/return validation runners. The current
smaller suite must regain the acceptance cases those runners exercised; test
count alone is not a quality measure.

## Validation performed in this review

Platform: the existing macOS Apple Silicon environment, Python 3.12, MuJoCo
CGL renderer; BiGym2 checkout `d71501359b215b9d78e27dcb70b220412d474dfa`, clean.
No Google API call was made for this audit.

| Check | Result | What it establishes |
| --- | --- | --- |
| Manipulation pytest before review fix | 38 passed | Existing unit/controller suite passes |
| Manipulation pytest after review fix | 41 passed | Adds success, failure and missing-status propagation checks |
| Ruff | Passed after import formatting correction | Static checks only |
| `network/test_protocol.py` | `all passed` | Standalone signing, tamper, replay, identity and expiry assertions |
| `network/test_policy.py` | `all passed` | Existing standalone policy assertions |
| Physical skill trials, sorting scene, seed 7 | **1/5 passed** | Current controller fails several basic tasks |
| Fresh English/Gemini acceptance | Not run | No current model task-success or cost claim |
| Clean-machine setup; live MQTT household demo | Not run | Existing environment tests do not prove these |

The network checks are standalone assertion scripts; collecting them with
pytest produces “no tests ran,” so they were also executed directly with their
declared dependencies. No broker was required.

### Physical results

| Task | Outcome | Observed failure or evidence |
| --- | --- | --- |
| Blue on red | Fail | XY error 0.1179 m; blue bottom 0.050 m below the red top, at table level |
| Cylinder on orange | Fail at pickup | Measured object as 30 × 9 × 10 cm; narrowest width 9.2 cm rejected against 8.3 cm opening |
| Cylinder next to green | Fail at pickup | Same geometry/gripper rejection |
| Release/drop red | Pass | Object fell 0.91 m; this is release, not controlled floor placement |
| Red back | Fail | First placement could not reach; subsequent pickup/return encountered an arm sweep rejection; final error 0.369 m |

No robot fall was reported in these five trials. MuJoCo emitted
`ARB_clip_control unavailable while mjDEPTH_ZEROFAR requested`; depth accuracy
needs investigation on this renderer. It is not established as the cause of
these failures.

The harness derives image bounding boxes from simulator segmentation, then
passes RGB-D observations to the skills. It bypasses model object localization
and English planning. Its truth measurements are useful diagnostic evidence,
**not a complete autonomy benchmark**. Current pass predicates are also too
weak: proximity/height do not prove stable support, release, correct contact,
or preservation of a neighbouring object. `red_back` is not the requested
stack-red-on-blue then unstack-only-red regression.

[Machine-readable evidence and source fingerprints](validation/manipulation-review-2026-10-29.json)
record the pre-fix physical run. The subsequent change affects result reporting,
not physical motion; the five-task physical suite was not rerun after it.

## Changes made during this review

1. Fixed `execute(move_object)` reporting `ok: true` unconditionally after
   `place` returned a failed verification. The wrapper now requires explicit
   placement success and forwards failure stage/suggestion. Added three regression
   cases, including a missing status that must fail closed.
2. Corrected one Ruff import-formatting violation.
3. Replaced the README's `RESULTS_TABLE` placeholder and unverified reliability
   claims with measured results and limitations; documented first-time setup.
4. Excluded MuJoCo runtime logs and environment variants from source publication.
   Preserved source, dependency lock, calibration tables and third-party notice.

This review does **not** claim to repair the physical failures.

## Debugging and token use

The app exposes Activity and Full debug trace. Local evidence under
`sim/sam-g1-manipulation/runs/agent/` includes `events.jsonl`; each `actions-*` directory
records `implementation.json`, camera images, `decisions.jsonl`,
`api-usage.jsonl` and action traces. Decisions contain the public explanation,
chosen tool/arguments, latency, model version and usage metadata. These are
observable decisions and measurements, not access to hidden model reasoning.

The loop already keeps only recent image pairs (default at most four image
instances in model history), returns compact tool results, and runs motion
locally between model decisions. It does not simply send a new image on a fixed
10-second timer. Text history can still grow within the step budget. The
configured model ID is an implementation setting, not a model-availability or
pricing claim; this review did not validate its API availability.

Next measurements should report tokens, image instances, API latency, physical
execution time, retries and verified task completion together. Saving tokens
by accepting an unverified “done” is not an improvement. Do not add another
model or a VLA until the current failures are isolated and a compatible policy
has a measured benefit on this robot and Mac setup.

## Gaps and ordered next steps

### 1. Make task success an independent, reliable verdict

Add machine-checked end-to-end predicates separate from the model's `finish`
message. A completed place must require release, settled support, correct
spatial relation, no fall and no unintended object movement. Make timeouts and
failed intermediate stages fail the mission; stop dependent steps when a prior
place fails instead of attempting another pickup while still holding it.

Restore the exact user cases in one environment, with varied layouts and
paraphrases: blue beside red; yellow cylinder beside orange; object onto floor;
red on blue followed by **only red** back to its observed original location.
Keep oracle-box skill tests separate from fresh-English tests. Record source,
assets, calibration hashes, model version, seed and thresholds for every run.
Explicitly report unsupported floor pickup/placement rather than substituting
a drop or a tabletop move. Add same-colour objects, occlusion, interruption,
stale frames and remembered-location ambiguity.

**Exit criterion:** every reported success has an independent physical verdict;
unsupported/failed requests never become success, and an agreed seeded suite
passes before broader claims are made.

### 2. Repair perception and physical execution using the failing evidence

- Inspect the cylinder crop, RGB-D components, support subtraction and frame
  transforms: the observed 30 cm extent suggests contamination or erroneous
  reconstruction, but the precise cause is not established. Do not widen the
  gripper threshold to hide it.
- Trace reference identity and target refresh during stacking. The blue trial's
  final commanded target was approximately `(0.6081, -0.2190)`, while the red
  reference started near `(0.4906, -0.2200)`. Determine why the target shifted
  before changing arm tuning.
- Reconcile planned stance/posture with measured settled feet, shoulder and
  hand pose under load. Fix collision/reach planning and carry transitions;
  preserve margins rather than relaxing checks until tests pass.
- Reintroduce isolated top-object grasps and lower-object displacement checks.
  Verify controlled descent, contact transfer, release and stable placement.

**Exit criterion:** repeatable perception and physical success on randomized
positions, with failure traces that identify the failing layer and no oracle
object coordinates available to the deployed agent.

### 3. Connect the execution prototype to the original trust project

Implement a typed adapter from an **approved action** to executor result through
the existing `network/` gatekeeper. Preserve the shared message contract; agree
any protocol extension with the network team. Correlate request identity,
allow/partial/deny decision, execution epoch, motion evidence and final result.
Unsupported actions must return a capability failure. Do not let remote free
text bypass owner policy or let a model override a denial.

Test valid, forbidden, forged, replayed and expired requests, plus disconnect,
timeout and cancellation while holding an object. Rejected requests must cause
no motion. Define safe interruption/holding behavior explicitly; local “drop”
convenience must not become an unauthorized remote capability.

**Exit criterion:** two isolated households demonstrate an approved physical
action and independently verifiable refusal/no-motion cases in the same trace.
Measure trust false-allow/false-deny rates and decision overhead separately from
robot execution success.

### 4. Make the result reproducible and mergeable

Validate `bash sim/sam-g1-manipulation/setup.sh` on a clean checkout; keep the pinned
BiGym dependency and lockfile reproducible. Add calibration compatibility
checks and CI unit/static checks plus selected physical trials where graphics
are supported. Keep dependency clones, credentials, images, raw prompts,
usage logs and local archives out of Git. Publish curated benchmark summaries.

Merge to main only after the physical acceptance gaps and runtime integration
boundary are reviewed. General humanoid autonomy, running/jumping, dexterous
hands, ROS MCP and VDA5050 are outside this change and are not prerequisites for
the trust-network demonstration.

## Reproduce this audit

From `sim/sam-g1-manipulation/` after setup:

```bash
MUJOCO_GL=cgl uv run --no-sync python -m pytest --capture=sys -q
uv run --no-sync ruff check .
MUJOCO_GL=cgl uv run --no-sync python skill_trials.py --seed 7
```

From `network/`, install `requirements.txt` in an isolated Python environment,
then run `python test_protocol.py` and `python test_policy.py`. The native lab
launch is `uv run --no-sync python view.py agent` from `sim/sam-g1-manipulation/` with
a Google API key supplied through the environment.
