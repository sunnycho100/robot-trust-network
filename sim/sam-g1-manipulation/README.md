# Robot Lab: experimental camera-guided G1 manipulation

A Unitree G1 (29 joints, two Dex1 two-finger grippers) in MuJoCo, with
GR00T-WBC locomotion and Gemini selecting generic physical actions from camera
observations. The native window combines simulation, commands and debug traces.

**This is an experimental execution prototype.** The October 29 audit passed
only 1 of 5 physical trials (release/drop); stacking, cylinder pickup and return
failed. It is not yet connected to the signed household trust flow. See the
[implementation review and next steps](../../docs/manipulation-status-2026-10-29.md)
for the scope, evidence and remaining gaps. This is a VLM plus controllers,
not a trained VLA policy.

## Run

First install `uv`, then run `bash setup.sh` from this directory. Setup checks
out the pinned BiGym2 dependency into a sibling `bigym2-lab` directory and installs
the locked Python 3.12 environment. It refuses to overwrite a different dependency
revision. A clean-machine install has not been validated in this audit.

With `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your environment:

```bash
uv run --no-sync python view.py agent          # the Robot Lab window
zsh -ic 'uv run --no-sync python view.py agent' # if the key lives in ~/.zshrc
```

One window: the simulation on the left (drag to orbit, right-drag to pan,
scroll to zoom), the command box, **Activity** (what the robot decided, did
and measured, in plain language) and **Full debug trace** (the same, with
numbers) on the right. You can send a new instruction at any time: it replaces
the current one, so "drop it" or "open your hand" acts right away.

Headless checks (no window, no API key):

```bash
# Physical skill trials on the sorting scene; judged from simulator truth
MUJOCO_GL=cgl uv run --no-sync python skill_trials.py --seed 7
MUJOCO_GL=cgl uv run --no-sync python skill_trials.py cylinder_on_orange --seed 8 --record front  # + GIF

# The full English pipeline (real Gemini calls), still headless
MUJOCO_GL=cgl uv run --no-sync python agent_trials.py "put the blue box on the red box" "drop it"

MUJOCO_GL=cgl uv run --no-sync python -m pytest -q   # unit and controller tests
```

`build_posture_table.py` and `build_gait_envelope.py` re-measure the robot's
body (below); run them only after changing the robot or controller.

## How it works

```
agent.py        Gemini: look at the head/overview images, pick ONE tool, read the result, repeat
skills.py       pick_up, place, move_object, release, grip, go_to, turn, walk, set_posture, move_hand
planner.py      where to stand, how to bend, which way to walk, will the arm hit anything
perception.py   object size/position/support from RGB-D, chosen by the model's 2D box
body.py         walking, turning, posture, joint-space arm swings, Cartesian hand servo
controller.py   GR00T-WBC legs + waist, Mink arm IK, Ruckig smoothing, gravity feed-forward
agent_server.py physics thread, cameras, safety monitor, message log; desktop_window.py UI
```

For object actions, the model selects a 2D box in an image; the skills measure
geometry from depth. Lower-level tools also accept bounded relative motions. Every
skill result is a sentence with what was measured and, on failure, why and
what to try instead.

### Intended pickup execution

1. **Measure** the object in the chosen frame (size, top, what it rests on).
2. **Choose a stance and posture** together. Candidates around the object are
   ranked by real walking distance (around furniture), reach comfort and room
   around the body. Each is checked with the measured posture table: the
   body's surface must keep clear of everything seen; the arm must reach every
   hand pose with **3 cm of reach to spare**, because the real body settles a
   little differently from the table; and the arm at its approach pose must
   keep 5 cm from other objects, to reduce the risk of reaching across a neighbouring
   object. These checks are not a collision-free execution guarantee. Planning work is fixed (not timed), so a scene always gives the
   same plan.
3. **Navigate to the stance**: turn toward the next waypoint and walk
   forward, slow down near furniture, come to rest, and step the last 25 cm
   precisely. In tight gaps the robot turns on the spot first and side-steps
   in. If something comes within 5 cm of the stepping legs it steers away; at
   1.5 cm it stops, steps clear and plans another way.
4. **Confirm the reach from where it actually stands**; if the arm can't reach
   from there, it picks another place to stand.
5. **Reach and bend in one motion**, in the order people use: reaching down,
   the arm goes out first and the body bends after; standing up, the body
   rises first. Large arm moves go straight to the joint angles the planner
   solved (as MoveIt does for transfer moves); only the last few centimetres
   use the Cartesian hand servo. The whole sweep of the arm (its real
   collision shapes) is checked against what the cameras see, and the free
   arm is parked beside the body when bending would carry it into a table.
6. **Grasp** from above, check both finger pads feel force, lift, and bring
   the object to the chest while standing up.

### Intended placement execution

The same stance, walk and reach steps bring the object 8 cm over the spot
(measured again up close; "on top" aims at the centre of the top face). Then
the hand lowers slowly until the fingers stop carrying the object's weight
(or the hand stops moving down), seats it, loosens, opens, lifts clear and
steps back. Finally the robot looks (head camera, then overview) for the
object by its colour: it reports success only if it sees it at the spot,
upright; otherwise it says what it saw ("it may have tipped over").

## Calibration notes from the implementation

| Fact | Value | Consequence |
| --- | --- | --- |
| Forward walking deadband | < 0.07 m/s does nothing | docking steps use ≥ 0.09 m/s |
| Stopping distance after a stop command | ~0.6–0.8 s × speed | slow final approach, then stepped docking |
| Turning on the spot from rest | drifts 3–11 cm, stops 5–11° short | come to rest first; second, slower correction pass |
| Stepping hip/thigh vs standing | reaches 6 cm further forward | walking checks use the measured gait envelope |
| Arm reach, shoulder to pinch point | 0.56 m fully stretched | stances are planned within 0.53 m |
| Arm tracking during a fast swing | lags its command by up to 6 cm | reach-then-bend ordering, 3.5 cm sweep margin |
| Right shoulder-roll actuator range | asset copies the left arm's (−1.59…2.25 vs joint −2.25…1.59) | IK plans only angles the motor can command |

The actuator/joint range mismatch is addressed by planning within their
intersection. The other calibration values describe the current controller;
they are not task-level accuracy guarantees.

## Results

October 29 review, sorting scene, seed 7, current rewrite before the reporting-only
fix. The harness selects image boxes using simulator segmentation; this tests
physical skills, **not Gemini's English understanding or visual localization**.

| Trial | Result | Evidence |
| --- | --- | --- |
| Blue on red | Fail | 11.79 cm XY error; blue remained at table height |
| Cylinder on orange | Fail | Measured width 9.2 cm exceeded 8.3 cm gripper opening |
| Cylinder next to green | Fail | Same pickup rejection |
| Release/drop | Pass | Red object fell 91 cm; not controlled floor placement |
| Red back | Fail | Initial placement failed; final return error 36.9 cm |

41 unit/controller tests and Ruff pass after the review fix. No fresh Gemini
end-to-end success rate is claimed. See [machine-readable evidence](../../docs/validation/manipulation-review-2026-10-29.json).

## Known limits

- Placement is unreliable; the trial above missed a stacking target by 11.79 cm.
  Calibration tolerances must not be presented as achieved task precision.
- Very low surfaces (the knee-high rear platform) need a deep crouch and a
  full lean at the edge of the arm's reach; those placements are the least
  reliable.
- Two-handed carrying, picking from the floor, and objects wider than the
  open gripper (8.3 cm) are not supported; the robot says so instead of trying.
- The model's reasoning takes a few seconds per step; the motions themselves
  run on the robot's own controllers.

## History

The previous build (a 5,000-line `grounded_motion.py`, walking-height
profiles, validation suites and their tests) has local-only archives (ignored and not distributed in this repository) in
`runs/backups/codex-build-20261024.tar.gz` and
`runs/backups/codex-leftovers-20261025.tar.gz`.
