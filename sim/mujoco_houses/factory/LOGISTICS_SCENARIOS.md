# Mixed-load fulfillment concept

One order, several object geometries, different robot teams. Avoid repeating the
same conveyor cell as the main story. Keep material provenance and event-gated
handoffs from the metal workflow.

## Clearance and side-grip revision

Current recording target: `sofa_side --mass 2`. The seat is fixed at 0.28 x
0.43 x 0.18 m. The back meets the seat top without overlapping volumes.
Initial arm opening is fitted to actual hand collision surfaces using signed
geom distance; the seat is not resized to the hands. Wrist rotation presents
opposed hand surfaces, and motor-limited arm gravity feedforward reduces sag.
No equality constraints or runtime object teleporting are used.

Eight-second run: 1.17 m travel, 98.3% hand-supported steps, 19.62 N mean
vertical hand support, all hand contacts classified left/A or right/B, no
bottom contacts, no torso/leg contacts, minimum object-floor clearance 0.425 m,
maximum contact penetration 0.255 mm. Report: `outputs/g1_sofa_side_verified.json`.

The updated T-frame crossbar is 1.12 x 0.56 m; the stem meets its edge without
overlapping. The initial load clears the visible hands. All robot body collision
geometry participates. Three-carrier run: 1.55 m travel, 3.80 mm maximum contact
penetration, minimum floor clearance 0.963 m. Two carriers still fail with these
positions/controllers. Earlier measurements below describe the older geometry.

```bash
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython -m unitree_demo.g1_logistics \
  --scenario sofa_side --mass 2 --duration 8 \
  --video outputs/g1_sofa_side.mp4 --report outputs/g1_sofa_side.json

PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython -m unitree_demo.g1_logistics \
  --scenario team_fixture --mass 3 --carriers 3 --duration 8 \
  --video outputs/g1_three_person_frame_v2.mp4 --report outputs/g1_three_person_frame_v2.json
```

| Scene | Team | Purpose | Status |
|---|---|---|---|
| Upholstered module | 1 G1, opposed side contact | Compact furniture module transport | Updated `sofa_side` passes at 2 kg / 8 s; fixed width 0.43 m |
| Equipment crate | 2 G1, palms under the base | Shared gravity load, then AMR handoff | 5 kg held-start carry tested for 8 s; handoff/pickup not implemented |
| Eccentric T-frame | 3 G1 | Support a load whose center of mass is beyond the two-carrier support region | 3 kg / 8 s passes with 3 carriers, fails with 2; cornering not implemented |
| Finished order | AMRs + pallet arm | Consolidation, shipping staging | Existing metal workflow is the validated handoff baseline |
| Blocked shipping lane | Dispatch coordinator | Reassign carrier / reroute around an occupied aisle | Proposed, not implemented |

`unitree_demo.g1_logistics` keeps the original G1 carry unchanged and provides
isolated contact-only prototypes. It uses no weld/connect grasp constraints.
Objects start in hand-contact poses: this is not a floor-to-carry pickup demo.
The sofa is a primitive modular furniture proxy, not a full-scale sofa model.

Measured 5 kg bottom-support run: about 0.69 m horizontal displacement, average
hand vertical contact support 49.09 N, support > half weight for 98.8% of steps,
all recorded hand/object contacts on the bottom face, both pelvis heights > .75 m
at the end. Payload minimum center height was .968 m; it sagged from 1.236 m.
The 8 kg test failed. Do not advertise an 8 kg load rating or stable heavy
handling based on this prototype.

Additional measured prototypes:

- Sofa module: 0.40 x 0.76 m seat, foam/furniture proxy, 2 kg; 2.19 m travel,
  support > half weight for 99.1% of steps, mean vertical hand force 19.61 N.
  The robot holds it low with bent elbows and upward palms. This is not a
  full-size sofa or a side-friction grip. The 5 kg test failed.
- T-frame: 25% of mass in crossbar, 75% in the long stem; 3 kg, 3 carriers;
  1.09 m travel, mean vertical hand force 29.42 N, support > half weight for
  98.8% of steps. Same initial object and two-carrier poses with the rear
  carrier omitted: the object drops. This is a controlled comparison of
  these poses/controllers, not proof that no two-robot arrangement could
  ever carry the object. The 6 kg / 3-carrier test failed.

```bash
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython -m unitree_demo.g1_logistics \
  --scenario sofa_underneath --mass 2 --duration 8 \
  --video outputs/g1_solo_sofa.mp4 --report outputs/g1_solo_sofa.json

PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython -m unitree_demo.g1_logistics \
  --scenario team_fixture --mass 3 --carriers 3 --duration 8 \
  --video outputs/g1_three_person_frame.mp4 --report outputs/g1_three_person_frame.json
```

```bash
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython -m unitree_demo.g1_logistics \
  --scenario bottom_crate --mass 5 --duration 8 \
  --video outputs/g1_bottom_5kg.mp4 --report outputs/g1_bottom_5kg.json
```

The recording uses the shared tqdm frame progress display. Preserve upstream
G1 licensing when distributing model assets. No new external assets are used.
