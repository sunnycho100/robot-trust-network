# Populated factory scene

This scene extends the existing Depot world and existing G1 controllers. It is
a scene-composition preview, not a commissioned production scheduler.

Run from `sim/mujoco_houses` with the existing sibling virtual environment:

```sh
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython \
  -m unitree_demo.flexible_factory.plant_view \
  --duration 30 --view overview \
  --video outputs/factory_dense_overview.mp4
```

The video writer reports progress with tqdm. This records one continuous view;
it does not join clips. `--view machining`, `receiving`, `assembly`, or `shipping`
selects a closer static camera. `--labels` enables optional 3D section captions.
Without `--video`, the command performs physics checks without a renderer.

The 24 × 14 m operating footprint contains stocked rack bays, receiving pallets,
sheet bundles, a CNC enclosure, cleaning bench, assembly tooling, rework bench,
QC stand, roller conveyor, packing station, shipping stock, tool cabinets, WIP
carts, bollards and charging docks. The main 2.2 m traffic aisle remains clear.
Cartons include tape, seams, label panels and barcode graphics; pallets have
individual deck boards and blocks. These props are original MJCF geometry,
not downloaded textured marketplace models. The conveyor is stationary in this
scene preview.

Fleet: six existing G1s, two TurtleBots, two Jackals, two Huskys, the existing QC Panda, two
UR5e cells, one assembly Panda, two UR10e arms (packing plus a shared-panel
partner) and two Spot quadrupeds (20 total). Spots use a three-support-foot
crawl with measured body balance,
stance contact-force allocation and swing-foot control. Only joint actuators
drive them; their bases remain free, with no base welds, external base forces
or runtime qpos replay. The brisk gait uses a larger stride and an inset support
triangle rather than moving to its centroid on every step. Both straight
inspection-area routes are now 1.5 m, then the robots park. This does not implement
visual inspection or arbitrary-path navigation. No external policy download
is needed. Positions, headings and `walk_distance` are configurable under
`quadrupeds` in `realistic_layout.json`; zero distance keeps the standing mode.
Use `--view quadrupeds` for a closer view of the shared-floor Spot. Independent
walking checks use `python -m unitree_demo.flexible_factory.spot_walk`.
`--pace crawl` on that probe retains the previous slow gait for comparison.
UR arms include the existing roqsim Robotiq model. Fixed cells now execute finite
physical work orders: CNC fixture loading, machining dwell and unloading;
cleaning fixture loading and output; two-component stacking in assembly; and
two-place packing. Processing is simplified. These are measured pick/place
transfers with proximity-gated assisted grasps, not contact-only gripper grasping
or a complete production scheduler. Parts exist from initialization, with no
respawning or runtime qpos replay. One TurtleBot presents a part to QC, triggering
the existing Panda handoff; the other mobile resources enter CNC supply,
assembly replenishment, QC-to-rework, bulk stock and shipping-to-packaging
routes. Routes are finite and distinct, not identical permanent loops.
Measured local yielding inhibits forward motion toward nearby robots; this is
commissioning waypoint navigation, not a global traffic planner. Extra tote
loads are rigid-mounted cargo; downstream unloading is not yet implemented.
G1s perform the preserved single, pair and
three-robot transports, measured placement, release and return. Loads are held
at initialization; this does not demonstrate autonomous source pickup.

## Faster motion and cooperative manipulation

TurtleBot is capped at 0.25 m/s within its native wheel-control range. Jackal
handles the longer replenishment/rework routes at up to 0.60 m/s; Husky is
0.55 m/s. The source model's explicit tyre/floor friction and manifest yaw-slip
calibrations (Jackal 1.7, Husky 3.0) are preserved. Wheel targets are scaled
together at actuator limits, preserving curvature. Robots align before
translating into tight turns. These are physical wheel-controller commands,
not video speed changes. Machining/cleaning dwell times are shortened
demo timers; arm commands use a larger joint-target increment where stable.

The three-G1 fixture route now travels approximately 2 m, and the two-G1 crate
approximately 1 m. The pair tracks a common measured body formation to reduce
lateral drift. The single G1 retains its proven short side-grip/placement route;
longer candidates slipped or failed placement and are not the default.

After the packing arm finishes its two box placements, it forms a temporary team
with a second UR10e. Both approach opposite marked points on one 1.5 kg panel,
attach proximity-gated soft assisted grasps, lift/translate synchronously,
lower until table contact is measured, release together and retract. The shared
panel is never teleported. Reports include both grasp-constraint force magnitudes
and inter-arm penetration. These are constraint forces, not certified payload
or balanced-load ratings. `--view cooperative_arms` gives a close static view.
Standalone commissioning uses `python -m unitree_demo.flexible_factory.dual_arm`.
The existing packing arm is reused; this does not add another independent loop.

The cleaning cell is now south beside sheet handling, with its original robot,
bench, tooling, cart and actual parts translated together. The single G1 works
north beside machining/QC. Mobile routes cross the main aisle and enter workcell
access spurs, breaking the previous arm/humanoid spatial division. For the full
set of commissioning routes, use `--duration 105`; 30–45 seconds shows concurrent
activity without waiting for every delivery to finish.

Assets remain referenced from the user's existing `../roqsim_factory_upstream`
checkout, retaining all original license files. Depot attribution: Rafi Abdullah
and Louise Poubel / Open Robotics, CC-BY-4.0. Robot and gripper licenses remain in
their respective roqsim model directories. No additional download is required.

Layout configuration: `factory/realistic_layout.json`. Equipment placement:
`plant_scene.py`. Stock and props: `plant_props.py`. Preview: `plant_view.py`.
Existing showcase and cooperative-carry entry points remain available.

## Architectural and equipment finish

The latest finish adds an open-front industrial shell: a tall rear wall with
clerestory glazing, low side walls, service trunk and dock door. The roof is open
for the camera. Coated work areas and a dark main logistics aisle use one muted
graphite/grey/teal palette. Original A1–H1 signs and floor graphics identify the
process areas; texture sources live in `assets/factory_graphics`.

Equipment detailing includes CNC access panels, glazing, a controller-driven
tower indicator, hydraulic press ram/platen, cleaning extraction hood, QC
camera/light gantry and conveyor supports. These additions are visual geometry;
robot dynamics, existing contact surfaces and workpiece handling are retained.
The indicator shows green while its CNC work order is active and grey afterward.
No additional process physics is implied by the equipment skins.

The clean HUD shows readable process names and actual work-order status.
`--view hero` selects a lower oblique whole-factory camera; `overview` retains
the higher layout-review angle. Both record one continuous shot.
`--diagnostics` adds internal controller state text when troubleshooting.
`--labels` remains optional because the scene now has physical wayfinding signs.
MuJoCo rendering still runs in the user's graphical terminal; headless physics
checks do not validate materials, light exposure or final camera composition.

```sh
PYTHONPATH=src ../coohoi_mujoco/.venv/bin/mjpython \
  -m unitree_demo.flexible_factory.plant_view \
  --duration 30 --view hero --video outputs/factory_professional.mp4
```
