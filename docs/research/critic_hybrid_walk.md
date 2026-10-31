# Critic: H1 walk in the delivery_area hybrid, 2026-10-31

**Verdict.** The walk proves the frame chain works (USDZ, baked collider, levelling, navmap, A*) and that the Poisson floor holds an H1 on one open route with no ground plane. It is not yet evidence of collisions, of visual consistency, of several robots, or of the customer pipeline. The 82% versus 2% support test is circular and cannot carry the README sentence "the collider is the one on real geometry". Two CPU checks and three short GPU runs fix most of that.

## A. What one H1 run shows

**Is evidence of (from the logs in docs/evidence/overnight/fleet):**
- Levelling is right: navmap floor z -0.00, floor normal tilt 3.5 deg corrected, base z 0.86 to 0.89 m. That matches H1 on the stock warehouse floor (0.83 to 0.90 in isaac_run_plan.log), so no sinking or floating.
- The collider floor is continuous and walkable along one 11 m path.

**Is not evidence of:**
- **Collisions.** level_scene picks start and goal inside a component with 0.8 m clearance, and NavMap inflates H1 by 0.5 m. The longest leg, (-0.6, -2.1) to (4.3, -3.6), is 5.1 m of straight open floor. The robot probably never came within 0.5 m of anything solid. Remove the USDZ and the run is identical: the splat played no physical role.
- **Floor fidelity where it matters.** Floor height p95 is 8.2 cm above the plane (log), so there is relief, likely curbs. Poisson smooths a step into a ramp, which flatters a blind flat-ground policy. The README already found Poisson-filled ground with a 64 cm p95 against the laser. Nobody checked whether this path crosses filled ground.
- **Safety from holes.** NavMap blocks only cells with points in the 0.1 to 1.8 m band; a cell with no points is free. A* will route over a hole in the collider, and with no ground plane that is a fall. This path missed one by luck, not by design. Obstacles under 10 cm are also invisible to the planner, which matters more for Go2 than H1.
- **No fall.** Base z is printed every 5 s: eight samples for the whole run. A stumble and recovery between prints is invisible.
- **Repeatability.** The policy and physics are deterministic, so a new seed with the same start changes nothing. Variation needs new start and goal pairs and jittered spawn poses.
- **Several robots.** Zero tested in a reconstruction.

**Smallest set that makes the claim honest (each under 30 min on an L4):**
1. **Path audit (CPU).** The DA navmap.npz was not kept: the copy in evidence is the warehouse one (310 by 469 cells, DA is 504 by 361). Rebuild it and report the trajectory's clearance to band points, the floor profile under it, and laser floor support under it.
2. **Near-obstacle batch.** 10 start and goal pairs sampled from free cells, at least half chosen so A* passes within 0.6 m of collider points, plus 6 straight walks into walls from 1 m (3 where splat and collider agree, 3 where they disagree by over 0.5 m). Spawn several H1s at once, 3 m apart, to fit the time. Log base z, roll and pitch at 50 Hz, and the minimum base distance to the collider.
3. **Fleet in the hybrid.** 2 H1, Go2 and Spot with `--plan`, crossing routes, scored with the critic_coordination metrics. First check Go2's normal height: it sat at 0.14 to 0.20 m on the flat warehouse, which looks low for a stand height near 0.3 m (unverified), so its fall bar is undefined today.
4. **Same batch on the TSDF collider** (section D), so the switch is decided on walks, not only on F1.

## B. Is the 82% versus 2% test sound?

No. Three problems, most serious first:
- **Circular.** The points P are the PatchMatch fusion the Poisson collider was built from, and Poisson interpolates P. "Collider hit within 5 cm of P" is close to true by construction. If PatchMatch itself were wrong (edge fattening, specular or window points), the collider would still score high. The 88% on agreeing pixels shows the ceiling of that construction: about 12% of the collider lies off its own points even where both agree, which is roughly the invented Poisson share.
- **Selection forces the 2%.** Pixels are chosen where the splat sits over 10 cm behind the collider. If P is a thin shell at the collider, the splat hit is over 10 cm from P along the ray, so the 2% follows from the bucket, not from a measurement of the splat.
- **3D nearest neighbour is lenient.** A Poisson lid or bridge across a gap touches real points at its rim and passes the 5 cm test there. It checks "near a point", not "first surface along this ray".

**The readout also biases the result.** Expected depth divided by opacity is a mean over every Gaussian on the ray, so semi-transparent background (sky, far facade under the 30 m cut) drags it deep. A median gap of 1.03 m is far too large for the "one sigma behind" centre bias in critic_grut_result. A long tail fits better. The median-depth probe (opacity crossing 0.5) from critic_tsdf_collider action 3 is still open and should come first.

**Independent check, using what exists.** `depth_check.py` already samples ETH3D's laser depth maps onto the undistorted grid and settles ray versus z depth. Per training view and per pixel, compare four depths along the same ray: laser, collider, splat expected, splat median. Report, for the collider-nearer bucket: the share with the collider within 5 cm of the laser, the share with the splat within 5 cm, and the share with no laser depth. Then run the ETH3D evaluator on the exact collider mesh (confirm it is the one behind the README's Open3D Poisson 0.71 and 0.84) and write its per-point accuracy cloud (check the flag names). That finds invented surfaces in laser-observed space. Report the unobserved share separately: the evaluator does not penalize surfaces where the laser saw nothing, which is exactly where Poisson invents.

## C. Visual consequence in Isaac

How RTX composites a NuRec volume with meshes in 6.2 is unverified. If it ray-marches particles and stops at the first mesh hit, per particle, then:
- **Splat behind collider (69%):** the robot draws in front of the splat wall, which is fine for occlusion. But it stops at the collider, so in any view with parallax it halts short of the visible wall, up to the 1.03 m median gap. That is "bumping into nothing".
- **Collider farther or missing (21%):** the splat sits in front. A robot walks into visible pallets, or its feet vanish into a splat floor raised above the collider floor. This is the worse case for a customer and the smaller one to check.
- If the readout above is a tail artifact, the splat's real front surface may sit at the collider, and none of this shows.

It is worth checking, because a customer sees it in the first ten seconds. **Cheap version (Isaac, about 15 min):** place 1 m grey cubes with no physics on the collider surface at 6 wall sites (3 agree, 3 disagree). Render from a training camera with and without them, plus a camera 30 deg off. Use an instance or depth output for the cube mask. The share of mask pixels whose colour is unchanged from the cube-free render is the share covered by the splat. **3DGRUT-only version:** for each collider hit, the splat's accumulated opacity before t_mesh minus 5 cm. Over 0.5 means a robot standing at the collider gets painted over.

## D. Switch the ETH3D collider to TSDF on PatchMatch depth?

Not blind, and not before pm_walk lands. The synthetic choice was made on ground-truth and splat depth, never on PatchMatch.
- **Expected benefit.** TSDF carves the space the cameras saw, so Poisson lids, bridges and bulges go away (the likely source of the unsupported 12%). There is no global 12% density trim, which deletes thin, sparsely sampled posts first. On DA the floor benefit is small: the Poisson floor already scored 0.25 cm median against the laser where seen.
- **Risks.** (1) TSDF leaves holes wherever PatchMatch has no depth (low texture, gloss, beyond the 12 m cut). With no ground plane, and a planner that treats empty cells as free, a hole is a fall. Fix the planner first (action 3). (2) Integrate COLMAP's geometric (filtered) depth maps, not the photometric ones, or edge fattening becomes surfaces in front of walls. (3) Pin Open3D 0.19, as noted in the README.
- **Decide on bars:** ETH3D F1 at 5 cm within 0.02 of Poisson, a bigger collider share within 5 cm of the laser on the disagreement pixels, under 2% floor holes along the batch paths, and the near-obstacle batch passing. For a real factory floor (epoxy, gloss), TSDF holes are likely, so pair it with the plane fill from floor_clip.py where SfM shows floor, and holes where the guard sees a pit.

## E. What a reviewer would call out first

1. **Poses are ETH3D's laser-registered poses.** They are metric and near perfect. The customer path (phone, GLOMAP or ARKit, no laser) has never run end to end into Isaac. GLOMAP poses already exist (44 of 44, 2.3 cm), so build the hybrid from them once and state the scale source.
2. **The camera is locked to a training view** to hide floaters. Any orbit or follow camera will show them. Say so in the demo, or keep the camera path near training cameras.
3. **Lighting.** The robots are lit by an arbitrary 1000 dome while the splat has baked sun, and they cast no shadow on it, so the result reads as composited. Check whether a matte or shadow-catcher setting on the collider works with NuRec in 6.2 (unverified).
4. **One robot, no job.** The product claim is several robots coordinating.
5. **The collider is Poisson** while the team's own synthetic results say TSDF. That inconsistency is easy to spot.
6. **Collider approximation.** Confirm the collider prim is a static triangle mesh (approximation none), not a hull or decomposition. The floor height says yes for the floor, but walls are untested (what add_mesh_to_usdz sets is unverified).
7. **Blind policies.** The stock policies have no height scan, so the scene reaches them only through the planner and contacts. Say that plainly.

## Ranked actions, cheapest first

1. **Keep the evidence.** Save the per-run navmap.npz, the trajectory, and base z, roll and pitch at 50 Hz. Bar: files saved per run; H1 minimum base z at least 0.75 m (flat baseline 0.83 to 0.90).
2. **Path audit (CPU).** Clearance, floor profile and laser floor support along tonight's trajectory. Bar: numbers reported. If minimum clearance is over 0.5 m, the claim becomes "open-floor walk".
3. **Floor-support mask in NavMap.** Block cells with no collider point within 5 cm of floor z, after a 3 by 3 closing. Bar: a hole case in test_navmap.py fails today and passes after; DA free area drops under 5%.
4. **Median-depth splat_vs_collider (GPU, about 10 min).** Bar: if the collider-nearer share falls under 30%, the 69% was a readout artifact; rewrite the README sentence.
5. **Laser-ray check (B, about 10 min).** Bar: collider within 5 cm of the laser on at least 60% of collider-nearer pixels with laser depth (PatchMatch gets 66% overall), and the splat on under 20%. Under 50% for the collider retracts "the collider is on real geometry".
6. **Cube occlusion render (C, about 15 min).** Bar: under 5% of cube pixels covered at agreeing sites. Record the visible gap at disagreeing sites.
7. **Near-obstacle and wall batch (A2).** Bar: 10 of 10 pairs arrive, 0 falls, at least 5 runs pass within 0.3 to 0.6 m of an obstacle, and 6 of 6 wall walks stop with no pass-through.
8. **Fleet in the hybrid (A3).** Bar: all arrive, 0 falls, minimum surface gap over 0.3 m, final distance to goal under 0.35 m (not latched).
9. **TSDF collider arm (D), after pm_walk.** Bar: section D.
10. **GLOMAP-pose hybrid end to end.** Bar: scale within 1% of ETH3D, and the action 7 subset passes.
