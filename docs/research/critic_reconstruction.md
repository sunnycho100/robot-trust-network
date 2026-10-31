# Critic: factory reconstruction plan, 2026-10-31

Verdict: **the method stack is fine; the test scene and the success checks are not.** Church spends most of tonight's L4 on a scene with no published baseline. Whole-scene F1 and "H1 does not fall through" cannot tell a good collision mesh from a bad one. The synthetic route is the right testbed long term, but it needs fixes and cannot run on Colab.

## 1. Test scene

Church is a poor first pick:

- **No calibration.** The 2DGS README reports T&T F1 on six scenes (Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck) and leaves Church out (verified). GOF and PGSR likely use the same six (unverified).
- **Not factory-like where it matters.** Robots need open floor, walls, and obstacles at 0 to 2 m. Church is mostly pews and ornate detail in dim light (from memory, unverified). Its openness is in ceiling height, which no robot uses.
- **Cost.** The plan's own estimate (2.5 to 3.5 h) is the whole budget. gdown on public Drive files can hit quota (unverified for these files). Step 5 needs Isaac Sim, which Colab lacks.

Evidence per GPU hour, ranked:

1. **ETH3D delivery_area.** The PatchMatch, vid2scene and floor scores already exist, and its 38 and 6 split matches 3DGRUT's `test_split_interval: 8` (verified in `configs/dataset/colmap.yaml`).
2. **Meetingroom.** An indoor floor with furniture, plus published F1 to check our harness against.
3. **Synthetic warehouse.** The only route with exact per-object GT, but it needs the Isaac box.
4. Church.

## 2. Appearance method and the head to head

3DGRUT is right for integration reasons (ParticleField USDZ, collider attach), not quality. 3DGUT rasterizes, so the L4's lack of RT cores is irrelevant. The real Colab risk is the install, which builds CUDA extensions (verified) with an unknown Colab build time (unverified). Skip splatfacto: it adds nerfstudio overhead and has no USDZ path.

The head to head is not fair as designed:

- **"Same iterations" is the wrong control.** A capped MCMC run and gsplat's default densification end with different Gaussian counts. Compare against gsplat's MCMC strategy at the same cap (from memory, unverified).
- **One scorer.** Score saved renders with one script. Each repo's LPIPS backbone and SSIM code can differ (unverified).
- **Matched per-image appearance.** PPISP and gsplat's appearance embedding have no code for held-out views. Disable both, or handle both the same way.
- **Matched inputs.** Same downsample factor, same init points. Report wall clock and peak VRAM.
- **The quoted PSNRs (27.4 and 29.0) are 1.6 dB apart** under different protocols. That is "unknown", not "equal".

A gap under about 1 dB would not change the pick, so the head to head is optional tonight.

## 3. Collision metrics

Whole-scene Chamfer and F1 weight ceilings like floors, and mean Chamfer is dominated by floaters. 2 cm is stricter than obstacles need, yet looser than a foot needs on the floor. Keep F1 at 5 cm only for comparison with published numbers. Decide on these, on a 5 cm grid over the area the cameras saw:

- **Floor height error.** Median and 95th percentile, the share of false bumps over 2 cm (trip hazard), and the share of holes (GT floor with no recon surface within 5 cm).
- **Traversable area IoU.** A cell is occupied if any surface sits in the height band (0.1 to 1.8 m for the humanoid, 0.05 to 0.5 m for the wheeled robot). Free space is floor that is not occupied, eroded by robot radius. Report false-free (collision danger) apart from false-occupied (blocked paths).
- **Obstacle recall above 10 cm.** Per GT prim on the synthetic route, with thin posts, rack uprights and legs as a separate row, since TSDF and Poisson erase them first.
- **Wall planes.** Plane RMS and angle error. `check_collisions.py` already computes ground error and surface IoU at 0.5 and 1.0 m; aim it at GT instead of our own mesh.

Two gaps:

- **The Ground Plane makes "H1 does not fall through" vacuous**, because it catches every hole. Walk without it, and check for a double floor where the plane and the recon floor overlap.
- **Metric scale.** COLMAP has none, and the T&T similarity alignment quietly supplies it. A customer capture needs its own scale source (phone LiDAR, ARKit poses, or a scale bar). Report which one each result used.

## 4. Synthetic route (capture_walkthrough.py)

The pose math is correct and unit tested. The problems:

- **Test leak.** `backproject` seeds points3D from every frame, test frames included (lines 190 to 192). Seed from training frames only.
- **Realism leak.** GT depth gives a dense init on texture-poor floors, and GT poses skip SfM failure, the main real-factory risk. Make SfM from scratch the headline run and keep GT poses as the upper bound.
- **Easy held-out views.** Every 8th frame of a 120-frame loop sits 3 degrees from a training view, so the test measures interpolation. Render a separate test path at 0.4 and 1.0 m and a different radius. The plan's 0.4 m loop is missing: `--height` takes one value.
- **Floor coverage.** The cameras are nearly level, with roughly 50 degrees of vertical FOV, so a 1.6 m camera never sees the floor within about 3.4 m. Add a loop pitched down 20 to 30 degrees.
- **Eyes inside geometry.** The loop comes from percentile vertex bounds, which follow vertex density, so the centre can fall inside a rack. Reject frames with near-zero minimum depth.
- **Units and up axis.** `metersPerUnit` and `upAxis` are never read from the asset (unverified for `full_warehouse.usd`). Assert both.
- **GT mesh is not visible GT.** It keeps guide and proxy meshes, drops PointInstancers, and includes hidden interiors. Use back-projected GT depth (1 cm voxels) for recall, and the mesh for per-prim recall.
- **Too clean.** Add exposure jitter, blur, noise, JPEG, and a plain grey epoxy floor variant. Check for temporal ghosting from the per-frame pose jumps (unverified).
- **End-to-end check.** Reproject frame i's depth into frame i+1, with a median error under 1 cm.

## 5. Tonight (L4, 3 h)

1. **0:00 to 0:30, install 3DGRUT** and run a 2k-iteration smoke test on delivery_area. Not training by 0:30: use gsplat and move the USDZ question to the L40S.
2. **0:30 to 1:05, 3DGUT MCMC on delivery_area**, existing split, shared scorer. Compare against vid2scene's 19.9 PSNR and 0.32 LPIPS.
3. **1:05 to 1:45, gsplat 2DGS plus TSDF on delivery_area.** If it loses to PatchMatch on F1 at 5 cm (0.56) and on the floor 95th percentile (1.0 cm), retire it.
4. **1:45 to 2:50, Meetingroom**: GLOMAP, PatchMatch at half resolution, T&T evaluator, floor metrics. If registration is under 90% or alignment fails by 2:20, pull what exists.
5. **2:50 to 3:00, pull results.**

**Brev L40S tomorrow:** fix and render the synthetic capture, run it from SfM with the robot metrics, load USDZ plus collider in Isaac Sim 6.2, and walk H1 and a wheeled robot without a Ground Plane. Then Church, PGSR, and the gsplat MCMC head to head.
