# Critic: 3DGUT result on delivery_area, 2026-10-31

**Verdict.** The appearance result is real. The geometry number is a fair verdict on volumetric 3DGUT with expected depth, not on splat geometry: no geometry-aware splat has been tested. The hybrid is still the right default, mostly for license reasons. The queued synthetic run tests the wrong variable.

## 1. Is F1 0.07 at 2 cm fair?

**No bug in the depth math** (checked against `/private/tmp/3dgrut`):

- `pred_dist` is the raw sum of `hitT * weight` (`gaussianParticles.cuh` line 412, unnormalized in `finalizeRay`), so dividing by opacity is right.
- `hitT` is the Euclidean distance to the point of peak response, so the unit direction is right.
- Intrinsics recovery is self-consistent, and `normalize_world_space: false` keeps the COLMAP frame.

**One real risk:** the intrinsics come from the last batch's rays and are reused for every view. If `dslr_calibration_undistorted/cameras.txt` has more than one camera, the 1% check compares the wrong pixels. Check for one line, or use per-view intrinsics.

**The numbers point at the primitive, not the code.** Implied accuracy is about 19% at 2 cm and 44% at 5 cm: most points that passed a three-view 1% check are still over 2 cm off. That is consistent bias, not noise. It fits 3DGUT's depth: `hitT` lands at each Gaussian's centre, roughly one sigma behind the visible front, and MCMC Gaussians are several cm across. With `k_buffer_size: 0`, blending follows a global sort by centre, so depth at overlaps varies by view and the 1% filter drops pixels. Check the per-view "kept X of Y" lines: under 20% kept means the filter explains the low completeness. The 1M cap and 1/3 resolution cost a little, not ten times.

**Red flag:** vid2scene's splat depth also scored 0.07 and 0.16 through the same fusion design. Worth action 1.

**The fair test:**

- Median depth still lands on Gaussian centres. TSDF instead of filter plus Poisson raises completeness but keeps the bias.
- The fair test is a **surfel primitive** with **depth distortion and normal consistency losses**, **median depth**, then **TSDF**: 2DGS. gsplat's `simple_trainer_2dgs.py` is Apache-licensed.
- PGSR, GOF and MILo only bound the upside: all non-commercial.
- 3DGRUT has an undocumented `primitive_type: trisurfel`, but no geometry losses, so not yet.

## 2. Architecture

Keep "splat for looks, PatchMatch for collision" as the default:

- **License.** Every strong geometry splat is non-commercial (RaDe-GS likely too, unverified). Only gsplat 2DGS is clean, and it is the weakest.
- **Accuracy.** On Tanks and Temples the best geometry splats roughly match MVS, not beat it, and surfels usually lose some PSNR.
- **Registration is free:** same poses.

The real costs: two pipelines, and visual-collision mismatch (bumping into nothing, walking through visible splats). Also, delivery_area is textured and outdoors, which flatters PatchMatch. Grey epoxy floors and shiny metal break PatchMatch first, which is where a regularized splat could win. Neither is tested there.

**Cheapest decisive experiment (L4, about 1.5 h):**

1. **0:00 to 0:10.** If the 3DGUT checkpoint survived, render depth on the 38 training views and run `depth_check.py` against the laser depth: signed error histogram, share within 2 and 5 cm. A peak a few cm behind the surface confirms the centre bias.
2. **0:10 to 1:05.** gsplat 2DGS, same images at 1/3 resolution, same 38 and 6 split, normal and distortion losses on, 20k steps.
3. **1:05 to 1:30.** Median depth into a 1 cm TSDF, ETH3D evaluator at 2 and 5 cm, `depth_check.py`, and test metrics with 3DGUT's scorer.

**Set thresholds first:** one model stays alive only if F1 at 5 cm is at least 0.75 and PSNR is within 1 dB of 23.04. Otherwise keep the hybrid and stop spending GPU on splat geometry until the synthetic floor results.

## 3. Queued synthetic experiment

- **Wrong variable.** Random versus triangulated init, with the same 3DGUT and the same extraction, cannot change the architecture. Both will likely fail geometry for the reason in section 1. Make triangulated init the only 3DGUT arm; random init is a lower bound if time is left.
- **The collider is never scored.** Without a PatchMatch arm on known poses, the hybrid's actual collision mesh never meets `score_robot.py`. Add it, and 2DGS if tomorrow passes.
- **The oracle skips the fusion.** `score_robot.py oracle` back-projects GT depth at 1 cm with no 1% filter; splat depth goes through `grut_points.py` at 5 mm with it. The gap mixes depth quality with filtering. Run GT depth through the same filter and voxel size.
- **One seed per arm.** Differences of a few points are noise.
- **Still open from the earlier critique:** held-out frames sit on the training loop, renders are clean, no grey floor variant. The grey floor decides the product; add it before scaling up.

## Ranked actions

1. **Bias check.** Read "kept X of Y", run `depth_check.py` on 3DGUT depth if the checkpoint exists, confirm one camera in `cameras.txt`.
2. **2DGS test.** Run gsplat 2DGS on delivery_area with the thresholds above, and decide on the hybrid from that.
3. **Rework the synthetic queue.** Add PatchMatch on known poses, run the oracle through the same filter, make triangulated init the headline.
4. **Grey floor variant.** Score PatchMatch and the splat on a plain epoxy floor.
5. **Mismatch metric.** Before walking robots in the hybrid USDZ, compare splat depth with the raycast collider per view: share of pixels off by more than 10 cm.
