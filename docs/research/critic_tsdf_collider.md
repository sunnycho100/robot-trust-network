# Critic: TSDF collider and splat depth on the synthetic warehouse

## 1. Poisson verdict: right default, unproven cause

TSDF is the right mesher for posed depth maps since it carves free space. But "Poisson buries the floor" is not shown. Free IoU 0.00 follows from the 94% floor holes alone: with no floor cells, the 0.3 m erosion leaves nothing free. A median floor error of 2.7 mm means the floor is correct where it exists. So the floor is either missing (trim, cluster filter or iso drift) or lidded, and only a lid matches the README.

STATE.md 05:33 gives Poisson free IoU 0.99994, likely the points JSON mislabelled. Confirm the README source.

**Cheapest check:** cast rays straight down from 2 m above each GT floor cell into the existing walk_oracle Poisson mesh. Count first hits at the floor, in the 0.1 to 1.8 m band (a lid), or none (missing).

## 2. Splat verdict: fair for this readout, not yet for the splat

The extraction is correct. The 3DGUT kernel accumulates hitT times weight (hitT is Euclidean, to each Gaussian's max response), so pred_dist divided by opacity is the expected depth, and the opacity > 0.5 mask caps any division effect at 2x, which cannot produce 4.2. Two confounds:

- The ratio script is not committed. grut_depth.py names outputs view_0000 to view_0209 by train-loader index, the capture frame_00000 to frame_00239 with every 8th held out. Pairing by index makes the offset grow over the walk, which fits ratios climbing from 1.04 to 4.2. The TSDF floor error is name-independent and stands.
- The floor is glossy; splats model reflections as mirror geometry below it. Depth 10 to 15% long from a 1.5 m camera puts the floor 15 to 20 cm low, as measured.

## 3. Oracle setup

Scoring the TSDF on the depth it integrated is circular but fine as a ceiling. Real gaps: false_free (collision danger) is missing from the README, and perfect depth still loses 7% of obstacle cells. The 12 m depth_trunc against unbounded GT points inflates TSDF holes. pm_walk leaks held-out frames to PatchMatch.

## 4. Next actions, cheapest first

1. Downward raycast on the Poisson oracle. Bar: "lid" only if over 50% of floor cells hit the band. Also report TSDF oracle false_free, bar at most 0.01.
2. Commit the ratio script, pair by name, split floor pixels out. Bar: a median within 1 plus or minus 0.03 retracts the README sentence.
3. Median-depth probe in torch: 2k pixels in 5 views, the depth where accumulated opacity crosses 0.5. Bar: 70% of floor pixels within 5 cm, then rerun grut_depth with it.
4. If "missing", rerun Poisson with trim 0 and depth 11. Bar: holes at most 10%, free IoU at least 0.9.
5. pm_walk. Bar: floor p95 at most 2 cm, holes at most 15%, free IoU at least 0.85, false_free at most 0.02, recall at least 0.85.
6. 2DGS on ETH3D. Bar: F1 at 5 cm at least 0.75, PSNR within 1 dB.
7. Matte floor recapture. Bar: splat floor error under 3 cm.

## Response (overnight loop, 06:10 CDT)

- Section 2, frame pairing: not an issue. The ratio script paired each rendered view with the capture frame by camera pose (nearest centre), not by index; every pair had a 0.0000 position gap and a 0.0000 maximum rotation-matrix difference (view 0 matched frame_00001 because frame 0 is held out). The ratios 1.04 to 4.2 stand. Splitting floor and non-floor pixels and the median-depth probe (actions 2 and 3) are still open.
- Section 1: agreed. README now says Poisson loses 94% of the floor cells and that lid versus missing is unchecked. The README Poisson row is from score_oracle_mesh.json; the STATE line was mislabelled and is fixed. Downward raycast (action 1) is next.
