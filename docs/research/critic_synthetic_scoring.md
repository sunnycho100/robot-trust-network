# Critic: synthetic scoring (score_robot.py and walk_queue.sh)

**Verdict.** The metrics in score_robot.py have the right shape, but no number from tonight's run should be trusted yet. In a scratch run against the unmodified `score()`, a perfect reconstruction of a flat floor sampled the way `load_points` samples (uniform random, 2 cm spacing) got free IoU 0.66 with 20% false-free, and 0.0 at a third of that density. In the same run, the floor detector picked the ceiling once it had 5% more area than the floor. The demo misses both because its regular grid never leaves a cell empty. Random init is the fair call, but its box is too small. Triangulated points should be the headline init. The queue fits 3DGRUT's loader, but if the capture fails it silently scores stale data.

## Findings

**1. Blocker: the erosion blows up sampling gaps.** About 0.2% of cells get no floor sample. Each one erodes a disk of about 113 cells at a 0.3 m radius, so free IoU drops and both false rates rise by about 20%. That noise hides any real difference. Fix: `ndimage.binary_closing(m, np.ones((3, 3), bool))` on both floor masks before eroding. Tested: perfect scores come back at full density and at a third. Add a uniform random plane to `demo()`.

**2. Blocker: the z-histogram mode can return the ceiling.** In a warehouse the ceiling and roof faces each have about the floor's area. Fix: build the histogram only from GT points below the lowest camera centre (`c[:, 2].min()` from `--poses`) and inside the region.

**3. Major: scoring runs against the whole GT mesh, not what the cameras saw.** Unseen floor under racks and pallets, hidden interiors and proxy meshes count as GT. Holes and false-occupied go up and recall goes down. Fix: back-project the training-frame depth PNGs, voxel them at 1 cm and pass that as `gt`. Crop to the region before sampling, so a skybox cannot eat the 30M cap.

**4. Major: the 0.3 m recon floor tolerance counts low objects as bumps.** Pallet decks (about 0.14 m) and base plates fall inside the window, which pushes bumps and p95 up. Fix: leave out cells with GT surface 2 to 30 cm above the floor, `is_floor & ~binary_dilation(occupied_map(gt, lo, shape, z_floor, (0.02, 0.3)), np.ones((3, 3)))`. Report dips (`err < -0.02`) too.

**5. Minor: double counting.** One floor hole counts once in `floor_holes` and about 113 times in `false_occupied`, and `false_free` can go above 1. Document this and never add metrics together. Units and frames are fine.

**6. Minor: `images.txt` parsing skips every other camera.** The parsers drop empty POINTS2D lines and then take `[::2]`, so they read every other camera. The box barely moves, but `poses.txt` comes out half-length. Fix: remove `l and` from all three filters.

**7. Major: the random init box (question 2).** Random init is right because the seeded points are GT depth. But the box assumes the floor is at z = 0 and leaves out the walls the outward loop sees. MCMC only moves dead Gaussians onto live ones, so far walls get painted onto the box faces. Those faces sit on the scoring region's edge and pass the depth check. Fix: use the GT 2nd to 98th percentile bounds or 3x the camera extent.

COLMAP `point_triangulator` with the known poses is worth about 20 minutes as the headline. Random init is the lower bound and GT seeding the leaky upper bound. Its sparse points (dense on texture, thin on plain floor) are what a real capture gets. Steps: `feature_extractor` (single PINHOLE camera, known params), then `exhaustive_matcher`, then `point_triangulator`. The database image IDs must match `images.txt`, so verify. 3DGRUT reads `points3D.bin` first, so give the model its own folder.

**8. Major: Poisson depth 10 at warehouse scale (question 3).** Over a 40 to 60 m box this gives 4 to 6 cm voxels. Uprights and pallet gaps blur, and the global density trim cuts sparse real geometry before invented surface. It works for floor and walls, not thin obstacles. **Most informative next experiment: an oracle ablation.** Run GT depth through the same fusion, Poisson and scorer, with no training. It checks the scorer (after fixes 1 to 4 it should come out near perfect) and gives the mesher's ceiling. Then compare Poisson with a 2 cm TSDF on the same depth, then run 2DGS plus TSDF on this capture. PatchMatch goes last, since it is slow and clean synthetic images flatter it.

**9. Major: walk_queue.sh scores stale data if the capture fails (question 4).** `$WK` is queue1's old folder, and `gt_mesh.ply` gets written before any frame renders. So the gate passes after a crash and the old frames get used. Fix: `rm -rf $WK` first, gate on `grep -q "\[walk\] done" /content/walk.log`, and `md5sum` the Colab copy of `capture_walkthrough.py`.

**10. Minor: smaller queue risks.** There is no `pipefail`. The checkpoint glob could pick up an older run. Colab's idle disconnect is a risk. `grut_points.py` needs about 5.5 GB of RAM.

Checked against the 3DGRUT source and fine:

- `images/` at `downsample_factor=1`, through the symlink.
- PINHOLE, empty POINTS2D lines, and the `points3D.txt` fallback.
- Every 8th frame by name, matching `test.txt`.
- Pixel centres at +0.5, matching `cx = w/2`.
- The `render.py` arguments.
