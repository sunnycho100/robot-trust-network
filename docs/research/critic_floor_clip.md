# Critic: floor clip and cluster filter on the splat collider

## 1. Is the floor clip legitimate?

No ground truth enters it: the floor height comes from SfM points through the scorer's `find_floor`, a shared definition, not shared data. But it leans on exact poses, known Z up and metric scale, and a perfectly flat floor. With a planar prior on a planar floor, floor error 1.3 mm measures the SfM plane, not the splat. Report it that way.

On a real capture, most dangerous first:

- **Geometry below the plane gets floored.** `clip()` snaps every point under floor + 2 cm onto the plane, so a dock pit, down ramp or trench gets a phantom floor: false free over a fall hazard.
- Up ramps and mezzanines sit above the plane, so their reflections stay holes (safe).
- SfM on a low-texture glossy floor also triangulates reflections, which can win the histogram mode.
- SfM gravity and scale are arbitrary; a 1 degree tilt is 17 cm over 10 m.

Minimal robust version: RANSAC plane on SfM points (`segment_plane`, as in `level_scene.py`), then a 0.5 m grid of local floor heights. Clip a pixel only where its cell has SfM floor support within 3 cm; elsewhere leave a hole. Holes are safe, invented floor is not.

## 2. Is clipping hiding obstacles?

Not by itself: after the clip, false free (0.001) and recall (0.93) equal GT-depth TSDF. The losses come with the cluster filter. Two blind spots remain: the scorer ignores 2 to 10 cm, exactly where the clip flattens base plates, forks and curbs; and per-cell recall cannot tell thinned rack edges from whole missing pallets.

False free 0.009 is nine times GT-depth TSDF, and recall 0.85 sits on the bar after tuning on GT. Not a pass yet. Check false free cells as blobs (m², distance to nearest GT obstacle), recall per GT obstacle and per height bin (0.1 to 0.3, 0.3 to 1.0, 1.0 to 1.8 m), and recall in a 0.02 to 0.10 m band.

## 3. Cluster filter

"Obstacles on the floor stay joined to it" fails for boxes on shelves and anything the TSDF leaves a gap under, the likely cause of the recall loss. Picking 2000 from two values by GT score is selection on the test scene, and triangle count scales with voxel size (2000 at 2 cm is roughly 0.4 m²).

GT-free rule: keep a component that comes within 5 cm of the local floor or exceeds 1 m² of area; drop the rest only if seen in fewer than 3 training views or within 1 m of the camera path, where splat floaters sit. Freeze it, then run a second Isaac scene without retuning.

## 4. Splat-only against PatchMatch

One L4 hour is small against capture time, so cost only breaks ties. Prefer splat-only if, on the warehouse and a second scene: false free at most 0.01 with no blob over 0.1 m², recall at least PatchMatch minus 0.03 and at least 0.8 per height bin, free IoU at least 0.85, and walks matching PatchMatch with zero pass-throughs. Otherwise PatchMatch. pm_walk sees all 240 frames, a small edge for it. The walks are weak evidence: starts and goals need 1 m clearance, so add one path within 0.4 m of racks.

## 5. Next actions, cheapest first

1. False free blobs on the k2000 mesh. Bar: none over 0.1 m², 90% of cells within 10 cm of a GT obstacle.
2. Recall per obstacle and height bin, k2000 against no filter. Bar: no obstacle over 0.1 m² below 50% recall.
3. Dropped cluster audit. Bar: under 20% of dropped triangles within 5 cm of GT surface.
4. Pit case in the `floor_clip.py` demo: a 1 m pit must stay a hole. Bar: fails today, passes after the guard.
5. RANSAC plane plus support guard. Bar: warehouse scores within 0.01.
6. GT-free cluster rule. Bar: false free at most 0.01, recall at least 0.88.
7. Second scene with a level change, settings frozen. Bar: section 4.

## Verdict

The floor clip is a GT-free, sound fix for the glossy floor here and costs nothing in safety on this scene. It is still a flat-floor prior that will invent floor over any drop on a real site, so it needs the local support guard before a real capture. The cluster filter is the real source of the false free and recall loss, its threshold was tuned on the test scene, and its premise fails for shelf contents. Treat free IoU 0.957 as promising on one scene, not a collider claim, until the blob, per-obstacle and second-scene checks pass.
