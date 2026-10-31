# CarDA cobot cell (rebuild of SIM_HANDOFF v2, WS30)

The handoff's own package (`out/sim_package`, `replay.py`) was not shared, so this rebuilds its WS30 baseline replay
from the PDF's numbers, Doosan's M0617 URDF and CarDA's XSens mocap. Analysis and results:
[`docs/research/sim_handoff_analysis.md`](../../docs/research/sim_handoff_analysis.md).

| File | What it does |
|---|---|
| `carda_cell.py` | M0617 on a pedestal and track, door box fitted to the worker's screw points, worker as bone capsules from the BVH, IK waypoints joined in joint space, P1 / P1s / P2 / P3 interlocks, capsule distance and ISO/TS 15066 SSM KPIs, optional video. `--selfcheck` runs without data. |
| `g1_cell.py` | Our G1 (Unitree's published walking policy, as in `sim/mujoco_houses`) takes the worker's place: walks to the door, holds 4 s, leaves; the arm runs P2 or P1. |
| `colab_job.sh` | The Colab L4 run: KPI matrix, three replay videos, two G1 videos. |

## Colab

```bash
colab new -s handoff --gpu L4
# on the VM: CarDA WS30 archive (922 MB, CC BY 4.0), only the BVH is needed
mkdir -p /content/carda /content/out && curl -fL -o /content/carda/ws30.rar "https://zenodo.org/api/records/14644367/files/ws30%20-%20svo%20-%20mp4%20-%20bvh.rar/content"
cd /content/carda && unrar x -o+ ws30.rar '*.bvh' '*.xlsx' bvh/
pip install -q mujoco opencv-python-headless
```

Upload this folder to `/content/handoff/`, and for `g1_cell.py` a tarball of `sim/mujoco_houses` with
`src/unitree_demo/{__init__,g1_policy,coop_lift}.py`, `assets/g1_23dof.xml`, `motion.pt` and the G1 meshes to
`/content/robosim/mujoco_houses/`. Then `cd /content; setsid nohup bash handoff/colab_job.sh > job.log 2>&1 &`.
The whole job took about 20 minutes on the L4 (CPU bound; set `OMP_NUM_THREADS`, or torch and numpy oversubscribe the 12 cores).
