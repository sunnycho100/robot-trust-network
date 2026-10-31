# Colab recipes (overnight 2026-10-31)

Scripts the overnight loop ran on a Colab L4 through the `colab` CLI. Each has a one-line usage comment at the top. Results and what failed are in [`docs/research/overnight_report_2026-10-31.md`](../../../docs/research/overnight_report_2026-10-31.md).

Start here:
- `isaac_pip.sh`, then `isaac_run.sh`: Isaac Sim 6.2 headless on Colab through pip, recording a robot example
- `grut_setup.sh`, `grut_run.sh`, `grut_post.sh`: 3DGRUT splat training and USDZ export
- `pm_da.sh`, `da_tsdf.sh`: PatchMatch stereo colliders (Poisson and TSDF) for the ETH3D delivery_area scan
- `dafleet.sh`, `dafill_fleet.sh`: three robots sharing a hand-off job in the reconstructed scan
- `lab_mappo.sh`: Isaac Lab 3.0 two-agent MAPPO training (no Isaac Sim)

The `*_queue.sh` files chain several jobs on one session. Launch with `cd /content; setsid nohup bash job.sh > job.log 2>&1 &`, with no `&&` before `setsid`, or the exec cell blocks until the job ends.
