"""Basic demo: one Unitree G1 walking on its published policy, driven from WASD."""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from .common import HumanCommandInput, add_run_arguments, check_run_arguments, ensure_macos_viewer_runtime
from .g1_policy import G1Controller, gait_phase, load_g1_config


def run(args: argparse.Namespace) -> None:
    cfg = load_g1_config()
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    model.opt.timestep = cfg.simulation_dt
    data = mujoco.MjData(model)
    g1 = G1Controller(model, cfg)
    fixed_command = np.asarray(args.command, dtype=np.float32)
    human_input = None if args.headless else HumanCommandInput("G1")
    viewer = None
    if not args.headless:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)

    print(f"Loaded Unitree G1 policy {cfg.policy_path.name} "
          f"({cfg.num_actions} actions, {1 / (cfg.simulation_dt * cfg.control_decimation):.0f} Hz)", flush=True)
    if human_input is not None:
        print("Focus the command-input window and hold a key; release it to stop.", flush=True)

    counter = 0
    wall_start = time.monotonic()
    try:
        while ((args.duration <= 0 or time.monotonic() - wall_start < args.duration)
               and (viewer is None or viewer.is_running())
               and (human_input is None or human_input.running)):
            step_start = time.monotonic()
            command = human_input.poll() if human_input is not None else fixed_command
            g1.apply(data)
            mujoco.mj_step(model, data)
            counter += 1
            if counter % cfg.control_decimation == 0:
                g1.update_policy(data, command, gait_phase(data))
            if viewer is not None:
                viewer.sync()
                remaining = cfg.simulation_dt - (time.monotonic() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
        base = data.qpos[g1.base_qpos:g1.base_qpos + 3]
        print(f"Finished at sim t={data.time:.2f}s, base=({base[0]:+.2f}, {base[1]:+.2f}, {base[2]:+.2f})", flush=True)
    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        if human_input is not None:
            human_input.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(parser)
    args = parser.parse_args()
    check_run_arguments(args)
    ensure_macos_viewer_runtime(args.headless, "unitree_demo.basic")
    run(args)


if __name__ == "__main__":
    main()
