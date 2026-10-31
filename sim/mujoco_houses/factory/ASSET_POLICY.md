# Factory asset policy

Robots: only cps-test-lab/roqsim or google-deepmind/mujoco_menagerie.
Base environment: roqsim Depot warehouse. Optional props: MIT-0 AWS Small Warehouse.
No marketplace downloads. No NC, ND, unknown or missing licences.
Functional equipment is locally authored MuJoCo primitives.

The demo references the adjacent upstream checkout without copying or stripping
its attribution. Keep that complete checkout when sharing or deploying assets.
Pinned inspected upstream commit: `6e6999be07e56ebf442b2f20225f33b97c3a9f70`.

| Asset | Attribution / licence file in upstream |
|---|---|
| Depot | `roqsim_scenes/src/roqsim_scenes/scenes/depot/CREDITS.txt`, `assets.lock.json`; CC-BY-4.0, Open Robotics, Rafi Abdullah and Louise Poubel |
| UR5e | `roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/ur5e/UR5e_LICENSE` |
| UR10e | `roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/ur10e/UR10e_LICENSE` |
| Panda | `roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/panda/PANDA_LICENSE` |
| Robotiq | `roqsim_manipulation_assets/src/roqsim_manipulation_assets/models/robotiq_2f85/ROBOTIQ_2F85_LICENSE` |
| Husky | `roqsim_mobile/src/roqsim_mobile/models/husky_a200/husky_a200_LICENSE` |
| Clearpath Jackal | `roqsim_mobile/src/roqsim_mobile/models/clearpath_jackal/clearpath_jackal_LICENSE` (BSD-3-Clause) |
| TurtleBot3 Waffle | `roqsim_mobile/src/roqsim_mobile/models/turtlebot3_waffle/turtlebot3_waffle_LICENSE` |
| Boston Dynamics Spot | `roqsim_quadruped/src/roqsim_quadruped/models/LICENSE.mujoco_menagerie` (BSD-3-Clause), provenance in `roqsim_quadruped/THIRD_PARTY.md`; model only, no external locomotion weights |

Also preserve upstream `LICENSE`, `NOTICE`, all relevant `THIRD_PARTY.md`,
and model manifests. Source: https://github.com/cps-test-lab/roqsim

Depot attribution for exported videos:
“Depot warehouse by Rafi Abdullah and Louise Poubel / Open Robotics,
CC-BY-4.0, adapted for MuJoCo by roqsim. Cutaway view and factory equipment added.”
https://fuel.gazebosim.org/1.0/OpenRobotics/models/Depot
https://creativecommons.org/licenses/by/4.0/
