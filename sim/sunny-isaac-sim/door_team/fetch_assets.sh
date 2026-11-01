#!/bin/bash
# Local copies of the stock robot assets for the laptop-side checks (test_rig.py, export_dots.py --check).
# The Isaac Sim render does not need these: it streams the same files from the NVIDIA asset root.
# Usage: bash fetch_assets.sh ASSET_DIR
set -eu
A=${1:?asset dir}; B=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/Robots
mkdir -p "$A" && cd "$A"
for f in UniversalRobots/ur20/ur20.usd UniversalRobots/ur20/configuration/ur20_base.usd UniversalRobots/ur20/configuration/ur20_physics.usd \
         UniversalRobots/ur20/configuration/ur20_robot_schema.usd UniversalRobots/ur20/configuration/ur20_sensor.usd \
         Unitree/G1/g1.usd Unitree/G1/configuration/g1_29dof_with_hand_rev_1_0_base.usd \
         Unitree/G1/configuration/g1_29dof_with_hand_rev_1_0_physics.usd Unitree/G1/configuration/g1_29dof_with_hand_rev_1_0_physics_simplified.usd \
         Unitree/G1/configuration/g1_29dof_with_hand_rev_1_0_robot.usd Unitree/G1/configuration/g1_29dof_with_hand_rev_1_0_sensor.usd \
         Unitree/G1/configuration/g1_left_hand_three_finger_hand_physics_left.usd Unitree/G1/configuration/g1_right_hand_three_finger_hand_physics_right.usd \
         Unitree/G1/configuration/three_finger_hand/three_finger_hand_base_left.usd Unitree/G1/configuration/three_finger_hand/three_finger_hand_base_right.usd \
         Unitree/G1/configuration/Materials/Materials.usd \
         Fanuc/crx10ia/crx10ia.usd Fanuc/crx10ia/configuration/crx10ia_base.usd Fanuc/crx10ia/configuration/crx10ia_physics.usd \
         Fanuc/crx10ia/configuration/crx10ia_sensor.usd; do
  mkdir -p "$(dirname "$f")"; [ -s "$f" ] || curl -sf -o "$f" "$B/$f"
done
du -sh "$A"
