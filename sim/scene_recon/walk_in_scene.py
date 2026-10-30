"""Deploy the walking Unitree G1 in a reconstructed scene and score it against a bare floor.

    python walk_in_scene.py SCENE_DIR [--episodes 20] [--video walk.mp4]

SCENE_DIR is a build_scene.py output (scene.xml, scene_info.json). Each episode starts the G1 at the
spawn point, picks a random reachable waypoint and steers to it with the published unitree_rl_gym
walking policy (sim/mujoco_houses). The same waypoints are then walked on an empty plane. A robot that
falls more often in the scene than on the plane points at bad collision geometry, not the policy.
Needs torch and the G1 assets from sim/mujoco_houses (./scripts/fetch_assets).
"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mujoco_houses" / "src"))
from unitree_demo.g1_policy import G1Controller, gait_phase, load_g1_config  # noqa: E402

FALL_HEIGHT = 0.35  # pelvis less than this above the lower foot means the robot is down (works on slopes too)


def build(scene_xml, cfg, spawn, z=0.0):
    spec = mujoco.MjSpec.from_file(str(scene_xml)) if scene_xml else mujoco.MjSpec.from_string(
        '<mujoco><worldbody><geom name="floor" type="plane" size="0 0 0.05"/></worldbody></mujoco>')
    frame = spec.worldbody.add_frame(name="g1_spawn", pos=(spawn[0], spawn[1], z))
    spec.attach(mujoco.MjSpec.from_file(str(cfg.robot_xml)), prefix="g1_", frame=frame)
    model = spec.compile()
    model.opt.timestep = cfg.simulation_dt
    return model


def fallen(model, data, g1):
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"g1_{s}_ankle_roll_link") for s in ("left", "right")]
    return data.xpos[g1.body_id][2] - min(data.xpos[f][2] for f in feet) < FALL_HEIGHT


def episode(model, cfg, start_yaw, goal, seconds, frames=None, renderer=None, camera=None):
    data = mujoco.MjData(model)
    g1 = G1Controller(model, cfg, "g1_")
    g1.reset(data)
    data.qpos[g1.base_qpos + 3:g1.base_qpos + 7] = [math.cos(start_yaw / 2), 0, 0, math.sin(start_yaw / 2)]
    mujoco.mj_forward(model, data)
    steps, command = int(seconds / cfg.simulation_dt), np.zeros(3, dtype=np.float32)
    for k in range(steps):
        g1.apply(data)
        mujoco.mj_step(model, data)
        if k % cfg.control_decimation:
            continue
        to_goal = goal - g1.position_xy(data)
        if np.linalg.norm(to_goal) < 0.3:
            return "reached", data.time
        if fallen(model, data, g1):
            return "fell", data.time
        err = (math.atan2(to_goal[1], to_goal[0]) - g1.yaw(data) + math.pi) % (2 * math.pi) - math.pi
        command[:] = [0.4 if abs(err) < 0.5 else 0.05, 0.0, float(np.clip(1.5 * err, -0.6, 0.6))]
        g1.update_policy(data, command, gait_phase(data))
        if renderer is not None and k % (cfg.control_decimation * 2) == 0:  # 25 fps
            camera.lookat[:] = data.xpos[g1.body_id]
            renderer.update_scene(data, camera)
            frames.write(renderer.render().tobytes())
    return "timeout", data.time


def wall_runs(scene_xml, cfg, walls, seconds=6.0):
    """Start 1 m in front of each wall box, walk straight at it: it must touch and never pass through."""
    touched = passed = 0
    for i, wall in enumerate(walls):
        n = np.array([math.cos(wall["yaw"]), math.sin(wall["yaw"])])  # box x axis = wall normal
        center = np.array(wall["pos"][:2])
        for side in (1, -1):
            start = center + side * 1.0 * n
            model = build(scene_xml, cfg, start)
            data = mujoco.MjData(model)
            g1 = G1Controller(model, cfg, "g1_")
            g1.reset(data)
            yaw = math.atan2(-side * n[1], -side * n[0])
            data.qpos[g1.base_qpos + 3:g1.base_qpos + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            wall_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wall_{i:02d}")
            hit = False
            for k in range(int(seconds / cfg.simulation_dt)):
                g1.apply(data)
                mujoco.mj_step(model, data)
                if k % cfg.control_decimation == 0:
                    g1.update_policy(data, np.array([0.4, 0.0, 0.0], dtype=np.float32), gait_phase(data))
                hit |= any(wall_geom in (c.geom1, c.geom2) for c in data.contact[:data.ncon])
            through = side * ((g1.position_xy(data) - center) @ n) < 0  # ended on the far side of the wall
            touched += hit
            passed += bool(through)
    return {"runs": 2 * len(walls), "touched": int(touched), "passed_through": int(passed)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", type=Path)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--video", type=Path, help="record the first 3 scene episodes to this mp4")
    a = p.parse_args()
    cfg = load_g1_config()
    info = json.loads((a.scene / "scene_info.json").read_text())
    spawn = np.array(info["spawn_xy"])
    rng = np.random.default_rng(0)
    results = {}
    # Half the goals sit in open floor, half 15-30 cm from walls and clutter, so the proxies get tested. All are
    # 1.5 to 6 m from the spawn, reachable within an episode (the nearest ones if a big site has too few).
    goals = []
    for k in ("open", "near"):
        g = np.array(info["waypoints"][k])
        if len(g):
            d = np.linalg.norm(g - spawn, axis=1)
            ok = (d > 1.5) & (d < 6.0)
            goals.append(g[ok] if ok.sum() >= 5 else g[np.argsort(np.abs(d - 3.0))[:10]])
    trials = [(rng.uniform(-math.pi, math.pi), g[rng.integers(len(g))]) for g in goals for _ in range(a.episodes // len(goals))]
    for name, xml in (("scene", a.scene / "scene.xml"), ("bare plane", None)):
        model = build(xml, cfg, spawn, info.get("spawn_z", 0.0) if xml else 0.0)
        frames = renderer = camera = None
        if a.video and xml:
            renderer, camera = mujoco.Renderer(model, 480, 640), mujoco.MjvCamera()
            camera.distance, camera.elevation, camera.azimuth = 3.0, -25.0, 135.0
            frames = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480",
                                       "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(a.video)], stdin=subprocess.PIPE)
        outcomes = []
        for i, (yaw, goal) in enumerate(trials):
            record = frames is not None and i < 3
            outcomes.append(episode(model, cfg, yaw, goal, a.seconds, frames.stdin if record else None,
                                    renderer if record else None, camera)[0])
        if frames is not None:
            frames.stdin.close()
            frames.wait()
        results[name] = {k: outcomes.count(k) for k in ("reached", "fell", "timeout")}
        print(f"{name:10s}: {results[name]}", flush=True)
    results["walk into walls"] = wall_runs(a.scene / "scene.xml", cfg, info["walls"])
    print(f"walk into walls: {results['walk into walls']}", flush=True)
    (a.scene / "walk_results.json").write_text(json.dumps(results, indent=2))
    walls_ok = results["walk into walls"]["passed_through"] == 0
    verdict = "PASS" if results["scene"]["fell"] <= results["bare plane"]["fell"] and walls_ok else "FAIL"
    print(f"{verdict}: falls in scene {results['scene']['fell']} vs bare plane {results['bare plane']['fell']} over {a.episodes} episodes")


if __name__ == "__main__":
    main()
