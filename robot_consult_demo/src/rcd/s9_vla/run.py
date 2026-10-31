"""s9: robot task plan ("video -> robot program draft") from the recommended allocation.

This is the step that SeeDo (Wang et al., "VLM See, Robot Do: Human Demo Video to Robot Action Plan via Vision
Language Model", arXiv:2410.08792) and Video-to-BT ("From Human Demonstration Videos to Behavior Trees", arXiv:2509.16611)
perform with a vision-language model: turn an observed human demonstration into an executable robot task plan /
behavior tree. Here NO VLA/VLM model is run (no GPU in this demo). The plan is rule-generated from the IE element
table: the robot-assigned element keys of the recommended s5 scenario, the s4 task-group program (targets from the
s1 hand-contact points, tool, tracking) and the s6a simulated timeline. This module is the plug-in point where a
VLA/VLM policy or a VLM-based plan generator would replace `steps_for_group`; the fixed primitive vocabulary and the
BehaviorTree.CPP v4 XML are the interface such a model would have to emit.

Inputs: out/s5/allocation.json (recommended scenario; if absent, a fixture derived from s4 TOPSIS rank 1 is used and
marked as such), out/s4/candidates.json, out/s6a/robot_task_times.json.
Output: out/s9/task_plan.json with, per station, flat `steps` (primitive, args, covers, est_s) and `bt_xml`.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from xml.dom import minidom

from rcd.common.io import ROOT, load_json, out_dir, save_json
from rcd.s4_catalog import process as P
from rcd.s4_catalog import scene as S

PRIMITIVES = ["move_j", "move_l", "track_conveyor_start", "track_conveyor_stop", "gripper_open", "gripper_close",
              "pick", "place", "screw", "capture_image", "wait_signal", "handover", "tool_change", "set_output"]
BT_NAME = {p: "".join(w.capitalize() for w in p.split("_")) for p in PRIMITIVES}  # move_j -> MoveJ
PORTS = {"move_j": ["target", "axis"], "move_l": ["target"], "track_conveyor_start": ["conveyor", "encoder"],
         "track_conveyor_stop": ["conveyor"], "gripper_open": ["width_mm"], "gripper_close": ["force_n"],
         "pick": ["object", "source"], "place": ["object", "target"], "screw": ["target", "torque_nm", "turns"],
         "capture_image": ["target", "check"], "wait_signal": ["signal", "timeout_s"], "handover": ["to", "reason"],
         "tool_change": ["tool"], "set_output": ["signal"]}
ORDER = {"pick_place": 0, "screwdriving": 1, "inspection": 2}


def _fixture_allocation(cand: dict, rtt: dict) -> dict:
    """Used only when out/s5/allocation.json is absent: TOPSIS rank-1 model of every group, all its keys to the robot."""
    have = {(e["model"], e["group_id"]) for e in rtt["entries"]}
    asg = []
    for g in cand["task_groups"]:
        best = [e for e in sorted((e for e in g["evaluated"] if e["passed"]), key=lambda e: e.get("rank") or 99)
                if (e["model"], g["group_id"]) in have]
        if best:
            asg += [{"element_key": k, "station": g["station"], "resource": best[0]["model"], "group_id": g["group_id"],
                     "time_s": 0.0} for k in g["element_keys"]]
    return {"scenarios": [{"name": "fixture_topsis_rank1", "recommended": True, "assignment": asg}]}


def steps_for_group(g: dict, sim_entry: dict | None, robot_keys: set[str]) -> tuple[list[dict], dict]:
    """Rule-based plan for one task group (the VLA/VLM plug-in point)."""
    G = {"geometry": g["geometry"], "targets_m": g["geometry"]["targets_m"], "process_type": g["process_type"],
         "tool": g["tool"], "program_est_s": g["geometry"]["program_est_s"], "mounting": g.get("mounting")}
    poses = S.task_poses(G, g["placement"]["base_xyz"])
    targets = {k: {"pos_m": [round(float(x), 4) for x in v[0]], "tool_axis": [round(float(x), 4) for x in v[1]],
                   "frame": "conveyor (tracked)" if v[2] else "world"} for k, v in poses.items()}
    ops = P.program(g["process_type"], len(g["geometry"]["targets_m"]), g["codes"])
    tl = (sim_entry or {}).get("timeline") or []
    st = g["station"]
    steps = []
    part = "screw" if g["process_type"] == "screwdriving" else "part"
    for i, op in enumerate(ops):
        k, a = op["op"], {}
        if k in ("move_j", "move_l"):
            prim, a = k, {"target": op["target"]}
        elif k == "track_start":
            prim, a = "track_conveyor_start", {"conveyor": f"{st}_line", "encoder": "line_encoder"}
        elif k == "track_stop":
            prim, a = "track_conveyor_stop", {"conveyor": f"{st}_line"}
        elif op["kind"] == "screw_pick":
            prim, a = "pick", {"object": "screw", "source": "screw_presenter"}
        elif op["kind"] == "rundown":
            prim, a = "screw", {"target": op["target"], "torque_nm": 3.0,
                                "turns": round(P.SCREW["engagement_mm"] / P.SCREW["thread_pitch_mm"], 1)}
        elif op["kind"] == "grip_close":
            prim, a = "pick", {"object": part, "source": "part_tray"}
        elif op["kind"] == "grip_open" and op.get("target"):
            prim, a = "place", {"object": part, "target": op["target"]}
        elif op["kind"] == "grip_open":
            prim, a = "gripper_open", {"width_mm": 85}
        else:
            prim, a = "capture_image", {"target": op.get("target"), "check": "presence_and_seating"}
        key = f"{st}:{op['code']}"
        steps.append({"primitive": prim, "args": a, "covers": [key] if key in robot_keys else [],
                      "est_s": tl[i]["dur_s"] if i < len(tl) else None, "group_id": g["group_id"]})
    if g.get("mounting") == "linear_track" and len(tl) > len(ops):
        steps.append({"primitive": "move_j", "args": {"target": "track_home", "axis": "linear_track"}, "covers": [],
                      "est_s": tl[-1]["dur_s"], "group_id": g["group_id"]})
    return steps, targets


def _bt_xml(tree_id: str, steps: list[dict]) -> str:
    root = ET.Element("root", BTCPP_format="4", main_tree_to_execute=tree_id)
    bt = ET.SubElement(root, "BehaviorTree", ID=tree_id)
    top = ET.SubElement(bt, "Sequence", name="cycle")
    cur_group, seq = None, top
    for s in steps:
        gid = s.get("group_id") or "station"
        if gid != cur_group:
            cur_group = gid
            seq = ET.SubElement(top, "Sequence", name=gid.replace(":", "_"))
        attrs = {k: str(v) for k, v in s["args"].items() if v is not None}
        attrs["name"] = s["id"]
        if s["primitive"] == "screw":  # retry once, then hand the screw over to the operator
            fb = ET.SubElement(seq, "Fallback", name=f"{s['id']}_or_handover")
            rt = ET.SubElement(fb, "RetryUntilSuccessful", num_attempts="2")
            ET.SubElement(rt, BT_NAME["screw"], **attrs)
            ET.SubElement(fb, BT_NAME["handover"], to="operator", reason="screw_fault", name=f"{s['id']}_handover")
        else:
            ET.SubElement(seq, BT_NAME[s["primitive"]], **attrs)
    model = ET.SubElement(root, "TreeNodesModel")
    for p in PRIMITIVES:
        a = ET.SubElement(model, "Action", ID=BT_NAME[p])
        for port in PORTS[p]:
            ET.SubElement(a, "input_port", name=port)
    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")


def run(cfg: dict) -> None:
    out = ROOT / cfg["out_root"]
    cand = load_json(out / "s4" / "candidates.json", contract="candidates")
    rtt = load_json(out / "s6a" / "robot_task_times.json", contract="robot_task_times")
    prov = {"s4/candidates.json": "real", "s6a/robot_task_times.json": "real"}
    p = out / "s5" / "allocation.json"
    if p.exists():
        al = load_json(p)
        prov["s5/allocation.json"] = "real"
    else:
        al = _fixture_allocation(cand, rtt)
        prov["s5/allocation.json"] = "fixture (derived from s4 TOPSIS rank 1; s5 output absent)"
    rec = next(s for s in al["scenarios"] if s.get("recommended"))
    groups = {g["group_id"]: g for g in cand["task_groups"]}
    key2group = {k: g["group_id"] for g in cand["task_groups"] for k in g["element_keys"]}
    sims = {(e["model"], e["group_id"]): e for e in rtt["entries"]}
    by_station: dict[str, list[dict]] = {}
    for a in rec["assignment"]:
        if a["resource"] != "human":
            by_station.setdefault(a["station"], []).append(a)
    plans = []
    for st, rows in sorted(by_station.items()):
        models = sorted({a["resource"] for a in rows})
        model = max(models, key=lambda m: sum(a["resource"] == m for a in rows))
        robot_keys = {a["element_key"] for a in rows}
        human_keys = sorted({a["element_key"] for a in rec["assignment"] if a["station"] == st and a["resource"] == "human"})
        gids = sorted({a.get("group_id") or key2group.get(a["element_key"]) for a in rows} - {None},
                      key=lambda gid: ORDER.get(groups[gid]["process_type"], 9) if gid in groups else 9)
        steps = [{"primitive": "wait_signal", "args": {"signal": "carrier_in_station", "timeout_s": cfg["line"]["takt_s"]},
                  "covers": [], "est_s": None, "group_id": "station"}]
        targets, tool = {}, None
        for gid in gids:
            g = groups.get(gid)
            if g is None:
                continue
            if tool is not None and tool != g["tool"]["name"]:
                steps.append({"primitive": "tool_change", "args": {"tool": g["tool"]["name"]}, "covers": [],
                              "est_s": None, "group_id": gid})
            tool = g["tool"]["name"]
            if human_keys and g["process_type"] != "pick_place":
                steps.append({"primitive": "wait_signal", "args": {"signal": "operator_ready", "timeout_s": 30},
                              "covers": [], "est_s": None, "group_id": gid})
            s, t = steps_for_group(g, sims.get((model, gid)), robot_keys)
            steps += s
            targets.update({f"{gid}/{k}": v for k, v in t.items()})
        steps.append({"primitive": "set_output", "args": {"signal": "robot_cycle_done"}, "covers": [], "est_s": None,
                      "group_id": "station"})
        if human_keys:
            steps.append({"primitive": "handover", "args": {"to": "operator", "reason": "remaining manual elements"},
                          "covers": [], "est_s": None, "group_id": "station"})
        covered = {k for s in steps for k in s["covers"]}
        for k in sorted(robot_keys - covered):  # robot key whose code the rule template does not name
            first = next((s for s in steps if s.get("group_id") == key2group.get(k)), steps[1])
            first["covers"].append(k)
            first.setdefault("note", "covers element(s) not named by the rule template")
        for i, s in enumerate(steps):
            s["id"] = f"{st}_s{i:03d}"
        tree_id = f"{st}_{model}".replace(" ", "_").replace("/", "_").replace(".", "_")
        est = [s["est_s"] for s in steps if s["est_s"] is not None]
        plans.append({"station": st, "model": model, "models_assigned": models, "groups": gids,
                      "robot_element_keys": sorted(robot_keys), "human_element_keys": human_keys,
                      "mounting": {gid: groups[gid].get("mounting") for gid in gids if gid in groups},
                      "steps": steps, "targets": targets, "est_cycle_s": round(sum(est), 2) if est else None,
                      "bt_xml": _bt_xml(tree_id, steps)})
        print(f"    {st}: {model}, {len(steps)} steps, covers {len({k for s in steps for k in s['covers']})}/{len(robot_keys)} robot keys")
    res = {"primitives": PRIMITIVES, "plans": plans, "scenario": rec["name"], "inputs": prov,
           "method": "rule-generated from the IE element table (s5 recommended assignment x s4 group programs x s6a "
                     "timeline); no VLA/VLM inference run. Plug-in point for a VLM plan generator as in SeeDo "
                     "(arXiv:2410.08792) / Video-to-BT (arXiv:2509.16611).",
           "bt_format": "BehaviorTree.CPP v4 XML (BTCPP_format=\"4\"); screw actions wrapped in Fallback(Retry x2, Handover)"}
    save_json(res, out_dir(cfg, "s9") / "task_plan.json", contract="task_plan")
