"""Generates the JSON Schemas in this folder. Edit here, then run: python contracts/_build_schemas.py"""
import json, pathlib
H = pathlib.Path(__file__).parent
num = {"type": "number"}; s = {"type": "string"}; i = {"type": "integer"}; b = {"type": "boolean"}
def obj(req, props=None, extra=True):
    return {"type": "object", "required": req, "properties": props or {}, "additionalProperties": extra}
def arr(item, mn=0): return {"type": "array", "items": item, "minItems": mn}
vec3 = arr(num, 3)

schemas = {
# ---------- S0: calibration (owner: 3D reconstruction teammate) ----------
"calibration": obj(["world_frame", "cameras"], {
    "world_frame": obj(["units", "up_axis", "origin_desc"]),
    "cameras": arr(obj(["id", "station", "side", "image_size", "K", "dist", "R", "t", "source", "reproj_rmse_px"], {
        "id": s, "station": s, "side": {"enum": ["in", "out"]}, "image_size": arr(i, 2),
        "K": arr(arr(num, 3), 3), "dist": arr(num), "R": arr(arr(num, 3), 3), "t": vec3,
        "source": s, "reproj_rmse_px": {"type": ["number", "null"]}}), 1)}),
# ---------- S1: scene graph (owner: 3D reconstruction teammate) ----------
"scene_graph": obj(["units", "stations"], {
    "units": {"const": "m"},
    "stations": arr(obj(["station", "cycles", "zones"], {
        "station": s,
        "zones": arr(obj(["name", "polygon_xy"])),
        "cycles": arr(obj(["cycle_id", "duration_s", "video", "human", "conveyor", "objects"], {
            "cycle_id": s, "duration_s": num,
            "video": obj(["in", "out", "fps"]),
            "human": obj(["joints_npz", "joint_names", "fps", "source"]),
            "conveyor": obj(["speed_mm_s", "method"]),
            "objects": arr(obj(["name", "positions_m"]))}), 1)}), 1)}),
# ---------- S2: work elements (owner: industrial engineer) ----------
"elements": obj(["taxonomy", "cycles"], {
    "taxonomy": arr(obj(["code", "name", "therblig", "description"]), 4),
    "cycles": arr(obj(["station", "cycle_id", "duration_s", "elements"], {
        "elements": arr(obj(["id", "code", "start_s", "end_s", "duration_s", "source", "confidence",
                             "hands", "travel_m", "trunk_flex_deg_mean", "subgoal"]), 1)}), 1)}),
"ergonomics": obj(["method", "cycles", "validation"], {
    "cycles": arr(obj(["station", "cycle_id", "postures_s"])),
    "validation": obj(["metric", "value", "n", "pairs"])}),
"time_study": obj(["params", "element_stats", "station_stats"], {
    "element_stats": arr(obj(["station", "code", "n", "mean_s", "std_s", "normal_s", "standard_s", "n_required"])),
    "station_stats": arr(obj(["station", "n_cycles", "observed_mean_s", "standard_s", "takt_s"]))}),
# ---------- S3: automatability (owner: industrial engineer) ----------
"automatability": obj(["criteria", "tasks"], {
    "criteria": arr(obj(["key", "weight", "description", "source"]), 6),
    "tasks": arr(obj(["station", "code", "scores", "total", "category", "rationale", "allocation"]), 1)}),
# ---------- S4: robot catalog + candidates (owner: robot simulation teammate) ----------
"robot_catalog": obj(["robots"], {
    "robots": arr(obj(["model", "maker", "type", "payload_kg", "reach_mm", "repeatability_mm", "weight_kg",
                       "dof", "joint_speed_deg_s", "collaborative", "conveyor_tracking", "price_usd_range",
                       "sim_model", "source_urls"]), 10)}),
"candidates": obj(["task_groups"], {
    "task_groups": arr(obj(["group_id", "station", "codes", "category", "requirements", "evaluated"], {
        "evaluated": arr(obj(["model", "passed", "reasons", "reachability"]))}), 1)}),
# ---------- S6A: robot motion simulation (owner: robot simulation teammate) ----------
"robot_task_times": obj(["method", "entries"], {
    "entries": arr(obj(["model", "group_id", "station", "cycle_time_s", "breakdown", "ik_success_rate", "sim"]), 1)}),
# ---------- S5: allocation optimisation (owner: industrial engineer) ----------
"allocation": obj(["solver", "scenarios"], {
    "scenarios": arr(obj(["name", "status", "objective", "robots", "assignment", "station_times", "checks"]), 3)}),
# ---------- S6B: discrete-event simulation (owner: industrial engineer) ----------
"des_results": obj(["replications", "scenarios", "validation"], {
    "scenarios": arr(obj(["name", "throughput_per_hr", "ci95_half_width", "utilization", "takt_violations_pct"]), 3)}),
# ---------- S7: ROI (owner: economics teammate) ----------
"roi": obj(["assumptions", "scenarios", "sensitivity"], {
    "assumptions": arr(obj(["name", "value", "unit", "source"]), 5),
    "scenarios": arr(obj(["name", "capex_usd", "annual_savings_usd", "payback_years", "npv_usd"]), 1),
    "sensitivity": arr(obj(["param", "low", "high", "npv_low", "npv_high"]), 3)}),
# ---------- S5X: design-space exploration (owner: industrial engineer) ----------
"design_space": obj(["factors", "criteria", "counts", "designs", "personas", "featured_comparisons"], {
    "criteria": arr(obj(["key", "label", "sense", "unit"]), 2),
    "counts": obj(["factor_runs", "unique_designs", "screen_pareto", "simulated", "final_pareto"]),
    "designs": arr(obj(["id", "label", "arms", "robots", "metrics", "pareto_screen", "dominated_by"], {
        "metrics": obj(["npv_usd", "capex_usd", "awkward_s", "max_cycle_s", "n_robots"])}), 2),
    "personas": arr(obj(["name", "weights", "best", "top3"]), 1),
    "featured_comparisons": arr(obj(["a", "b", "rows", "verdict"]))}),
# ---------- S9: robot task plan (owner: robot simulation teammate) ----------
"task_plan": obj(["primitives", "plans"], {
    "plans": arr(obj(["station", "model", "steps", "bt_xml"]), 1)}),
}
for name, sch in schemas.items():
    sch = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": name, **sch}
    (H / f"{name}.schema.json").write_text(json.dumps(sch, indent=2))
print("wrote", len(schemas), "schemas")
