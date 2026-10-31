"""Language model over the trained walking policies: Claude turns a plain-English job into station routes, and
multi_robot_scene.py (fleet.Jobs) executes them with the stock RL policies. Usage:

    python task_plan.py "Bring two loads from the dock to shelf A, the Go2 checks shelf B" \\
        --robot h1 -4 0 0 --robot go2 0 -4 90 --robot anymal 4 -3 135 \\
        --station dock 0 0 3 --station shelfA 0 3 0 --station shelfB -3 3 0
    python task_plan.py --plan plan.json ...    # skip the API, validate a saved plan

Prints the multi_robot_scene.py arguments on stdout (paste after the script name, add --stream on Brev).
Needs the anthropic package and a credential (ANTHROPIC_API_KEY or `ant auth login`) unless --plan is given.
"""
import argparse
import json
import sys

MODEL = "claude-opus-5-5"
SCHEMA = {
    "type": "object",
    "properties": {
        "routes": {"type": "array", "items": {
            "type": "object",
            "properties": {"robot": {"type": "integer"}, "stations": {"type": "array", "items": {"type": "string"}}},
            "required": ["robot", "stations"], "additionalProperties": False}},
        "reason": {"type": "string"},
    },
    "required": ["routes", "reason"], "additionalProperties": False,
}


def ask(job, robots, stations):
    import anthropic
    prompt = (
        f"Job: {job}\n\nRobots (index, kind, x, y): {json.dumps([[i, k, x, y] for i, (k, x, y, _) in enumerate(robots)])}\n"
        f"Stations (name, x, y, dwell seconds): {json.dumps([[n, x, y, d] for n, (x, y, d) in stations.items()])}\n\n"
        "Plan which robot visits which stations, in order. Each robot walks from where it stands through its "
        "stations; a station holds one robot at a time, so others wait until it leaves. Lower robot index has "
        "right of way. Leave idle robots out. H1 is a humanoid, Go2 a small quadruped, ANYmal and Spot larger "
        "quadrupeds. Keep total time short. Give a one-sentence reason."
    )
    resp = anthropic.Anthropic().beta.messages.create(
        model=MODEL, max_tokens=4000, output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"},
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason != "end_turn":
        sys.exit(f"no plan: stop_reason {resp.stop_reason} {getattr(resp, 'stop_details', None)}")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def to_args(plan, robots, stations):
    """Validated multi_robot_scene.py arguments. Raises ValueError on a plan the fleet cannot run."""
    seen, stops = set(), []
    for r in plan["routes"]:
        i = r["robot"]
        if not 0 <= i < len(robots) or i in seen:
            raise ValueError(f"bad or repeated robot index {i}")
        seen.add(i)
        for name in r["stations"]:
            if name not in stations:
                raise ValueError(f"unknown station {name!r}")
            x, y, d = stations[name]
            stops.append(f"--stop {i} {x:g} {y:g} {d:g}")
    if not stops:
        raise ValueError("empty plan")
    # ponytail: fleet.Jobs has no deadlock check; two routes swapping stations would wait forever
    bots = [f"--robot {k} {x:g} {y:g} {yaw:g} 0 0 0" for k, x, y, yaw in robots]
    return " ".join(bots + stops)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job", nargs="?", default="")
    p.add_argument("--robot", nargs=4, action="append", required=True, metavar=("KIND", "X", "Y", "YAW"))
    p.add_argument("--station", nargs=4, action="append", required=True, metavar=("NAME", "X", "Y", "DWELL"))
    p.add_argument("--plan", help="saved plan JSON instead of calling the API")
    a = p.parse_args()
    robots = [(k, float(x), float(y), float(yaw)) for k, x, y, yaw in a.robot]
    stations = {n: (float(x), float(y), float(d)) for n, x, y, d in a.station}
    plan = json.load(open(a.plan)) if a.plan else ask(a.job, robots, stations)
    print(json.dumps(plan), file=sys.stderr)
    print(to_args(plan, robots, stations))


def demo():
    robots = [("h1", -4, 0, 0), ("go2", 0, -4, 90)]
    stations = {"dock": (0, 0, 3), "shelfA": (0, 3, 0)}
    ok = {"routes": [{"robot": 0, "stations": ["dock", "shelfA"]}, {"robot": 1, "stations": ["dock"]}], "reason": ""}
    assert to_args(ok, robots, stations) == ("--robot h1 -4 0 0 0 0 0 --robot go2 0 -4 90 0 0 0 "
                                             "--stop 0 0 0 3 --stop 0 0 3 0 --stop 1 0 0 3")
    for bad in ({"routes": [{"robot": 2, "stations": ["dock"]}]}, {"routes": [{"robot": 0, "stations": ["roof"]}]},
                {"routes": [{"robot": 0, "stations": ["dock"]}, {"robot": 0, "stations": ["shelfA"]}]}):
        try:
            to_args(bad, robots, stations)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    print("ok")


if __name__ == "__main__":
    demo() if len(sys.argv) == 1 else main()
