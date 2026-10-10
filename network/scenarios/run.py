"""Scenario runner. Builds the households, runs every scenario, prints the table. Host side, stdlib only."""
import json, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE = ["docker", "compose", "-f", os.path.join(HERE, "compose.yml")]
RUN_DIR = os.path.join(HERE, ".run")
A, D, B, C = "robot-a@house-1", "robot-d@house-1", "robot-b@house-2", "robot-c@house-3"


def sh(*args, check=True):
    done = subprocess.run(args, capture_output=True, text=True)
    if check and done.returncode:
        raise RuntimeError(f"{' '.join(args[:6])} failed: {done.stderr.strip()[-500:]}")
    return done


def client(*args):
    """Run client.py in house 1 and return its JSON line."""
    out = sh(*COMPOSE, "exec", "-T", "house-1", "python", "client.py", *args).stdout
    return json.loads(out.strip().splitlines()[-1])


def container_ip(service):
    name = sh(*COMPOSE, "ps", "-q", service).stdout.strip()
    return sh("docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name).stdout.strip()


def events(house):
    path = os.path.join(RUN_DIR, house, "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def setup():
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    os.makedirs(RUN_DIR)
    sh("docker", "build", "-q", "-t", "trust-net-house", "-f", os.path.join(HERE, "Dockerfile"), os.path.dirname(HERE))
    sh("docker", "run", "--rm", "--network", "none", "-v", f"{RUN_DIR}:/out", "trust-net-house", "python", "keygen.py", "/out")
    sh(*COMPOSE, "up", "-d")
    for _ in range(50):  # wait until both receiving houses are online
        if all(any(e["kind"] == "online" for e in events(h)) for h in ("house-2", "house-3")):
            return
        time.sleep(0.2)
    raise RuntimeError("houses did not come online")


def teardown():
    sh(*COMPOSE, "down", "-t", "1", check=False)


# Each scenario returns (expected, got, passed).

def s11_isolation():
    direct = client("probe", container_ip("house-2"), "1883")
    by_name = client("probe", "house-2", "1883")
    broker = client("probe", "broker", "1883")
    via_broker = client("request", A, B, "status")
    got = (f"direct {'open' if direct['reachable'] else 'blocked'}, by name {'open' if by_name['reachable'] else 'blocked'}, "
           f"broker {'open' if broker['reachable'] else 'blocked'}, signed via broker {via_broker['result']}")
    passed = (not direct["reachable"] and not by_name["reachable"] and broker["reachable"]
              and via_broker["result"] != "TIMEOUT" and via_broker.get("reply_matches"))
    return "house 2 unreachable, broker path works", got, passed


SCENARIOS = [
    ("S11", "House 1 tries to reach house 2 directly", s11_isolation),
]


def main():
    try:
        setup()
        rows = []
        for sid, name, fn in SCENARIOS:
            try:
                expected, got, passed = fn()
            except Exception as e:  # a crashed scenario is a failed row, not a crashed run
                expected, got, passed = "-", f"error: {e!r}", False
            rows.append((sid, name, expected, got, passed))
            print(f"{'PASS' if passed else 'FAIL'}  {sid:<4} {name}\n      expected: {expected}\n      got:      {got}", flush=True)
        failed = [r for r in rows if not r[4]]
        print(f"\n{len(rows) - len(failed)} of {len(rows)} scenarios passed")
        return 1 if failed else 0
    finally:
        if "--keep" not in sys.argv:
            teardown()


if __name__ == "__main__":
    sys.exit(main())
