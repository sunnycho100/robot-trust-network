"""Scenario runner. Builds the households, runs every scenario, prints the table. Host side, stdlib only."""
import json, os, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
COMPOSE = ["docker", "compose", "-f", os.path.join(HERE, "compose.yml")]
RUN_DIR = os.path.join(HERE, ".run")
A, D, B, C = "robot-a@house-1", "robot-d@house-1", "robot-b@house-2", "robot-c@house-3"
TALLY = []  # (should_allow, was_allowed) for every request a scenario judged


def judge(should_allow, result):
    TALLY.append((should_allow, result == "ALLOW"))


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


def wait_events(house, nonce, count=1, timeout=3.0):
    """The verdict events a house logged for one message nonce."""
    deadline = time.monotonic() + timeout
    while True:
        found = [e for e in events(house) if e.get("nonce") == nonce]
        if len(found) >= count or time.monotonic() > deadline:
            return found
        time.sleep(0.1)


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

def background_client(*args):
    return subprocess.Popen([*COMPOSE, "exec", "-T", "house-1", "python", "client.py", *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(proc, timeout=20):
    out, _ = proc.communicate(timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def last_event(house, kind, since=0, **match):
    """Newest event of this kind among those logged after position `since` in the house log."""
    found = [e for e in events(house)[since:] if e["kind"] == kind
             and all(e.get(k) == v for k, v in match.items())]
    return found[-1] if found else None


def wait_event(house, kind, since=0, timeout=5.0, **match):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = last_event(house, kind, since, **match)
        if found:
            return found
        time.sleep(0.05)
    return None


def owner(house, command, robot):
    sh(*COMPOSE, "exec", "-T", house, "python", "owner.py", command, robot)


def s8_revoked_mid_task():
    since = len(events("house-2"))
    proc = background_client("task", A, B, "join_route", "--seconds", "6")
    started = wait_event("house-2", "task_started", since, **{"from": A})
    time.sleep(1.0)
    owner("house-2", "revoke", A)
    revoked = wait_event("house-2", "revoked", since)
    stopped = started and wait_event("house-2", "task_stopped", since, task=started["task"])
    finish(proc)
    owner("house-2", "activate", A)  # the owner pairs A again for the scenarios after this one
    if not (started and revoked and stopped):
        return "stops within 1 s of revocation", f"started {bool(started)}, revoked {bool(revoked)}, stopped {stopped}", False
    delay = stopped["at"] - revoked["at"]
    return ("robot B stops within 1 s of revocation",
            f"stopped {delay * 1000:.0f} ms after revocation ({stopped['reason']}, {stopped['steps']} steps acted on)",
            stopped["reason"] == "sender key revoked" and 0 <= delay <= 1.0)


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


def trust(house, robot):
    """Current score, or the owner's starting score if the house has not changed anything yet."""
    for name in ("trust.json", "rules.json"):
        path = os.path.join(RUN_DIR, house, name)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return data.get("trust", data).get(robot, 0)
    return 0


def short(r):
    return f"{r['result']} ({r['reason']})"


def s1_neighbor_allowed():
    r = client("request", A, B, "join_route")
    judge(True, r["result"])
    return ("ALLOW, signed reply to this exact request",
            f"{short(r)}, reply matches request: {r.get('reply_matches')}",
            r["result"] == "ALLOW" and r.get("reply_matches") is True)


def s2_owners_differ():
    b, c = client("request", A, B, "join_route"), client("request", A, C, "join_route")
    judge(True, b["result"])
    judge(False, c["result"])
    return ("house 2 ALLOW, house 3 PARTIAL or DENY", f"house 2 {short(b)}, house 3 {short(c)}",
            b["result"] == "ALLOW" and c["result"] in ("PARTIAL", "DENY"))


def s3_forbidden_at_full_trust():
    before = trust("house-3", A)  # the owner starts A at 100
    r = client("request", A, C, "unlock_door")
    judge(False, r["result"])
    after = trust("house-3", A)
    return ("DENY even at trust 100, and A loses trust for asking",
            f"{short(r)}, A's trust {before} -> {after}", r["result"] == "DENY" and after < before)


def s4_forged_messages():
    expected = {"unsigned": "no signature", "spoof": "bad signature", "tamper": "bad signature",
                "wrong_to": "not addressed to me", "replay": "replay"}
    got = {}
    for kind in expected:
        sent = client("attack", kind, A, B, C)
        found = wait_events("house-2", sent["nonce"], sent["copies"])
        last = found[-1] if found else None  # for replay, the second copy is the attack
        got[kind] = last["reason"] if last else "no verdict"
        judge(False, "ALLOW" if last and last["ok"] else "REJECTED")
    return ("all REJECTED: " + ", ".join(f"{k} {v}" for k, v in expected.items()),
            ", ".join(f"{k} {v}" for k, v in got.items()), got == expected)


def s5_new_robot_information_only():
    info, light = client("request", D, B, "status"), client("request", D, B, "turn_on_light")
    judge(True, info["result"])
    judge(False, light["result"])
    return ("status ALLOW, turn_on_light DENY", f"status {short(info)}, turn_on_light {short(light)}",
            info["result"] == "ALLOW" and light["result"] == "DENY")


def s6_trust_is_earned():
    before = client("request", D, B, "carry_together")
    judge(False, before["result"])
    start, clean = trust("house-2", D), 0
    while clean < 30:
        r = client("request", D, B, "status")
        judge(True, r["result"])
        clean += 1
        if trust("house-2", D) >= 60:
            break
        time.sleep(1.0)  # house 2 counts at most one gain per second
    after = client("request", D, B, "carry_together")
    judge(True, after["result"])
    return ("carry_together DENY, then ALLOW after clean interactions",
            f"{short(before)}, trust {start} -> {trust('house-2', D)} after {clean} clean requests, then {short(after)}",
            before["result"] == "DENY" and after["result"] == "ALLOW")


def s7_forgery_does_not_hurt_victim():
    before = trust("house-2", A)
    for kind in ("unsigned", "spoof", "tamper", "unsigned", "spoof"):
        sent = client("attack", kind, A, B, C)
        wait_events("house-2", sent["nonce"])
        judge(False, "REJECTED")
    after_attacks = trust("house-2", A)
    still = client("request", A, B, "join_route")
    judge(True, still["result"])
    return ("A's trust unchanged, A still allowed",
            f"A's trust {before} -> {after_attacks} after 5 forgeries, then {short(still)}",
            after_attacks == before and still["result"] == "ALLOW")


SCENARIOS = [
    ("S1", "Known neighbor asks within its trust", s1_neighbor_allowed),
    ("S2", "Same request, stricter house", s2_owners_differ),
    ("S3", "Forbidden action at maximum trust", s3_forbidden_at_full_trust),
    ("S4", "Unsigned, spoofed, tampered, wrong to, replayed", s4_forged_messages),
    ("S5", "New robot asks for information", s5_new_robot_information_only),
    ("S6", "New robot earns help-level access", s6_trust_is_earned),
    ("S7", "Forged messages claiming to be robot A", s7_forgery_does_not_hurt_victim),
    ("S8", "Robot A's key revoked mid-task", s8_revoked_mid_task),
    ("S11", "House 1 tries to reach house 2 directly", s11_isolation),
]


def numbers():
    blocked = [allowed for should, allowed in TALLY if not should]
    allowed = [allowed for should, allowed in TALLY if should]
    print(f"unsafe requests blocked: {blocked.count(False)} of {len(blocked)}")
    print(f"safe requests allowed:   {allowed.count(True)} of {len(allowed)}")
    costs = [e["verify_ms"] + e.get("decide_ms", 0) for h in ("house-2", "house-3")
             for e in events(h) if e["kind"] == "verdict" and e["ok"]]
    if len(costs) >= 2:
        p95 = statistics.quantiles(costs, n=20)[-1]
        print(f"verify and decide time:  median {statistics.median(costs):.2f} ms, p95 {p95:.2f} ms "
              f"over {len(costs)} accepted messages")


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
        print()
        numbers()
        print(f"\n{len(rows) - len(failed)} of {len(rows)} scenarios passed")
        return 1 if failed else 0
    finally:
        if "--keep" not in sys.argv:
            teardown()


if __name__ == "__main__":
    sys.exit(main())
