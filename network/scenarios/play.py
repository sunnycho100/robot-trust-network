"""Talk to the households yourself. Run: ./play.sh  (type help inside)"""
import os, shlex, subprocess, sys, time
import run
from nl import ACTIONS, interpret, interpret_route, room_of
from run import A, B, C, D, COMPOSE, client, events, owner, sh, trust, wait_events

ROBOTS = {"a": A, "d": D, "b": B, "c": C}
HOUSE = {B: "house-2", C: "house-3"}
ATTACKS = ("forge", "unsigned", "tamper", "replay", "wrong_to")  # forge = spoof: claims to be you, wrong key
HELP = """
  any sentence       e.g.  could you come walk with me?   please let me in
                     a model turns it into one action, then it is sent signed with your sentence
  <action> [words]   or name the action yourself, e.g.  join_route   unlock_door   carry_together
  as a | as d        speak as robot A (trusted neighbor) or robot D (new, trust 0)
  as stranger        speak as robot-x, a robot nobody has paired with
  to b | to c        talk to robot B (house 2) or robot C (house 3, stricter owner)
  forge <action>     someone pretends to be you, signed with the wrong key
  unsigned <action>  your request with the signature removed
  tamper <action>    your real request, changed to unlock_door on the way
  replay <action>    your real request, sent twice
  wrong_to <action>  your real request meant for the other robot, bounced here
  task <words> [s]   start a task for s seconds (default 15) while you keep typing, e.g.  task walk with me 10
  route <rooms>      visit rooms in house 2, e.g.  route 현관 거실 안방   or  route 거실 지나서 안방까지 갈게
                     house 2 releases the Base you may enter now; you stop at the end of it (decision point)
  approve <room>     as house 2's owner, let the visitor into one more room: a signed update extends the Base
  fake_update        a stranger sends a forged update that opens every room
  bad_update         house 2's software sends a real update that skips the decision point
  replay_update      a real update captured off the broker, sent again
  map                open the route map page (green Base, grey Horizon, red closed)
  revoke | activate  as the receiving house's owner, revoke or restore the current speaker's key
  trust              every house's trust scores
  log                the last few things the receiving house did
  help | quit
""".rstrip()


def say(e):
    """One house log line in plain words."""
    who = (e.get("from") or e.get("robot") or "?").split("@")[0]
    if e["kind"] == "verdict":
        said = f" (\"{e['request']}\")" if e.get("request") and e["request"] != e.get("action") else ""
        return f"{who} asked {e.get('action')}{said}: {e['result']} ({e['reason']})"
    if e["kind"] == "task_started":
        return f"started {e['action']} for {who}"
    if e["kind"] == "task_stopped":
        return f"stopped {e['action']} for {who}: {e['reason']}, after {e['steps']} steps"
    if e["kind"] in ("revoked", "active"):
        return f"owner set {who} to {e['kind']}"
    return None


def understand(words):
    """(action, sentence) from what was typed: an action name as is, anything else through the model."""
    if not words:
        return "status", "status"
    if words[0] in ACTIONS:
        return words[0], " ".join(words)
    sentence = " ".join(words)
    action, how = interpret(sentence)
    print(f"  understood as {action or 'nothing a robot can do'} (by {how})")
    return action, sentence


KO = {"front_door": "현관", "living_room": "거실", "kitchen": "주방", "bedroom": "안방", "safe_room": "금고방"}


def follow(since, until, timeout):
    """Print what the visiting robot reports, live, until a note contains one of `until`."""
    seen, deadline = since, time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = events("house-1")[seen:]
        seen += len(new)
        for e in new:
            if e["kind"] != "visit":
                continue
            note = e["note"]
            for room, ko in KO.items():
                note = note.replace(room, f"{ko}")
            print(f"  [{e['status']}] {note}")
            if any(u in e["note"] for u in until):
                return e
        time.sleep(0.2)
    print("  (no news yet; it keeps going in the background)")


def is_up():
    return "house-2" in sh(*COMPOSE, "ps", "--services", "--status", "running", check=False).stdout


def verdict(house, nonce, copies=1):
    found = wait_events(house, nonce, copies)
    return found[-1] if found else None


def main():
    os.environ.update(HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()))
    if "--fresh" in sys.argv or not is_up():
        print("starting the households (about 20 s the first time) ...")
        run.setup()
    me, peer, tasks = A, B, []
    print(f"ready. You are {me}, talking to {peer}. Type help for commands.")
    while True:
        try:
            line = input(f"\n{me.split('@')[0]} -> {peer.split('@')[0]} > ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "quit"
        if not line:
            continue
        word, *rest = shlex.split(line)
        house = HOUSE[peer]
        if word in ("quit", "exit"):
            print("bye. The households keep running; ./play.sh picks them up again, docker compose down stops them.")
            return
        if word == "help":
            print(HELP)
        elif word == "as" and rest:
            me = "robot-x@house-9" if rest[0] == "stranger" else ROBOTS.get(rest[0], me)
            print(f"  you are now {me}")
        elif word == "to" and rest and rest[0] in ("b", "c"):
            peer = ROBOTS[rest[0]]
            print(f"  now talking to {peer}")
        elif word == "trust":
            for h, robot in (("house-2", B), ("house-3", C)):
                print(f"  {robot}: " + ", ".join(f"{r.split('@')[0]} {trust(h, r)}" for r in (A, D)))
        elif word == "log":
            lines = [say(e) for e in events(house)]
            for text in [t for t in lines if t][-6:]:
                print(f"  {text}")
        elif word in ("revoke", "activate"):
            if me not in (A, D):
                print("  only paired robots can be revoked")
                continue
            since = len(events(house))
            owner(house, word, me)
            print(f"  {house}'s owner {word}d {me}")
            stopped = word == "revoke" and run.wait_event(house, "task_stopped", since, timeout=1.5, **{"from": me})
            if stopped:
                print(f"  {peer.split('@')[0]} {say(stopped)}")
        elif word == "map":
            subprocess.run(["open", os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.html")])
            print("  opened the route map in your browser")
        elif word == "route" and rest:
            if me not in (A, D) or peer != B:
                print("  visits are set up for robot A or D visiting house 2 (as a, to b)")
                continue
            rooms = [room_of(w) for w in rest]
            if all(rooms):
                how = None
            else:
                rooms, how = interpret_route(" ".join(rest))
                print(f"  understood the route as {' > '.join(KO.get(r, r) for r in rooms) or 'no rooms'} (by {how})")
                if not rooms:
                    continue
            sh(*COMPOSE, "exec", "-T", "house-1", "pkill", "-f", "client.py visit", check=False)  # one visit at a time
            since = len(events("house-1"))
            tasks.append(run.background_client("visit", me, peer, *rooms, "--pace", "1.5", "--seconds", "180"))
            follow(since, ("waiting at", "arrived", "DENY"), timeout=20)
        elif word == "approve" and rest:
            room = room_of(rest[0]) or rest[0]
            since = len(events("house-1"))
            owner("house-2", "approve", me, room)
            print(f"  house 2's owner approved {KO.get(room, room)} for {me.split('@')[0]}")
            follow(since, ("waiting at", "arrived", "rejected"), timeout=15)
        elif word in ("fake_update", "replay_update", "bad_update"):
            since = len(events("house-1"))
            if word == "bad_update":
                owner("house-2", "bad_update", me)
            else:
                sent = client("route_attack", word, me, peer)
                if not sent.get("sent"):
                    print(f"  {sent['note']}")
                    continue
            follow(since, ("rejected", "accepted"), timeout=6)
        elif word == "task" and rest:
            if me not in (A, D):
                print("  a stranger cannot hold a task: its requests never pass the checks")
                continue
            seconds = rest.pop() if len(rest) > 1 and rest[-1].isdigit() else "15"
            action, text = understand(rest)
            if not action:
                continue
            since = len(events(house))
            tasks.append(run.background_client("task", me, peer, action, "--seconds", seconds))
            started = run.wait_event(house, "task_started", since, timeout=6, **{"from": me})
            if started:
                print(f"  {peer.split('@')[0]} is doing {action} for you, {seconds} s of signed steps. Try revoke, then log.")
            else:
                e = run.last_event(house, "verdict", since, **{"from": me})
                print(f"  not started: " + (f"{e['result']} ({e['reason']})" if e else "no answer"))
        elif word in ATTACKS:
            if me not in (A, D):
                print("  switch to as a or as d first: these attacks misuse a real robot's name")
                continue
            action, text = understand(rest)
            if not action:
                continue
            other = C if peer == B else B
            sent = client("attack", "spoof" if word == "forge" else word, me, peer, other, "--action", action, "--text", text)
            e = verdict(house, sent["nonce"], sent["copies"])
            print(f"  no reply (rejected messages get silence). {peer.split('@')[0]}'s log: "
                  + (f"{e['result']} ({e['reason']})" if e else "nothing logged"))
        elif me == "robot-x@house-9":
            action, text = understand(shlex.split(line))
            if not action:
                continue
            sent = client("attack", "stranger", A, peer, C, "--action", action, "--text", text)
            e = verdict(house, sent["nonce"])
            print(f"  no reply. {peer.split('@')[0]}'s log: " + (f"{e['result']} ({e['reason']})" if e else "nothing logged"))
        else:
            action, text = understand(shlex.split(line))
            if not action:
                continue
            before = trust(house, me)
            r = client("request", me, peer, action, "--text", text)
            after = trust(house, me)
            change = f"{before} -> {after}" if after != before else f"{after}, unchanged"
            if r["result"] in ("TIMEOUT", "BAD_REPLY"):
                print(f"  {r['result']}: {r['reason']}")
            else:
                print(f"  {peer.split('@')[0]} says: {r['result']} ({r['reason']}), signed reply checked")
            print(f"  {me.split('@')[0]}'s trust at {peer.split('@')[0]}: {change}")


if __name__ == "__main__":
    main()
