"""Small authorization layer used after cryptographic verification."""
from dataclasses import dataclass


NEVER_ALLOWED = {"unlock_door", "disable_camera", "share_private_video"}
MINIMUM_TRUST = {
    "status": 0,
    "verify_package": 30,
    "turn_on_light": 10,
    "mirror_motion": 30,
    "join_route": 40,
    "open_door": 50,
    "deliver_parcel": 60,
    "carry_together": 60,
}


@dataclass(frozen=True)
class Decision:
    result: str
    reason: str


def decide(message, trust_score, rules=None):
    """Return ALLOW, PARTIAL or DENY. Call only after Verifier.check succeeds.

    rules is one house owner's {"never": [...], "minimum": {action: score}}. It can only make
    the house stricter: extra forbidden actions, or higher thresholds than the defaults.
    """
    rules = rules or {}
    minimum = {a: max(t, rules.get("minimum", {}).get(a, t)) for a, t in MINIMUM_TRUST.items()}
    action = message.get("action")
    if not action:
        return Decision("DENY", "missing action")
    if action in NEVER_ALLOWED or action in rules.get("never", ()):
        return Decision("DENY", "blocked by owner safety rule")
    # A signed neighbor may look. Unlocking stays restricted at every trust score.
    if action == "verify_and_unlock":
        if trust_score >= minimum["verify_package"]:
            return Decision("PARTIAL", "may inspect the entrance, but may not unlock the door")
        return Decision("DENY", f"trust {trust_score} is below threshold {minimum['verify_package']}")
    needed = minimum.get(action)
    if needed is None:
        return Decision("DENY", "unknown action")
    if trust_score >= needed:
        return Decision("ALLOW", f"trust {trust_score} meets threshold {needed}")
    if action == "carry_together" and trust_score >= 40:
        return Decision("PARTIAL", "may follow the route, but may not handle the load")
    if action == "deliver_parcel" and trust_score >= 40:
        return Decision("PARTIAL", "may walk to the door, but may not move the parcel")
    return Decision("DENY", f"trust {trust_score} is below threshold {needed}")


@dataclass(frozen=True)
class Release:
    result: str
    base: list     # rooms the visitor may enter now, in order (VDA 5050 "Base", released)
    horizon: list  # rooms it asked for that wait for more trust or the owner (VDA 5050 "Horizon")
    reason: str
    closed: tuple = ()  # rooms the owner keeps closed; any of them denies the whole route


def release(route, trust_score, rooms, approved=()):
    """Split a requested route into a released Base and a waiting Horizon.

    rooms is the owner's map: room -> minimum trust, "owner" (needs the owner's approval) or
    "never". The Base is the longest start of the route the visitor may enter now; it ends at
    the decision point, where the visitor stops until an update releases more.
    """
    blocked = [room for room in route if rooms.get(room) == "never"]
    if blocked:
        return Release("DENY", [], [], f"{blocked[0]} is closed to visitors", tuple(blocked))
    base = []
    for room in route:
        rule = rooms.get(room, "owner")  # a room not on the owner's map waits for the owner
        allowed = room in approved if rule == "owner" else trust_score >= rule
        if not allowed:
            break
        base.append(room)
    horizon = route[len(base):]
    if not base:
        return Release("DENY", [], horizon, f"trust {trust_score} does not cover {route[0]}")
    if horizon:
        need = rooms.get(horizon[0], "owner")
        why = "the owner's approval" if need == "owner" else f"trust {need}"
        return Release("PARTIAL", base, horizon, f"stop at {base[-1]}: {horizon[0]} needs {why}")
    return Release("ALLOW", base, [], "whole route released")
