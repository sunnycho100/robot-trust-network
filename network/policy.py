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


def decide(message, trust_score):
    """Return ALLOW, PARTIAL or DENY. Call only after Verifier.check succeeds."""
    action = message.get("action")
    if not action:
        return Decision("DENY", "missing action")
    if action in NEVER_ALLOWED:
        return Decision("DENY", "blocked by owner safety rule")
    # A signed neighbor may look. Unlocking stays restricted at every trust score.
    if action == "verify_and_unlock":
        if trust_score >= MINIMUM_TRUST["verify_package"]:
            return Decision("PARTIAL", "may inspect the entrance, but may not unlock the door")
        return Decision("DENY", f"trust {trust_score} is below threshold {MINIMUM_TRUST['verify_package']}")
    needed = MINIMUM_TRUST.get(action)
    if needed is None:
        return Decision("DENY", "unknown action")
    if trust_score >= needed:
        return Decision("ALLOW", f"trust {trust_score} meets threshold {needed}")
    if action == "carry_together" and trust_score >= 40:
        return Decision("PARTIAL", "may follow the route, but may not handle the load")
    if action == "deliver_parcel" and trust_score >= 40:
        return Decision("PARTIAL", "may walk to the door, but may not move the parcel")
    return Decision("DENY", f"trust {trust_score} is below threshold {needed}")
