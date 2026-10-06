"""Checks for the authorization layer: python test_policy.py"""
from policy import decide


assert decide({"action": "mirror_motion"}, 70).result == "ALLOW"
assert decide({"action": "carry_together"}, 70).result == "ALLOW"
assert decide({"action": "carry_together"}, 45).result == "PARTIAL"
assert decide({"action": "carry_together"}, 10).result == "DENY"
assert decide({"action": "unlock_door"}, 100).result == "DENY"
assert decide({"action": "turn_on_light"}, 30).result == "ALLOW"
assert decide({"action": "open_door"}, 55).result == "ALLOW"
assert decide({"action": "open_door"}, 30).result == "DENY"
assert decide({"action": "carry_together"}, 55).result == "PARTIAL"
assert decide({"action": "deliver_parcel"}, 70).result == "ALLOW"
assert decide({"action": "deliver_parcel"}, 55).result == "PARTIAL"
assert decide({"action": "deliver_parcel"}, 30).result == "DENY"
assert decide({"action": "verify_package"}, 55).result == "ALLOW"
assert decide({"action": "verify_package"}, 10).result == "DENY"
assert decide({"action": "verify_and_unlock"}, 100).result == "PARTIAL"
assert decide({"action": "verify_and_unlock"}, 10).result == "DENY"
assert decide({"action": "not_registered"}, 100).result == "DENY"

print("all passed")
