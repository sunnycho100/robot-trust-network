"""Checks for order updates: python test_orders.py"""
from orders import check_update

order = {"id": "o1", "by": "robot-b@house-2", "update_id": 0, "base": ["front_door", "living_room"]}
good = {"from": "robot-b@house-2", "order_id": "o1", "order_update_id": 1, "base": ["living_room", "bedroom"]}
assert check_update(order, good) == (True, "ok")
assert check_update(order, dict(good, order_id="o2"))[1] == "not this order"
assert check_update(order, dict(good, order_update_id=0))[1] == "update id is not newer"
assert check_update(order, dict(good, base=["front_door", "bedroom"]))[1] == "does not start at the decision point living_room"
assert check_update(order, dict(good, base=[]))[1] == "does not start at the decision point living_room"
assert check_update(order, dict(good, **{"from": "robot-c@house-3"}))[1] == "not from the house that released this order"
print("all passed")
