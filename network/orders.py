"""Order updates, after VDA 5050: the visiting robot's checks before it extends its route.

An order keeps one id. Each update carries a higher update id, comes from the house that released
the order, and its Base starts at the last released room (the stitching node), so a released part
can never be changed afterwards. Call only after Verifier.check has passed.
"""


def check_update(order, msg):
    """order is what the visitor holds: {"id", "by", "update_id", "base"}. Returns (ok, reason)."""
    if msg.get("from") != order["by"]:
        return False, "not from the house that released this order"
    if msg.get("order_id") != order["id"]:
        return False, "not this order"
    update_id = msg.get("order_update_id")
    if not isinstance(update_id, int) or update_id <= order["update_id"]:
        return False, "update id is not newer"
    base = msg.get("base")
    if not isinstance(base, list) or not base or base[0] != order["base"][-1]:
        return False, f"does not start at the decision point {order['base'][-1]}"
    return True, "ok"
