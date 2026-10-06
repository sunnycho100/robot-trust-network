"""Run one robot: it listens on its inbox, verifies every request, and sends what you type.

    python robot.py a     # robot A, house 1
    python robot.py b     # robot B, house 2

Every verdict is also published to verdicts/<house>/<robot> so the link monitor can show it.
"""
import json, sys
import paho.mqtt.client as mqtt
from protocol import BROKER_PORT, ROBOTS, Verifier, inbox, load_private_key, load_registry, make_message

me_short = sys.argv[1] if len(sys.argv) > 1 else ""
if me_short not in ROBOTS:
    sys.exit("usage: python robot.py a|b")
me = ROBOTS[me_short]
peer = next(r for k, r in ROBOTS.items() if k != me_short)
name = me.split("@")[0]
verdict_topic = inbox(me).replace("requests/", "verdicts/", 1)

private_key = load_private_key(me)
verifier = Verifier(me, load_registry())


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(inbox(me))
    print(f"{name} online, listening on {inbox(me)}. Type a message for {peer.split('@')[0]} and press Enter.")


def on_message(client, userdata, m):
    try:
        msg = json.loads(m.payload)
    except ValueError:
        msg = m.payload.decode(errors="replace")
    ok, reason = verifier.check(msg)
    claimed = msg.get("from", "?") if isinstance(msg, dict) else "?"
    text = msg.get("request", "") if isinstance(msg, dict) else msg
    print(f"\n  {'ACCEPTED' if ok else 'REJECTED'} ({reason}) from {claimed}: {text}")
    client.publish(verdict_topic, json.dumps({"robot": me, "from": claimed, "request": text,
                                              "result": "ACCEPTED" if ok else "REJECTED", "reason": reason}))


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=name)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", BROKER_PORT)
client.loop_start()

try:
    for line in sys.stdin:
        if line.strip():
            client.publish(inbox(peer), json.dumps(make_message(private_key, me, peer, line.strip()), ensure_ascii=False))
            print(f"  sent (signed) to {peer.split('@')[0]}")
except KeyboardInterrupt:
    pass
client.disconnect()  # disconnect first, so loop_stop does not wait on a live connection
client.loop_stop()
print(f"\n{name} offline")
