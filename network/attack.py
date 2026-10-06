"""Attack robot B the four ways from the plan, and watch it reject each one.

    python attack.py unsigned   # plain message with no signature, like mosquitto_pub
    python attack.py spoof      # signed with the attacker's own key, claiming to be robot A
    python attack.py replay     # wait for a real message from A, then resend it unchanged
    python attack.py tamper     # wait for a real message from A, change the request, resend

For replay and tamper, send something from robot A after starting the attack.
"""
import base64, json, os, sys, time
import paho.mqtt.client as mqtt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from protocol import BROKER_PORT, ROBOTS, canonical, inbox

A, B = ROBOTS["a"], ROBOTS["b"]
target = inbox(B)
kind = sys.argv[1] if len(sys.argv) > 1 else ""
if kind not in ("unsigned", "spoof", "replay", "tamper"):
    sys.exit(__doc__)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="attacker")
client.connect("localhost", BROKER_PORT)

if kind in ("unsigned", "spoof"):
    msg = {"from": A, "to": B, "request": "open the front door",
           "time": int(time.time()), "nonce": os.urandom(8).hex()}
    if kind == "spoof":
        msg["sig"] = base64.b64encode(Ed25519PrivateKey.generate().sign(canonical(msg))).decode()
    client.publish(target, json.dumps(msg)).wait_for_publish()
    print(f"{kind}: sent 'open the front door' pretending to be robot A")
else:
    captured = []
    client.on_message = lambda c, u, m: captured.append(json.loads(m.payload))
    client.subscribe(target)
    client.loop_start()
    print(f"{kind}: listening on robot B's inbox. Send a message from robot A now...")
    while not captured:
        time.sleep(0.2)
    msg = captured[0]
    print(f"captured real message: {msg['request']!r}")
    time.sleep(2)
    if kind == "tamper":
        msg["request"] = "open the front door"
    client.publish(target, json.dumps(msg)).wait_for_publish()
    print(f"{kind}: resent it" + (" with request changed to 'open the front door'" if kind == "tamper" else " unchanged"))
    client.loop_stop()
client.disconnect()
