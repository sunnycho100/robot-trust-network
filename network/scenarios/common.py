"""Shared by every house: state folder, broker connection, event log."""
import json, os, threading, time
import paho.mqtt.client as mqtt
import protocol

STATE = os.environ.get("STATE", "/state")
protocol.KEYS_DIR = STATE  # each house only sees its own keys and registry
QOS = 1  # broker acks each publish; a redelivered copy is rejected as a replay (decisions D13)
CONNECT_S, SUBSCRIBE_S = 5.0, 10.0  # worst case before a request is even sent (decisions D30)


def connect(client_id, topics=(), on_message=None, keepalive=5, on_ready=None):
    """Connect, subscribe, and return only once the broker has confirmed the subscription.

    on_ready runs after every (re)subscription, so a house can log each time it is back online.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv5)
    client.reconnect_delay_set(min_delay=0.5, max_delay=2)  # come back quickly after a cut link
    client.connect_timeout = CONNECT_S
    subscribed = threading.Event()

    def on_subscribe(*_):
        subscribed.set()
        if on_ready:
            on_ready()

    client.on_connect = lambda c, *_: topics and c.subscribe([(t, QOS) for t in topics])
    client.on_subscribe = on_subscribe
    if on_message:
        client.on_message = on_message
    client.connect(os.environ.get("BROKER", "broker"), protocol.BROKER_PORT, keepalive=keepalive)
    client.loop_start()
    if topics and not subscribed.wait(SUBSCRIBE_S):
        raise OSError("broker never confirmed the subscription")
    return client


def log(event):
    """Append-only event log the scenario runner reads from outside the house."""
    event["at"] = time.time()
    with open(os.path.join(STATE, "events.jsonl"), "a") as f:
        f.write(json.dumps(event) + "\n")


def parse(payload):
    try:
        return json.loads(payload)
    except ValueError:
        return payload.decode(errors="replace")
