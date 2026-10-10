"""Shared by every house: state folder, broker connection, event log."""
import json, os, time
import paho.mqtt.client as mqtt
import protocol

STATE = os.environ.get("STATE", "/state")
protocol.KEYS_DIR = STATE  # each house only sees its own keys and registry
QOS = 1  # broker acks each publish; a redelivered copy is rejected as a replay (decisions D13)


def connect(client_id, topics=(), on_message=None, keepalive=5):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv5)
    client.on_connect = lambda c, *_: topics and c.subscribe([(t, QOS) for t in topics])
    if on_message:
        client.on_message = on_message
    client.connect(os.environ.get("BROKER", "broker"), protocol.BROKER_PORT, keepalive=keepalive)
    client.loop_start()
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
